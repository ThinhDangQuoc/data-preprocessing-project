#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess user parquet chunks (Task 1 -> Task 4).

Supports:
  - Single file: --parquet /path/to/sales_pers.user_chunk_0.parquet
  - Directory:   --dataset_dir /path --pattern "sales_pers.user_chunk_*.parquet"

Pipeline:
  Task 1: Drop unneeded/meta columns; prefer `user_id` over `customer_id`
  Task 2: NULL handling (objects -> 'Unknown', numerics -> median)
  Task 3: Redundancy checks; convert numeric-coded categoricals to categorical
  Task 4: Normalize numeric (except IDs/timestamps/date cols) & encode categoricals
          - OHE for low-card (<= threshold)
          - Frequency encoding for high-card

Outputs (in --out_dir):
  - <same_basename>.parquet           (processed per-file)
  - user_transformers.joblib          (global scaler, OHE list, freq mappings)
  - user_drop_report.json
  - user_cardinality.json
  - user_corr_report.csv              (optional; last file scanned)

Usage examples:
  # Single file
  python preprocess_user.py \
    --parquet /content/data/sales_pers.user_chunk_0.parquet \
    --out_dir /content/out_user \
    --write_corr

  # Directory (consistent encoders across all chunks)
  python preprocess_user.py \
    --dataset_dir /content/data \
    --pattern "sales_pers.user_chunk_*.parquet" \
    --out_dir /content/out_user \
    --concat_all \
    --write_corr
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --------------------------
# Config
# --------------------------

LOW_CARD_DEFAULT_THRESHOLD = 20

# Drop these system/meta columns if present
USER_DROP_ALWAYS = [
    "sync_error_message",
    "sync_status_id",
    "last_sync_date",
    "updated_date",
    "timestamp",
    "is_deleted",
]

# We keep created_date for future featurization (recency), but exclude from scaling/encoding in this script.
OBJ_EXCLUDE = []  # add any free-text columns you want to keep raw

# --------------------------
# Helpers
# --------------------------

def read_parquet(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p)

def is_id_or_ts_or_date(col: str) -> bool:
    c = col.lower()
    return (
        c.endswith("_id")             # user_id, customer_id, location_id
        or c == "user_id"
        or c == "customer_id"
        or "timestamp" in c
        or c == "created_date"
        or c == "install_date"        # often epoch; we won't scale here
    )

def split_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return num_cols, obj_cols

def clip_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    if series.isna().all():
        return series
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lower, upper)

def looks_like_user_file(path: Path, pattern_glob: str) -> bool:
    name = path.name.lower()
    return ("user_chunk" in name) and path.match(pattern_glob)

def detect_numeric_categoricals(df: pd.DataFrame, max_ratio: float = 0.001, max_abs: int = 64) -> List[str]:
    """
    Heuristic: numeric columns that behave like categorical codes (e.g., 'location').
    - if nunique / rows <= max_ratio OR nunique <= max_abs  => treat as categorical
    """
    candidates = []
    n = max(1, len(df))
    for c in df.select_dtypes(include=[np.number]).columns:
        if is_id_or_ts_or_date(c):
            continue
        k = df[c].nunique(dropna=False)
        if (k / n) <= max_ratio or k <= max_abs:
            candidates.append(c)
    return candidates

def one_hot_low_card(df: pd.DataFrame, ohe_cols: List[str]) -> pd.DataFrame:
    if not ohe_cols:
        return df
    return pd.get_dummies(df, columns=ohe_cols, prefix=[f"{c}_ohe" for c in ohe_cols], dummy_na=False)

def apply_frequency_mapping(series: pd.Series, mapping: Dict[str, float]) -> pd.Series:
    return series.astype(str).map(mapping).fillna(0.0)

# --------------------------
# Core per-file cleaning (Tasks 1 & 2 & 3 prep)
# --------------------------

def clean_user_file(df: pd.DataFrame, iqr_factor: float, drop_report: Dict) -> pd.DataFrame:
    # Task 1: drop system/meta
    to_drop = [c for c in USER_DROP_ALWAYS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
        drop_report.setdefault("dropped_system_meta", []).extend(to_drop)

    # Prefer user_id over customer_id (keep user_id for joins with purchases)
    if "customer_id" in df.columns and "user_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # Task 2: NULL handling
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna("Unknown")
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Task 2: Outlier capping (mostly not needed here; clip small numeric fields if present)
    for c in df.select_dtypes(include=[np.number]).columns:
        if is_id_or_ts_or_date(c):
            continue
        # light capping in case of weird spikes on numeric-coded columns
        df[c] = clip_iqr(df[c], factor=iqr_factor)

    # Task 3 prep: convert numeric-coded categoricals to object for proper encoding
    num_as_cat_cols = detect_numeric_categoricals(df)
    if num_as_cat_cols:
        drop_report.setdefault("numeric_as_categorical", {})["detected"] = num_as_cat_cols
        for c in num_as_cat_cols:
            df[c] = df[c].astype("Int64").astype(str)  # preserve codes as strings

    return df

def update_global_cardinality_and_freq(
    df: pd.DataFrame,
    global_cardinality: Dict[str, int],
    global_freq_counts: Dict[str, Dict[str, int]],
    obj_cols_exclude: List[str],
    id_like_cols: List[str],
):
    for c in df.select_dtypes(include=["object"]).columns:
        if c in obj_cols_exclude or c in id_like_cols:
            continue
        vc = df[c].value_counts(dropna=False)
        # cardinality
        global_cardinality[c] = max(global_cardinality.get(c, 0), int(vc.shape[0]))
        # counts for frequency mapping
        g = global_freq_counts.setdefault(c, {})
        for k, v in vc.items():
            key = str(k) if pd.notna(k) else "__NA__"
            g[key] = g.get(key, 0) + int(v)

def build_freq_mappings(global_freq_counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    mappings = {}
    for col, cnts in global_freq_counts.items():
        total = float(sum(cnts.values())) or 1.0
        mappings[col] = {k: v / total for k, v in cnts.items()}
    return mappings

# --------------------------
# Two-pass processing (dir mode) for consistent encoders/scaler
# --------------------------

def process_dir(
    dataset_dir: str,
    out_dir: str,
    pattern: str,
    iqr_factor: float,
    low_cardinality_threshold: int,
    concat_all: bool,
    write_corr: bool,
):
    dataset = Path(dataset_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in dataset.rglob("*.parquet") if looks_like_user_file(p, pattern)])
    if not files:
        raise FileNotFoundError(f"No user parquet found in '{dataset_dir}' matching '{pattern}'")

    print(f"Found {len(files)} user chunk(s).")

    drop_report = {}
    global_cardinality: Dict[str, int] = {}
    global_freq_counts: Dict[str, Dict[str, int]] = {}

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler_fitted = False

    id_like_cols = ["user_id"]  # we already drop customer_id

    # ---------- PASS 1: fit global stats ----------
    for fp in tqdm(files, desc="Pass 1/2 (fit)"):
        df = read_parquet(fp).copy()
        df = clean_user_file(df, iqr_factor=iqr_factor, drop_report=drop_report)

        if write_corr:
            num_corr = df.select_dtypes(include=[np.number]).corr()
            num_corr.to_csv(out / "user_corr_report.csv")

        num_cols, obj_cols = split_types(df)

        # numeric to scale (exclude IDs/timestamps/date-like)
        numeric_for_scaling = [c for c in num_cols if not is_id_or_ts_or_date(c)]
        if numeric_for_scaling:
            scaler.partial_fit(df[numeric_for_scaling])
            scaler_fitted = True

        # gather global categorical stats (exclude IDs & OBJ_EXCLUDE)
        update_global_cardinality_and_freq(
            df, global_cardinality, global_freq_counts,
            obj_cols_exclude=OBJ_EXCLUDE,
            id_like_cols=id_like_cols
        )

    # Decide OHE vs Freq globally
    ohe_cols = sorted([c for c, k in global_cardinality.items()
                       if (k <= low_cardinality_threshold) and (c not in OBJ_EXCLUDE)])
    freq_cols = sorted([c for c, k in global_cardinality.items()
                        if (k > low_cardinality_threshold) and (c not in OBJ_EXCLUDE)])

    # Build global frequency mappings
    freq_mappings = build_freq_mappings(global_freq_counts)

    # Save cardinality choices
    with open(out / "user_cardinality.json", "w", encoding="utf-8") as f:
        json.dump({
            "low_cardinality_threshold": low_cardinality_threshold,
            "ohe_cols": ohe_cols,
            "freq_cols": freq_cols,
            "global_cardinality": global_cardinality
        }, f, ensure_ascii=False, indent=2)

    # ---------- PASS 2: transform & save ----------
    processed_paths = []
    for fp in tqdm(files, desc="Pass 2/2 (transform)"):
        df = read_parquet(fp).copy()
        df = clean_user_file(df, iqr_factor=iqr_factor, drop_report=drop_report)

        # Task 4: numeric scaling
        numeric_for_scaling = [c for c in df.select_dtypes(include=[np.number]).columns if not is_id_or_ts_or_date(c)]
        if scaler_fitted and numeric_for_scaling:
            df[numeric_for_scaling] = scaler.transform(df[numeric_for_scaling])

        # Task 4: categorical encoding
        # Frequency encode high-card
        for c in freq_cols:
            if c in df.columns and c not in id_like_cols:
                df[c] = apply_frequency_mapping(df[c], freq_mappings.get(c, {}))

        # One-hot low-card (ignoring IDs)
        ohe_cols_present = [c for c in ohe_cols if (c in df.columns and c not in id_like_cols)]
        df = one_hot_low_card(df, ohe_cols_present)

        # Save per-file
        out_path = out / fp.name
        df.to_parquet(out_path, index=False)
        processed_paths.append(str(out_path))

    # Save transformers
    dump({
        "scaler": scaler if scaler_fitted else None,
        "ohe_cols": ohe_cols,
        "freq_cols": freq_cols,
        "freq_mappings": freq_mappings,
        "id_like_cols": id_like_cols,
        "params": {
            "iqr_factor": iqr_factor,
            "low_cardinality_threshold": low_cardinality_threshold,
            "pattern": pattern
        }
    }, out / "user_transformers.joblib")

    with open(out / "user_drop_report.json", "w", encoding="utf-8") as f:
        json.dump(drop_report, f, ensure_ascii=False, indent=2)

    if concat_all:
        dfs = [pd.read_parquet(p) for p in processed_paths]
        pd.concat(dfs, axis=0, ignore_index=True).to_parquet(out / "user_all_processed.parquet", index=False)

    print("\n=== DONE (Task 1-4 for user chunks) ===")
    print(f"Files processed: {len(processed_paths)}")
    print(f"Saved to: {out}")
    if concat_all:
        print(f"Concatenated: {out/'user_all_processed.parquet'}")


def process_single_file(parquet_path: str, out_dir: str, iqr_factor: float,
                        low_cardinality_threshold: int, write_corr: bool):
    p = Path(parquet_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")

    drop_report = {}

    df = read_parquet(p).copy()
    df = clean_user_file(df, iqr_factor=iqr_factor, drop_report=drop_report)

    if write_corr:
        num_corr = df.select_dtypes(include=[np.number]).corr()
        num_corr.to_csv(out / "user_corr_report.csv")

    # Decide encodings from this file alone (single-file mode)
    num_cols, obj_cols = split_types(df)

    # numeric scaling
    numeric_for_scaling = [c for c in num_cols if not is_id_or_ts_or_date(c)]
    scaler = StandardScaler(with_mean=True, with_std=True)
    if numeric_for_scaling:
        df[numeric_for_scaling] = scaler.fit_transform(df[numeric_for_scaling])

    # cardinality
    id_like_cols = ["user_id"]
    cardinality = {c: df[c].nunique(dropna=False) for c in obj_cols if c not in OBJ_EXCLUDE and c not in id_like_cols}
    ohe_cols = sorted([c for c, k in cardinality.items() if k <= low_cardinality_threshold])
    freq_cols = sorted([c for c, k in cardinality.items() if k > low_cardinality_threshold])

    # build freq mappings + transform
    freq_mappings = {}
    for c in freq_cols:
        vc = df[c].value_counts(dropna=False)
        total = float(vc.sum()) or 1.0
        freq_mappings[c] = {str(k) if pd.notna(k) else "__NA__": v / total for k, v in vc.items()}
        df[c] = apply_frequency_mapping(df[c], freq_mappings[c])

    # one-hot
    ohe_cols_present = [c for c in ohe_cols if c in df.columns]
    df = one_hot_low_card(df, ohe_cols_present)

    # Save outputs
    out_path = out / p.name
    df.to_parquet(out_path, index=False)

    dump({
        "scaler": scaler,
        "ohe_cols": ohe_cols,
        "freq_cols": freq_cols,
        "freq_mappings": freq_mappings,
        "id_like_cols": id_like_cols,
        "params": {
            "iqr_factor": iqr_factor,
            "low_cardinality_threshold": low_cardinality_threshold
        }
    }, out / "user_transformers.joblib")

    with open(out / "user_drop_report.json", "w", encoding="utf-8") as f:
        json.dump(drop_report, f, ensure_ascii=False, indent=2)

    with open(out / "user_cardinality.json", "w", encoding="utf-8") as f:
        json.dump({
            "low_cardinality_threshold": low_cardinality_threshold,
            "ohe_cols": ohe_cols,
            "freq_cols": freq_cols,
            "cardinality": cardinality
        }, f, ensure_ascii=False, indent=2)

    print("\n=== DONE (Task 1-4 single user file) ===")
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--parquet", help="Path to a single user chunk parquet")
    mode.add_argument("--dataset_dir", help="Directory containing many user chunks")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--pattern", default="*user_chunk_*.parquet",
                    help="Glob pattern for directory mode (used with --dataset_dir)")
    ap.add_argument("--iqr_factor", type=float, default=1.5, help="IQR factor for outlier capping (light)")
    ap.add_argument("--low_cardinality_threshold", type=int, default=LOW_CARD_DEFAULT_THRESHOLD,
                    help="<= threshold → OHE; > threshold → frequency encoding")
    ap.add_argument("--concat_all", action="store_true", help="Concat processed files (dir mode only)")
    ap.add_argument("--write_corr", action="store_true", help="Write one correlation CSV for reference")

    args = ap.parse_args()

    if args.parquet:
        process_single_file(
            parquet_path=args.parquet,
            out_dir=args.out_dir,
            iqr_factor=args.iqr_factor,
            low_cardinality_threshold=args.low_cardinality_threshold,
            write_corr=args.write_corr
        )
    else:
        process_dir(
            dataset_dir=args.dataset_dir,
            out_dir=args.out_dir,
            pattern=args.pattern,
            iqr_factor=args.iqr_factor,
            low_cardinality_threshold=args.low_cardinality_threshold,
            concat_all=args.concat_all,
            write_corr=args.write_corr
        )

if __name__ == "__main__":
    main()
