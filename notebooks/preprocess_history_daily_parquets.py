#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess purchase_history_daily parquet (Task 1 -> Task 4).

Supports:
  - Single file: --parquet /path/to/chunk.parquet
  - Directory of chunks: --dataset_dir /path --pattern "purchase_history_daily_chunk_*.parquet"

Pipeline:
  Task 1: Drop unneeded/redundant columns (system/meta, event_value, customer_id)
  Task 2: NULL handling + IQR outlier capping for price/discount/quantity
  Task 3: Redundancy checks (reports) and ID-safe handling
  Task 4: Normalize numeric (except IDs/timestamps/dates) & encode categoricals
          - OHE for low-card (<= threshold), Frequency-encode for high-card

Outputs (in --out_dir):
  - <same_basename>.parquet (processed per-file)
  - purchase_daily_transformers.joblib (global scaler, OHE list, freq-mappings)
  - purchase_daily_drop_report.json
  - purchase_daily_cardinality.json
  - purchase_daily_corr_report.csv (optional, last file scanned)

Usage examples:
  # Single file
  python preprocess_purchase_daily.py \
    --parquet /content/data/sales_pers.purchase_history_daily_chunk_0.parquet \
    --out_dir /content/out_purchase

  # Directory (consistent encoders across all chunks)
  python preprocess_purchase_daily.py \
    --dataset_dir /content/data \
    --pattern "sales_pers.purchase_history_daily_chunk_*.parquet" \
    --out_dir /content/out_purchase \
    --concat_all
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

# Always drop these (system/meta/noise) if present
PURCHASE_DROP_ALWAYS = [
    "is_deleted",
    "created_date",
    "updated_date",
    "event_value",     # often verbose log metadata
    "customer_id",     # redundant with user_id for modeling
]

# Treat as free-text -> never encode (keep raw); not many in this dataset, but keep here for extensibility
OBJ_EXCLUDE = []


# --------------------------
# Helpers
# --------------------------

def read_parquet(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p)

def to_float_from_object(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r'[^0-9.\-]', '', regex=True)
         .replace({'': np.nan, '.': np.nan, '-.': np.nan})
         .astype(float)
    )

def clip_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    if series.isna().all():
        return series
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lower, upper)

def is_id_or_ts_or_date(col: str) -> bool:
    c = col.lower()
    return (
        c.endswith("_id")        # e.g., user_id, item_id, location_id
        or c == "user_id"
        or c == "item_id"
        or "timestamp" in c      # UNIX seconds/millis
        or c == "date_key"       # yyyymmdd
    )

def split_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return num_cols, obj_cols

def looks_like_purchase_file(path: Path, pattern_glob: str) -> bool:
    name = path.name.lower()
    return ("purchase_history_daily_chunk" in name) and path.match(pattern_glob)

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

def apply_frequency_mapping(series: pd.Series, mapping: Dict[str, float]) -> pd.Series:
    return series.astype(str).map(mapping).fillna(0.0)

def one_hot_low_card(df: pd.DataFrame, ohe_cols: List[str]) -> pd.DataFrame:
    if not ohe_cols:
        return df
    return pd.get_dummies(df, columns=ohe_cols, prefix=[f"{c}_ohe" for c in ohe_cols], dummy_na=False)


# --------------------------
# Core per-file cleaning (Tasks 1 & 2)
# --------------------------

def clean_purchase_file(df: pd.DataFrame, iqr_factor: float, drop_report: Dict) -> pd.DataFrame:
    # Task 1: drop columns
    to_drop = [c for c in PURCHASE_DROP_ALWAYS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
        drop_report.setdefault("dropped_system_meta", []).extend(to_drop)

    # Convert strings to numeric
    if "price" in df.columns and df["price"].dtype == "object":
        df["price"] = to_float_from_object(df["price"])
    if "discount" in df.columns and df["discount"].dtype == "object":
        df["discount"] = to_float_from_object(df["discount"])

    # Task 2: NULL handling
    # Objects → 'Unknown'
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna("Unknown")

    # Numerics → median
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Task 2: Outlier capping
    for c in ["price", "discount", "quantity"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = clip_iqr(df[c], factor=iqr_factor)

    return df


# --------------------------
# Two-pass processing for directory (consistent encoders/scaler)
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

    files = sorted([p for p in dataset.rglob("*.parquet") if looks_like_purchase_file(p, pattern)])
    if not files:
        raise FileNotFoundError(f"No purchase-history daily parquet found in '{dataset_dir}' matching '{pattern}'")

    print(f"Found {len(files)} purchase-history chunk(s).")

    drop_report = {}
    global_cardinality: Dict[str, int] = {}
    global_freq_counts: Dict[str, Dict[str, int]] = {}

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler_fitted = False

    # IDs to exclude from encoding
    id_like_cols = ["user_id", "item_id"]

    # ---------- PASS 1: fit global stats ----------
    for fp in tqdm(files, desc="Pass 1/2 (fit)"):
        df = read_parquet(fp).copy()
        df = clean_purchase_file(df, iqr_factor=iqr_factor, drop_report=drop_report)

        if write_corr:
            num_corr = df.select_dtypes(include=[np.number]).corr()
            num_corr.to_csv(out / "purchase_daily_corr_report.csv")

        num_cols, obj_cols = split_types(df)

        # numeric to scale (exclude IDs/timestamps/date_key)
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
    with open(out / "purchase_daily_cardinality.json", "w", encoding="utf-8") as f:
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
        df = clean_purchase_file(df, iqr_factor=iqr_factor, drop_report=drop_report)

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

    # Save transformers for future transform-only
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
    }, out / "purchase_daily_transformers.joblib")

    with open(out / "purchase_daily_drop_report.json", "w", encoding="utf-8") as f:
        json.dump(drop_report, f, ensure_ascii=False, indent=2)

    if concat_all:
        dfs = [pd.read_parquet(p) for p in processed_paths]
        pd.concat(dfs, axis=0, ignore_index=True).to_parquet(out / "purchase_all_processed.parquet", index=False)

    print("\n=== DONE (Task 1-4 for purchase history) ===")
    print(f"Files processed: {len(processed_paths)}")
    print(f"Saved to: {out}")
    if concat_all:
        print(f"Concatenated: {out/'purchase_all_processed.parquet'}")


def process_single_file(parquet_path: str, out_dir: str, iqr_factor: float,
                        low_cardinality_threshold: int, write_corr: bool):
    p = Path(parquet_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")

    drop_report = {}

    df = read_parquet(p).copy()
    df = clean_purchase_file(df, iqr_factor=iqr_factor, drop_report=drop_report)

    # Reports
    if write_corr:
        num_corr = df.select_dtypes(include=[np.number]).corr()
        num_corr.to_csv(out / "purchase_daily_corr_report.csv")

    # Decide encodings from this file alone (single-file mode)
    num_cols, obj_cols = split_types(df)
    id_like_cols = ["user_id", "item_id"]

    # numeric scaling
    numeric_for_scaling = [c for c in num_cols if not is_id_or_ts_or_date(c)]
    scaler = StandardScaler(with_mean=True, with_std=True)
    if numeric_for_scaling:
        df[numeric_for_scaling] = scaler.fit_transform(df[numeric_for_scaling])

    # cardinality
    cardinality = {c: df[c].nunique(dropna=False) for c in obj_cols if c not in OBJ_EXCLUDE and c not in id_like_cols}
    ohe_cols = sorted([c for c, k in cardinality.items() if k <= low_cardinality_threshold])
    freq_cols = sorted([c for c, k in cardinality.items() if k > low_cardinality_threshold])

    # build freq mappings
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
    }, out / "purchase_daily_transformers.joblib")

    with open(out / "purchase_daily_drop_report.json", "w", encoding="utf-8") as f:
        json.dump(drop_report, f, ensure_ascii=False, indent=2)

    with open(out / "purchase_daily_cardinality.json", "w", encoding="utf-8") as f:
        json.dump({
            "low_cardinality_threshold": low_cardinality_threshold,
            "ohe_cols": ohe_cols,
            "freq_cols": freq_cols,
            "cardinality": cardinality
        }, f, ensure_ascii=False, indent=2)

    print("\n=== DONE (Task 1-4 single file) ===")
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--parquet", help="Path to a single purchase_history_daily chunk parquet")
    mode.add_argument("--dataset_dir", help="Directory containing many purchase_history_daily chunks")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--pattern", default="*purchase_history_daily_chunk_*.parquet",
                    help="Glob pattern for directory mode (used with --dataset_dir)")
    ap.add_argument("--iqr_factor", type=float, default=1.5, help="IQR factor for outlier capping")
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
