#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess all item_chunk_*.parquet in a directory (Task 1 -> Task 4).

Two-pass pipeline for consistency across chunks:
- Pass 1: scan & fit global encoders/scaler (without saving outputs)
- Pass 2: transform each chunk and save processed parquet to out_dir

Outputs (in --out_dir):
- <same_basename>.parquet              (processed per-file outputs)
- items_dir_transformers.joblib        (scaler, mappings, params)
- items_dir_drop_report.json           (dropped columns + reasons)
- items_dir_cardinality.json           (OHE vs Freq decisions)
- items_dir_corr_report.csv (optional per-file if you enable--write_corr, uses the last file inspected)

Usage:
python preprocess_items_dir.py \
  --dataset_dir /content \
  --pattern "item_chunk_*.parquet" \
  --out_dir /content/out \
  --low_cardinality_threshold 20 \
  --concat_all
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --------------------------
# Helpers
# --------------------------

ITEM_DEFAULT_DROP = [
    "sync_error_message", "sync_status_id", "last_sync_date", "updated_date",
    "is_deleted", "p_id", "category_id"  # usually redundant with level IDs
    # keep image_url if you need vision later
]

LOW_CARD_DEFAULT_THRESHOLD = 20

def looks_like_item_file(path: Path, pattern_glob: str) -> bool:
    # We filter by both: user-provided glob pattern and "item_chunk" substring
    name = path.name.lower()
    return ("item_chunk" in name) and path.match(pattern_glob)

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

def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return num_cols, obj_cols

def is_id_or_timestamp(col: str) -> bool:
    c = col.lower()
    return c.endswith("id") or ("timestamp" in c)

def safe_read_parquet(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p)

def ensure_description_merged(df: pd.DataFrame, drop_report: Dict):
    if "description_new" in df.columns:
        if "description" in df.columns:
            df["description"] = df["description_new"].fillna(df["description"])
        else:
            df["description"] = df["description_new"]
        df.drop(columns=["description_new"], inplace=True)
        drop_report["description_new"] = "Merged into description."

def initial_clean_and_drop(df: pd.DataFrame, drop_report: Dict) -> pd.DataFrame:
    # Drop fully missing weight if present
    to_drop = []
    if "weight" in df.columns and df["weight"].isna().mean() >= 0.999:
        to_drop.append("weight")
        drop_report["weight"] = "100% missing -> drop."

    # Candidate system/meta drops
    to_drop.extend([c for c in ITEM_DEFAULT_DROP if c in df.columns])

    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
        drop_report["dropped_system_meta"] = to_drop
    return df

def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    if "price" in df.columns and df["price"].dtype == "object":
        df["price"] = to_float_from_object(df["price"])
    return df

def basic_impute_and_cap(df: pd.DataFrame, iqr_factor: float) -> pd.DataFrame:
    # Fill objects with 'Unknown'
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna("Unknown")
    # Fill numerics with median
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # IQR capping for price
    if "price" in df.columns and pd.api.types.is_numeric_dtype(df["price"]):
        df["price"] = clip_iqr(df["price"], factor=iqr_factor)
    return df

def update_global_cardinality_and_freq(
    df: pd.DataFrame,
    global_cardinality: Dict[str, int],
    global_freq_counts: Dict[str, Dict[str, int]],
    obj_cols_exclude: List[str]
):
    for c in df.select_dtypes(include=["object"]).columns:
        if c in obj_cols_exclude:
            continue
        vc = df[c].value_counts(dropna=False)
        # cardinality
        global_cardinality[c] = max(global_cardinality.get(c, 0), int(vc.shape[0]))
        # frequency counts
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
# Main pipeline (two-pass)
# --------------------------

def process_items_dir(
    dataset_dir: str,
    out_dir: str,
    pattern: str = "item_chunk_*.parquet",
    iqr_factor: float = 1.5,
    low_cardinality_threshold: int = LOW_CARD_DEFAULT_THRESHOLD,
    drop_highly_correlated: bool = False,
    corr_threshold: float = 0.95,
    concat_all: bool = False,
    write_corr: bool = False
):
    dataset = Path(dataset_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # find candidate files
    all_files = sorted([p for p in dataset.rglob("*.parquet") if looks_like_item_file(p, pattern)])
    if not all_files:
        raise FileNotFoundError(f"No item parquet found in '{dataset_dir}' matching pattern '{pattern}'")

    print(f"Found {len(all_files)} item chunk(s).")

    drop_report = {}
    global_cardinality: Dict[str, int] = {}
    global_freq_counts: Dict[str, Dict[str, int]] = {}

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler_fitted = False

    # columns that we will never OHE/freq encode even if object: textual free-form
    obj_exclude = ["description", "image_url"]  # keep raw for future text/vision

    # --------------------------
    # PASS 1: fit global stats
    # --------------------------
    for fp in tqdm(all_files, desc="Pass 1/2 (fit)"):
        df = safe_read_parquet(fp).copy()

        # Task 1
        ensure_description_merged(df, drop_report)
        df = initial_clean_and_drop(df, drop_report)

        # price cleanup
        df = clean_price(df)

        # Task 2
        df = basic_impute_and_cap(df, iqr_factor=iqr_factor)

        # Task 3 (optional numeric corr report per last file)
        if write_corr:
            num_corr = df.select_dtypes(include=[np.number]).corr()
            num_corr.to_csv(out / "items_dir_corr_report.csv")

        # prepare for Task 4: decide numeric to scale
        num_cols, obj_cols = detect_columns(df)

        # update global cardinality & frequencies (for object cols except excluded)
        update_global_cardinality_and_freq(df, global_cardinality, global_freq_counts, obj_exclude)

        # choose numeric columns to scale (skip IDs/timestamps)
        numeric_for_scaling = [c for c in num_cols if not is_id_or_timestamp(c)]

        # partial fit scaler
        if numeric_for_scaling:
            scaler.partial_fit(df[numeric_for_scaling])
            scaler_fitted = True

    # Decide OHE vs Freq globally
    ohe_cols = sorted([c for c, k in global_cardinality.items()
                       if (k <= low_cardinality_threshold) and (c not in obj_exclude)])
    freq_cols = sorted([c for c, k in global_cardinality.items()
                        if (k > low_cardinality_threshold) and (c not in obj_exclude)])

    # Build global frequency mappings
    freq_mappings = build_freq_mappings(global_freq_counts)

    # Save cardinality decisions
    with open(out / "items_dir_cardinality.json", "w", encoding="utf-8") as f:
        json.dump({
            "low_cardinality_threshold": low_cardinality_threshold,
            "ohe_cols": ohe_cols,
            "freq_cols": freq_cols,
            "global_cardinality": global_cardinality
        }, f, ensure_ascii=False, indent=2)

    # --------------------------
    # PASS 2: transform + save
    # --------------------------
    processed_paths = []
    for fp in tqdm(all_files, desc="Pass 2/2 (transform)"):
        df = safe_read_parquet(fp).copy()

        # Task 1
        ensure_description_merged(df, drop_report)
        df = initial_clean_and_drop(df, drop_report)

        # price cleanup
        df = clean_price(df)

        # Task 2
        df = basic_impute_and_cap(df, iqr_factor=iqr_factor)

        # Task 4
        # (A) numeric scaling
        numeric_for_scaling = [c for c in df.select_dtypes(include=[np.number]).columns if not is_id_or_timestamp(c)]
        if scaler_fitted and numeric_for_scaling:
            df[numeric_for_scaling] = scaler.transform(df[numeric_for_scaling])

        # (B) frequency encoding for high-card object columns
        for c in freq_cols:
            if c in df.columns:
                df[c] = apply_frequency_mapping(df[c], freq_mappings.get(c, {}))

        # (C) one-hot for low-card object columns
        # Ensure we do not try to OHE columns that were freq-encoded
        ohe_cols_present = [c for c in ohe_cols if c in df.columns]
        df = one_hot_low_card(df, ohe_cols_present)

        # Save per-file
        out_path = out / fp.name  # keep same basename
        df.to_parquet(out_path, index=False)
        processed_paths.append(str(out_path))

    # Save transformers (for future transform-only runs)
    dump({
        "numeric_cols_for_scaling": "computed_per_file (non-ID numeric columns)",
        "scaler": scaler if scaler_fitted else None,
        "ohe_cols": ohe_cols,
        "freq_cols": freq_cols,
        "freq_mappings": freq_mappings,
        "obj_exclude": obj_exclude,
        "params": {
            "iqr_factor": iqr_factor,
            "low_cardinality_threshold": low_cardinality_threshold,
            "drop_highly_correlated": drop_highly_correlated,
            "corr_threshold": corr_threshold,
            "pattern": pattern
        }
    }, out / "items_dir_transformers.joblib")

    with open(out / "items_dir_drop_report.json", "w", encoding="utf-8") as f:
        json.dump(drop_report, f, ensure_ascii=False, indent=2)

    # Optional concat
    if concat_all:
        dfs = [pd.read_parquet(p) for p in processed_paths]
        pd.concat(dfs, axis=0, ignore_index=True).to_parquet(out / "items_all_processed.parquet", index=False)

    print("\n=== DONE (Task 1-4 across directory) ===")
    print(f"Files processed: {len(processed_paths)}")
    print(f"Saved to: {out}")
    if concat_all:
        print(f"Concatenated file: {out/'items_all_processed.parquet'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="Directory containing many item_chunk_*.parquet")
    ap.add_argument("--out_dir", required=True, help="Where to write processed outputs")
    ap.add_argument("--pattern", default="sales_pers.item_chunk_*.parquet", help="Glob pattern to match item parquet files")
    ap.add_argument("--iqr_factor", type=float, default=1.5, help="IQR factor for outlier capping (price)")
    ap.add_argument("--low_cardinality_threshold", type=int, default=LOW_CARD_DEFAULT_THRESHOLD,
                    help="<= threshold → OHE; > threshold → frequency encoding")
    ap.add_argument("--drop_highly_correlated", action="store_true", help="(reserved) not used in dir mode")
    ap.add_argument("--corr_threshold", type=float, default=0.95, help="(reserved) not used in dir mode")
    ap.add_argument("--concat_all", action="store_true", help="Concatenate processed files into items_all_processed.parquet")
    ap.add_argument("--write_corr", action="store_true", help="Write one numeric corr report from the last scanned file")
    args = ap.parse_args()

    process_items_dir(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        pattern=args.pattern,
        iqr_factor=args.iqr_factor,
        low_cardinality_threshold=args.low_cardinality_threshold,
        drop_highly_correlated=args.drop_highly_correlated,
        corr_threshold=args.corr_threshold,
        concat_all=args.concat_all,
        write_corr=args.write_corr
    )

if __name__ == "__main__":
    main()
