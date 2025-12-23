from __future__ import annotations
import argparse
import os
import gc
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from xgboost import XGBRanker

# Import your custom modules
from preprocess import *
from utils import *
from feature import *
from generate_candidates import *

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system"
MODEL_PATH = os.path.join(BASE_DIR, "outputs/xgb_ranker.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/inference")

TRANSACTIONS_GLOB = f"{BASE_DIR}/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = f"{BASE_DIR}/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = f"{BASE_DIR}/dataset/sales_pers.user_chunk*.parquet"

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
os.environ["POLARS_MAX_THREADS"] = str(MAX_WORKERS)
pl.Config.set_streaming_chunk_size(1000000)

ALL_SCORES = [
    "feat_pop_score",
    "feat_cat_rank_score",
    "feat_cf_score",
    "feat_trend_score",
    "feat_i2v_score",
    "feat_repurchase_score", 
    "feat_new_score"
]

# ============================================================================
# HELPER: LOAD & CLEAN DATA
# ============================================================================

def load_historical_transactions_from_pickle(pickle_path: str, verbose: bool = True) -> pl.LazyFrame:
    if verbose: print(f"\n[Data Load] Loading history from Pickle: {pickle_path}")
    try:
        with open(pickle_path, 'rb') as f:
            hist_data = pickle.load(f)
        
        if isinstance(hist_data, dict):
            hist_pd = pd.DataFrame(hist_data)
        elif isinstance(hist_data, pd.DataFrame):
            hist_pd = hist_data
        else:
            raise ValueError(f"Unknown format in pickle: {type(hist_data)}")
        
        # DEBUG: Print ID types
        if 'customer_id' in hist_pd.columns:
            sample_id = hist_pd['customer_id'].iloc[0]
            if verbose: print(f"  Sample ID from Pickle: {sample_id} (Type: {type(sample_id)})")

        lf = pl.from_pandas(hist_pd).lazy()
        
        # FORCE CAST TO STRING
        if "customer_id" in lf.columns:
            lf = lf.with_columns(pl.col("customer_id").cast(pl.Utf8).str.strip_chars())
        if "item_id" in lf.columns:
            lf = lf.with_columns(pl.col("item_id").cast(pl.Utf8).str.strip_chars())
            
        if "created_date" not in lf.columns and "timestamp" in lf.columns:
            lf = lf.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("created_date"))

        return lf
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        raise

# ============================================================================
# HELPER: DATASET BUILDER
# ============================================================================

def build_dataset_inference(
    split_config: dict,
    trans_clean: pl.LazyFrame,
    items_clean: pl.LazyFrame,
    users_clean: pl.LazyFrame,
    verbose: bool = True
) -> pl.LazyFrame:
    
    if verbose:
        print(f"\nBuilding Inference Dataset...")
        print(f"  History Window: {split_config['hist_start'].date()} -> {split_config['hist_end'].date()}")

    candidates_lazy = generate_candidates(
        transactions=trans_clean,
        items=items_clean,
        users=users_clean,
        hist_start=split_config["hist_start"],
        hist_end=split_config["hist_end"],
        max_candidates_per_user=100,
        verbose=verbose
    )

    if verbose: print(f"  [Optimization] Materializing candidates...")
    candidates_df = candidates_lazy.collect()
    
    # DEBUG: Check IDs in candidates
    if verbose and candidates_df.height > 0:
        sample_cand_id = candidates_df["customer_id"][0]
        print(f"  Sample Candidate ID: '{sample_cand_id}'")

    if candidates_df.height == 0:
        print("  WARNING: No candidates generated!")
        schema = candidates_df.schema
        return pl.DataFrame([], schema=schema).lazy().with_columns([
            pl.lit(0.0).alias(s) for s in ALL_SCORES if s not in schema
        ])

    candidates = candidates_df.lazy()
    
    features = build_features_robust(
        candidates=candidates,
        transactions=trans_clean,
        items=items_clean,
        hist_start=split_config["hist_start"],
        hist_end=split_config["hist_end"],
        verbose=verbose
    )
    
    existing_cols = features.columns
    exprs = [pl.col("customer_id"), pl.col("item_id")]
    for score in ALL_SCORES:
        if score in existing_cols:
            exprs.append(pl.col(score).cast(pl.Float32))
        else:
            exprs.append(pl.lit(0.0, dtype=pl.Float32).alias(score))
            
    return features.select(exprs)

# ============================================================================
# HELPER: PREDICTION (With ID Matching Debug)
# ============================================================================

def predict_top_k(
    model: XGBRanker,
    lf: pl.LazyFrame,
    features: List[str],
    gt_users: List[str],
    top_k: int = 10,
    batch_size: int = 500_000,
    verbose: bool = True
) -> Dict[Any, List[Any]]:
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Inference: Generating Predictions")
        print(f"{'='*60}")

    # FORCE CAST Candidate IDs to String before filtering
    lf = lf.with_columns(pl.col("customer_id").cast(pl.Utf8))
    
    # Filter
    df_test = (
        lf
        .filter(pl.col("customer_id").is_in(gt_users))
        .select(["customer_id", "item_id"] + features)
        .collect()
    )
    
    total_rows = len(df_test)
    if verbose: 
        print(f"  Predicting on {total_rows:,} candidate rows...")
        if total_rows == 0:
            print("  [DEBUG] 0 rows found! Checking ID Mismatch...")
            # Check what IDs exist in candidates vs GT
            cand_ids = lf.select("customer_id").unique().head(5).collect()["customer_id"].to_list()
            print(f"    - First 5 GT IDs:        {gt_users[:5]}")
            print(f"    - First 5 Candidate IDs: {cand_ids}")
            print("    * Ensure these match exactly (strings, no spaces).")

    if total_rows == 0:
        return {}

    all_scores = np.zeros(total_rows, dtype=np.float32)
    
    for i in range(0, total_rows, batch_size):
        end = min(i + batch_size, total_rows)
        X_batch = df_test[i:end].select(features).to_numpy()
        all_scores[i:end] = model.predict(X_batch)
        if i % (batch_size * 5) == 0: gc.collect()

    if verbose: print("  Ranking Top-K...")
    
    results_df = (
        df_test
        .select(["customer_id", "item_id"])
        .with_columns(pl.Series("score", all_scores))
        .sort(["customer_id", "score"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .head(top_k)
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id"))
    )

    results = {}
    for row in results_df.iter_rows():
        results[row[0]] = row[1] 

    del df_test, all_scores, results_df
    gc.collect()
    return results

# ============================================================================
# HELPER: COLD START
# ============================================================================

def create_cold_start_recommendations(trans_lazy, hist_start, hist_end, top_k=50):
    trend_start = max(hist_start, hist_end - timedelta(days=30))
    print(f"  Generating cold-start items (Trending since {trend_start.date()})...")

    trending = (
        trans_lazy
        .filter((pl.col("created_date") >= trend_start) & (pl.col("created_date") <= hist_end))
        .group_by("item_id")
        .agg([
            pl.len().alias("recent_sales"),
            pl.col("customer_id").n_unique().alias("recent_buyers")
        ])
        .with_columns((pl.col("recent_sales") * pl.col("recent_buyers").log1p()).alias("trend_score"))
        .sort("trend_score", descending=True)
        .head(top_k)
        .select("item_id")
        .collect()
    )
    
    # If list is empty, fallback to just popular
    if trending.height == 0:
        print("  Warning: No trend data, falling back to global popularity.")
        trending = (
            trans_lazy.group_by("item_id").len().sort("len", descending=True).head(top_k).select("item_id").collect()
        )

    return trending["item_id"].to_list()

# ============================================================================
# HELPER: EVALUATION (FIXED for Arrays)
# ============================================================================

def build_history_dict(transactions, hist_start, hist_end):
    hist = (
        transactions
        .filter(pl.col("created_date").is_between(hist_start, hist_end))
        .group_by("customer_id")
        .agg(pl.col("item_id").unique())
        .collect()
    )
    return {row[0]: row[1] for row in hist.iter_rows()}

def precision_at_k(pred, gt, hist, filter_bought=False, K=10):
    precisions = []
    
    for user in gt.keys():
        if user not in pred: continue
        
        # --- FIX: Handle Dictionary, List, or Numpy Array in GT ---
        raw_val = gt[user]
        if isinstance(raw_val, dict):
            raw_items = raw_val.get("list_items", [])
        else:
            raw_items = raw_val
            
        # Flatten NumPy arrays or nested lists
        if isinstance(raw_items, np.ndarray):
            raw_items = raw_items.flatten().tolist()
        elif isinstance(raw_items, list):
            # Check if inner elements are arrays
            if len(raw_items) > 0 and isinstance(raw_items[0], np.ndarray):
                flat_list = []
                for x in raw_items:
                    if isinstance(x, np.ndarray):
                        flat_list.extend(x.flatten().tolist())
                    else:
                        flat_list.append(x)
                raw_items = flat_list

        # Convert to strings to be safe
        gt_items = set(str(x) for x in raw_items)
        
        # Filter History
        if filter_bought and user in hist:
            hist_items = set(str(x) for x in hist[user])
            gt_items -= hist_items
            
        if not gt_items: 
            precisions.append(0.0)
            continue
            
        # Score
        pred_items = set(str(x) for x in pred[user][:K])
        hits = len(pred_items & gt_items)
        precisions.append(hits / K)
        
    return float(np.mean(precisions)) if precisions else 0.0
# ============================================================================
# CRITICAL FIX: Feature Alignment for Inference
# ============================================================================

def build_dataset_inference(
    split_config: dict,
    trans_clean: pl.LazyFrame,
    items_clean: pl.LazyFrame,
    users_clean: pl.LazyFrame,
    verbose: bool = True
) -> pl.LazyFrame:
    """
    FIXED: Generate ALL features (not just scores) to match training
    """
    
    if verbose:
        print(f"\nBuilding Inference Dataset...")
        print(f"  History Window: {split_config['hist_start'].date()} -> {split_config['hist_end'].date()}")

    # 1. Generate Candidates
    candidates_lazy = generate_candidates(
        transactions=trans_clean,
        items=items_clean,
        users=users_clean,
        hist_start=split_config["hist_start"],
        hist_end=split_config["hist_end"],
        max_candidates_per_user=100,
        verbose=verbose
    )

    if verbose: 
        print(f"  [Optimization] Materializing candidates...")
    candidates_df = candidates_lazy.collect()
    
    if verbose and candidates_df.height > 0:
        sample_cand_id = candidates_df["customer_id"][0]
        print(f"  Sample Candidate ID: '{sample_cand_id}'")

    if candidates_df.height == 0:
        print("  WARNING: No candidates generated!")
        return pl.DataFrame([]).lazy()

    candidates = candidates_df.lazy()
    
    # 2. Build FULL Features (same as training)
    features = build_features_robust(
        candidates=candidates,
        transactions=trans_clean,
        items=items_clean,
        hist_start=split_config["hist_start"],
        hist_end=split_config["hist_end"],
        verbose=verbose
    )
    
    # ⭐ CRITICAL: Return ALL features, not just score columns
    # The model was trained on the full feature set
    return features


def predict_top_k(
    model: XGBRanker,
    lf: pl.LazyFrame,
    features: List[str],  # ← This should contain ALL 26 features
    gt_users: List[str],
    top_k: int = 10,
    batch_size: int = 500_000,
    verbose: bool = True
) -> Dict[Any, List[Any]]:
    """
    FIXED: Use the correct feature list for prediction
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Inference: Generating Predictions")
        print(f"{'='*60}")
        print(f"  [DEBUG] Model expects {len(features)} features")

    # Cast and filter
    lf = lf.with_columns(pl.col("customer_id").cast(pl.Utf8))
    
    df_test = (
        lf
        .filter(pl.col("customer_id").is_in(gt_users))
        .select(["customer_id", "item_id"] + features)
        .collect()
    )
    
    total_rows = len(df_test)
    if verbose: 
        print(f"  Predicting on {total_rows:,} candidate rows...")
        if total_rows > 0:
            # Verify feature count
            feature_cols = [c for c in df_test.columns if c not in ["customer_id", "item_id"]]
            print(f"  [DEBUG] DataFrame has {len(feature_cols)} feature columns")
            if len(feature_cols) != len(features):
                print(f"  [ERROR] Feature mismatch! Expected {len(features)}, got {len(feature_cols)}")
                print(f"  Expected: {features[:5]}...")
                print(f"  Got: {feature_cols[:5]}...")

    if total_rows == 0:
        return {}

    # Predict
    all_scores = np.zeros(total_rows, dtype=np.float32)
    
    for i in range(0, total_rows, batch_size):
        end = min(i + batch_size, total_rows)
        X_batch = df_test[i:end].select(features).to_numpy()
        
        # Debug first batch
        if i == 0 and verbose:
            print(f"  [DEBUG] First batch shape: {X_batch.shape}")
        
        all_scores[i:end] = model.predict(X_batch)
        
        if i % (batch_size * 5) == 0: 
            gc.collect()

    if verbose: 
        print("  Ranking Top-K...")
    
    # Rest of the function remains the same...
    results_df = (
        df_test
        .select(["customer_id", "item_id"])
        .with_columns(pl.Series("score", all_scores))
        .sort(["customer_id", "score"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .head(top_k)
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id"))
    )

    results = {}
    for row in results_df.iter_rows():
        results[row[0]] = row[1] 

    del df_test, all_scores, results_df
    gc.collect()
    return results
# ============================================================================
# MAIN
# ============================================================================

def run_inference(
    target_date_str: str,
    history_start_str: str = None,
    history_data_path: str = None, 
    gt_file: str = None,
    output_filename: str = "predictions.json",
    top_k: int = 10
):
    start_time = datetime.now()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dates
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    hist_end = target_date - timedelta(days=1)
    if history_start_str:
        hist_start = datetime.strptime(history_start_str, "%Y-%m-%d")
    else:
        hist_start = target_date - timedelta(days=365)
        
    print("="*60)
    print(f"INFERENCE RUN: {target_date_str}")
    print(f"History Data: {history_data_path}")
    print("="*60)

    # 2. Load Data
    print("\n[1/6] Loading Data...")
    raw_items = pl.scan_parquet(ITEMS_PATH).with_columns(pl.col("item_id").cast(pl.Utf8))
    raw_users = pl.scan_parquet(USERS_GLOB).with_columns(pl.col("customer_id").cast(pl.Utf8))

    if history_data_path and os.path.exists(history_data_path):
        raw_trans = load_historical_transactions_from_pickle(history_data_path, verbose=True)
    else:
        print("Using default parquet.")
        raw_trans = pl.scan_parquet(TRANSACTIONS_GLOB).with_columns(pl.col("customer_id").cast(pl.Utf8))

    # 3. Target Users
    target_user_ids = []
    if gt_file:
        print(f"\n[3/6] Loading Ground Truth...")
        gt_data = load_ground_truth_fixed(gt_file, verbose=True)
        target_user_ids = list(gt_data.keys())
        
        print(f"  ✓ Targeting {len(target_user_ids):,} users")
        print(f"  First 5 GT IDs: {target_user_ids[:5]}")
        
        raw_users = raw_users.filter(pl.col("customer_id").is_in(target_user_ids))

    # 4. Build Dataset (FIXED - generates all features)
    infer_config = {
        "hist_start": hist_start,
        "hist_end": hist_end,
        "target_start": None,
        "target_end": None
    }
    
    features_lf = build_dataset_inference(
        infer_config, raw_trans, raw_items, raw_users, verbose=True
    )

    # 5. Load Model and Get Feature Names
    print(f"\n[XGBoost] Loading model {MODEL_PATH}...")
    model = XGBRanker()
    model.load_model(MODEL_PATH)
    
    # ⭐ CRITICAL: Get the ACTUAL feature names the model was trained on
    # Option 1: If you saved them during training
    feature_names_path = os.path.join(OUTPUT_DIR, "feature_names.json")
    
    if os.path.exists(feature_names_path):
        print(f"  Loading feature names from {feature_names_path}")
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
    else:
        # Option 2: Extract from the feature LazyFrame
        print(f"  Inferring feature names from dataset...")
        feature_names = [
            c for c in features_lf.collect_schema().names() 
            if c not in ["customer_id", "item_id", "Y", "created_date", "item_token"]
        ]
    
    print(f"  Model uses {len(feature_names)} features:")
    print(f"    {feature_names[:5]}... (showing first 5)")
    
    # 6. Predict
    preds = predict_top_k(
        model, features_lf, feature_names, target_user_ids, top_k, verbose=True
    )

    # 7. Cold Start
    print("\n[Post-Process] Handling Cold Start...")
    cold_start_items = create_cold_start_recommendations(raw_trans, hist_start, hist_end, top_k)
    cold_start_items = [str(x) for x in cold_start_items]
    
    final_results = {}
    fallback_count = 0
    
    for uid in target_user_ids:
        if uid in preds and len(preds[uid]) > 0:
            final_results[uid] = preds[uid]
        else:
            final_results[uid] = cold_start_items
            fallback_count += 1
            
    print(f"  Used fallback for {fallback_count:,} users.")

    # 8. Save
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n✓ Saved: {out_path}")

    # 9. Evaluation
    if gt_file:
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        hist_dict = build_history_dict(raw_trans, hist_start, hist_end)
        
        p10 = precision_at_k(final_results, gt_data, hist_dict, False, top_k)
        p10_filt = precision_at_k(final_results, gt_data, hist_dict, True, top_k)
        
        print(f"  Precision@{top_k} (Raw):      {p10:.4f}")
        print(f"  Precision@{top_k} (No Prev):  {p10_filt:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--history_start", type=str, default="None")
    parser.add_argument("--history_data", type=str, default="/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/01-2025.pkl")
    parser.add_argument("--gt", type=str, default=None)
    parser.add_argument("--output", type=str, default="predictions.json")
    parser.add_argument("--top_k", type=int, default=10)
    
    args = parser.parse_args()
    
    run_inference(
        args.date, args.history_start, args.history_data, 
        args.gt, args.output, args.top_k
    )