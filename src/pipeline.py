from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import os
import gc
import pickle
import json
import warnings
import numpy as np
import polars as pl
from xgboost import XGBRanker
from gensim.models import Word2Vec
from preprocess import *
from utils import *
from feature import *
from generate_candidates import *
warnings.filterwarnings('ignore')
import os
import polars as pl
from generate_candidates import * 
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# Best: set environment variable before Polars does heavy work
os.environ["POLARS_MAX_THREADS"] = str(MAX_WORKERS)



# GLOBAL PERFORMANCE CONFIG
pl.Config.set_streaming_chunk_size(1000000)
# Use 'spawn' or 'fork' depending on OS, but standard is usually fine.
# Setting env var for Polars parallelization
os.environ["POLARS_MAX_THREADS"] = str(os.cpu_count())

"""
CRITICAL FIXES FOR CANDIDATE GENERATION
========================================

Problems Fixed:
1. Low Recall@80 (53%) - Missing too many GT items
2. Weak collaborative filtering (only 3 co-occurrences)
3. Item2Vec too restrictive (only 10 neighbors)
4. Missing repeat purchase patterns
"""

import polars as pl
from datetime import datetime
from gensim.models import Word2Vec
import os





# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system"
TRANSACTIONS_GLOB = f"{BASE_DIR}/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = f"{BASE_DIR}/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = f"{BASE_DIR}/dataset/sales_pers.user_chunk*.parquet"
GT_PKL_PATH = f"{BASE_DIR}/groundtruth.pkl"
OUTPUT_DIR = f"{BASE_DIR}/outputs"

# Line 89 - UPDATE THIS
ALL_SCORES = [
    "feat_pop_score",
    "feat_cat_rank_score",
    "feat_cf_score",
    "feat_trend_score",
    "feat_i2v_score",
    "feat_repurchase_score",  # ADDED - was "feat_repeat_score"
]

def standardize(df: pl.LazyFrame, active_score_col: str) -> pl.LazyFrame:
    # Always output: customer_id, item_id, then ALL_SCORES in fixed order
    exprs = [
        pl.col("customer_id"),
        pl.col("item_id"),
    ]

    for sc in ALL_SCORES:
        if sc == active_score_col:
            exprs.append(pl.col(active_score_col).cast(pl.Float32).alias(sc))
        else:
            exprs.append(pl.lit(0.0, dtype=pl.Float32).alias(sc))

    return df.select(exprs)



@dataclass(frozen=True)
class DataSplit:
    """Time-based train/val/test split configuration"""
    name: str
    hist_start: datetime
    hist_end: datetime
    target_start: datetime | None
    target_end: datetime | None
    
    def __repr__(self):
        if self.target_start is None:
            return f"{self.name}: hist=[{self.hist_start.date()}→{self.hist_end.date()}], target=GT_FILE"
        return f"{self.name}: hist=[{self.hist_start.date()}→{self.hist_end.date()}], target=[{self.target_start.date()}→{self.target_end.date()}]"

# # ============================================================================
# # MAIN INTEGRATION FUNCTION
# # ============================================================================

# ============================================================================
# DATASET BUILDER
# ============================================================================

# ============================================================================
# FIXED: DATASET BUILDER
# ============================================================================

def build_dataset_v2(
    split_name: str,
    split_config: dict,
    trans_clean: pl.LazyFrame,
    items_clean: pl.LazyFrame,
    users_clean: pl.LazyFrame,
    is_train: bool = True,
    verbose: bool = True,
    *,
    max_train_items_per_user: int = 100,
    seed: int = 42,
    do_recall_check: bool = False,
    recall_k: int = 100,
    allow_repurchase: bool = True,
) -> pl.LazyFrame:
    
    if verbose:
        print(f"\n{'='*60}\nBuilding Dataset V3 FIXED: {split_name}\n{'='*60}")
    
    # 1. Generate Candidates (Lazy)
    candidates_lazy = generate_candidates(
    transactions=trans_clean,
    items=items_clean,
    users=users_clean,
    hist_start=split_config["hist_start"],
    hist_end=split_config["hist_end"],
    max_candidates_per_user=100,
    verbose=True
)

    
    # -----------------------------------------------------------------------
    # CRITICAL OPTIMIZATION: Break Lineage
    # We collect candidates here to prevent the complex generation graph 
    # from being duplicated inside every feature function.
    # -----------------------------------------------------------------------
    if verbose: print(f"  [Optimization] Materializing candidates to break graph lineage...")
    
    # Collect to memory (it's small: users * 100 rows)
    candidates_df = candidates_lazy.collect()
    
    # Convert back to Lazy for feature engineering, but now it's a leaf node!
    candidates = candidates_df.lazy()
    
    if verbose: print(f"  [Optimization] Candidates materialized: {candidates_df.height:,} rows.")
    # -----------------------------------------------------------------------

    # 2. Recall check (optional)
    if do_recall_check and split_config.get("target_start"):
        recall_at_k_candidates(candidates, trans_clean,
            split_config["target_start"], split_config["target_end"], K=recall_k, verbose=True)
    
    # 3. Filter history if needed
    if not allow_repurchase:
        if verbose: print("  [Filtering] Removing previously purchased items...")
        hist_pairs = (
            trans_clean
            .filter(pl.col("created_date") <= split_config["hist_end"])
            .select(["customer_id", "item_id"]).unique()
        )
        candidates = candidates.join(hist_pairs, on=["customer_id", "item_id"], how="anti")
    
    # 4. Build features (Now safe because candidates is simple)
    features = build_features_robust(
        candidates=candidates,
        transactions=trans_clean,
        items=items_clean,
        hist_start=split_config["hist_start"],
        hist_end=split_config["hist_end"],
        verbose=verbose
    )
    
    # 5. Train/Val Logic (Same as before)
    if is_train:
        return sample_train_pairs_v3_FIXED(
            candidates_with_features=features,
            trans_clean=trans_clean,
            target_start=split_config["target_start"],
            target_end=split_config["target_end"],
            max_items_per_user=max_train_items_per_user,
            seed=seed,
            verbose=verbose,
        )
    
    # Val/Test: Add labels
    if split_config.get("target_start") is not None:
        targets = (
            trans_clean
            .filter(pl.col("created_date").is_between(split_config["target_start"], split_config["target_end"]))
            .select(["customer_id", "item_id"]).unique()
            .with_columns(pl.lit(1).cast(pl.UInt8).alias("Y"))
        )
        features = features.join(targets, on=["customer_id", "item_id"], how="left").with_columns(pl.col("Y").fill_null(0))
    
    return features


# ============================================================================
# NUMPY PREPARATION
# ============================================================================

# ============================================================================
# FIXED: NUMPY PREPARATION
# ============================================================================

def prepare_for_xgb_v2(
    lf: pl.LazyFrame,
    is_train: bool = True,
    verbose: bool = True,
    *,
    filter_no_positive_users: bool = False,  # ← CHANGED: Don't filter in train
    rand_seed: int = 42,
):
    """
    FIXED: Don't filter users in training (already sampled properly)
    """
    if verbose:
        print(f"\n[XGB Prep V3 FIXED] is_train={is_train}")
    
    exclude_cols = ["customer_id", "item_id", "Y", "created_date", "item_token"]
    
    df = lf.collect()
    
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt8]
    ]
    
    if verbose:
        print(f"  Features: {len(feature_cols)}")
    
    # ⭐ ONLY filter no-positive users in VAL/TEST, not TRAIN
    if "Y" in df.columns and filter_no_positive_users and not is_train:
        pos_users = (
            df.filter(pl.col("Y") == 1)
            .select("customer_id")
            .unique()
        )
        before = df.height
        df = df.join(pos_users, on="customer_id", how="inner")
        if verbose and before > df.height:
            print(f"  Filtered no-positive users: {before:,} -> {df.height:,} rows")
    
    # Sort by user
    df_final = df.with_columns(pl.lit(np.random.rand(len(df))).alias("shuffle_key"))
    df_final = df_final.sort(["customer_id", "shuffle_key"])
    
    # Build arrays
    X = df_final.select(feature_cols).to_numpy()
    y = df_final["Y"].to_numpy() if "Y" in df_final.columns else None
    query_ids = df_final["customer_id"].to_numpy()
    _, groups = np.unique(query_ids, return_counts=True)
    
    if verbose:
        n_samples = df_final.height
        n_groups = len(groups)
        avg_group = n_samples / n_groups if n_groups > 0 else 0
        if y is not None:
            pos_count = int(y.sum())
            pos_rate = pos_count / n_samples if n_samples > 0 else 0
            print(f"  Output: {n_samples:,} rows, {n_groups:,} groups")
            print(f"  Positive: {pos_count:,} ({pos_rate:.2%})")
            print(f"  Avg items/group: {avg_group:.1f}")
        else:
            print(f"  Output: {n_samples:,} rows, {n_groups:,} groups (no labels)")
    
    return X, y, groups, feature_cols, df_final.select(["customer_id", "item_id"])



# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    X_train, y_train, groups_train,
    X_val, y_val, groups_val,
    verbose: bool = True
) -> XGBRanker:
    """
    FIXED: Prevent overfitting with proper regularization
    """
    
    print("\n[Training XGBRanker V3 - Anti-Overfitting]")
    
    model = XGBRanker(
        objective='rank:ndcg',
        n_estimators=2000,       # Increased (we'll use early stopping)
        learning_rate=0.05,      # REDUCED (was 0.1) - slower = better generalization
        max_depth=4,             # REDUCED (was 5) - prevent memorization
        subsample=0.7,           # REDUCED (was 0.8) - more randomness
        colsample_bytree=0.7,    # REDUCED (was 0.8) - feature sampling
        colsample_bylevel=0.7,   # NEW - per-level feature sampling
        reg_lambda=5.0,          # INCREASED (was 2.0) - stronger L2
        reg_alpha=1.0,           # INCREASED (was 0.5) - stronger L1
        min_child_weight=10,     # INCREASED (was 5) - larger leaf nodes
        gamma=0.5,               # NEW - minimum loss reduction for split
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
        eval_metric='ndcg@10',
        early_stopping_rounds=10,  # NEW - stop if no improvement for 30 rounds
    )
    
    model.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_group=[groups_train, groups_val],
        verbose=10,
    )
    
    # Report best iteration
    best_iteration = model.best_iteration
    best_score = model.best_score
    print(f"\n✓ Best iteration: {best_iteration}")
    print(f"✓ Best validation NDCG@10: {best_score:.4f}")
    
    return model


# ============================================================================
# FIX 4: COLD-START FALLBACK STRATEGY
# ============================================================================

def create_cold_start_recommendations(
    items: pl.LazyFrame,
    transactions: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    top_k: int = 100,
    verbose: bool = True
) -> List[Any]:
    """
    Generate fallback recommendations for cold-start users
    
    Strategy:
    1. Trending items (recent popularity spike)
    2. New items (launched recently)
    3. Popular items (safety net)
    """
    
    recent_date = hist_end - timedelta(days=14)
    
    # 1. Trending items (high recent activity)
    trending = (
        transactions
        .filter(
            (pl.col("created_date") >= recent_date) &
            (pl.col("created_date") <= hist_end)
        )
        .group_by("item_id")
        .agg([
            pl.len().alias("recent_sales"),
            pl.col("customer_id").n_unique().alias("recent_buyers")
        ])
        .with_columns([
            (pl.col("recent_sales") * pl.col("recent_buyers").log1p())
            .alias("trend_score")
        ])
        .sort("trend_score", descending=True)
        .head(40)
        .select("item_id")
        .collect()
    )
    
    # 2. New items (launched in last 30 days)
    new_items = (
        transactions
        .filter(pl.col("created_date") <= hist_end)
        .group_by("item_id")
        .agg([
            pl.col("created_date").min().alias("first_sale")
        ])
        .filter(
            (hist_end - pl.col("first_sale")).dt.total_days() <= 30
        )
        .join(
            transactions
            .filter(
                (pl.col("created_date") >= recent_date) &
                (pl.col("created_date") <= hist_end)
            )
            .group_by("item_id")
            .len()
            .rename({"len": "sales"}),
            on="item_id",
            how="inner"
        )
        .sort("sales", descending=True)
        .head(30)
        .select("item_id")
        .collect()
    )
    
    # 3. Overall popular items
    popular = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) &
            (pl.col("created_date") <= hist_end)
        )
        .group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(30)
        .select("item_id")
        .collect()
    )
    
    # Combine: trending > new > popular
    cold_start_items = (
        pl.concat([trending, new_items, popular], how="vertical")
        .unique()
        .head(top_k)
        .to_series()
        .to_list()
    )
    
    if verbose:
        print(f"\n[Cold-Start Fallback] Created {len(cold_start_items)} recommendations")
    
    return cold_start_items


# ============================================================================
# EVALUATION
# ============================================================================

def build_history_dict(
    transactions: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose: bool = True
) -> Dict[Any, List[Any]]:
    """Build user purchase history"""
    if verbose:
        print("  Building history dict...")
    
    hist = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").unique())
        .collect()  # ✅ Regular collect
    )
    
    # ✅ More efficient dict construction
    hist_dict = {row[0]: row[1] for row in hist.iter_rows()}
    
    if verbose:
        print(f"  History: {len(hist_dict):,} users")
    
    return hist_dict


def predict_top_k(
    model: XGBRanker,
    lf: pl.LazyFrame,
    features: List[str],
    gt_users: List[Any],
    top_k: int = 10,
    batch_size: int = 500_000,  # Process 500k rows at a time (adjust based on RAM)
    verbose: bool = True
) -> Dict[Any, List[Any]]:
    """
    Optimized inference function.
    
    Strategy:
    1. Materialize the relevant test features into memory ONCE.
    2. Run XGBoost prediction in large vector batches.
    3. Use Polars (Rust) to sort and extract top-k items efficiently.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Inference: Generating Predictions")
        print(f"{'='*60}")
        print("  [1/4] Materializing test data...")

    # -------------------------------------------------------------------------
    # 1. Materialize Data (The bottleneck breaker)
    # -------------------------------------------------------------------------
    # We filter for only the users we need to predict for, then collect.
    # This executes the Feature Engineering DAG exactly once.
    df_test = (
        lf
        .filter(pl.col("customer_id").is_in(gt_users))
        .select(["customer_id", "item_id"] + features)
        .collect()
    )
    
    total_rows = len(df_test)
    if verbose:
        print(f"  [2/4] Predicting on {total_rows:,} candidate rows...")
        print(f"        (Batch size: {batch_size:,})")

    if total_rows == 0:
        return {}

    # -------------------------------------------------------------------------
    # 2. Vectorized Prediction
    # -------------------------------------------------------------------------
    # We pre-allocate a numpy array for scores to avoid memory fragmentation
    all_scores = np.zeros(total_rows, dtype=np.float32)
    
    # Iterate through the DataFrame in chunks (CPU-bound)
    # This prevents creating a massive NumPy array copy of the features if RAM is tight
    for i in range(0, total_rows, batch_size):
        end = min(i + batch_size, total_rows)
        
        # Slice feature columns and convert to numpy
        # Note: We assume feature columns are floats. If not, this might copy.
        X_batch = df_test[i:end].select(features).to_numpy()
        
        # Predict
        all_scores[i:end] = model.predict(X_batch)
        
        # Periodic Garbage Collection for very large loops
        if i % (batch_size * 5) == 0:
            gc.collect()

    # -------------------------------------------------------------------------
    # 3. Efficient Top-K Extraction
    # -------------------------------------------------------------------------
    if verbose:
        print("  [3/4] Ranking and extracting Top-K items...")

    # Attach scores and sort
    # Polars is much faster at "Sort-Group-Head" than Python dictionaries
    top_k_df = (
        df_test
        .select(["customer_id", "item_id"])          # Drop feature cols to save RAM
        .with_columns(pl.Series("score", all_scores)) # Attach predictions
        .sort(["customer_id", "score"], descending=[False, True]) # Sort by User then Score
        .group_by("customer_id", maintain_order=True) # Group
        .head(top_k)                                  # Take top K
    )

    # -------------------------------------------------------------------------
    # 4. Convert to Output Dictionary
    # -------------------------------------------------------------------------
    if verbose:
        print("  [4/4] Formatting results...")

    results = {}
    
    # Aggregating to list is the fastest way to bridge Polars -> Python Dict
    # This results in: customer_id | [item1, item2, item3...]
    final_agg = (
        top_k_df
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id"))
    )

    # Fast iteration over rows
    for row in final_agg.iter_rows():
        results[row[0]] = row[1]  # row[0] is user_id, row[1] is list of items

    # Fill in missing users (if any GT users had no candidates generated)
    missing_count = 0
    for user in gt_users:
        if user not in results:
            results[user] = []
            missing_count += 1

    if verbose:
        print(f"  Done. Predictions generated for {len(results):,} users.")
        if missing_count > 0:
            print(f"  Warning: {missing_count} users had 0 candidates (returned empty list).")

    # Clean up memory
    del df_test, all_scores, top_k_df, final_agg
    gc.collect()

    return results

def precision_at_k_customer(
    pred,
    gt,
    hist,
    filter_bought_items: bool = False,  # CHANGED DEFAULT to False
    K: int = 10,
    *,
    return_stats: bool = True,
):
    """
    MODIFIED: Customer-compatible Precision@K with repurchase support
    """
    precisions = []
    cold_start_users = []
    nusers = len(gt.keys())

    n_missing_hist = 0
    n_missing_pred = 0
    n_empty_relevant_after_filter = 0

    for user in gt.keys():
        missing_hist = (user not in hist)
        missing_pred = (user not in pred)

        if missing_hist or missing_pred:
            cold_start_users.append(user)
            if missing_hist:
                n_missing_hist += 1
            if missing_pred:
                n_missing_pred += 1
            continue

        val = gt[user]
        gt_items = val["list_items"] if isinstance(val, dict) else val
        relevant_items = set(gt_items)

        # MODIFIED: Only filter if explicitly requested
        if filter_bought_items and user in hist:
            relevant_items -= set(hist[user])

        if not relevant_items:
            n_empty_relevant_after_filter += 1

        hits = len(set(pred[user][:K]) & relevant_items)
        precisions.append(hits / K)

    mean_precision = float(np.mean(precisions)) if precisions else 0.0

    if not return_stats:
        return mean_precision, cold_start_users

    evaluated = len(precisions)
    stats = {
        "total_gt_users": nusers,
        "evaluated_users": evaluated,
        "cold_start_users": len(cold_start_users),
        "missing_hist": n_missing_hist,
        "missing_pred": n_missing_pred,
        "coverage_rate": (evaluated / nusers) if nusers else 0.0,
        "empty_relevant_after_filter": n_empty_relevant_after_filter,
        "precision_at_k": mean_precision,
        "K": K,
        "filter_bought_items": filter_bought_items,
    }

    return mean_precision, cold_start_users, stats

def precision_at_k(
    pred: Dict[Any, List[Any]],
    gt: Dict[Any, Any],
    hist: Dict[Any, List[Any]],
    filter_bought_items: bool = False,  # CHANGED DEFAULT to False
    K: int = 10,
    verbose: bool = True
) -> Tuple[float, List[Any], Dict[str, int]]:
    """
    MODIFIED: Default to NOT filtering bought items (allow repurchase)
    """
    precisions = []
    missing_preds_users = []
    
    for user in gt.keys():
        if user not in pred:
            missing_preds_users.append(user)
            continue
        
        # Get ground truth
        if isinstance(gt[user], dict):
            gt_items = gt[user]['list_items']
        else:
            gt_items = gt[user]
        
        relevant_items = set(gt_items)
        
        # MODIFIED: Only filter if explicitly requested
        if filter_bought_items and user in hist:
            past_items = set(hist.get(user, []))
            relevant_items -= past_items
            # if verbose and len(relevant_items) == 0:
                # print(f"  Warning: User {user} has no new items after filtering")
        
        if not relevant_items:
            precisions.append(0.0)
            continue    
        
        # Compute precision
        hits = len(set(pred[user][:K]) & relevant_items)
        precisions.append(hits / K)
    
    # Statistics
    nusers = len(gt.keys())
    n_missing = len(missing_preds_users)
    n_evaluated = len(precisions)
    mean_precision = np.mean(precisions) if precisions else 0.0
    
    stats = {
        'total_gt_users': nusers,
        'evaluated_users': n_evaluated,
        'users_without_preds': n_missing,
        'coverage_rate': n_evaluated / nusers if nusers > 0 else 0.0,
        'mean_precision': mean_precision,
        'filter_bought_items': filter_bought_items
    }
    
    if verbose:
        print(f"\n  Evaluation Statistics:")
        print(f"  ├─ Total GT users: {stats['total_gt_users']:,}")
        print(f"  ├─ Evaluated: {stats['evaluated_users']:,}")
        print(f"  ├─ Missing Preds: {stats['users_without_preds']:,}")
        print(f"  ├─ Filter Bought: {filter_bought_items}")
        print(f"  └─ Precision@{K}: {stats['mean_precision']:.4f}")
    
    return mean_precision, missing_preds_users, stats

def predict_with_cold_start_fallback(
    model: XGBRanker,
    lf: pl.LazyFrame,
    features: List[str],
    gt_users: List[Any],
    hist_dict: Dict[Any, List[Any]],
    cold_start_items: List[Any],
    top_k: int = 10,
    batch_size: int = 500_000,
    verbose: bool = True
) -> Dict[Any, List[Any]]:
    """
    Predict with fallback for cold-start users
    """
    
    # Regular prediction (same as before)
    results = predict_top_k(
        model, lf, features, gt_users,
        top_k=top_k, batch_size=batch_size, verbose=verbose
    )
    
    # Fill in cold-start users
    cold_start_count = 0
    for user in gt_users:
        if user not in hist_dict or len(hist_dict[user]) == 0:
            # Cold-start user - use fallback
            if user not in results or len(results[user]) == 0:
                results[user] = cold_start_items[:top_k]
                cold_start_count += 1
    
    if verbose:
        print(f"\n[Cold-Start Fallback] Applied to {cold_start_count:,} users")
    
    return results

# ============================================================================
# MODIFIED: MAIN PIPELINE
# ============================================================================

# ... (Previous imports and functions remain the same) ...

# ============================================================================
# MODIFIED: MAIN PIPELINE
# ============================================================================

def main():
    """
    Modified pipeline using historical GT for test window
    """
    
    # SETTINGS
    RECREATE_PREPROCESSED = True
    PROCESSED_DIR = f"{BASE_DIR}/processed_v1"
    TRAIN_SAMPLE_RATE = 0.2
    VAL_SAMPLE_RATE = 0.2
    TEST_SAMPLE_RATE = 1.0
    
    # FILE PATHS
    HISTORICAL_GT_PATH = f"{BASE_DIR}/01-2025.pkl"  # ← Historical window
    CURRENT_GT_PATH = f"{BASE_DIR}/final_groundtruth.pkl"     # ← Evaluation target
    
    print("="*60)
    print("RECOMMENDATION SYSTEM PIPELINE")
    print("Using Historical GT as Test Window")
    print("="*60)
    
    # ========================================================================
    # 1. Load Ground Truth (02-2025.pkl)
    # ========================================================================
    print("\n[1/8] Loading Ground Truth (02-2025)...")
    with open(CURRENT_GT_PATH, 'rb') as f:
        gt = pickle.load(f)
    
    # FIX 1: Ensure GT IDs are strings to match the casted Parquet data later
    gt_user_ids = [str(k) for k in gt.keys()]
    print(f"  GT Users (Feb 2025): {len(gt_user_ids):,}")
    
    # ========================================================================
    # 2. Initialize Raw Data
    # ========================================================================
    print("\n[2/8] Loading Raw Data...")
    raw_trans = pl.scan_parquet(TRANSACTIONS_GLOB)
    raw_items = pl.scan_parquet(ITEMS_PATH)
    raw_users = pl.scan_parquet(USERS_GLOB)
    print("  ✓ Data loaded (lazy)")
    
    # ========================================================================
    # 3. Create Splits (Modified with Historical GT)
    # ========================================================================
    print("\n[3/8] Creating Time Splits...")
    ROBUST_SPLITS = create_realistic_splits_with_historical_gt(
        raw_trans,
        historical_gt_path=HISTORICAL_GT_PATH,
        verbose=True
    )

    # ========================================================================
    # 6. Process TEST Split (WITH HISTORICAL GT)
    # ========================================================================
    print("\n[6/8] Processing TEST Split (Using Historical GT)...")
    
    # FIX 2: Cast 'customer_id' to String BEFORE filtering
    # The error happened here because raw_users['customer_id'] was Int32 
    # but gt_user_ids was List[String].
    gt_users_only = (
        raw_users
        .with_columns(pl.col("customer_id").cast(pl.Utf8))  # <--- CRITICAL FIX
        .filter(pl.col("customer_id").is_in(gt_user_ids))
    )
    
    # Use modified preprocessing
    te_trans, te_items, te_users = get_preprocessed_data_with_historical_gt(
        "test", 
        ROBUST_SPLITS['test'], 
        raw_trans, 
        raw_items, 
        gt_users_only,  # Now contains correct types
        output_dir=PROCESSED_DIR, 
        recreate=RECREATE_PREPROCESSED,
        sample_rate=TEST_SAMPLE_RATE,
        verbose=True
    )
    
    # Build test dataset with MODIFIED config
    test_config_modified = {
        'hist_start': datetime(2024, 12, 1),   # Last 2 months before Jan 2025
        'hist_end': datetime(2025, 1, 31),     # End of historical GT period
        'target_start': None,
        'target_end': None
    }
    
    test_ds = build_dataset_v2(
        "test", 
        test_config_modified,  # Use modified config
        te_trans, 
        te_items, 
        te_users, 
        is_train=False,
        verbose=True,
        allow_repurchase=True
    )
    
    # ========================================================================
    # 4. Process TRAIN Split
    # ========================================================================
    print("\n[4/8] Processing TRAIN Split...")
    t_trans, t_items, t_users = get_preprocessed_data(
        "train", ROBUST_SPLITS['train'], 
        raw_trans, raw_items, raw_users,
        output_dir=PROCESSED_DIR, 
        recreate=RECREATE_PREPROCESSED,
        sample_rate=TRAIN_SAMPLE_RATE
    )
    
    train_ds = build_dataset_v2(
        "train",
        ROBUST_SPLITS["train"],
        t_trans, t_items, t_users,
        is_train=True,
        verbose=True,
        seed=42,
        do_recall_check=True,
        recall_k=80,
        allow_repurchase=True
    )
    
    # ========================================================================
    # 5. Process VAL Split
    # ========================================================================
    print("\n[5/8] Processing VAL Split...")
    v_trans, v_items, v_users = get_preprocessed_data(
        "val", ROBUST_SPLITS['val'], 
        raw_trans, raw_items, raw_users,
        output_dir=PROCESSED_DIR, 
        recreate=RECREATE_PREPROCESSED,
        sample_rate=VAL_SAMPLE_RATE
    )
    
    val_ds = build_dataset_v2(
        "val",
        ROBUST_SPLITS["val"],
        v_trans, v_items, v_users,
        is_train=False,
        verbose=True,
        do_recall_check=True,
        recall_k=80,
        allow_repurchase=True
    )
    val_ds = sample_validation_data(
        val_ds, 
        target_positive_rate=0.05,
        verbose=True
    )
    
    # ========================================================================
    # 7. Train Model (Same as before)
    # ========================================================================
    print("\n[7/8] Training Model...")
    
    X_train, y_train, g_train, feats, _ = prepare_for_xgb_v2(
        train_ds, is_train=True, verbose=True
    )
    
    X_val, y_val, g_val, _, _ = prepare_for_xgb_v2(
        val_ds, is_train=False, verbose=True, filter_no_positive_users=True
    )
    
    model = train_model(
        X_train, y_train, g_train, 
        X_val, y_val, g_val,
        verbose=True
    )
    model.save_model(f"{OUTPUT_DIR}/xgb_ranker.json")
    
    del X_train, y_train, g_train, X_val, y_val, g_val, train_ds, val_ds
    gc.collect()
    
    # ========================================================================
    # 8. Predict & Evaluate
    # ========================================================================
    print("\n[8/8] Generating Predictions...")
    
    cold_start_items = create_cold_start_recommendations(
        te_items, te_trans,
        test_config_modified['hist_start'],
        test_config_modified['hist_end'],
        verbose=True
    )
    
    # Build history from HISTORICAL GT + transaction data
    hist_dict = build_history_dict(
        te_trans,
        test_config_modified['hist_start'],
        test_config_modified['hist_end'],
        verbose=True
    )
    
    preds = predict_with_cold_start_fallback(
        model, test_ds, feats, gt_user_ids,
        hist_dict, cold_start_items,
        top_k=10, verbose=True
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Make sure to handle keys as matching types (String) in evaluation
    gt_str_keys = {str(k): v for k, v in gt.items()}
    
    p_at_10_no_filter, _, stats_no_filter = precision_at_k(
        pred=preds, gt=gt_str_keys, hist=hist_dict,
        filter_bought_items=False, K=10, verbose=True
    )
    
    p_at_10_filtered, _, stats_filtered = precision_at_k(
        pred=preds, gt=gt_str_keys, hist=hist_dict,
        filter_bought_items=True, K=10, verbose=True
    )
    
    p_at_10_customer, cold_start_users, stats_customer = precision_at_k_customer(
        pred=preds, gt=gt_str_keys, hist=hist_dict,
        filter_bought_items=False, K=10, return_stats=True
    )
    
    # ========================================================================
    # 9. Save Results
    # ========================================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    eval_results = {
        'precision_at_10': {
            'with_repurchase': float(p_at_10_no_filter),
            'without_repurchase': float(p_at_10_filtered),
            'customer_metric': float(p_at_10_customer)
        },
        'statistics': {
            'no_filter': stats_no_filter,
            'filtered': stats_filtered,
            'customer': stats_customer
        },
        'historical_gt_used': HISTORICAL_GT_PATH,
        'evaluation_gt_used': CURRENT_GT_PATH
    }
    
    eval_path = f"{OUTPUT_DIR}/evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"✓ Evaluation saved: {eval_path}")
    
    output_path = f"{OUTPUT_DIR}/predictions.json"
    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in preds.items()}, f, indent=2)
    print(f"✓ Predictions saved: {output_path}")
    
    # Feature importance
    print("\n" + "="*60)
    print("TOP 10 FEATURE IMPORTANCE")
    print("="*60)
    
    importance = model.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(
        importance.items(), key=lambda x: x[1], reverse=True
    )
    
    for feat, score in sorted_importance[:10]:
        feat_idx = int(feat.replace('f', ''))
        feat_name = feats[feat_idx] if feat_idx < len(feats) else feat
        print(f"  {feat_name}: {score:.2f}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Final Precision@10 (with repurchase): {p_at_10_no_filter:.4f}")
    print(f"Final Precision@10 (no repurchase):   {p_at_10_filtered:.4f}")

if __name__ == "__main__":
    main()