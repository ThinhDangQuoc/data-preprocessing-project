from __future__ import annotations
import pickle
import json
import os
import gc
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import polars as pl
from xgboost import XGBRanker

# Import your modules
# Ensure feature_builder.py is the optimized version provided in the previous turn
from feature_builder import generate_candidates, build_ranking_features
from collaborative_filter import CollaborativeFilter

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE_DIR = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system"
TRANSACTIONS_GLOB = f"{BASE_DIR}/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = f"{BASE_DIR}/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = f"{BASE_DIR}/dataset/sales_pers.user_chunk*.parquet"
GT_PKL_PATH = f"{BASE_DIR}/groundtruth.pkl"

OUTPUT_DIR = f"{BASE_DIR}/outputs"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "xgbranker_enhanced_discovery.pkl")
SUBMISSION_SAVE_PATH = os.path.join(OUTPUT_DIR, "submission_enhanced_discovery.json")

# Mapping for display (We will dynamically detect actual count later)
FEATURE_NAMES_MAP = {
    "X_1": "User Total Txns", "X_2": "User Unique Items", "X_3": "User Avg Spend",
    "X_4": "User Days Inactive", "X_5": "User Diversity Score", "X_6": "Item Global Count",
    "X_7": "Item Unique Users", "X_8": "Item Avg Price", "X_9": "Item Trend Score",
    "X_10": "User-Item Buy Count", "X_11": "User-Item Total Qty", "X_12": "User-Item Days Since Last Buy",
    "X_13": "User-Item Purchase Span", "X_14": "User-Category Buy Count", "X_15": "Price Ratio",
    "X_16": "Category Affinity", "X_17": "User-Item Loyalty", "X_18": "Item Penetration"
}

@dataclass(frozen=True)
class FeatureWindow:
    history_start: datetime
    history_end: datetime
    recent_start: datetime | None
    recent_end: datetime | None

# Standard split logic
TRAIN_WINDOW = FeatureWindow(
    history_start=datetime(2024, 11, 1), history_end=datetime(2024, 12, 15),
    recent_start=datetime(2024, 12, 16), recent_end=datetime(2024, 12, 20),
)
VAL_WINDOW = FeatureWindow(
    history_start=datetime(2024, 12, 1), history_end=datetime(2024, 12, 20),
    recent_start=datetime(2024, 12, 21), recent_end=datetime(2024, 12, 30),
)
# Test window implies inference for future (or holdout set)
TEST_WINDOW = FeatureWindow(
    history_start=datetime(2024, 12, 1), history_end=datetime(2024, 12, 30),
    recent_start=None, recent_end=None,
)

RANK_KS = (5, 10)
SAMPLE_USERS_FRACTION = 0.5 
MAX_CANDIDATES_PER_USER = 50 
CF_N_FACTORS = 50 
CF_ENABLED = True

# Tuning parameters
CANDIDATE_PARAMS = {
    "n_trending": 100,
    "n_popular": 150,
    "n_user_history": 50,
    "n_similar_items": 50,
    "n_new_items": 30,
    "n_price_band": 30,
}

# ==============================================================================
# HELPERS
# ==============================================================================

def load_ground_truth(pkl_path: str) -> Dict[Any, Dict[str, List[Any]]]:
    print(f"Loading Ground Truth from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        gt_raw = pickle.load(f)
    return {cid: {"list_items": item_list} for cid, item_list in gt_raw.items()}

def _scan_sources() -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    transactions = pl.scan_parquet(TRANSACTIONS_GLOB)
    items = pl.scan_parquet(ITEMS_PATH)
    users = pl.scan_parquet(USERS_GLOB)
    return transactions, items, users

def get_sampled_users(users_lf: pl.LazyFrame, fraction: float, seed: int = 42) -> pl.LazyFrame:
    """
    Downsample users at the source LazyFrame level.
    This prevents generating candidates for users we won't use.
    """
    if fraction >= 1.0:
        return users_lf
    
    print(f"  > Downsampling users to {fraction:.0%} (Push-down optimization)...")
    return users_lf.filter(pl.col("customer_id").hash(seed=seed) % 100 < (fraction * 100))

# ==============================================================================
# PIPELINE BUILDER
# ==============================================================================

def _build_two_stage_lazy(
    window: FeatureWindow,
    *,
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame, # Expected to be already filtered if needed
    cf_model: CollaborativeFilter = None,
    gt_dict: Dict = None,
) -> pl.LazyFrame:
    
    mode = "TRAIN/VAL" if window.recent_start else "TEST"
    print(f"\nBuilding Lazy Graph for {mode} | Hist: {window.history_start.date()} -> {window.history_end.date()}")
    
    # 1. Candidate Generation
    candidates = generate_candidates(
        transactions=transactions, items=items, users=users,
        begin_hist=window.history_start, end_hist=window.history_end,
        gt_dict=gt_dict, cf_model=cf_model, 
        n_cf_candidates=100,
        **CANDIDATE_PARAMS
    )

    # 1.5 Inject Hard Positives (Train Only)
    if window.recent_start is not None:
        actual_purchases = (
            transactions
            .filter(
                (pl.col("created_date") >= window.recent_start) & 
                (pl.col("created_date") <= window.recent_end)
            )
            .select(["customer_id", "item_id"])
            .unique()
            # Must join with our target user subset to ensure we don't re-introduce filtered users
            .join(users.select("customer_id"), on="customer_id")
            .with_columns(
                pl.lit("injected_truth").cast(pl.Categorical).alias("candidate_source")
            )
        )
        candidates = pl.concat([candidates, actual_purchases], how="diagonal") \
                       .unique(subset=["customer_id", "item_id"])

    # 2. Base Feature Engineering
    ranked_lf = build_ranking_features(
        candidates=candidates, transactions=transactions, items=items, users=users,
        end_hist=window.history_end,
        # Note: ranking feature builder in previous step didn't take recent start/end
        # If your feature_builder has labels generation inside, pass them.
        # Assuming the optimized version provided previously which calculates labels internally if provided:
    )
    
    # 2.5 Add Labels (Y) if we have ground truth info
    if window.recent_start:
        # Create labels based on actual purchases in recent window
        labels = (
            transactions
            .filter(
                (pl.col("created_date") >= window.recent_start) &
                (pl.col("created_date") <= window.recent_end)
            )
            .select(["customer_id", "item_id"])
            .unique()
            .with_columns(pl.lit(1).cast(pl.UInt8).alias("Y"))
        )
        ranked_lf = ranked_lf.join(labels, on=["customer_id", "item_id"], how="left")
        ranked_lf = ranked_lf.with_columns(pl.col("Y").fill_null(0))
    elif "Y" not in ranked_lf.schema:
        # Test mode
        ranked_lf = ranked_lf.with_columns(pl.lit(0).cast(pl.UInt8).alias("Y"))

    return ranked_lf

def prepare_data_pipeline(
    lf: pl.LazyFrame, 
    is_train: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    
    print("  > collecting data from graph...")
    # OPTIMIZATION: Cast to Float32 inside Polars to save 50% RAM on materialization
    # Dynamically find X columns
    schema = lf.schema
    feature_cols = sorted([c for c in schema.keys() if c.startswith("X_") and c != "X-1" and c != "X_0"], 
                          key=lambda x: int(x.split('_')[1]))
    
    print(f"  > Found {len(feature_cols)} features.")

    # Cast to Float32 immediately
    lf = lf.with_columns([pl.col(c).cast(pl.Float32) for c in feature_cols])

    df = lf.collect()
    
    if is_train and MAX_CANDIDATES_PER_USER:
        print(f"  > Downsampling negatives (Keep Top {MAX_CANDIDATES_PER_USER} per user)...")
        # Efficient stratified sampling
        positives = df.filter(pl.col("Y") == 1)
        negatives = df.filter(pl.col("Y") == 0)
        
        if len(negatives) > 0:
            negatives = (
                negatives
                .with_columns(pl.lit(np.random.rand(len(negatives))).alias("__rnd"))
                .sort(["X-1", "__rnd"]) # Sorting by ID then Random
                .set_sorted("X-1")
                .group_by("X-1", maintain_order=True)
                .head(MAX_CANDIDATES_PER_USER)
                .drop("__rnd")
            )
            df = pl.concat([positives, negatives])
            del positives, negatives
            gc.collect()

    print("  > Sorting by Customer ID for XGBRanker group...")
    df = df.sort("X-1")
    
    print("  > converting to numpy...")
    X = df.select(feature_cols).to_numpy() # Already Float32
    y = df["Y"].to_numpy().astype(np.float32)
    
    # Calculate groups efficiently
    # Run Length Encoding on Sorted Customer IDs
    groups = (
        df.select(pl.col("X-1").rle())
        .unnest("X-1")
        .select("len")
        .to_numpy()
        .flatten()
        .astype(np.uint32)
    )
    
    print(f"  > Data Shape: X={X.shape}, Groups={len(groups)}")
    
    del df
    gc.collect()
    return X, y, groups, feature_cols

# ==============================================================================
# TRAINING
# ==============================================================================

def train_ranker_optimized(
    transactions, items, users, cf_model
) -> Tuple[XGBRanker, List[str]]:
    
    # 1. Filter Users EARLY (Push-down)
    # This prevents building candidates for users we will just drop later
    train_users = get_sampled_users(users, SAMPLE_USERS_FRACTION, seed=42)
    val_users = get_sampled_users(users, 0.2, seed=99) # Smaller val set

    # 2. Build Lazy Pipelines
    train_lf = _build_two_stage_lazy(
        TRAIN_WINDOW, transactions=transactions, items=items, users=train_users, cf_model=cf_model
    )
    val_lf = _build_two_stage_lazy(
        VAL_WINDOW, transactions=transactions, items=items, users=val_users, cf_model=cf_model
    )
    
    # 3. Materialize
    print("\n[Pipeline] Processing Training Data...")
    X_train, y_train, groups_train, feat_cols = prepare_data_pipeline(train_lf, is_train=True)
    
    print("\n[Pipeline] Processing Validation Data...")
    X_val, y_val, groups_val, _ = prepare_data_pipeline(val_lf, is_train=True)

    # 4. Train
    print("\n" + "="*80 + "\n=== Training XGBRanker ===\n" + "="*80)
    model = XGBRanker(
        objective='rank:ndcg',
        n_estimators=150,      # Increased but relies on early stopping
        learning_rate=0.05,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        tree_method='hist',    # Fast histogram optimized
        random_state=42,
        n_jobs=-1,
        eval_metric='ndcg@10',
        early_stopping_rounds=15
    )
    
    model.fit(
        X_train, y_train, 
        group=groups_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_group=[groups_train, groups_val],
        verbose=True
    )
    
    return model, feat_cols

# ==============================================================================
# INFERENCE (BATCHED)
# ==============================================================================

def batched_inference_pipeline(
    model: XGBRanker, 
    test_lf: pl.LazyFrame, 
    feature_cols: List[str],
    top_k: int = 10,
    batch_size_users: int = 10_000
) -> Dict:
    """
    Processes test data in chunks of users to ensure we never OOM.
    """
    print(f"\n[Pipeline] Starting Batched Inference (User Batch: {batch_size_users})...")
    
    # Get all unique users in test set
    all_test_users = test_lf.select("X-1").unique().collect()["X-1"].to_list()
    total_users = len(all_test_users)
    print(f"  > Total Test Users: {total_users:,}")
    
    final_preds = {}
    
    # Iterate in chunks
    for i in range(0, total_users, batch_size_users):
        chunk_users = all_test_users[i : i + batch_size_users]
        print(f"  > Processing Batch {i//batch_size_users + 1} ({len(chunk_users)} users)...")
        
        # Filter LazyFrame to this chunk
        chunk_lf = test_lf.filter(pl.col("X-1").is_in(chunk_users))
        
        # Collect only this chunk
        df_chunk = chunk_lf.select(["X-1", "X_0"] + feature_cols).collect()
        
        if df_chunk.is_empty():
            continue
            
        X_chunk = df_chunk.select(feature_cols).to_numpy().astype(np.float32)
        
        # Predict
        # Note: XGBoost isn't great at predicting empty arrays, ensure data exists
        if X_chunk.shape[0] > 0:
            scores = model.predict(X_chunk)
            df_chunk = df_chunk.with_columns(pl.Series("score", scores))
            
            # Select Top K
            # Sort by User, then Score Descending
            # Then head(k) per user
            best_items = (
                df_chunk
                .select(["X-1", "X_0", "score"])
                .sort(["X-1", "score"], descending=[False, True])
                .group_by("X-1")
                .agg(pl.col("X_0").head(top_k))
            )
            
            # Update Dict
            batch_res = {row[0]: row[1].to_list() for row in best_items.iter_rows()}
            final_preds.update(batch_res)
        
        # Clean up
        del df_chunk, X_chunk
        gc.collect()

    return final_preds

# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    transactions, items, users = _scan_sources()
    
    # Train CF Model
    cf_model = None
    if CF_ENABLED:
        print("\n[CF Training] Enhanced collaborative filtering...")
        cf_model = CollaborativeFilter(n_factors=CF_N_FACTORS, min_support=2)
        cf_model.fit(
            transactions=transactions,
            begin_date=TRAIN_WINDOW.history_start,
            end_date=TRAIN_WINDOW.recent_end,
        )
    
    # Train Ranker
    model, feature_cols = train_ranker_optimized(transactions, items, users, cf_model)
    
    # Save Model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f: pickle.dump(model, f)
    
    # Inference
    test_gt = load_ground_truth(GT_PKL_PATH)
    
    # Build Test Graph (For all target users in GT)
    # Note: We pass the target dictionary so generate_candidates filters internally
    test_lf = _build_two_stage_lazy(
        TEST_WINDOW, transactions=transactions, items=items, users=users, 
        cf_model=cf_model, gt_dict=test_gt
    )
    
    # Run Batched Inference
    preds = batched_inference_pipeline(model, test_lf, feature_cols, top_k=10)
    
    # Evaluation
    print("\n" + "="*80 + "\nFINAL EVALUATION\n" + "="*80)
    # Get history for filtering
    hist_items = (
        transactions
        .filter(pl.col("created_date") <= TEST_WINDOW.history_end)
        .select(["customer_id", "item_id"]).unique()
        .group_by("customer_id").agg(pl.col("item_id"))
        .collect()
    )
    hist_dict = {row[0]: set(row[1]) for row in hist_items.iter_rows()}
    
    def precision_at_k(pred, gt, hist, k):
        precisions = []
        for user, data in gt.items():
            if user not in pred: 
                precisions.append(0.0)
                continue
            
            gt_items = set(data['list_items'])
            # Filter bought
            user_hist = hist.get(user, set())
            gt_items = gt_items - user_hist
            
            if not gt_items: continue
            
            rec_items = pred[user][:k]
            hits = len(set(rec_items) & gt_items)
            precisions.append(hits / k)
        return np.mean(precisions) if precisions else 0.0

    for k in RANK_KS:
        p = precision_at_k(preds, test_gt, hist_dict, k)
        print(f"  P@{k}: {p:.4f}")
        
    with open(SUBMISSION_SAVE_PATH, 'w') as f: 
        json.dump({str(k): v for k, v in preds.items()}, f)
        
    print(f"\n✅ Pipeline Complete. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()