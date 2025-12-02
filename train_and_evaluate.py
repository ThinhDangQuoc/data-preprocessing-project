from __future__ import annotations
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, List, Any
import numpy as np
import polars as pl
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import your enhanced feature builder
from feature_builder import build_feature_label_table

# --- CONFIGURATION ---
TRANSACTIONS_GLOB = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.user_chunk*.parquet"

# Define the number of features created in builder
NUM_FEATURES = 29 

@dataclass(frozen=True)
class FeatureWindow:
    history_start: datetime
    history_end: datetime
    recent_start: datetime
    recent_end: datetime

TRAIN_WINDOW = FeatureWindow(
    history_start=datetime(2024, 11, 1),
    history_end=datetime(2024, 12, 1),
    recent_start=datetime(2024, 12, 2),
    recent_end=datetime(2024, 12, 6),
)

VAL_WINDOW = FeatureWindow(
    history_start=datetime(2024, 11, 4),
    history_end=datetime(2024, 12, 1),
    recent_start=datetime(2024, 12, 4),
    recent_end=datetime(2024, 12, 6),
)

TEST_WINDOW = FeatureWindow(
    history_start=datetime(2024, 12, 9),
    history_end=datetime(2024, 12, 12),
    recent_start=datetime(2024, 12, 13),
    recent_end=datetime(2024, 12, 19),
)

RANK_KS = (5, 10, 20)

# --- NEW: DYNAMIC GROUND TRUTH GENERATOR ---
def generate_ground_truth(
    transactions: pl.LazyFrame, 
    window: FeatureWindow,
    customer_col: str = "customer_id",
    item_col: str = "item_id",
    time_col: str = "created_date"
) -> Dict[Any, Dict[str, List[Any]]]:
    """
    Generates ground truth from the transactions that occurred during the 
    'recent' (target) period of the provided window.
    """
    print(f"Generating Ground Truth for window: {window.recent_start} to {window.recent_end}")
    
    gt_df = (
        transactions
        .filter(pl.col(time_col) >= window.recent_start)
        .filter(pl.col(time_col) <= window.recent_end)
        .select([customer_col, item_col])
        .unique() # We care if they bought the item, not how many times (for Precision@K)
        .group_by(customer_col)
        .agg(pl.col(item_col).alias("list_items"))
        .collect()
    )
    
    # Format: {customer_id: {'list_items': [item_id, item_id, ...]}}
    gt_dict = {}
    for row in gt_df.iter_rows(named=True):
        gt_dict[row[customer_col]] = {"list_items": row["list_items"]}
        
    print(f"Ground truth generated for {len(gt_dict)} users.")
    return gt_dict

# --- EVALUATION LOGIC ---
def precision_at_k(pred, gt, hist, filter_bought_items=True, K=10):
    precisions = []
    ncold_start = 0
    cold_start_users = []
    
    for user in gt.keys():
        # Check if user exists in our prediction set
        if user not in pred:
            # If the user is in GT but not in Pred, it usually means 
            # we didn't generate candidates for them (Cold Start or filtering).
            # We count this as 0 precision.
            precisions.append(0.0)
            ncold_start += 1
            cold_start_users.append(user)
            continue
        
        gt_items = gt[user]['list_items']
        relevant_items = set(gt_items)
        
        # Filter items already seen in history (if configured)
        if filter_bought_items and user in hist:
            relevant_items -= set(hist[user])
            
        # If no relevant items remain after filtering (e.g. user only bought rebuy items), skip
        if len(relevant_items) == 0:
            continue
        
        if len(pred[user]) == 0:
            precisions.append(0.0)
            continue
            
        hits = len(set(pred[user][:K]) & relevant_items)
        precisions.append(hits / K)
    
    if not precisions:
        return 0.0, cold_start_users
    return np.mean(precisions), cold_start_users

# --- PIPELINE FUNCTIONS ---
def _scan_sources() -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    transactions = pl.scan_parquet(TRANSACTIONS_GLOB)
    items = pl.scan_parquet(ITEMS_PATH)
    users = pl.scan_parquet(USERS_GLOB)
    return transactions, items, users

def _build_table(
    window: FeatureWindow,
    *,
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame
) -> pl.DataFrame:
    print(f"Building table for window: History {window.history_start}-{window.history_end} | Target {window.recent_start}-{window.recent_end}")
    return build_feature_label_table(
        transactions,
        items,
        users,
        window.history_start,
        window.history_end,
        window.recent_start,
        window.recent_end,
        transaction_time_col="created_date",
        customer_id_col="customer_id",
        price_col="price",
        quantity_col="quantity"
    ).collect()

def extract_features(table: pl.DataFrame) -> np.ndarray:
    feature_cols = [f"X_{i}" for i in range(1, NUM_FEATURES + 1)]
    return table.select(feature_cols).to_numpy().astype(np.float64)

def train_model(
    train_table: pl.DataFrame, 
    model_type: str = "xgboost",
    val_table: pl.DataFrame = None
) -> Tuple[Any, StandardScaler]:
    
    print("  -> Preprocessing training data...")
    X = extract_features(train_table)
    y = train_table["Y"].to_numpy()
    
    # Calculate scale_pos_weight for imbalance
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"  -> Class balance: {neg_count} Neg, {pos_count} Pos. Scale Weight: {scale_pos_weight:.2f}")
    
    scaler = StandardScaler()
    scaler.fit(X) 
    
    eval_set = None
    if val_table is not None:
        X_val = extract_features(val_table)
        y_val = val_table["Y"].to_numpy()
        eval_set = [(X, y), (X_val, y_val)]
    
    print("  -> Starting XGBoost training...")
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=50 if eval_set else None
    )
    
    fit_params = {'verbose': False}
    if eval_set:
        fit_params['eval_set'] = eval_set
    
    model.fit(X, y, **fit_params)
    
    if eval_set and hasattr(model, 'best_iteration'):
        print(f"  -> Best iteration: {model.best_iteration}")

    # --- FEATURE IMPORTANCE WITH PERCENTAGES ---
    importances = model.feature_importances_
    feature_names = [f"X_{i}" for i in range(1, NUM_FEATURES + 1)]
    
    # Sort indices descending
    sorted_indices = np.argsort(importances)[::-1]
    
    print(f"\n  -> Top 10 Feature Importances:")
    for i in range(min(10, len(importances))):
        idx = sorted_indices[i]
        imp_percent = importances[idx] * 100
        print(f"     {feature_names[idx]}: {imp_percent:.2f}%")
        
    
    return model, scaler
def score_table(model: Any, scaler: StandardScaler, table: pl.DataFrame) -> pl.DataFrame:
    X = extract_features(table)
    scores = model.predict_proba(X)[:, 1]
    return table.with_columns(pl.Series("score", scores))

def _build_history_dict(transactions: pl.LazyFrame, window: FeatureWindow) -> Dict[Any, List[Any]]:
    # Items bought BEFORE the prediction window (History)
    # Used to filter out items user has already bought if we want purely new recommendations
    hist_df = (
        transactions
        .filter(pl.col("created_date") < window.recent_start) 
        .select(["customer_id", "item_id"])
        .unique()
        .group_by("customer_id")
        .agg(pl.col("item_id"))
        .collect()
    )
    hist_dict = {cid: items.to_list() for cid, items in zip(hist_df["customer_id"], hist_df["item_id"])}
    return hist_dict

def _build_prediction_dict(scored: pl.DataFrame, max_k: int) -> Dict[Any, List[Any]]:
    # Get top K items per user based on score
    pred_df = (
        scored
        .sort(["X-1", "score"], descending=[False, True])
        .group_by("X-1")
        .agg(pl.col("X_0").head(max_k))
    )
    pred_dict = {cid: items.to_list() for cid, items in zip(pred_df["X-1"], pred_df["X_0"])}
    return pred_dict

def main() -> None:
    transactions, items, users = _scan_sources()
    
    # 1. Build Data Tables
    # We do not use ground truth for Training, only for final Evaluation
    print("\n--- 1. Building Feature Tables ---")
    train_table = _build_table(TRAIN_WINDOW, transactions=transactions, items=items, users=users)
    val_table = _build_table(VAL_WINDOW, transactions=transactions, items=items, users=users)
    
    # 2. Train Model
    print("\n--- 2. Training Model ---")
    model, scaler = train_model(
        train_table, 
        model_type="xgboost",
        val_table=val_table
    )
    
    # 3. Prepare Test Data
    print("\n--- 3. Preparing Test Evaluation ---")
    test_table = _build_table(TEST_WINDOW, transactions=transactions, items=items, users=users)
    
    # --- HERE IS THE CHANGE ---
    # Generate Ground Truth dynamically for the Test Window
    test_ground_truth = generate_ground_truth(
        transactions, 
        TEST_WINDOW,
        customer_col="customer_id",
        item_col="item_id",
        time_col="created_date"
    )
    
    # 4. Score Test Data
    print("\n--- 4. Scoring Test Data ---")
    scored_test = score_table(model, scaler, test_table)
    
    # 5. Evaluate
    print(f"\n--- 5. Final Evaluation on Test Window ({TEST_WINDOW.recent_start.date()} - {TEST_WINDOW.recent_end.date()}) ---")
    
    # Build history dict to optionally filter out items the user has bought in the past
    # (depending on business logic, usually we filter them out for discovery)
    hist_dict = _build_history_dict(transactions, TEST_WINDOW)
    
    # Get predictions (Top 20 to cover all K metrics)
    pred_dict = _build_prediction_dict(scored_test, 20)
    
    for k in RANK_KS:
        # filter_bought_items=True means we don't reward recommending items the user 
        # already bought in the 'history' window.
        p, _ = precision_at_k(pred_dict, test_ground_truth, hist_dict, filter_bought_items=True, K=k)
        print(f"  Precision@{k}: {p:.4f}")

if __name__ == "__main__":
    main()