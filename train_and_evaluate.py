from __future__ import annotations
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, List, Any
import numpy as np
import polars as pl
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

from feature_builder import build_feature_label_table, build_feature_label_table_with_gt

# --- CONFIGURATION ---
TRANSACTIONS_GLOB = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.user_chunk*.parquet"
GT_PKL_PATH = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/groundtruth.pkl"

# Define the number of features created in builder
NUM_FEATURES = 29 

@dataclass(frozen=True)
class FeatureWindow:
    history_start: datetime
    history_end: datetime
    recent_start: datetime
    recent_end: datetime

# Train: Nov 1 - Dec 12 history | Dec 13-19 target
TRAIN_WINDOW = FeatureWindow(
    history_start=datetime(2024, 11, 1),
    history_end=datetime(2024, 12, 12),
    recent_start=datetime(2024, 12, 13),
    recent_end=datetime(2024, 12, 19),
)

# === FIX 1: CLOSE THE VALIDATION GAP ===
# Val: History must extend to Dec 19 so there is no gap before Target (Dec 20)
VAL_WINDOW = FeatureWindow(
    history_start=datetime(2024, 11, 1),
    history_end=datetime(2024, 12, 19),  # Fixed: Was Dec 12, now Dec 19
    recent_start=datetime(2024, 12, 20),
    recent_end=datetime(2024, 12, 26),
)

# Test: Extended history (Nov 1 - Dec 30) | GT labels
TEST_WINDOW = FeatureWindow(
    history_start=datetime(2024, 11, 1),
    history_end=datetime(2024, 12, 30),
    recent_start=None,
    recent_end=None,
)

RANK_KS = (5, 10, 20)

def load_ground_truth(pkl_path: str) -> Dict[Any, Dict[str, List[Any]]]:
    print(f"Loading Ground Truth from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        gt_raw = pickle.load(f)
    
    gt_dict = {customer_id: {"list_items": item_list} for customer_id, item_list in gt_raw.items()}
    print(f"Ground truth loaded for {len(gt_dict)} users.")
    return gt_dict

def generate_ground_truth(
    transactions: pl.LazyFrame,
    window: FeatureWindow,
    customer_col: str = "customer_id",
    item_col: str = "item_id",
    time_col: str = "created_date"
) -> Dict[Any, Dict[str, List[Any]]]:
    print(f"Generating Ground Truth for: {window.recent_start} to {window.recent_end}")
    gt_df = (
        transactions
        .filter(pl.col(time_col) >= window.recent_start)
        .filter(pl.col(time_col) <= window.recent_end)
        .select([customer_col, item_col])
        .unique()
        .group_by(customer_col)
        .agg(pl.col(item_col).alias("list_items"))
        .collect()
    )
    
    gt_dict = {row[customer_col]: {"list_items": row["list_items"]} for row in gt_df.iter_rows(named=True)}
    print(f"Ground truth generated for {len(gt_dict)} users.")
    return gt_dict

def precision_at_k(pred, gt, hist, filter_bought_items=False, K=10):
    """
    Calculates Precision@K.
    
    Args:
        filter_bought_items: If True, removes items previously bought from the relevant set.
                             CRITICAL: Set this to False if your candidate generation includes
                             historical items (re-purchases). Otherwise, you punish the model
                             for correctly predicting a re-order.
    """
    precisions = []
    ncold_start = 0
    
    for user in gt.keys():
        if user not in pred:
            precisions.append(0.0)
            ncold_start += 1
            continue
        
        gt_items = gt[user]['list_items']
        relevant_items = set(gt_items)
        
        # Only filter if explicitly requested (usually False for grocery/retail)
        if filter_bought_items and user in hist:
            relevant_items -= set(hist[user])
        
        if len(relevant_items) == 0:
            continue
        
        if len(pred[user]) == 0:
            precisions.append(0.0)
            continue
        
        hits = len(set(pred[user][:K]) & relevant_items)
        precisions.append(hits / K)
    
    return np.mean(precisions) if precisions else 0.0, ncold_start

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
    users: pl.LazyFrame,
    gt_dict: Dict = None,
    include_gt_in_candidates: bool = True
) -> pl.DataFrame:
    """
    Build feature table with control over GT inclusion in candidates.
    """
    if gt_dict is not None:
        print(f"Building table with GT | History: {window.history_start.date()}-{window.history_end.date()} | Include GT: {include_gt_in_candidates}")
        return build_feature_label_table_with_gt(
            transactions,
            items,
            users,
            window.history_start,
            window.history_end,
            gt_dict,
            transaction_time_col="created_date",
            customer_id_col="customer_id",
            price_col="price",
            quantity_col="quantity",
            include_gt_in_candidates=include_gt_in_candidates
        ).collect()
    else:
        print(f"Building table | History: {window.history_start.date()}-{window.history_end.date()} | Target: {window.recent_start.date()}-{window.recent_end.date()}")
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
    val_table: pl.DataFrame = None
) -> XGBClassifier:
    print("\n=== Training XGBoost ===")
    X = extract_features(train_table)
    y = train_table["Y"].to_numpy()

    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print(f"Train: {len(y)} samples | Pos: {pos_count} ({(y==1).mean():.4f}) | Neg: {neg_count}")
    print(f"Scale Weight: {scale_pos_weight:.2f}")
    
    eval_set = None
    if val_table is not None:
        X_val = extract_features(val_table)
        y_val = val_table["Y"].to_numpy()
        eval_set = [(X, y), (X_val, y_val)]
        print(f"Val: {len(y_val)} samples | Pos: {(y_val==1).sum()} ({(y_val==1).mean():.4f})")
    
    model = XGBClassifier(
        n_estimators=300,
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
        print(f"Best iteration: {model.best_iteration}")
    
    # Feature importance
    importances = model.feature_importances_
    feature_names = [f"X_{i}" for i in range(1, NUM_FEATURES + 1)]
    sorted_indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 Features:")
    for i in range(min(10, len(importances))):
        idx = sorted_indices[i]
        print(f"  {feature_names[idx]}: {importances[idx]*100:.2f}%")
    
    return model

def score_table(model: XGBClassifier, table: pl.DataFrame) -> pl.DataFrame:
    X = extract_features(table)
    scores = model.predict_proba(X)[:, 1]
    return table.with_columns(pl.Series("score", scores))

def _build_history_dict(transactions: pl.LazyFrame, window: FeatureWindow) -> Dict[Any, List[Any]]:
    hist_df = (
        transactions
        .filter(pl.col("created_date") <= window.history_end)
        .select(["customer_id", "item_id"])
        .unique()
        .group_by("customer_id")
        .agg(pl.col("item_id"))
        .collect()
    )
    return {cid: items.to_list() for cid, items in zip(hist_df["customer_id"], hist_df["item_id"])}

def _build_prediction_dict(scored: pl.DataFrame, max_k: int) -> Dict[Any, List[Any]]:
    pred_df = (
        scored
        .sort(["X-1", "score"], descending=[False, True])
        .group_by("X-1")
        .agg(pl.col("X_0").head(max_k))
    )
    return {cid: items.to_list() for cid, items in zip(pred_df["X-1"], pred_df["X_0"])}

def main() -> None:
    transactions, items, users = _scan_sources()
    
    # === STRATEGY 1: Train with GT INCLUDED (Baseline) ===
    print("\n" + "="*80)
    print("STRATEGY 1: Train with GT items included in candidates (Baseline)")
    print("="*80)
    
    # Generate GT for Train/Val
    train_gt = generate_ground_truth(transactions, TRAIN_WINDOW)
    val_gt = generate_ground_truth(transactions, VAL_WINDOW)
    
    # Build tables WITH GT in candidates
    train_table_with_gt = _build_table(
        TRAIN_WINDOW, 
        transactions=transactions, 
        items=items, 
        users=users, 
        gt_dict=train_gt,
        include_gt_in_candidates=True  # Include GT items for training label coverage
    )
    
    val_table_with_gt = _build_table(
        VAL_WINDOW,
        transactions=transactions,
        items=items,
        users=users,
        gt_dict=val_gt,
        include_gt_in_candidates=True
    )
    
    # Train model
    model_baseline = train_model(train_table_with_gt, val_table_with_gt)
    
    # Test with GT EXCLUDED (realistic scenario)
    test_ground_truth = load_ground_truth(GT_PKL_PATH)
    test_table_baseline = _build_table(
        TEST_WINDOW,
        transactions=transactions,
        items=items,
        users=users,
        gt_dict=test_ground_truth,
        include_gt_in_candidates=False  # EXCLUDE GT items from test candidates
    )
    
    scored_test_baseline = score_table(model_baseline, test_table_baseline)
    hist_dict = _build_history_dict(transactions, TEST_WINDOW)
    pred_dict_baseline = _build_prediction_dict(scored_test_baseline, 20)
    
    print("\n--- Test Results (Strategy 1: Baseline) ---")
    for k in RANK_KS:
        # FIX: filter_bought_items=False because candidates include history (re-purchases allowed)
        p, ncold = precision_at_k(pred_dict_baseline, test_ground_truth, hist_dict, filter_bought_items=False, K=k)
        print(f"  P@{k}: {p:.4f} | Cold start: {ncold}")
    
    # === STRATEGY 2: Train WITHOUT GT (Realistic Training) ===
    print("\n" + "="*80)
    print("STRATEGY 2: Train without GT items in candidates (Realistic)")
    print("="*80)
    
    # Use standard function for train/val
    train_table_realistic = _build_table(TRAIN_WINDOW, transactions=transactions, items=items, users=users)
    val_table_realistic = _build_table(VAL_WINDOW, transactions=transactions, items=items, users=users)
    
    model_realistic = train_model(train_table_realistic, val_table_realistic)
    
    # Test same way
    test_table_realistic = _build_table(
        TEST_WINDOW,
        transactions=transactions,
        items=items,
        users=users,
        gt_dict=test_ground_truth,
        include_gt_in_candidates=False
    )
    
    scored_test_realistic = score_table(model_realistic, test_table_realistic)
    pred_dict_realistic = _build_prediction_dict(scored_test_realistic, 20)
    
    print("\n--- Test Results (Strategy 2: Realistic) ---")
    for k in RANK_KS:
        # FIX: filter_bought_items=False
        p, ncold = precision_at_k(pred_dict_realistic, test_ground_truth, hist_dict, filter_bought_items=False, K=k)
        print(f"  P@{k}: {p:.4f} | Cold start: {ncold}")
    
    # === STRATEGY 3: Hybrid (Optional) ===
    print("\n" + "="*80)
    print("STRATEGY 3: Hybrid - Balance GT inclusion vs generalization")
    print("="*80)
    
    # Merge tables with resampling
    train_hybrid = pl.concat([
        train_table_realistic.sample(fraction=0.7, seed=42),
        train_table_with_gt.sample(fraction=0.3, seed=42)
    ]).unique()
    
    model_hybrid = train_model(train_hybrid, val_table_realistic)
    
    scored_test_hybrid = score_table(model_hybrid, test_table_realistic)
    pred_dict_hybrid = _build_prediction_dict(scored_test_hybrid, 20)
    
    print("\n--- Test Results (Strategy 3: Hybrid) ---")
    for k in RANK_KS:
        # FIX: filter_bought_items=False
        p, ncold = precision_at_k(pred_dict_hybrid, test_ground_truth, hist_dict, filter_bought_items=False, K=k)
        print(f"  P@{k}: {p:.4f} | Cold start: {ncold}")
    
    # === FINAL COMPARISON ===
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    # FIX: Ensure all strategies use the same metric settings (False to include re-orders)
    print("\nStrategy 1 (Baseline - GT in training candidates):")
    for k in RANK_KS:
        p, _ = precision_at_k(pred_dict_baseline, test_ground_truth, hist_dict, filter_bought_items=False, K=k)
        print(f"  P@{k}: {p:.4f}")
    
    print("\nStrategy 2 (Realistic - No GT in candidates):")
    for k in RANK_KS:
        p, _ = precision_at_k(pred_dict_realistic, test_ground_truth, hist_dict, filter_bought_items=False, K=k)
        print(f"  P@{k}: {p:.4f}")
    
    print("\nStrategy 3 (Hybrid):")
    for k in RANK_KS:
        p, _ = precision_at_k(pred_dict_hybrid, test_ground_truth, hist_dict, filter_bought_items=False, K=k)
        print(f"  P@{k}: {p:.4f}")

if __name__ == "__main__":
    main()