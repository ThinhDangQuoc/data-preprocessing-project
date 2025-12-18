from __future__ import annotations
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, List, Any
import numpy as np
import polars as pl
import warnings

# --- IMPORTS ---
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

# Assumes feature_builder.py exists in your local directory
from feature_builder import build_feature_label_table, build_feature_label_table_with_gt

# --- CONFIGURATION ---
TRANSACTIONS_GLOB = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.user_chunk*.parquet"
GT_PKL_PATH = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/groundtruth.pkl"

NUM_FEATURES = 29 

@dataclass(frozen=True)
class FeatureWindow:
    history_start: datetime
    history_end: datetime
    recent_start: datetime | None
    recent_end: datetime | None

TRAIN_WINDOW = FeatureWindow(
    history_start=datetime(2024, 11, 1),
    history_end=datetime(2024, 12, 12),
    recent_start=datetime(2024, 12, 13),
    recent_end=datetime(2024, 12, 19),
)

TEST_WINDOW = FeatureWindow(
    history_start=datetime(2024, 11, 1),
    history_end=datetime(2024, 12, 30),
    recent_start=None,
    recent_end=None,
)

RANK_KS = (5, 10, 20)

# --- DATA LOADING & UTILS ---

def load_ground_truth(pkl_path: str) -> Dict[Any, Dict[str, List[Any]]]:
    print(f"Loading Ground Truth from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        gt_raw = pickle.load(f)
    
    gt_dict = {customer_id: {"list_items": item_list} for customer_id, item_list in gt_raw.items()}
    print(f"Ground truth loaded for {len(gt_dict)} users.")
    return gt_dict

def precision_at_k(pred, gt, hist, filter_bought_items=False, K=10):
    precisions = []
    ncold_start = 0
    
    users = list(gt.keys())
    for user in tqdm(users, desc=f"Evaluating P@{K}", leave=False):
        if user not in pred:
            precisions.append(0.0)
            ncold_start += 1
            continue
        
        gt_items = gt[user]['list_items']
        relevant_items = set(gt_items)
        
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

def _build_table(window: FeatureWindow, *, transactions: pl.LazyFrame, items: pl.LazyFrame, users: pl.LazyFrame, gt_dict: Dict = None, include_gt_in_candidates: bool = False) -> pl.DataFrame:
    if gt_dict is not None:
        print(f"Building Test table | History: {window.history_start.date()}-{window.history_end.date()}")
        return build_feature_label_table_with_gt(
            transactions, items, users, window.history_start, window.history_end, gt_dict,
            transaction_time_col="created_date", customer_id_col="customer_id", price_col="price", quantity_col="quantity",
            include_gt_in_candidates=include_gt_in_candidates
        ).collect()
    else:
        print(f"Building Train table | History: {window.history_start.date()}-{window.history_end.date()}")
        return build_feature_label_table(
            transactions, items, users, window.history_start, window.history_end, window.recent_start, window.recent_end,
            transaction_time_col="created_date", customer_id_col="customer_id", price_col="price", quantity_col="quantity"
        ).collect()

def extract_features(table: pl.DataFrame) -> np.ndarray:
    feature_cols = [f"X_{i}" for i in range(1, NUM_FEATURES + 1)]
    # OPTIMIZATION: Convert to float32 immediately to save 50% RAM
    return table.select(feature_cols).to_numpy().astype(np.float32)

# --- OPTIMIZED MODEL TRAINING ---

def train_logistic_sgd(train_table: pl.DataFrame) -> Any:
    print("\n--- Training Logistic Regression (SGD) ---")
    X = extract_features(train_table)
    y = train_table["Y"].to_numpy()

    # Create a pipeline: Scaler -> Model
    # SGDClassifier with loss='log_loss' is mathematically equivalent to Logistic Regression
    # but uses Stochastic Gradient Descent, which is much faster for large data.
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss='log_loss', 
            penalty='l2',
            alpha=0.0001,
            class_weight='balanced',
            max_iter=1000,
            n_jobs=-1,
            random_state=42,
            early_stopping=True, # Automatically uses 10% data for validation to stop early
            verbose=1
        )
    )
    
    print("Fitting model...")
    model.fit(X, y)
    return model

def train_svm_sgd(train_table: pl.DataFrame) -> Any:
    print("\n--- Training Linear SVM (SGD) ---")
    X = extract_features(train_table)
    y = train_table["Y"].to_numpy()

    # SGDClassifier with loss='hinge' is mathematically equivalent to LinearSVC
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss='hinge', 
            penalty='l2',
            alpha=0.0001,
            class_weight='balanced',
            max_iter=10000,
            n_jobs=-1,
            random_state=42,
            early_stopping=True,
            verbose=1
        )
    )
    
    print("Fitting model...")
    model.fit(X, y)
    return model

# --- SCORING & UTILS ---

def score_table(model: Any, table: pl.DataFrame) -> pl.DataFrame:
    X = extract_features(table)
    
    # Model is now a Pipeline, so it handles Scaling automatically.
    
    if hasattr(model, "predict_proba"):
        # Logistic Regression (SGD log_loss)
        scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        # SVM (SGD hinge) - Hinge loss doesn't provide probabilities, use decision distance
        scores = model.decision_function(X)
    else:
        # Fallback for Pipeline wrapper
        scores = model.decision_function(X)
        
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
    # Sort by score descending
    pred_df = (
        scored
        .sort(["X-1", "score"], descending=[False, True])
        .group_by("X-1")
        .agg(pl.col("X_0").head(max_k))
    )
    return {cid: items.to_list() for cid, items in zip(pred_df["X-1"], pred_df["X_0"])}

def main() -> None:
    transactions, items, users = _scan_sources()
    
    print("\n" + "="*80)
    print("FAST LINEAR MODEL TRAINING: SGDClassifier")
    print("="*80)
    
    # 1. Build Tables
    print("Step 1/2: Building Data Tables...")
    train_table = _build_table(TRAIN_WINDOW, transactions=transactions, items=items, users=users, gt_dict=None)
    
    # Load GT and Build Test Table
    test_ground_truth = load_ground_truth(GT_PKL_PATH)
    test_table = _build_table(
        TEST_WINDOW,
        transactions=transactions,
        items=items,
        users=users,
        gt_dict=test_ground_truth,
        include_gt_in_candidates=False 
    )
    
    hist_dict = _build_history_dict(transactions, TEST_WINDOW)

    # 2. Define Models to Run
    # Note: We no longer need to pass a separate scaler function
    models_to_run = [
        ("LogisticRegression_SGD", lambda: train_logistic_sgd(train_table)),
        ("LinearSVM_SGD", lambda: train_svm_sgd(train_table))
    ]

    # 3. Training & Evaluation Loop
    results = {}

    for model_name, train_func in models_to_run:
        print(f"\n{'='*20} Processing {model_name} {'='*20}")
        
        # Train (returns Pipeline object)
        model = train_func()
        
        # Score
        print(f"Scoring Test Data with {model_name}...")
        scored_test = score_table(model, test_table)
        
        # Build predictions
        pred_dict = _build_prediction_dict(scored_test, 20)
        
        # Evaluate
        print(f"Evaluating {model_name}...")
        model_res = {}
        for k in RANK_KS:
            p, ncold = precision_at_k(pred_dict, test_ground_truth, hist_dict, filter_bought_items=False, K=k)
            model_res[k] = p
            print(f"  {model_name} P@{k}: {p:.4f} (Cold: {ncold})")
        results[model_name] = model_res

    print("\n" + "="*80)
    print("FINAL SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Model':<25} | {'P@5':<10} | {'P@10':<10} | {'P@20':<10}")
    print("-" * 65)
    for model_name, res in results.items():
        print(f"{model_name:<25} | {res[5]:.4f}     | {res[10]:.4f}     | {res[20]:.4f}")

if __name__ == "__main__":
    main()