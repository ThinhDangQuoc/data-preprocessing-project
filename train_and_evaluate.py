from __future__ import annotations
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import polars as pl
import warnings

# --- MODELS ---
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier # Fast approximation for SVM
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# Import your enhanced feature builder
from feature_builder import build_feature_label_table

# --- CONFIGURATION ---
TRANSACTIONS_GLOB = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system/dataset/sales_pers.user_chunk*.parquet"

NUM_FEATURES = 29 
RANK_KS = (5, 10, 20)

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

# --- UTILS ---
def generate_ground_truth(transactions: pl.LazyFrame, window: FeatureWindow) -> Dict[Any, Dict[str, List[Any]]]:
    gt_df = (
        transactions
        .filter(pl.col("created_date") >= window.recent_start)
        .filter(pl.col("created_date") <= window.recent_end)
        .select(["customer_id", "item_id"])
        .unique()
        .group_by("customer_id")
        .agg(pl.col("item_id").alias("list_items"))
        .collect()
    )
    return {row["customer_id"]: {"list_items": row["list_items"]} for row in gt_df.iter_rows(named=True)}

def precision_at_k(pred, gt, hist, filter_bought_items=True, K=10):
    precisions = []
    for user in gt.keys():
        if user not in pred:
            precisions.append(0.0)
            continue
        
        gt_items = set(gt[user]['list_items'])
        if filter_bought_items and user in hist:
            gt_items -= set(hist[user])
            
        if not gt_items: continue # Skip if no relevant items left
        if not pred[user]: 
            precisions.append(0.0)
            continue
            
        hits = len(set(pred[user][:K]) & gt_items)
        precisions.append(hits / K)
    
    return np.mean(precisions) if precisions else 0.0

def _scan_sources() -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    return pl.scan_parquet(TRANSACTIONS_GLOB), pl.scan_parquet(ITEMS_PATH), pl.scan_parquet(USERS_GLOB)

def _build_table(window: FeatureWindow, transactions, items, users) -> pl.DataFrame:
    print(f"Building table: H({window.history_start.date()}-{window.history_end.date()}) -> T({window.recent_start.date()})")
    return build_feature_label_table(
        transactions, items, users,
        window.history_start, window.history_end,
        window.recent_start, window.recent_end,
        transaction_time_col="created_date",
        customer_id_col="customer_id",
        price_col="price", quantity_col="quantity"
    ).collect()

def extract_features(table: pl.DataFrame) -> np.ndarray:
    return table.select([f"X_{i}" for i in range(1, NUM_FEATURES + 1)]).to_numpy().astype(np.float64)

# --- MODULAR TRAINING FUNCTION ---

def get_model_pipeline(model_type: str, scale_pos_weight: float) -> Any:
    """Returns a scikit-learn compatible pipeline or model based on type."""
    
    # 1. XGBoost (Gradient Boosting)
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
            eval_metric='logloss', early_stopping_rounds=50
        )

    # 2. LightGBM (Gradient Boosting - often faster/better than XGB)
    elif model_type == "lgbm":
        return LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8,
            num_leaves=31, scale_pos_weight=scale_pos_weight, 
            random_state=42, n_jobs=-1, verbose=-1,
            early_stopping_rounds=50
        )

    # 3. Logistic Regression (Linear)
    # Needs Imputation and Scaling
    elif model_type == "logistic":
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                solver='liblinear', 
                class_weight='balanced', 
                random_state=42,
                max_iter=1000
            ))
        ])

    # 4. SVM (Support Vector Machine)
    # Standard SVC is O(N^3) and too slow for large RecSys.
    # We use SGDClassifier with loss='modified_huber' which approximates SVM 
    # but supports predict_proba and is O(N).
    elif model_type == "svm":
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', SGDClassifier(
                loss='modified_huber', # Allows probability estimates
                penalty='l2',
                alpha=1e-4, 
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            ))
        ])

    # 5. Decision Tree (Simple rule-based)
    elif model_type == "tree":
        return DecisionTreeClassifier(
            max_depth=10, 
            min_samples_leaf=20,
            class_weight='balanced', 
            random_state=42
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model_generic(
    train_table: pl.DataFrame, 
    model_type: str,
    val_table: pl.DataFrame = None
) -> Any:
    
    print(f"  -> Training {model_type.upper()}...")
    X = extract_features(train_table)
    y = train_table["Y"].to_numpy()
    
    # Calculate Class Balance
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Get model structure
    model = get_model_pipeline(model_type, scale_pos_weight)
    
    # Fit Logic
    if model_type in ["xgboost", "lgbm"]:
        # GBMs support native validation sets
        eval_set = None
        if val_table is not None:
            X_val = extract_features(val_table)
            y_val = val_table["Y"].to_numpy()
            eval_set = [(X_val, y_val)]
        
        # XGBoost requires eval_set to be list of tuples, LGBM same
        # Note: Scikit-Learn pipelines don't take eval_set in fit easily
        model.fit(X, y, eval_set=eval_set)
        
    else:
        # Sklearn models (Logistic, SVM, Tree) don't use eval_set for early stopping 
        # in the standard API (SGD does support partial_fit but let's keep it simple)
        model.fit(X, y)

    return model

def score_table(model: Any, table: pl.DataFrame) -> pl.DataFrame:
    X = extract_features(table)
    # Handle Pipeline vs raw model
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        # Fallback if probability not supported (shouldn't happen with this setup)
        scores = model.decision_function(X) 
        
    return table.with_columns(pl.Series("score", scores))

def _build_history_dict(transactions: pl.LazyFrame, window: FeatureWindow) -> Dict[Any, List[Any]]:
    hist_df = (
        transactions
        .filter(pl.col("created_date") < window.recent_start) 
        .select(["customer_id", "item_id"])
        .unique()
        .collect()
    )
    # Optimized grouping
    return {k: list(v) for k, v in hist_df.group_by("customer_id", maintain_order=False).agg("item_id").iter_rows()}

def _build_prediction_dict(scored: pl.DataFrame, max_k: int) -> Dict[Any, List[Any]]:
    pred_df = (
        scored
        .sort(["X-1", "score"], descending=[False, True])
        .group_by("X-1")
        .agg(pl.col("X_0").head(max_k))
    )
    return {row["X-1"]: row["X_0"] for row in pred_df.iter_rows(named=True)}

# --- MAIN ---

def main() -> None:
    transactions, items, users = _scan_sources()
    
    # 1. Build Data
    train_table = _build_table(TRAIN_WINDOW, transactions, items, users)
    val_table = _build_table(VAL_WINDOW, transactions, items, users)
    test_table = _build_table(TEST_WINDOW, transactions, items, users)
    
    # 2. Evaluation Setup
    test_ground_truth = generate_ground_truth(transactions, TEST_WINDOW)
    hist_dict = _build_history_dict(transactions, TEST_WINDOW)
    
    # 3. Model Loop
    models_to_run = ["logistic", "tree", "xgboost", "lgbm", "svm"]
    results = {}

    print(f"\n{'='*10} STARTING MODEL COMPARISON {'='*10}")

    for m_name in models_to_run:
        print(f"\n>>> Running: {m_name}")
        
        # Train
        try:
            model = train_model_generic(train_table, m_name, val_table)
            
            # Score
            scored_test = score_table(model, test_table)
            
            # Evaluate
            pred_dict = _build_prediction_dict(scored_test, 20)
            
            metrics = {}
            for k in RANK_KS:
                p = precision_at_k(pred_dict, test_ground_truth, hist_dict, filter_bought_items=True, K=k)
                metrics[f"P@{k}"] = p
            
            results[m_name] = metrics
            print(f"    Scores: {metrics}")
            
        except Exception as e:
            print(f"    Failed to run {m_name}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Final Summary
    print(f"\n{'='*10} FINAL SUMMARY {'='*10}")
    print(f"{'Model':<15} | {'P@5':<10} | {'P@10':<10} | {'P@20':<10}")
    print("-" * 55)
    for m_name, metrics in results.items():
        print(f"{m_name:<15} | {metrics['P@5']:<10.4f} | {metrics['P@10']:<10.4f} | {metrics['P@20']:<10.4f}")

if __name__ == "__main__":
    main()