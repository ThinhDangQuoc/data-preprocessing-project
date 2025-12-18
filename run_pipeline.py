from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
import gc
import pickle
import json
import warnings
import numpy as np
import polars as pl
from xgboost import XGBRanker
from datetime import datetime

# Import from other modules
from config import BASE_DIR, TRANSACTIONS_GLOB, ITEMS_PATH, USERS_GLOB, GT_PKL_PATH, OUTPUT_DIR, SPLITS
from data_pipeline import build_dataset

warnings.filterwarnings('ignore')

# ============================================================================
# NUMPY PREPARATION
# ============================================================================

def prepare_for_xgb(
    lf: pl.LazyFrame,
    is_train: bool = True,
    hard_neg_ratio: int = 10,
    easy_neg_ratio: int = 10,
    hard_neg_col: str = "X_14",
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pl.DataFrame]:
    """Convert to numpy for XGBoost with Hard Negative Mining"""
    if verbose:
        print(f"  Materializing data (Hard Negative Mode)...")
    
    df = lf.collect(streaming=True)
    
    if is_train and "Y" in df.columns:
        pos = df.filter(pl.col("Y") == 1)
        neg = df.filter(pl.col("Y") == 0)
        n_pos = len(pos)
        
        if n_pos > 0:
            if hard_neg_col in df.columns:
                hard_pool = neg.filter(pl.col(hard_neg_col) <= 5)
                easy_pool = neg.filter(pl.col(hard_neg_col) > 5)
            else:
                hard_pool = neg
                easy_pool = pl.DataFrame([])
            
            n_hard_target = n_pos * hard_neg_ratio
            n_easy_target = n_pos * easy_neg_ratio
            
            if len(hard_pool) > 0:
                n_hard_actual = min(len(hard_pool), n_hard_target)
                hard_sampled = hard_pool.sample(n=n_hard_actual, seed=42, shuffle=True)
            else:
                hard_sampled = pl.DataFrame([], schema=neg.schema)
            
            if len(easy_pool) > 0:
                n_easy_actual = min(len(easy_pool), n_easy_target)
                easy_sampled = easy_pool.sample(n=n_easy_actual, seed=42, shuffle=True)
            else:
                n_remaining = (n_pos * (hard_neg_ratio + easy_neg_ratio)) - len(hard_sampled)
                n_actual = min(len(neg), n_remaining)
                easy_sampled = neg.sample(n=n_actual, seed=42, shuffle=True)
            
            df = pl.concat([pos, hard_sampled, easy_sampled], how="vertical")
            
            if verbose:
                print(f"  [Sampling] Pos: {len(pos):,}")
                print(f"  [Sampling] Hard Neg: {len(hard_sampled):,} (Target: {n_hard_target:,})")
                print(f"  [Sampling] Easy Neg: {len(easy_sampled):,} (Target: {n_easy_target:,})")
                print(f"  [Sampling] Total: {len(df):,}")
            
            del pos, neg, hard_pool, easy_pool, hard_sampled, easy_sampled
            gc.collect()
    
    elif not is_train and "Y" in df.columns:
        pos = df.filter(pl.col("Y") == 1)
        neg = df.filter(pl.col("Y") == 0)
        
        target_neg = min(len(neg), len(pos) * 100)
        neg_sampled = neg.sample(n=target_neg, seed=42, shuffle=True)
        
        df = pl.concat([pos, neg_sampled], how="vertical")
        if verbose:
            print(f"  [Val-Sample] Reduced to {len(df):,} rows")
        
        del pos, neg, neg_sampled
        gc.collect()
    
    df = df.sort("customer_id")
    feature_cols = [c for c in df.columns if c.startswith("X_")]
    
    if not feature_cols:
        raise ValueError("No columns starting with 'X_' found!")
    
    X = df.select(feature_cols).to_numpy()
    y = df["Y"].to_numpy() if "Y" in df.columns else None
    
    groups = (
        df
        .select("customer_id")
        .select(pl.col("customer_id").rle_id())
        .group_by("customer_id")
        .len()
        .sort("customer_id")
        .select("len")
        .to_numpy()
        .flatten()
    )
    
    id_df = df.select(["customer_id", "item_id"])
    
    if verbose:
        print(f"  Shape: X={X.shape}, groups_count={len(groups)}")
        if y is not None:
            print(f"  Avg Group Size: {len(X)/len(groups):.1f}")
    
    return X, y, groups, feature_cols, id_df


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    verbose: bool = True
) -> XGBRanker:
    """Train XGBoost ranker"""
    if verbose:
        print("\n" + "="*60)
        print("Training XGBRanker")
        print("="*60)
    
    model = XGBRanker(
        objective='rank:ndcg',
        n_estimators=20,
        max_depth=6,
        learning_rate=0.05,
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    
    model.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_val, y_val)],
        eval_group=[groups_val],
        verbose=verbose
    )
    
    return model


def predict_top_k(
    model: XGBRanker,
    lf: pl.LazyFrame,
    features: List[str],
    gt_users: List[Any],
    top_k: int = 10,
    batch_size: int = 10000,
    verbose: bool = True
) -> Dict[Any, List[Any]]:
    """Batched inference for GT users"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Inference - {len(gt_users):,} users")
        print(f"{'='*60}")
    
    results = {}
    
    for i in range(0, len(gt_users), batch_size):
        batch_users = gt_users[i:i+batch_size]
        batch_lf = lf.filter(pl.col("customer_id").is_in(batch_users))
        df_batch = batch_lf.collect(streaming=True)
        
        if df_batch.is_empty():
            continue
        
        X_batch = df_batch.select(features).to_numpy()
        scores = model.predict(X_batch)
        
        df_batch = df_batch.with_columns(pl.Series("score", scores))
        
        preds = (
            df_batch
            .sort(["customer_id", "score"], descending=[False, True])
            .group_by("customer_id", maintain_order=True)
            .agg(pl.col("item_id").head(top_k))
        )
        
        for row in preds.iter_rows():
            results[row[0]] = row[1]
        
        if verbose and (i + batch_size) % 50000 == 0:
            print(f"  Processed {min(i+batch_size, len(gt_users)):,}/{len(gt_users):,}")
        
        del df_batch, X_batch, scores, preds
        gc.collect()
    
    if verbose:
        print(f"  Done: {len(results):,} predictions")
    
    return results


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
        .group_by("customer_id")
        .agg(pl.col("item_id").unique())
        .collect(streaming=True)
    )
    
    hist_dict = {row[0]: row[1] for row in hist.iter_rows()}
    
    if verbose:
        print(f"  History: {len(hist_dict):,} users")
    
    return hist_dict


def precision_at_k(
    pred: Dict[Any, List[Any]],
    gt: Dict[Any, Any],
    hist: Dict[Any, List[Any]],
    filter_bought_items: bool = True,
    K: int = 10,
    verbose: bool = True
) -> Tuple[float, List[Any], Dict[str, int]]:
    """Compute Precision@K"""
    precisions = []
    missing_preds_users = []
    
    for user in gt.keys():
        if user not in pred:
            missing_preds_users.append(user)
            continue
        
        if isinstance(gt[user], dict):
            gt_items = gt[user]['list_items']
        else:
            gt_items = gt[user]
        
        relevant_items = set(gt_items)
        
        if filter_bought_items:
            past_items = set(hist.get(user, []))
            relevant_items -= past_items
        
        if not relevant_items:
            precisions.append(0.0)
            continue
        
        hits = len(set(pred[user][:K]) & relevant_items)
        precisions.append(hits / K)
    
    nusers = len(gt.keys())
    n_missing = len(missing_preds_users)
    n_evaluated = len(precisions)
    mean_precision = np.mean(precisions) if precisions else 0.0
    
    stats = {
        'total_gt_users': nusers,
        'evaluated_users': n_evaluated,
        'users_without_preds': n_missing,
        'coverage_rate': n_evaluated / nusers if nusers > 0 else 0.0,
        'mean_precision': mean_precision
    }
    
    if verbose:
        print(f"\n  Evaluation Statistics:")
        print(f"  ├─ Total GT users: {stats['total_gt_users']:,}")
        print(f"  ├─ Evaluated: {stats['evaluated_users']:,}")
        print(f"  ├─ Missing Preds: {stats['users_without_preds']:,}")
        print(f"  └─ Precision@{K}: {stats['mean_precision']:.4f}")
    
    return mean_precision, missing_preds_users, stats


def main():
    print("\n" + "="*60)
    print("TWO-STAGE RECOMMENDATION PIPELINE")
    print("="*60)
    
    print("\nLoading ground truth...")
    with open(GT_PKL_PATH, 'rb') as f:
        gt = pickle.load(f)
    
    gt_user_ids = list(gt.keys())
    print(f"  Found {len(gt_user_ids):,} users in ground truth")
    
    print("\nLoading data sources...")
    trans = pl.scan_parquet(TRANSACTIONS_GLOB)
    items = pl.scan_parquet(ITEMS_PATH)
    users_all = pl.scan_parquet(USERS_GLOB)
    
    schema = trans.collect_schema()
    cid_type = schema["customer_id"]
    
    gt_users_lf = pl.LazyFrame({"customer_id": gt_user_ids}).with_columns(
        pl.col("customer_id").cast(cid_type)
    )
    
    SAMPLE_USERS = 0.1
    
    # Build datasets
    train_lf = build_dataset(
        SPLITS['train'], trans, users_all, items,
        is_train=True, sample_users=SAMPLE_USERS
    )
    
    val_lf = build_dataset(
        SPLITS['val'], trans, users_all, items,
        is_train=True, sample_users=SAMPLE_USERS
    )
    
    print(f"\nBuilding TEST dataset (GT users: {len(gt_user_ids):,})")
    test_lf = build_dataset(
        SPLITS['test'], trans, gt_users_lf, items,
        is_train=False, sample_users=1.0
    )
    
    # Prepare for XGBoost
    X_train, y_train, g_train, feats, _ = prepare_for_xgb(
        train_lf, is_train=True,
        hard_neg_ratio=15,
        easy_neg_ratio=15,
        hard_neg_col="X_14"
    )
    
    X_val, y_val, g_val, _, _ = prepare_for_xgb(val_lf, is_train=False)
    
    # Train
    model = train_model(X_train, y_train, g_train, X_val, y_val, g_val)
    
    del X_train, y_train, g_train, X_val, y_val, g_val, train_lf, val_lf
    gc.collect()
    
    # Predict
    preds = predict_top_k(model, test_lf, feats, gt_users=gt_user_ids)
    
    # Build history for eval
    hist_dict = build_history_dict(
        trans,
        SPLITS['test'].hist_start,
        SPLITS['test'].hist_end
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    
    p_at_10, missing_users, stats = precision_at_k(
        pred=preds,
        gt=gt,
        hist=hist_dict,
        filter_bought_items=True,
        K=10
    )
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    eval_results = {
        'precision_at_10': float(p_at_10),
        'statistics': stats,
        'n_missing_users': len(missing_users),
        'sample_missing_users': [str(u) for u in missing_users[:10]]
    }
    
    eval_path = f"{OUTPUT_DIR}/evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    output_path = f"{OUTPUT_DIR}/predictions.json"
    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in preds.items()}, f, indent=2)
    
    print(f"\n✓ Results saved to {OUTPUT_DIR}")
    
    print("\n" + "="*60)
    print("Top 10 Feature Importance")
    print("="*60)
    
    importance = model.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    for feat, score in sorted_importance[:10]:
        feat_idx = int(feat.replace('f', ''))
        feat_name = feats[feat_idx] if feat_idx < len(feats) else feat
        print(f"  {feat_name}: {score:.2f}")

if __name__ == "__main__":
    main()