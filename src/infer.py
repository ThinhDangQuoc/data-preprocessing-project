import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import os
import gc
from datetime import datetime
from typing import List, Any, Tuple, Dict

# Import definitions from your main pipeline file
# WE ASSUME THE FILE YOU PROVIDED IS SAVED AS "recsys_pipeline.py"
from pipeline import (
    BASE_DIR, TRANSACTIONS_GLOB, ITEMS_PATH, USERS_GLOB, GT_PKL_PATH, OUTPUT_DIR,
    generate_candidates, 
    build_features_robust,
    create_cold_start_recommendations,
    build_history_dict,
    predict_with_cold_start_fallback,
    precision_at_k,
    precision_at_k_customer,
    get_preprocessed_data,
    DataSplit,
)

def load_feature_names(df: pl.DataFrame, exclude_cols):
    """Dynamically extract feature columns to match training order"""
    return [
        c for c in df.columns 
        if c not in exclude_cols 
        and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt8]
    ]

def main_inference():
    print("="*60)
    print("INFERENCE & EVALUATION ONLY MODE")
    print("="*60)

    # 1. Configuration for TEST
    # ----------------------------------------------------------------
    TEST_SPLIT_CONFIG = {
        'hist_start': datetime(2024, 11, 1),
        'hist_end': datetime(2024, 12, 30),
        'target_start': None, # No target window for generation, we predict for future
        'target_end': None
    }
    
    # Exclude columns that are not features
    EXCLUDE_COLS = ["customer_id", "item_id", "Y", "created_date", "item_token"]

    # 2. Load Ground Truth
    # ----------------------------------------------------------------
    print("\n[1/6] Loading Ground Truth...")
    with open(GT_PKL_PATH, 'rb') as f:
        gt = pickle.load(f)
    gt_user_ids = list(gt.keys())
    print(f"  Total GT Users to predict: {len(gt_user_ids):,}")

    # 3. Load Raw Data
    # ----------------------------------------------------------------
    print("\n[2/6] Loading Raw Data...")
    raw_trans = pl.scan_parquet(TRANSACTIONS_GLOB)
    raw_items = pl.scan_parquet(ITEMS_PATH)
    raw_users = pl.scan_parquet(USERS_GLOB)

    # Optimization: Filter users to only those in GT to save RAM
    target_users_lf = raw_users.filter(pl.col("customer_id").is_in(gt_user_ids))

    # 4. Process Data (Preprocessing)
    # ----------------------------------------------------------------
    print("\n[3/6] Preprocessing for Test Split...")
    # Reuse the logic from your pipeline to ensure cleaning is identical
    te_trans, te_items, te_users = get_preprocessed_data(
        "test", TEST_SPLIT_CONFIG,
        raw_trans, raw_items, target_users_lf,
        output_dir=f"{BASE_DIR}/processed_v1",
        recreate=False, # Use cached if available
        sample_rate=1.0 
    )

    # 5. Generate Candidates & Features
    # ----------------------------------------------------------------
    print("\n[4/6] Generating Candidates & Features...")
    
    # A. Generate Candidates
    candidates_lazy = generate_candidates(
        te_trans, te_items, te_users,
        TEST_SPLIT_CONFIG["hist_start"], TEST_SPLIT_CONFIG["hist_end"],
        max_candidates_per_user=100, # Match training
        include_repurchase_candidates=True,
        verbose=True
    )
    
    # Materialize candidates to break lineage (Optimization)
    candidates_df = candidates_lazy.collect()
    candidates = candidates_df.lazy()
    
    # B. Build Features
    features_lazy = build_features_robust(
        candidates, te_trans, te_items,
        TEST_SPLIT_CONFIG["hist_start"], TEST_SPLIT_CONFIG["hist_end"],
        verbose=True
    )
    
    # C. Collect Feature Matrix (needed to extract column names)
    print("  Materializing feature matrix for inference...")
    df_features = features_lazy.collect()
    
    # Extract feature names dynamically to ensure exact match with data types
    feature_cols = load_feature_names(df_features, EXCLUDE_COLS)
    print(f"  Feature columns found: {len(feature_cols)}")
    print(f"  First 5: {feature_cols[:5]}")

    # 6. Load Model & Predict
    # ----------------------------------------------------------------
    print("\n[5/6] Loading Model & Running Inference...")
    
    model_path = f"{OUTPUT_DIR}/xgb_ranker.json"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    
    ranker = xgb.XGBRanker()
    ranker.load_model(model_path)
    
    # Create Cold Start Fallback Items
    cold_start_items = create_cold_start_recommendations(
        te_items, te_trans,
        TEST_SPLIT_CONFIG['hist_start'], TEST_SPLIT_CONFIG['hist_end'],
        verbose=True
    )

    # Build history for logic (filtering/repurchase)
    hist_dict = build_history_dict(
        te_trans,
        TEST_SPLIT_CONFIG['hist_start'], TEST_SPLIT_CONFIG['hist_end'],
        verbose=True
    )

    # RUN PREDICTION
    preds = predict_with_cold_start_fallback(
        model=ranker,
        lf=df_features.lazy(), # Pass back as LazyFrame as expected by your function
        features=feature_cols,
        gt_users=gt_user_ids,
        hist_dict=hist_dict,
        cold_start_items=cold_start_items,
        top_k=10,
        batch_size=100_000,
        verbose=True
    )

    # 7. Evaluation
    # ----------------------------------------------------------------
    print("\n[6/6] Evaluation...")
    
    # A. Standard Metric (filter previously bought = False)
    print("\n--- Precision@10 (Allow Repurchase) ---")
    p_at_10, _, stats = precision_at_k(
        pred=preds, gt=gt, hist=hist_dict,
        filter_bought_items=False, 
        K=10, verbose=True
    )

    # B. Standard Metric (filter previously bought = True)
    print("\n--- Precision@10 (Exclude Repurchase) ---")
    p_at_10_filtered, _, _ = precision_at_k(
        pred=preds, gt=gt, hist=hist_dict,
        filter_bought_items=True, 
        K=10, verbose=True
    )

    # C. Detailed Customer Metric
    print("\n--- Customer Metric Details ---")
    p_customer, cold_start_users, stats_cust = precision_at_k_customer(
        pred=preds, gt=gt, hist=hist_dict,
        filter_bought_items=False,
        K=10, return_stats=True
    )
    
    print(f"  Precision@10 (Customer Avg): {p_customer:.4f}")
    print(f"  Cold Start Users: {len(cold_start_users)}")

    # 8. Save Final Predictions
    # ----------------------------------------------------------------
    out_file = f"{OUTPUT_DIR}/final_inference_predictions.json"
    with open(out_file, "w") as f:
        json.dump({str(k): v for k, v in preds.items()}, f, indent=2)
    
    print(f"\n✓ Saved predictions to: {out_file}")
    print("Done.")

if __name__ == "__main__":
    main_inference()