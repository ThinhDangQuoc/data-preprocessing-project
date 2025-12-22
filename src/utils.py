import os
import polars as pl
from pathlib import Path
from preprocess import preprocess_pipeline
from datetime import datetime, timedelta
from typing import List, Any, Tuple, Dict
def get_preprocessed_data(
    split_name: str,
    split_config: dict,
    raw_trans: pl.LazyFrame,
    raw_items: pl.LazyFrame,
    raw_users: pl.LazyFrame,
    output_dir: str = "processed_data",
    recreate: bool = False,
    sample_rate: float = 1.0,
    verbose: bool = True
):
    """
    Checks for existing preprocessed files. 
    If they don't exist (or recreate=True), runs the pipeline and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths based on split name
    trans_path = Path(output_dir) / f"trans_{split_name}.parquet"
    items_path = Path(output_dir) / f"items_{split_name}.parquet"
    users_path = Path(output_dir) / f"users_{split_name}.parquet"
    
    paths_exist = trans_path.exists() and items_path.exists() and users_path.exists()
    
    if not recreate and paths_exist:
        if verbose: print(f"  [Cache] Loading preprocessed {split_name} data from {output_dir}...")
        return (
            pl.scan_parquet(str(trans_path)),
            pl.scan_parquet(str(items_path)),
            pl.scan_parquet(str(users_path))
        )

    if verbose: print(f"  [Pipeline] Recreating {split_name} preprocessed data...")
    
    # Run the user's provided logic
    trans_clean, items_quality, users_sampled = preprocess_pipeline(
        raw_trans, raw_items, raw_users,split_name,
        split_config, verbose
    )
    
    # Materialize and Save
    if verbose: print(f"  [IO] Saving processed files to {output_dir}...")
    
    # We collect and write here to ensure the "Preprocessing" stage is finished
    trans_clean.collect().write_parquet(trans_path)
    items_quality.collect().write_parquet(items_path)
    users_sampled.collect().write_parquet(users_path)
    
    return (
        pl.scan_parquet(str(trans_path)),
        pl.scan_parquet(str(items_path)),
        pl.scan_parquet(str(users_path))
    )
    
def sample_train_pairs_v3_FIXED(
    candidates_with_features: pl.LazyFrame,  # ← CHANGED: After features
    trans_clean: pl.LazyFrame,
    *,
    target_start: datetime,
    target_end: datetime,
    max_items_per_user: int = 50,  # ← Keep substantial ranking context
    min_items_per_user: int = 10,  # ← Ensure minimum ranking quality
    seed: int = 42,
    verbose: bool = True,
) -> pl.LazyFrame:
    """
    FIXED: Sample training data AFTER feature engineering
    
    Key Changes:
    1. Accept candidates WITH features already computed
    2. Keep 30-50 items per user (not 2.2!)
    3. Balance positives/negatives while maintaining ranking context
    4. Ensure every user has enough items to learn ranking
    """
    
    # 1. Add labels to feature-rich candidates
    targets = (
        trans_clean
        .filter((pl.col("created_date") >= target_start) & (pl.col("created_date") <= target_end))
        .select(["customer_id", "item_id"])
        .unique()
        .with_columns(pl.lit(1).cast(pl.UInt8).alias("Y"))
    )
    
    cand_labeled = (
        candidates_with_features
        .join(targets, on=["customer_id", "item_id"], how="left")
        .with_columns(pl.col("Y").fill_null(0).cast(pl.UInt8))
    )
    
    # 2. Keep only users with >=1 positive
    pos_users = (
        cand_labeled
        .filter(pl.col("Y") == 1)
        .select("customer_id")
        .unique()
    )
    cand_labeled = cand_labeled.join(pos_users, on="customer_id", how="inner")
    
    # 3. Separate positives and negatives
    pos_df = cand_labeled.filter(pl.col("Y") == 1)
    neg_df = cand_labeled.filter(pl.col("Y") == 0)
    
    # 4. SMART SAMPLING STRATEGY
    # Goal: Keep 30-50 items per user with balanced positive rate (15-25%)
    
    user_pos_counts = (
        pos_df
        .group_by("customer_id")
        .len()
        .rename({"len": "n_pos"})
    )
    
    user_neg_counts = (
        neg_df
        .group_by("customer_id")
        .len()
        .rename({"len": "n_neg_available"})
    )
    
    # Calculate sampling targets
    sampling_plan = (
        user_pos_counts
        .join(user_neg_counts, on="customer_id", how="inner")
        .with_columns([
            # Target: 40 total items per user
            # With 20% positive rate: 8 pos + 32 neg
            pl.min_horizontal(
                pl.col("n_pos"),
                pl.lit(max_items_per_user * 0.2)  # Cap positives at 20% of max
            ).cast(pl.Int32).alias("n_pos_keep"),
            
            # Negatives: Fill remaining slots (30-40 items)
            pl.max_horizontal(
                pl.min_horizontal(
                    pl.col("n_neg_available"),
                    pl.lit(max_items_per_user) - pl.col("n_pos")
                ),
                pl.lit(min_items_per_user)  # Ensure minimum
            ).cast(pl.Int32).alias("n_neg_keep")
        ])
    )
    
    # 5. Sample positives (if user has too many)
    pos_sampled = (
        pos_df
        .join(sampling_plan.select(["customer_id", "n_pos_keep"]), on="customer_id", how="inner")
        .with_columns(pl.arange(0, pl.len()).shuffle(seed=seed).alias("__rnd"))
        .sort("__rnd")
        .with_columns(
            pl.col("__rnd").rank("dense").over("customer_id").alias("__rank")
        )
        .filter(pl.col("__rank") <= pl.col("n_pos_keep"))
        .drop(["__rnd", "__rank", "n_pos_keep"])
    )
    
    # 6. Sample negatives
    neg_sampled = (
        neg_df
        .join(sampling_plan.select(["customer_id", "n_neg_keep"]), on="customer_id", how="inner")
        .with_columns(pl.arange(0, pl.len()).shuffle(seed=seed + 1).alias("__rnd"))
        .sort("__rnd")
        .with_columns(
            pl.col("__rnd").rank("dense").over("customer_id").alias("__rank")
        )
        .filter(pl.col("__rank") <= pl.col("n_neg_keep"))
        .drop(["__rnd", "__rank", "n_neg_keep"])
    )
    
    # 7. Combine
    sampled = (
        pl.concat([pos_sampled, neg_sampled], how="vertical")
        .sort(["customer_id", "Y"], descending=[False, True])
    )
    
    if verbose:
        s = sampled.select([
            pl.len().alias("rows"),
            pl.col("customer_id").n_unique().alias("users"),
            pl.col("Y").sum().alias("pos")
        ]).collect()
        rows = int(s["rows"][0])
        users = int(s["users"][0])
        pos = int(s["pos"][0])
        pos_rate = pos / rows if rows > 0 else 0
        avg_items = rows / users if users > 0 else 0
        
        print(f"\n[FIXED Train Sampling]")
        print(f"  Users: {users:,}")
        print(f"  Total items: {rows:,}")
        print(f"  Avg items/user: {avg_items:.1f} (Target: {max_items_per_user})")
        print(f"  Positives: {pos:,} ({pos_rate:.1%})")
    
    return sampled


# ============================================================================
# FIX 3: BALANCE VAL POSITIVE RATE
# ============================================================================

def sample_validation_data(
    val_features: pl.LazyFrame,
    target_positive_rate: float = 0.10,  # 10% positive (5x current)
    seed: int = 42,
    verbose: bool = True
) -> pl.LazyFrame:
    """
    Balance validation set to have reasonable positive rate
    
    Strategy:
    - Keep ALL positives (these are precious!)
    - Undersample negatives to achieve target positive rate
    """
    
    df = val_features.collect()
    
    pos_df = df.filter(pl.col("Y") == 1)
    neg_df = df.filter(pl.col("Y") == 0)
    
    n_pos = len(pos_df)
    n_neg_needed = int(n_pos * (1.0 / target_positive_rate - 1))
    n_neg_available = len(neg_df)
    
    if n_neg_needed >= n_neg_available:
        # Not enough negatives to balance, keep all
        if verbose:
            print(f"[Val Balancing] Keeping all negatives ({n_neg_available:,})")
        return val_features
    
    # Sample negatives per user to maintain user representation
    user_pos_counts = (
        pos_df
        .group_by("customer_id")
        .len()
        .rename({"len": "n_pos"})
    )
    
    user_neg_counts = (
        neg_df
        .group_by("customer_id")
        .len()
        .rename({"len": "n_neg"})
    )
    
    sampling_plan = (
        user_pos_counts
        .join(user_neg_counts, on="customer_id", how="inner")
        .with_columns([
            # Negatives per user proportional to their positives
            (pl.col("n_pos") * (1.0 / target_positive_rate - 1))
            .cast(pl.Int32)
            .alias("n_neg_target")
        ])
        .with_columns([
            # Don't exceed available negatives per user
            pl.min_horizontal(
                pl.col("n_neg_target"),
                pl.col("n_neg")
            ).alias("n_neg_keep")
        ])
    )
    
    neg_sampled = (
        neg_df.lazy()
        .join(sampling_plan.lazy().select(["customer_id", "n_neg_keep"]), 
              on="customer_id", how="inner")
        .with_columns(pl.arange(0, pl.len()).shuffle(seed=seed).alias("__rnd"))
        .sort("__rnd")
        .with_columns(
            pl.col("__rnd").rank("dense").over("customer_id").alias("__rank")
        )
        .filter(pl.col("__rank") <= pl.col("n_neg_keep"))
        .drop(["__rnd", "__rank", "n_neg_keep"])
    )
    
    balanced = pl.concat([pos_df.lazy(), neg_sampled], how="vertical")
    
    if verbose:
        stats = balanced.select([
            pl.len().alias("total"),
            pl.col("Y").sum().alias("pos")
        ]).collect()
        total = stats["total"][0]
        pos = stats["pos"][0]
        print(f"\n[Val Balancing]")
        print(f"  Before: {len(df):,} rows ({n_pos/len(df):.2%} positive)")
        print(f"  After: {total:,} rows ({pos/total:.2%} positive)")
    
    return balanced

# ============================================================================
# 1. OPTIMIZED DATA TYPES
# ============================================================================

def optimize_dtypes(df: pl.LazyFrame, verbose: bool = True) -> pl.LazyFrame:
    """
    Reduce memory by using appropriate dtypes
    
    Savings: ~50% memory reduction
    """
    schema = df.collect_schema()
    
    optimizations = []
    for col, dtype in schema.items():
        if dtype == pl.Float64:
            optimizations.append(pl.col(col).cast(pl.Float32))
        elif dtype == pl.Int64 and col.endswith("_id"):
            # IDs can be categorical if cardinality is low
            optimizations.append(pl.col(col).cast(pl.Utf8).cast(pl.Categorical))
        elif dtype == pl.Int64:
            optimizations.append(pl.col(col).cast(pl.Int32))
    
    if optimizations:
        df = df.with_columns(optimizations)
        if verbose:
            print(f"  [Dtype] Optimized {len(optimizations)} columns")
    
    return df

# ============================================================================
# EVALUATION: Measure Balance
# ============================================================================

def evaluate_discovery_coverage(
    predictions: Dict,
    history: Dict,
    ground_truth: Dict,
    verbose: bool = True
) -> Dict:
    """
    Measure how well we balance repurchase vs discovery
    
    Metrics:
    - Repurchase Precision: % of predictions that are repurchases
    - Discovery Precision: % of predictions that are NEW items
    - Diversity: Unique items across all users
    - Serendipity: Correct NEW items predicted
    """
    
    total_recs = 0
    repurchase_count = 0
    discovery_count = 0
    correct_repurchase = 0
    correct_discovery = 0
    
    for user, pred_items in predictions.items():
        if user not in history or user not in ground_truth:
            continue
        
        hist_items = set(history[user])
        gt_items = set(ground_truth[user] if isinstance(ground_truth[user], list) 
                      else ground_truth[user]['list_items'])
        
        for item in pred_items:
            total_recs += 1
            
            # Check if repurchase or discovery
            if item in hist_items:
                repurchase_count += 1
                if item in gt_items:
                    correct_repurchase += 1
            else:
                discovery_count += 1
                if item in gt_items:
                    correct_discovery += 1
    
    metrics = {
        "repurchase_ratio": repurchase_count / total_recs if total_recs > 0 else 0,
        "discovery_ratio": discovery_count / total_recs if total_recs > 0 else 0,
        "repurchase_precision": correct_repurchase / repurchase_count if repurchase_count > 0 else 0,
        "discovery_precision": correct_discovery / discovery_count if discovery_count > 0 else 0,
        "total_recommendations": total_recs
    }
    
    if verbose:
        print("\n" + "="*60)
        print("DISCOVERY-REPURCHASE BALANCE")
        print("="*60)
        print(f"Repurchase: {metrics['repurchase_ratio']:.1%} "
              f"(Precision: {metrics['repurchase_precision']:.3f})")
        print(f"Discovery:  {metrics['discovery_ratio']:.1%} "
              f"(Precision: {metrics['discovery_precision']:.3f})")
    
    return metrics

    

def recall_at_k_candidates(candidates_lf: pl.LazyFrame, trans_lf: pl.LazyFrame,
                           target_start: datetime, target_end: datetime,
                           K: int = 80, verbose=True) -> float:
    # GT pairs in target window
    gt_pairs = (
        trans_lf
        .filter((pl.col("created_date") >= target_start) & (pl.col("created_date") <= target_end))
        .select(["customer_id","item_id"])
        .unique()
    )

    # take top-K candidates per user (if not already limited)
    cand_topk = (
        candidates_lf
        .group_by("customer_id", maintain_order=True)
        .head(K)
        .select(["customer_id","item_id"])
    )

    hit_users = cand_topk.join(gt_pairs, on=["customer_id","item_id"], how="inner") \
                         .select("customer_id").unique()

    n_hit = hit_users.collect().height
    n_gt_users = gt_pairs.select("customer_id").unique().collect().height
    recall = n_hit / n_gt_users if n_gt_users else 0.0

    if verbose:
        print(f"[Recall@{K}] GT users={n_gt_users:,} hit users={n_hit:,} recall={recall:.2%}")
    return recall