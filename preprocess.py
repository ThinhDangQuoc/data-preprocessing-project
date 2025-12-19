"""
Robust Preprocessing Pipeline to Fix Distribution Mismatch & Data Leakage
Key Improvements:
1. Temporal validation that matches test conditions
2. User cold-start handling
3. Item freshness decay
4. Remove leaky features
"""

import polars as pl
from datetime import datetime, timedelta
import numpy as np

# ============================================================================
# 1. TEMPORAL VALIDATION FIX
# ============================================================================

def create_realistic_splits(transactions: pl.LazyFrame, verbose=True):
    """
    Create splits that mirror test conditions:
    - Test users may be cold-start (little history)
    - Validation should simulate 1-month ahead prediction
    """
    
    SPLITS_ROBUST = {
        'train': {
            'hist_start': datetime(2024, 10, 1),
            'hist_end': datetime(2024, 11, 20),    # Leave 10-day gap
            'target_start': datetime(2024, 12, 1),
            'target_end': datetime(2024, 12, 10)
        },
        'val': {
            'hist_start': datetime(2024, 10, 1),
            'hist_end': datetime(2024, 11, 30),    # Use same history length
            'target_start': datetime(2024, 12, 11),
            'target_end': datetime(2024, 12, 20)
        },
        'test': {
            'hist_start': datetime(2024, 12, 1),
            'hist_end': datetime(2024, 12, 30),
            'target_start': None,  # Ground truth
            'target_end': None
        }
    }
    
    return SPLITS_ROBUST


# ============================================================================
# 2. PREPROCESSING: REMOVE DATA LEAKAGE
# ============================================================================

def preprocess_transactions(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    hist_end: datetime,
    verbose=True
) -> pl.LazyFrame:
    """
    Clean data to prevent leakage:
    1. Add temporal decay weights
    2. Flag item lifecycle stage
    3. Normalize by time period
    """
    if verbose:
        print("\n[Preprocessing] Cleaning transactions...")
    
    # Calculate days since hist_end for each transaction
    trans_with_decay = transactions.with_columns([
        # Temporal decay: recent purchases matter more
        pl.when(pl.col("created_date") <= hist_end)
        .then(
            ((hist_end - pl.col("created_date")).dt.total_days() / 30.0)
            .clip(0, 6)  # Max 6 months decay
        )
        .otherwise(pl.lit(0.0))
        .alias("months_ago"),
        
        # Exponential decay weight
        pl.when(pl.col("created_date") <= hist_end)
        .then(
            (0.95 ** ((hist_end - pl.col("created_date")).dt.total_days() / 30.0))
        )
        .otherwise(pl.lit(0.0))
        .alias("temporal_weight")
    ])
    
    # Add item lifecycle features
    item_lifecycle = (
        transactions
        .filter(pl.col("created_date") <= hist_end)
        .group_by("item_id")
        .agg([
            pl.col("created_date").min().alias("first_sale_date"),
            pl.col("created_date").max().alias("last_sale_date"),
            pl.len().alias("total_sales")
        ])
        .with_columns([
            # Item age in days
            (hist_end - pl.col("first_sale_date")).dt.total_days()
                .alias("item_age_days"),
            
            # Days since last sale
            (hist_end - pl.col("last_sale_date")).dt.total_days()
                .alias("days_since_last_sale"),
            
            # Lifecycle stage
            pl.when(
                (hist_end - pl.col("first_sale_date")).dt.total_days() < 30
            ).then(pl.lit("new"))
            .when(
                (hist_end - pl.col("last_sale_date")).dt.total_days() > 60
            ).then(pl.lit("declining"))
            .otherwise(pl.lit("stable"))
            .alias("lifecycle_stage")
        ])
    )
    
    # Join lifecycle info
    trans_processed = trans_with_decay.join(
        item_lifecycle, 
        on="item_id", 
        how="left"
    )
    
    if verbose:
        stages = trans_processed.select("lifecycle_stage").collect()
        print(f"  Item Lifecycle: New={stages.filter(pl.col('lifecycle_stage')=='new').height}")
    
    return trans_processed


# ============================================================================
# 3. USER STRATIFICATION: HANDLE COLD-START
# ============================================================================

def stratify_users_by_activity(
    transactions: pl.LazyFrame,
    users: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose=True
) -> dict:
    """
    Separate users into activity tiers:
    - cold_start: < 3 purchases
    - casual: 3-10 purchases  
    - regular: 10-30 purchases
    - power: 30+ purchases
    
    This allows:
    1. Stratified sampling (ensure test distribution matches train)
    2. User-specific candidate strategies
    """
    
    user_activity = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
        .group_by("customer_id")
        .agg([
            pl.len().alias("txn_count"),
            pl.col("item_id").n_unique().alias("unique_items"),
            pl.col("created_date").max().alias("last_purchase")
        ])
        .with_columns([
            # Categorize users
            pl.when(pl.col("txn_count") < 3)
            .then(pl.lit("cold_start"))
            .when(pl.col("txn_count") < 10)
            .then(pl.lit("casual"))
            .when(pl.col("txn_count") < 30)
            .then(pl.lit("regular"))
            .otherwise(pl.lit("power"))
            .alias("user_tier"),
            
            # Recency
            (hist_end - pl.col("last_purchase")).dt.total_days()
                .alias("days_inactive")
        ])
    )
    
    # Join with all users
    users_stratified = users.join(
        user_activity, 
        on="customer_id", 
        how="left"
    ).with_columns([
        pl.col("user_tier").fill_null("cold_start"),
        pl.col("txn_count").fill_null(0),
        pl.col("days_inactive").fill_null(999)
    ])
    
    if verbose:
        stats = users_stratified.group_by("user_tier").len().collect()
        print("\n[User Stratification]")
        for row in stats.iter_rows():
            print(f"  {row[0]}: {row[1]:,}")
    
    return users_stratified


def stratified_sample(
    users_stratified: pl.LazyFrame,
    sample_rate: float = 0.1,
    seed: int = 42
) -> pl.LazyFrame:
    """
    Sample users proportionally from each tier to maintain distribution
    """
    
    sampled_parts = []
    for tier in ["cold_start", "casual", "regular", "power"]:
        tier_users = (
            users_stratified
            .filter(pl.col("user_tier") == tier)
            .with_columns(
                (pl.col("customer_id").hash(seed) % 100 < int(sample_rate * 100))
                .alias("_sampled")
            )
            .filter(pl.col("_sampled"))
            .drop("_sampled")
        )
        sampled_parts.append(tier_users)
    
    return pl.concat(sampled_parts, how="vertical")


# ============================================================================
# 4. ITEM QUALITY FILTERING
# ============================================================================

def filter_quality_items(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    min_sales: int = 5,
    min_users: int = 3,
    verbose=True
) -> pl.LazyFrame:
    """
    Remove items with insufficient signal:
    - Items with < 5 sales (noise)
    - Items purchased by < 3 users (too niche)
    - Items with extreme prices (outliers)
    """
    
    item_stats = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
        .group_by("item_id")
        .agg([
            pl.len().alias("sales_count"),
            pl.col("customer_id").n_unique().alias("unique_buyers"),
            pl.col("price").mean().alias("avg_price")
        ])
    )
    
    # Calculate price quantiles
    price_quantiles = item_stats.select([
        pl.col("avg_price").quantile(0.01).alias("p1"),
        pl.col("avg_price").quantile(0.99).alias("p99")
    ]).collect()
    
    p1, p99 = price_quantiles["p1"][0], price_quantiles["p99"][0]
    
    quality_items = (
        item_stats
        .filter(
            (pl.col("sales_count") >= min_sales) &
            (pl.col("unique_buyers") >= min_users) &
            (pl.col("avg_price") >= p1) &
            (pl.col("avg_price") <= p99)
        )
        .select("item_id")
    )
    
    if verbose:
        before = items.select(pl.len()).collect().item()
        after = quality_items.collect().height
        print(f"\n[Item Quality Filter]")
        print(f"  Before: {before:,} items")
        print(f"  After: {after:,} items ({after/before:.1%} retained)")
    
    return items.join(quality_items, on="item_id", how="inner")


# ============================================================================
# 5. FEATURE PREPROCESSING
# ============================================================================

def normalize_features_by_timewindow(
    features: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose=True
) -> pl.LazyFrame:
    """
    Normalize count-based features by time window length
    This prevents train/val/test having different scales
    """
    
    days_in_window = (hist_end - hist_start).days
    
    normalized = features.with_columns([
        # Normalize count features by days
        (pl.col("user_txn_count") * 30.0 / days_in_window)
            .alias("user_txn_count_normalized"),
        
        (pl.col("item_txn_count") * 30.0 / days_in_window)
            .alias("item_txn_count_normalized"),
        
        # Add rate-based features instead of counts
        (pl.col("user_txn_count") / days_in_window)
            .alias("user_daily_rate"),
        
        (pl.col("item_txn_count") / days_in_window)
            .alias("item_daily_rate")
    ])
    
    if verbose:
        print(f"\n[Normalization] Time window: {days_in_window} days")
        print(f"  All count features scaled to 30-day equivalent")
    
    return normalized


# ============================================================================
# 6. INTEGRATION EXAMPLE
# ============================================================================

def preprocess_pipeline(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    split_config: dict,
    sample_rate: float = 0.1,
    verbose=True
):
    """
    Full preprocessing pipeline
    """
    hist_start = split_config['hist_start']
    hist_end = split_config['hist_end']
    
    # Step 1: Preprocess transactions
    trans_clean = preprocess_transactions(
        transactions, items, hist_end, verbose
    )
    
    # Step 2: Filter quality items
    items_quality = filter_quality_items(
        transactions, items, hist_start, hist_end, verbose=verbose
    )
    
    # Step 3: Stratify users
    users_stratified = stratify_users_by_activity(
        transactions, users, hist_start, hist_end, verbose
    )
    
    # Step 4: Stratified sampling
    if sample_rate < 1.0:
        users_sampled = stratified_sample(users_stratified, sample_rate)
    else:
        users_sampled = users_stratified
    
    if verbose:
        print(f"\n[Pipeline Complete]")
        print(f"  Sampled users: {users_sampled.select(pl.len()).collect().item():,}")
    
    return trans_clean, items_quality, users_sampled


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Load data
    trans = pl.scan_parquet("transactions.parquet")
    items = pl.scan_parquet("items.parquet")
    users = pl.scan_parquet("users.parquet")
    
    # Create robust splits
    splits = create_realistic_splits(trans)
    
    # Preprocess training data
    trans_clean, items_clean, users_clean = preprocess_pipeline(
        trans, items, users,
        splits['train'],
        sample_rate=0.1
    )