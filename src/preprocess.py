"""
Robust Preprocessing Pipeline to Fix Distribution Mismatch & Data Leakage
Key Improvements:
1. Temporal validation that matches test conditions
2. User cold-start handling
3. Item freshness decay
4. Remove leaky features
"""
import pickle
import polars as pl
from datetime import datetime, timedelta
import numpy as np
import polars as pl
def classify_user_intent(
    transactions: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose: bool = True
) -> pl.LazyFrame:
    """
    Classify users into intent profiles:
    - REPURCHASER: High repurchase rate, low exploration
    - EXPLORER: High exploration, low repurchase
    - BALANCED: Mix of both
    - COLD_START: Insufficient data
    """
    
    user_behavior = (
        transactions
        .filter(pl.col("created_date").is_between(hist_start, hist_end))
        .group_by("customer_id")
        .agg([
            pl.len().alias("total_txns"),
            pl.col("item_id").n_unique().alias("unique_items"),
            pl.col("item_id").len().alias("total_items")
        ])
        .with_columns([
            # Repurchase rate: How often they buy same items
            (1.0 - pl.col("unique_items") / pl.col("total_items"))
            .clip(0, 1)
            .alias("repurchase_rate"),
            
            # Exploration score: Variety relative to purchase volume
            (pl.col("unique_items").log1p() / pl.col("total_txns").log1p())
            .clip(0, 1)
            .alias("exploration_score")
        ])
        .with_columns([
            # Classify user intent
            pl.when(pl.col("total_txns") < 5)
            .then(pl.lit("COLD_START"))
            .when(
                (pl.col("repurchase_rate") > 0.4) & 
                (pl.col("exploration_score") < 0.6)
            )
            .then(pl.lit("REPURCHASER"))
            .when(
                (pl.col("exploration_score") > 0.7) & 
                (pl.col("repurchase_rate") < 0.3)
            )
            .then(pl.lit("EXPLORER"))
            .otherwise(pl.lit("BALANCED"))
            .alias("user_intent"),
            
            # Intent strength (confidence in classification)
            pl.max_horizontal(
                (pl.col("repurchase_rate") - 0.5).abs(),
                (pl.col("exploration_score") - 0.5).abs()
            ).alias("intent_strength")
        ])
    )
    
    if verbose:
        stats = user_behavior.group_by("user_intent").len().collect()
        print("\n[User Intent Classification]")
        for row in stats.iter_rows():
            print(f"  {row[0]}: {row[1]:,}")
    
    return user_behavior
def balanced_activity_sample(
    users_stratified: pl.LazyFrame,
    seed: int = 42,
    verbose: bool = True
) -> pl.LazyFrame:
    """
    Creates a 50/50 dataset of High Activity vs Low Activity users.
    
    Strategy:
    1. Keep ALL 'power' and 'regular' users (High Signal).
    2. Undersample 'cold_start' and 'casual' users to match the count of High Signal users.
    
    Why?
    - If we sample randomly, 90% of data is cold-start (Noise).
    - We need the model to learn deep item correlations from the Power users.
    """
    
    # 1. Define Groups
    # High Signal: The users who actually teach us what items go together
    high_signal_condition = pl.col("user_tier").is_in(["power", "regular"])
    
    # Low Signal: Users who mostly need fallback strategies (Popularity/Trend)
    low_signal_condition = pl.col("user_tier").is_in(["casual", "cold_start"])

    # 2. Separate the populations
    df_users = users_stratified.collect() # Materialize for exact counts (it's small enough: user IDs)
    
    high_signal_users = df_users.filter(high_signal_condition)
    low_signal_users = df_users.filter(low_signal_condition)
    
    n_high = len(high_signal_users)
    n_low_available = len(low_signal_users)
    
    # 3. Sample Low Signal to match High Signal (1:1 Ratio)
    # If we have 150k Active users, we pick 150k Inactive users.
    n_sample = min(n_high, n_low_available)
    
    # Sample using Polars sample (shuffle=True)
    low_signal_sampled = low_signal_users.sample(n=n_sample, seed=seed, shuffle=True)
    
    # 4. Combine
    balanced_df = pl.concat([high_signal_users, low_signal_sampled])
    
    if verbose:
        print(f"\n[Balanced Sampling Strategy]")
        print(f"  ├─ High Signal (Power/Regular): {n_high:,} (Kept 100%)")
        print(f"  ├─ Low Signal (Cold/Casual):    {n_sample:,} (Sampled from {n_low_available:,})")
        print(f"  └─ Total Training Users:        {len(balanced_df):,} (50/50 Split)")
        
    return balanced_df.lazy()
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
            'hist_start': datetime(2024, 9, 1),   # Sept 1
            'hist_end': datetime(2024, 10, 31),   # Oct 31 (60 days)
            'target_start': datetime(2024, 11, 5), # Nov 5 (5-day gap)
            'target_end': datetime(2024, 11, 14)   # Nov 14 (10 days)
        },
        'val': {
            'hist_start': datetime(2024, 10, 1),  # Oct 1
            'hist_end': datetime(2024, 11, 30),   # Nov 30 (60 days)
            'target_start': datetime(2024, 12, 5), # Dec 5 (5-day gap)
            'target_end': datetime(2024, 12, 14)   # Dec 14 (10 days)
        },
        'test': {
            # This depends on when your ground truth is from
            # Adjust based on your GT file date range
            'hist_start': datetime(2024, 11, 1),  # Nov 1
            'hist_end': datetime(2024, 12, 31),   # Dec 31 (60 days)
            'target_start': None,  # From ground truth file
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
            (pl.col("unique_buyers") >= min_users) 
            # (pl.col("avg_price") >= p1) &
            # (pl.col("avg_price") <= p99)
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
    split_name: str,       # 'train', 'val', or 'test'
    split_config: dict,
    verbose=True
):
    hist_start = split_config['hist_start']
    hist_end = split_config['hist_end']
    
    # 1. Clean Transactions
    trans_clean = preprocess_transactions(
        transactions, items, hist_end, verbose
    )
    
    # 2. Quality Filter Items
    items_quality = filter_quality_items(
        transactions, items, hist_start, hist_end, verbose=verbose
    )
    
    # 3. Stratify Users
    users_stratified = stratify_users_by_activity(
        transactions, users, hist_start, hist_end, verbose
    )
    
    # 4. SAMPLING LOGIC (The Fix)
    if split_name in ['train', 'val']:
        # For Train/Val: Enforce 50/50 balance to learn personalization
        if verbose: print(f"  Applying Balanced Sampling for {split_name}...")
        users_final = balanced_activity_sample(users_stratified, seed=42, verbose=verbose)
        
    else: 
        # For Test: Keep EVERYONE (100%) or use standard sampling
        # In reality, you usually test on 100% of target users
        if verbose: print(f"  Keeping all users for {split_name} (Evaluation Mode)...")
        users_final = users_stratified
    
    return trans_clean, items_quality, users_final
    
    
# ============================================================================
# MODIFIED: TEST SPLIT WITH HISTORICAL GROUND TRUTH AS HISTORY
# ============================================================================

def load_historical_transactions_from_pickle(
    pickle_path: str,
    verbose: bool = True
) -> pl.LazyFrame:
    """
    Load transaction data from pickle file (01-2025.pkl format)
    
    Args:
        pickle_path: Path to historical transaction pickle (e.g., "01-2025.pkl")
    
    Returns:
        LazyFrame with transaction data (same format as parquet files)
    """
    if verbose:
        print(f"\n[Loading Historical Transactions] {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        hist_data = pickle.load(f)
    
    # Convert to DataFrame
    import pandas as pd
    
    if isinstance(hist_data, dict):
        # Dict of Series → DataFrame
        hist_pd = pd.DataFrame(hist_data)
    elif isinstance(hist_data, pd.DataFrame):
        hist_pd = hist_data
    else:
        raise ValueError(f"Unexpected format in {pickle_path}: {type(hist_data)}")
    
    # Convert to Polars
    hist_df = pl.from_pandas(hist_pd).lazy()
    
    # Convert timestamp to created_date if needed
    if 'timestamp' in hist_df.columns and 'created_date' not in hist_df.columns:
        hist_df = hist_df.with_columns([
            pl.from_epoch(pl.col("timestamp"), time_unit="s")
            .alias("created_date")
        ])
    
    # CRITICAL FIX: Cast customer_id and item_id to String for consistency
    if 'customer_id' in hist_df.columns:
        hist_df = hist_df.with_columns([
            pl.col("customer_id").cast(pl.Utf8).alias("customer_id")
        ])
    
    if 'item_id' in hist_df.columns:
        hist_df = hist_df.with_columns([
            pl.col("item_id").cast(pl.Utf8).alias("item_id")
        ])
    
    if verbose:
        sample = hist_df.select([
            pl.col("customer_id").n_unique().alias("n_users"),
            pl.len().alias("n_transactions")
        ]).collect()
        print(f"  Users: {sample['n_users'][0]:,}")
        print(f"  Transactions: {sample['n_transactions'][0]:,}")
    
    return hist_df


def create_realistic_splits_with_historical_gt(
    transactions: pl.LazyFrame,
    historical_gt_path: str = None,
    verbose: bool = True
):
    """
    Modified split creation that uses historical GT for test window
    
    Args:
        transactions: Main transaction data
        historical_gt_path: Path to 01-2025.pkl (or None to use transaction data)
    """
    
    SPLITS = {
        'train': {
            'hist_start': datetime(2024, 9, 1),
            'hist_end': datetime(2024, 10, 31),
            'target_start': datetime(2024, 11, 5),
            'target_end': datetime(2024, 11, 14)
        },
        'val': {
            'hist_start': datetime(2024, 10, 1),
            'hist_end': datetime(2024, 11, 30),
            'target_start': datetime(2024, 12, 5),
            'target_end': datetime(2024, 12, 14)
        },
        'test': {
            # Use historical GT file if provided
            'use_historical_gt': historical_gt_path is not None,
            'historical_gt_path': historical_gt_path,
            
            # Fallback to transaction data if no GT
            'hist_start': datetime(2024, 11, 1),
            'hist_end': datetime(2024, 12, 31),
            
            # Target is Feb 2025 (from new GT file)
            'target_start': None,  # From 02-2025.pkl
            'target_end': None
        }
    }
    
    if verbose:
        print("\n[Split Configuration]")
        for name, cfg in SPLITS.items():
            print(f"\n{name.upper()}:")
            if name == 'test' and cfg.get('use_historical_gt'):
                print(f"  History: From GT file {cfg['historical_gt_path']}")
                print(f"  Target:  From current GT file (02-2025.pkl)")
            else:
                print(f"  History: {cfg['hist_start'].date()} → {cfg['hist_end'].date()}")
                if cfg.get('target_start'):
                    print(f"  Target:  {cfg['target_start'].date()} → {cfg['target_end'].date()}")
    
    return SPLITS


# ============================================================================
# MODIFIED: PREPROCESSING WITH HISTORICAL GT SUPPORT
# ============================================================================

def get_preprocessed_data_with_historical_gt(
    split_name: str,
    split_config: dict,
    raw_trans: pl.LazyFrame,
    raw_items: pl.LazyFrame,
    raw_users: pl.LazyFrame,
    output_dir: str,
    recreate: bool = False,
    sample_rate: float = 1.0,
    verbose: bool = True
):
    """
    Modified preprocessing that uses historical transaction pickle for test split
    """
    
    # Check if we should use historical transactions
    if split_name == 'test' and split_config.get('use_historical_gt'):
        if verbose:
            print(f"\n[Test Split] Using Historical Transaction File as History")
        
        # Load historical transactions from pickle
        historical_trans = load_historical_transactions_from_pickle(
            split_config['historical_gt_path'],
            verbose=verbose
        )
        
        # CRITICAL FIX: Cast raw_users customer_id to String to match historical_trans
        raw_users_casted = raw_users.with_columns([
            pl.col("customer_id").cast(pl.Utf8).alias("customer_id")
        ])
        
        # Also cast raw_items if needed
        raw_items_casted = raw_items
        if 'item_id' in raw_items.columns:
            raw_items_casted = raw_items.with_columns([
                pl.col("item_id").cast(pl.Utf8).alias("item_id")
            ])
        
        # Use ONLY the historical transactions (don't merge with raw_trans)
        # This ensures we're only using January 2025 data as history
        trans_clean = preprocess_transactions(
            historical_trans,
            raw_items_casted,
            hist_end=datetime(2025, 1, 31),  # End of January
            verbose=verbose
        )
        
        # Filter items based on historical period
        items_clean = filter_quality_items(
            historical_trans,
            raw_items_casted,
            hist_start=datetime(2025, 1, 1),   # All of January
            hist_end=datetime(2025, 1, 31),
            verbose=verbose
        )
        
        # Stratify users based on historical transactions
        users_clean = stratify_users_by_activity(
            historical_trans,
            raw_users_casted,  # Use casted version
            hist_start=datetime(2025, 1, 1),
            hist_end=datetime(2025, 1, 31),
            verbose=verbose
        )
        
        return trans_clean, items_clean, users_clean
    
    else:
        # Original preprocessing for train/val
        return get_preprocessed_data(
            split_name, split_config,
            raw_trans, raw_items, raw_users,
            output_dir, recreate, sample_rate, verbose
        )


"""
FIX: Ground Truth Pickle Loading
The issue is that the pickle contains a DataFrame with column headers,
not a clean dict of user_id -> items
"""

import pickle
import pandas as pd

def load_ground_truth_fixed(pickle_path: str, verbose: bool = True):
    """
    Properly load ground truth from pickle file
    
    Expected formats:
    1. Dict[user_id -> list of items]
    2. Dict[user_id -> {'list_items': [...]}]
    3. DataFrame with columns: customer_id, item_id (or list_items)
    """
    
    if verbose:
        print(f"\n[Loading GT] {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    if verbose:
        print(f"  Type: {type(raw_data)}")
        if isinstance(raw_data, dict):
            first_key = list(raw_data.keys())[0]
            print(f"  First key: '{first_key}' (type: {type(first_key)})")
    
    # Case 1: It's a DataFrame
    if isinstance(raw_data, pd.DataFrame):
        if verbose:
            print(f"  Format: DataFrame")
            print(f"  Columns: {raw_data.columns.tolist()}")
        
        # Check if it has the expected columns
        if 'customer_id' in raw_data.columns:
            if 'list_items' in raw_data.columns:
                # Format: customer_id | list_items
                gt_dict = dict(zip(
                    raw_data['customer_id'].astype(str).str.strip(),
                    raw_data['list_items']
                ))
            elif 'item_id' in raw_data.columns:
                # Format: customer_id | item_id (need to group)
                gt_dict = (
                    raw_data
                    .groupby('customer_id')['item_id']
                    .apply(list)
                    .to_dict()
                )
                # Ensure string keys
                gt_dict = {str(k).strip(): v for k, v in gt_dict.items()}
            else:
                raise ValueError(f"DataFrame missing expected columns. Has: {raw_data.columns.tolist()}")
        else:
            raise ValueError(f"DataFrame missing 'customer_id' column. Has: {raw_data.columns.tolist()}")
    
    # Case 2: It's already a dict
    elif isinstance(raw_data, dict):
        # Check if first key looks like a column name
        first_key = list(raw_data.keys())[0]
        
        if first_key in ['customer_id', 'item_id', 'list_items']:
            # BUG: This is actually a dict of Series (column-oriented format)
            if verbose:
                print(f"  Format: Dict of Series (converting to DataFrame)")
            
            df = pd.DataFrame(raw_data)
            
            # Now process as DataFrame
            if 'customer_id' in df.columns and 'list_items' in df.columns:
                gt_dict = dict(zip(
                    df['customer_id'].astype(str).str.strip(),
                    df['list_items']
                ))
            elif 'customer_id' in df.columns and 'item_id' in df.columns:
                gt_dict = (
                    df.groupby('customer_id')['item_id']
                    .apply(list)
                    .to_dict()
                )
                gt_dict = {str(k).strip(): v for k, v in gt_dict.items()}
            else:
                raise ValueError(f"Cannot parse dict-of-series format. Columns: {df.columns.tolist()}")
        
        else:
            # This is already in correct format: user_id -> items
            if verbose:
                print(f"  Format: Dict (user_id -> items)")
            
            # Just ensure string keys
            gt_dict = {str(k).strip(): v for k, v in raw_data.items()}
    
    else:
        raise ValueError(f"Unexpected ground truth format: {type(raw_data)}")
    
    # Validate
    if not gt_dict:
        raise ValueError("Ground truth is empty after parsing!")
    
    first_user = list(gt_dict.keys())[0]
    first_items = gt_dict[first_user]
    
    if verbose:
        print(f"  ✓ Loaded {len(gt_dict):,} users")
        print(f"  Sample user: '{first_user}'")
        print(f"  Sample items: {first_items[:3] if len(first_items) > 3 else first_items}")
    
    return gt_dict

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