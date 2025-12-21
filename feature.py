"""
Robust Feature Engineering to Fix Dominance & Overfitting

Key Fixes:
1. Regularize popularity features (log, percentile)
2. Add diversity features (exploration vs exploitation)
3. Time-aware features (decay, momentum)
4. User-item interaction quality
5. Remove leaky features
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import polars as pl
import numpy as np
from datetime import datetime, timedelta

# Helper to enforce memory efficiency
def _cast_float32(expr):
    return expr.cast(pl.Float32)

def _cast_uint32(expr):
    return expr.cast(pl.UInt32)
# ============================================================================
# 1. REGULARIZED POPULARITY FEATURES
# ============================================================================

# ============================================================================
# 1. REGULARIZED POPULARITY (Optimized)
# ============================================================================
def compute_regularized_popularity(
    transactions: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    alpha: float = 0.1,
    verbose=True
) -> pl.LazyFrame:
    """
    Optimized: Reduced precision to Float32/UInt32 and limited column selection.
    """
    days_in_window = max((hist_end - hist_start).days, 1)
    
    # Pre-filter time window once
    hist_trans = transactions.filter(pl.col("created_date").is_between(hist_start, hist_end))

    # 1. Global Stats (Computed cheaply)
    # We use a separate collect here to get scalars, avoiding a cross-join of a massive table in the lazy graph
    try:
        global_stats = (
            hist_trans.select([
                pl.col("item_id").n_unique().alias("n_items"),
                pl.len().alias("n_txns")
            ]).collect()
        )
        avg_txns = global_stats["n_txns"][0] / max(global_stats["n_items"][0], 1)
    except:
        avg_txns = 1.0

    return (
        hist_trans
        .group_by("item_id")
        .agg([
            pl.len().cast(pl.UInt32).alias("raw_count"),
            pl.col("customer_id").n_unique().cast(pl.UInt32).alias("unique_users")
        ])
        .with_columns([
            # Bayesian Smooth
            _cast_float32((pl.col("raw_count") + alpha * avg_txns) / (pl.col("raw_count") + alpha))
            .alias("smoothed_popularity"),
            
            # Log Scale
            _cast_float32(pl.col("raw_count").log1p() / pl.lit(np.log1p(global_stats["n_txns"][0])))
            .alias("log_popularity"),
            
            # Rate
            _cast_float32(pl.col("raw_count") * 30.0 / days_in_window)
            .alias("monthly_sales_rate")
        ])
        .select(["item_id", "smoothed_popularity", "log_popularity", "monthly_sales_rate", "unique_users"])
    )



# ============================================================================
# 2. DIVERSITY & EXPLORATION FEATURES
# ============================================================================

# ============================================================================
# 2. DIVERSITY FEATURES (Optimized)
# ============================================================================
def compute_diversity_features(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    candidates: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose=True
) -> pl.LazyFrame:
    """
    Optimized: Filters history to only 'candidate users' before complex aggregations.
    """
    # 1. Filter Transactions to only relevant Users (Huge memory saver)
    target_users = candidates.select("customer_id").unique()
    
    hist_trans = (
        transactions
        .filter(pl.col("created_date").is_between(hist_start, hist_end))
        .join(target_users, on="customer_id", how="inner") # <--- Critical Optimization
    )

    # 2. User Exploration (Lightweight)
    user_exploration = (
        hist_trans
        .join(items.select(["item_id", "category_id"]), on="item_id", how="left")
        .group_by("customer_id")
        .agg([
            pl.col("category_id").n_unique().cast(pl.UInt32).alias("categories_explored"),
            _cast_float32(1.0 - (pl.len() / pl.col("item_id").n_unique())).alias("exploration_score"),
            _cast_float32(pl.col("price").std().fill_null(0)).alias("price_exploration")
        ])
    )

    # 3. Item Novelty (Global, but simple)
    # Re-using raw transactions for item global stats is fine as it's a simple count
    item_novelty = (
        transactions
        .filter(pl.col("created_date").is_between(hist_start, hist_end))
        .group_by("item_id")
        .len()
        .select([
            pl.col("item_id"),
            _cast_float32(1.0 / (1.0 + pl.col("len").log1p())).alias("novelty_score")
        ])
    )

    # 4. Entropy (Expensive -> Optimized)
    # We aggregate on user+category, which is smaller than raw transactions
    user_category_entropy = (
        hist_trans
        .join(items.select(["item_id", "category_id"]), on="item_id", how="left")
        .filter(pl.col("category_id").is_not_null())
        .group_by(["customer_id", "category_id"])
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over("customer_id")).alias("p_cat")
        )
        .group_by("customer_id")
        .agg(
            _cast_float32((-(pl.col("p_cat") * pl.col("p_cat").log())).sum()).alias("category_entropy")
        )
    )

    return (
        candidates
        .select(["customer_id", "item_id"]) # Start clean
        .join(user_exploration, on="customer_id", how="left")
        .join(item_novelty, on="item_id", how="left")
        .join(user_category_entropy, on="customer_id", how="left")
        .select([
            "customer_id", "item_id",
            pl.col("exploration_score").fill_null(0.5),
            pl.col("novelty_score").fill_null(0.5),
            pl.col("category_entropy").fill_null(0.0),
            pl.col("categories_explored").fill_null(1),
            pl.col("price_exploration").fill_null(0.0)
        ])
    )
def compute_repurchase_features(
    transactions: pl.LazyFrame,
    candidates: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose: bool = True
) -> pl.LazyFrame:
    """
    Optimized Repurchase Features.
    """
    if verbose: print(f"  [Repurchase Features] Computing metrics...")

    # Get Schema types from candidates to ensure join safety
    try:
        cand_schema = candidates.collect_schema()
        cid_type = cand_schema["customer_id"]
        iid_type = cand_schema["item_id"]
    except:
        # Fallback if schema inference fails (rare)
        cid_type = pl.Utf8
        iid_type = pl.Utf8

    # 1. Prepare History (Filter & Cast)
    # We cast columns immediately to ensure the join works
    hist_trans = (
        transactions
        .filter(pl.col("created_date").is_between(hist_start, hist_end))
        .with_columns([
            pl.col("customer_id").cast(cid_type),
            pl.col("item_id").cast(iid_type)
        ])
    )

    # 2. Get User Stats (Global) - Filter users by candidates
    # Extract unique users from candidates (small operation)
    target_users = candidates.select("customer_id").unique()
    
    user_stats = (
        hist_trans
        .join(target_users, on="customer_id", how="inner") # Filter history to only relevant users
        .group_by("customer_id")
        .agg([
            pl.col("item_id").n_unique().cast(pl.UInt32).alias("uniq_items"),
            pl.len().cast(pl.UInt32).alias("cnt")
        ])
        .select([
            "customer_id",
            _cast_float32(pl.col("cnt") / pl.col("uniq_items")).alias("user_repurchase_propensity")
        ])
    )
    
    # 3. Get Item Stats (Global)
    item_stats = (
        hist_trans
        .group_by("item_id")
        .agg([
            pl.col("customer_id").n_unique().cast(pl.UInt32).alias("uniq_buy"),
            pl.len().cast(pl.UInt32).alias("cnt")
        ])
        .select([
            "item_id",
            _cast_float32(pl.col("cnt") / pl.col("uniq_buy")).alias("item_repurchase_rate")
        ])
    )

    # 4. User-Item Interaction (Heavy)
    # Filter history to ONLY pairs that exist in candidates
    candidate_pairs = candidates.select(["customer_id", "item_id"])
    
    user_item_history = (
        hist_trans
        .join(candidate_pairs, on=["customer_id", "item_id"], how="inner") # Exact join
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().cast(pl.UInt32).alias("past_cnt"),
            pl.col("created_date").max().alias("last_date"),
            pl.col("created_date").min().alias("first_date")
        ])
        .with_columns([
            (hist_end - pl.col("last_date")).dt.total_days().alias("days_since"),
            ((pl.col("last_date") - pl.col("first_date")).dt.total_days() / 
             pl.col("past_cnt").clip(1, None)).alias("avg_days")
        ])
        .select([
            "customer_id", "item_id",
            "past_cnt", "days_since",
            _cast_float32(pl.col("avg_days")).alias("avg_days_between_purchases"),
            
            # Inline Score Calculation
            _cast_float32(
                pl.when(pl.col("past_cnt") >= 3)
                .then(
                    pl.when(pl.col("avg_days") < 45).then(2.0)
                    .when(pl.col("avg_days") < 90).then(1.5)
                    .otherwise(1.0)
                ).otherwise(0.0)
            ).alias("purchase_regularity_score"),

            _cast_float32(
                pl.when(pl.col("days_since") < 30).then(2.0)
                .when(pl.col("days_since") < 60).then(1.5)
                .otherwise(1.0)
            ).alias("recency_boost")
        ])
    )

    # 5. Merge all
    return (
        candidates
        .select(["customer_id", "item_id"])
        .join(item_stats, on="item_id", how="left")
        .join(user_stats, on="customer_id", how="left")
        .join(user_item_history, on=["customer_id", "item_id"], how="left")
        .with_columns([
            pl.col("item_repurchase_rate").fill_null(1.0),
            pl.col("user_repurchase_propensity").fill_null(1.0),
            pl.col("past_cnt").fill_null(0),
            pl.col("days_since").fill_null(999),
            pl.col("purchase_regularity_score").fill_null(0.0),
            pl.col("recency_boost").fill_null(0.0),
        ])
        .with_columns(
            _cast_float32(
                pl.col("past_cnt").log1p() * 2.0 + 
                pl.col("recency_boost") * 1.5 + 
                pl.col("purchase_regularity_score") * 1.0 +
                (pl.col("user_repurchase_propensity") - 1.0) * 0.5
            ).alias("repurchase_composite_score")
        )
    )
# ============================================================================
# 3. TIME-AWARE FEATURES (MOMENTUM & DECAY)
# ============================================================================

# ============================================================================
# FIXED: TEMPORAL FEATURES (Fix correlation bug)
# ============================================================================

# ============================================================================
# 4. TEMPORAL FEATURES
# ============================================================================
def compute_temporal_features(
    transactions: pl.LazyFrame,
    candidates: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose=True
) -> pl.LazyFrame:
    mid_date = hist_start + (hist_end - hist_start) / 2
    
    # Pre-aggregate item counts per period (Low Cardinality: Items only)
    def get_sales(start, end, alias):
        return (
            transactions
            .filter(pl.col("created_date").is_between(start, end))
            .group_by("item_id")
            .len()
            .rename({"len": alias})
        )

    early = get_sales(hist_start, mid_date, "s_early")
    late = get_sales(mid_date, hist_end, "s_late")
    recent = get_sales(hist_end - timedelta(days=14), hist_end, "s_recent")

    item_momentum = (
        early
        .join(late, on="item_id", how="outer_coalesce")
        .join(recent, on="item_id", how="outer_coalesce")
        .fill_null(0)
        .select([
            "item_id",
            _cast_float32((pl.col("s_late") - pl.col("s_early")) / (pl.col("s_early") + 1)).alias("sales_momentum"),
            _cast_float32(pl.col("s_recent") / (pl.col("s_late") + 1)).alias("recent_acceleration")
        ])
    )
    
    # User engagement - Filter by candidates first
    target_users = candidates.select("customer_id").unique()
    user_eng = (
        transactions
        .filter(pl.col("created_date").is_between(hist_start, hist_end))
        .join(target_users, on="customer_id", how="inner")
        .group_by("customer_id")
        .agg(_cast_float32(pl.len() / (hist_end - hist_start).days).alias("purchase_frequency"))
    )
    
    return (
        candidates.select(["customer_id", "item_id"])
        .join(item_momentum, on="item_id", how="left")
        .join(user_eng, on="customer_id", how="left")
        .fill_null(0)
    )

# ============================================================================
# 4. USER-ITEM INTERACTION QUALITY
# ============================================================================

# ============================================================================
# 5. INTERACTION QUALITY
# ============================================================================
def compute_interaction_quality(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    candidates: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose=True
) -> pl.LazyFrame:
    # Minimal Item Lookup
    items_min = items.select(["item_id", "category_id", "price"])
    
    # 1. User Baskets - Filter by candidates first
    target_users = candidates.select("customer_id").unique()
    
    user_baskets = (
        transactions
        .filter(pl.col("created_date").is_between(hist_start, hist_end))
        .join(target_users, on="customer_id", how="inner")
        .join(items_min, on="item_id", how="left") # Price needed here
        .group_by("customer_id")
        .agg([
            pl.col("price").mean().alias("u_avg_price"),
            pl.col("price").std().fill_null(0).alias("u_price_std"),
            pl.col("category_id").n_unique().alias("u_div")
        ])
    )

    # 2. Join candidates
    return (
        candidates
        .join(user_baskets, on="customer_id", how="left")
        .join(items_min, on="item_id", how="left")
        .select([
            "customer_id", "item_id",
            _cast_float32(
                1.0 - ((pl.col("price") - pl.col("u_avg_price")).abs() / (pl.col("u_price_std") + 1.0))
                .clip(0, 3.0) / 3.0
            ).alias("price_fit_score"),
            
            _cast_float32(
                pl.when(pl.col("u_div") > 3).then(1.0).otherwise(0.5)
            ).alias("complementarity_bonus")
        ])
    )