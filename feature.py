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

# ============================================================================
# 1. REGULARIZED POPULARITY FEATURES
# ============================================================================

def compute_regularized_popularity(
    transactions: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    alpha: float = 0.1,
    verbose=True
) -> pl.LazyFrame:
    """
    Instead of raw counts, use:
    1. Log-scaled popularity
    2. Percentile-based ranks (less sensitive to outliers)
    3. Bayesian smoothing (add prior)
    """
    
    days_in_window = (hist_end - hist_start).days
    
    # Global statistics for Bayesian smoothing
    global_stats = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
        .select([
            pl.col("item_id").n_unique().alias("total_items"),
            pl.len().alias("total_txns")
        ])
        .collect()
    )
    
    avg_txns_per_item = global_stats["total_txns"][0] / global_stats["total_items"][0]
    
    item_pop = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
        .group_by("item_id")
        .agg([
            pl.len().alias("raw_count"),
            pl.col("customer_id").n_unique().alias("unique_users")
        ])
        .with_columns([
            # Bayesian smoothed popularity
            (
                (pl.col("raw_count") + alpha * avg_txns_per_item) /
                (pl.col("raw_count") + alpha)
            ).alias("smoothed_popularity"),
            
            # Log-scaled (prevent extreme values)
            (pl.col("raw_count").log1p() / pl.col("raw_count").log1p().max())
                .alias("log_popularity"),
            
            # Percentile rank (robust to outliers)
            (pl.col("raw_count").rank("dense") / pl.col("raw_count").rank("dense").max())
                .alias("popularity_percentile"),
            
            # Normalize by time window
            (pl.col("raw_count") * 30.0 / days_in_window)
                .alias("monthly_sales_rate")
        ])
        .select([
            "item_id",
            "smoothed_popularity",
            "log_popularity", 
            "popularity_percentile",
            "monthly_sales_rate",
            "unique_users"
        ])
    )
    
    if verbose:
        stats = item_pop.select([
            pl.col("log_popularity").mean(),
            pl.col("popularity_percentile").mean()
        ]).collect()
        print(f"[Regularized Popularity] Mean log_pop: {stats[0,0]:.3f}, Mean percentile: {stats[0,1]:.3f}")
    
    return item_pop


# ============================================================================
# 2. DIVERSITY & EXPLORATION FEATURES
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
    Encourage model to explore beyond popular items:
    1. User exploration score (how much they try new things)
    2. Item novelty score (how different from user's history)
    3. Category diversity
    """
    
    hist_trans = transactions.filter(
        (pl.col("created_date") >= hist_start) & 
        (pl.col("created_date") <= hist_end)
    )
    
    # User exploration propensity
    user_exploration = (
        hist_trans
        .join(items, on="item_id", how="left")
        .group_by("customer_id")
        .agg([
            # How many unique categories explored
            pl.col("category_id").n_unique().alias("categories_explored"),
            
            # Repeat purchase rate (inverse = exploration)
            (
                1.0 - (pl.len() / pl.col("item_id").n_unique())
            ).alias("exploration_score"),
            
            # Price variance (willing to try different price points)
            pl.col("price").std().alias("price_exploration")
        ])
        .with_columns([
            pl.col("price_exploration").fill_null(0.0)
        ])
    )
    
    # Item novelty score (inverse of popularity)
    item_novelty = (
        hist_trans
        .group_by("item_id")
        .len()
        .with_columns([
            # Novelty = 1 / (1 + log(popularity))
            (1.0 / (1.0 + pl.col("len").log1p()))
                .alias("novelty_score")
        ])
        .select(["item_id", "novelty_score"])
    )
    
    # User's historical category distribution
    user_category_entropy = (
        hist_trans
        .join(items, on="item_id", how="left")
        .filter(pl.col("category_id").is_not_null())
        .group_by(["customer_id", "category_id"])
        .len()
        .with_columns([
            # Probability of each category
            (pl.col("len") / pl.col("len").sum().over("customer_id"))
                .alias("p_cat")
        ])
        .group_by("customer_id")
        .agg([
            # Shannon entropy: -sum(p * log(p))
            (-(pl.col("p_cat") * pl.col("p_cat").log())).sum()
                .alias("category_entropy")
        ])
    )
    
    # Merge into candidates
    diversity_features = (
        candidates
        .join(user_exploration, on="customer_id", how="left")
        .join(item_novelty, on="item_id", how="left")
        .join(user_category_entropy, on="customer_id", how="left")
        .with_columns([
            pl.col("exploration_score").fill_null(0.5),
            pl.col("novelty_score").fill_null(0.5),
            pl.col("category_entropy").fill_null(0.0),
            pl.col("categories_explored").fill_null(1),
            pl.col("price_exploration").fill_null(0.0)
        ])
        .select([
            "customer_id",
            "item_id",
            "exploration_score",
            "novelty_score",
            "category_entropy",
            "categories_explored",
            "price_exploration"
        ])
    )
    
    if verbose:
        print(f"[Diversity Features] Added 5 exploration features")
    
    return diversity_features


# ============================================================================
# 3. TIME-AWARE FEATURES (MOMENTUM & DECAY)
# ============================================================================

# ============================================================================
# FIXED: TEMPORAL FEATURES (Fix correlation bug)
# ============================================================================

def compute_temporal_features(
    transactions: pl.LazyFrame,
    candidates: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose=True
) -> pl.LazyFrame:
    """
    FIXED: Removed invalid correlation, added proper temporal metrics
    """
    
    # Split history into periods
    mid_date = hist_start + (hist_end - hist_start) / 2
    recent_date = hist_end - timedelta(days=14)
    
    # Item momentum (sales trend)
    def get_period_sales(start, end, label):
        return (
            transactions
            .filter(
                (pl.col("created_date") >= start) & 
                (pl.col("created_date") <= end)
            )
            .group_by("item_id")
            .len()
            .rename({"len": f"sales_{label}"})
        )
    
    early_sales = get_period_sales(hist_start, mid_date, "early")
    late_sales = get_period_sales(mid_date, hist_end, "late")
    recent_sales = get_period_sales(recent_date, hist_end, "recent")
    
    item_momentum = (
        early_sales
        .join(late_sales, on="item_id", how="outer_coalesce")
        .join(recent_sales, on="item_id", how="outer_coalesce")
        .with_columns([
            pl.col("sales_early").fill_null(0),
            pl.col("sales_late").fill_null(0),
            pl.col("sales_recent").fill_null(0)
        ])
        .with_columns([
            # Momentum = (late - early) / (early + 1)
            (
                (pl.col("sales_late") - pl.col("sales_early")) /
                (pl.col("sales_early") + 1.0)
            ).alias("sales_momentum"),
            
            # Recent acceleration
            (
                pl.col("sales_recent") / 
                (pl.col("sales_late") + 1.0)
            ).alias("recent_acceleration"),
        ])
        .select([
            "item_id",
            "sales_momentum",
            "recent_acceleration",
        ])
    )
    
    # ========================================================================
    # FIX: User engagement (removed invalid correlation)
    # ========================================================================
    user_engagement = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
        .group_by("customer_id")
        .agg([
            # Days since last purchase
            (hist_end - pl.col("created_date").max()).dt.total_days()
                .alias("days_since_last_purchase"),
            
            # Purchase frequency (purchases per day)
            (pl.len() / (hist_end - hist_start).days)
                .alias("purchase_frequency")
        ])
    )
    
    # Merge
    temporal_features = (
        candidates
        .join(item_momentum, on="item_id", how="left")
        .join(user_engagement, on="customer_id", how="left")
        .with_columns([
            pl.col("sales_momentum").fill_null(0.0),
            pl.col("recent_acceleration").fill_null(1.0),
            pl.col("days_since_last_purchase").fill_null(999),
            pl.col("purchase_frequency").fill_null(0.0)
        ])
        .select([
            "customer_id",
            "item_id",
            "sales_momentum",
            "recent_acceleration",
            "days_since_last_purchase"
        ])
    )
    
    if verbose:
        print(f"  [Temporal Features] Added 3 time-aware features")
    
    return temporal_features

# ============================================================================
# 4. USER-ITEM INTERACTION QUALITY
# ============================================================================

def compute_interaction_quality(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    candidates: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose=True
) -> pl.LazyFrame:
    """
    Measure quality of potential match beyond just co-occurrence:
    1. Complementary products (bought together but different categories)
    2. Price consistency
    3. Timing patterns
    """
    
    hist_trans = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
        .join(items.select(["item_id", "category_id", "price"]), 
              on="item_id", how="left")
    )
    
    # User's average purchase basket
    user_baskets = (
        hist_trans
        .group_by("customer_id")
        .agg([
            pl.col("price").mean().alias("avg_basket_price"),
            pl.col("price").std().alias("basket_price_std"),
            pl.col("category_id").n_unique().alias("basket_category_diversity")
        ])
        .with_columns([
            pl.col("basket_price_std").fill_null(0.0)
        ])
    )
    
    # Candidate item info
    candidate_items = (
        candidates
        .select("item_id")
        .unique()
        .join(items.select(["item_id", "category_id", "price"]), 
              on="item_id", how="left")
    )
    
    # Compute match quality
    interaction_quality = (
        candidates
        .join(user_baskets, on="customer_id", how="left")
        .join(candidate_items, on="item_id", how="left")
        .with_columns([
            # Price fit score
            (
                1.0 - (
                    (pl.col("price") - pl.col("avg_basket_price")).abs() /
                    (pl.col("basket_price_std") + 1.0)
                ).clip(0, 3.0) / 3.0
            ).alias("price_fit_score"),
            
            # Category complementarity
            pl.when(pl.col("basket_category_diversity") > 3)
            .then(1.0)  # Users who explore deserve diverse recs
            .otherwise(0.5)
            .alias("complementarity_bonus")
        ])
        .select([
            "customer_id",
            "item_id",
            "price_fit_score",
            "complementarity_bonus"
        ])
    )
    
    if verbose:
        print(f"[Interaction Quality] Added 2 match quality features")
    
    return interaction_quality
