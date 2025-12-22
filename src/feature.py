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
# 5. MULTI-OBJECTIVE RANKING
# ============================================================================

def compute_final_ranking_score(
    features: pl.LazyFrame,
    verbose: bool = True
) -> pl.LazyFrame:
    """
    Compute final ranking score with balanced objectives
    
    Objectives:
    1. Accuracy (Repurchase)
    2. Discovery (Novelty)
    3. Serendipity (Surprise & Delight)
    """
    
    ranked = features.with_columns([
        # Multi-objective score
        (
            # Core signals (50%)
            pl.col("feat_repurchase_weighted").fill_null(0.0) * 0.25 +
            pl.col("feat_cf_score").fill_null(0.0) * 0.15 +
            pl.col("feat_i2v_score").fill_null(0.0) * 0.10 +
            
            # Discovery signals (30%)
            pl.col("feat_discovery_weighted").fill_null(0.0) * 0.20 +
            pl.col("feat_discovery_new_item").fill_null(0.0) * 0.05 +
            pl.col("feat_discovery_trending").fill_null(0.0) * 0.05 +
            
            # Serendipity (10%)
            pl.col("feat_serendipity_bonus").fill_null(0.0) * 0.10 +
            
            # Baseline (10%)
            pl.col("feat_pop_score").fill_null(0.0) * 0.05 +
            pl.col("feat_cat_rank_score").fill_null(0.0) * 0.05
        )
        .alias("final_ranking_score")
    ])
    
    if verbose:
        # Analyze score distribution by candidate type
        stats = ranked.group_by("candidate_type").agg([
            pl.col("final_ranking_score").mean().alias("avg_score"),
            pl.len().alias("count")
        ]).collect()
        
        print("\n[Ranking Scores by Type]")
        for row in stats.iter_rows():
            print(f"  {row[0]}: avg={row[1]:.3f}, count={row[2]:,}")
    
    return ranked

# ============================================================================
# 2. DIVERSITY & EXPLORATION FEATURES
# ============================================================================
def compute_balanced_features(
    candidates: pl.LazyFrame,
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    user_intent: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose: bool = True
) -> pl.LazyFrame:
    """
    Add features that balance repurchase vs discovery
    """
    
    if verbose:
        print("\n[Feature Engineering] Computing balanced features...")
    
    # Join user intent
    candidates_with_intent = candidates.join(
        user_intent.select([
            "customer_id",
            "user_intent",
            "repurchase_rate",
            "exploration_score"
        ]),
        on="customer_id",
        how="left"
    )
    
    # Add intent-aware weights
    features = candidates_with_intent.with_columns([
        # Repurchase features (boosted for REPURCHASER)
        pl.when(pl.col("user_intent") == "REPURCHASER")
        .then(pl.col("feat_repurchase_score") * 1.5)
        .otherwise(pl.col("feat_repurchase_score") * 0.8)
        .alias("feat_repurchase_weighted"),
        
        # Discovery features (boosted for EXPLORER)
        pl.when(pl.col("user_intent") == "EXPLORER")
        .then(pl.col("discovery_score") * 1.5)
        .when(pl.col("user_intent") == "COLD_START")
        .then(pl.col("discovery_score") * 2.0)  # Extra boost for cold-start
        .otherwise(pl.col("discovery_score") * 0.8)
        .alias("feat_discovery_weighted"),
        
        # Serendipity score: Discovery items for repurchasers (surprise!)
        pl.when(
            (pl.col("user_intent") == "REPURCHASER") &
            (pl.col("candidate_type") == "DISCOVERY")
        )
        .then(pl.lit(1.5))
        .otherwise(pl.lit(0.0))
        .alias("feat_serendipity_bonus")
    ])
    
    return features
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
# ============================================================================
# SOLUTION 2: BETTER FEATURE BALANCING
# ============================================================================


def build_features_robust(
    candidates: pl.LazyFrame,
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose: bool = True
) -> pl.LazyFrame:
    """
    MODIFIED: Include comprehensive repurchase features
    """
    if verbose:
        print(f"\n[Stage 2] Building Features with Repurchase Support...")
    
    days_in_window = max((hist_end - hist_start).days, 1)
    
    # 1. Regularized Popularity
    item_pop = compute_regularized_popularity(
        transactions, hist_start, hist_end, verbose=verbose
    )
    
    # 2. Diversity Features
    diversity_feats = compute_diversity_features(
        transactions, items, candidates, 
        hist_start, hist_end, verbose=verbose
    )
    
    # 3. Temporal Features
    temporal_feats = compute_temporal_features(
        transactions, candidates,
        hist_start, hist_end, verbose=verbose
    )
    
    # 4. Interaction Quality
    quality_feats = compute_interaction_quality(
        transactions, items, candidates,
        hist_start, hist_end, verbose=verbose
    )
    
    # 5. Repurchase Features
    repurchase_feats = compute_repurchase_features(
        transactions, candidates,
        hist_start, hist_end, verbose=verbose
    )
    
    # 6. FIXED: User Stats - Filter by candidates first
    target_users = candidates.select("customer_id").unique()
    
    user_daily_stats = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
        .join(target_users, on="customer_id", how="inner")  # ← FIX: Only compute for relevant users
        .group_by("customer_id")
        .agg([
            (pl.len() / days_in_window).cast(pl.Float32).alias("user_daily_purchase_rate"),
            pl.col("price").mean().cast(pl.Float32).alias("user_avg_spend"),
            pl.col("item_id").n_unique().cast(pl.UInt32).alias("user_item_diversity")
        ])
        .with_columns([
            pl.col("user_avg_spend").fill_null(0.0),
            pl.col("user_item_diversity").fill_null(1)
        ])
    )
    
    # 7. MERGE ALL
    features = (
        candidates
        .join(item_pop, on="item_id", how="left")
        .join(diversity_feats, on=["customer_id", "item_id"], how="left")
        .join(temporal_feats, on=["customer_id", "item_id"], how="left")
        .join(quality_feats, on=["customer_id", "item_id"], how="left")
        .join(repurchase_feats, on=["customer_id", "item_id"], how="left")
        .join(user_daily_stats, on="customer_id", how="left")
    )
    
    # 8. Final column selection with proper null handling
    final_cols = [
        # Candidate generation scores
        "feat_cf_score", "feat_i2v_score", "feat_repurchase_score", "feat_pop_score",
        "feat_cat_rank_score", "feat_trend_score",
        # Popularity features
        "smoothed_popularity", "log_popularity", "monthly_sales_rate",
        # Diversity features
        "exploration_score", "novelty_score", "category_entropy",
        # Temporal features
        "sales_momentum", "recent_acceleration",
        # Quality features
        "price_fit_score", "complementarity_bonus",
        # User stats
        "user_daily_purchase_rate", "user_avg_spend", "user_item_diversity",
        # Repurchase features
        "item_repurchase_rate", "user_repurchase_propensity", 
        "feat_new_score",  # ← ADD THIS
        "past_cnt", "days_since",
        "purchase_regularity_score", "repurchase_composite_score"
    ]
        
    output_cols = [pl.col("customer_id"), pl.col("item_id")]
    for col in final_cols:
        output_cols.append(
            pl.col(col).fill_null(0.0).cast(pl.Float32).alias(col)
        )
    
    return features.select(output_cols)
