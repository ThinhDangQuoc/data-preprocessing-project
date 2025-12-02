from __future__ import annotations
import argparse
from datetime import datetime, timedelta
from typing import Optional, Sequence
import polars as pl
import numpy as np

def _ensure_columns_exist(lf: pl.LazyFrame, columns: Sequence[str], source_name: str) -> None:
    schema_cols: Sequence[str] = list(lf.collect_schema().keys())
    missing = [col for col in columns if col not in schema_cols]
    if missing:
        raise ValueError(
            f"{source_name} must contain the following columns for feature construction: {missing}"
        )

def _validate_window(start: datetime, end: datetime, window_name: str) -> None:
    if start > end:
        raise ValueError(f"{window_name} start must precede its end (got {start} > {end}).")

def _resolve_user_key(
    users: pl.LazyFrame,
    customer_id_col: str,
    override: Optional[str],
    user_columns: Sequence[str],
) -> Optional[str]:
    if override and override in user_columns:
        return override
    for candidate in (customer_id_col, "customer_id", "user_id"):
        if candidate in user_columns:
            return candidate
    return None

def _build_time_filter(
    lf: pl.LazyFrame, time_col: str, start: datetime, end: datetime
) -> pl.LazyFrame:
    return lf.filter(
        (pl.col(time_col) >= pl.lit(start)) & (pl.col(time_col) <= pl.lit(end))
    )

def build_feature_label_table(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    begin_hist: datetime,
    end_hist: datetime,
    begin_recent: datetime,
    end_recent: datetime,
    *,
    transaction_time_col: str = "timestamp",
    customer_id_col: str = "customer_id",
    item_id_col: str = "item_id",
    item_brand_col: str = "brand",
    item_age_group_col: str = "age_group",
    item_category_col: str = "category",
    user_id_col: Optional[str] = None,
    price_col: str = "price",
    quantity_col: str = "quantity",
    n_popular_candidates: int = 50 # <--- NEW ARGUMENT
) -> pl.LazyFrame:
    """
    Enhanced feature builder with Price, Quantity, Temporal, and Behavior features.
    Includes Negative Sampling (Popularity based) to prevent data leakage.
    """
    _validate_window(begin_hist, end_hist, "Historical window")
    _validate_window(begin_recent, end_recent, "Recent window")
    
    # Validation including Price and Quantity
    _ensure_columns_exist(
        transactions, 
        [transaction_time_col, customer_id_col, item_id_col, price_col, quantity_col], 
        "transactions"
    )
    _ensure_columns_exist(
        items, [item_id_col, item_brand_col, item_age_group_col, item_category_col], "items"
    )
    
    users_schema = list(users.collect_schema().keys())
    
    # Time windows
    hist_transactions = _build_time_filter(transactions, transaction_time_col, begin_hist, end_hist)
    recent_transactions = _build_time_filter(transactions, transaction_time_col, begin_recent, end_recent)
    
    # Recency Split
    hist_duration = (end_hist - begin_hist).days
    mid_hist = begin_hist + timedelta(days=hist_duration // 2)
    hist_recent_window = _build_time_filter(transactions, transaction_time_col, mid_hist, end_hist)
    
    # Item attributes
    brand_key = "_feature_brand"
    age_key = "_feature_age_group"
    category_key = "_feature_category"
    
    item_attrs = items.select([
        item_id_col,
        pl.coalesce(pl.col(item_brand_col), pl.lit("Unknown")).alias(brand_key),
        pl.coalesce(pl.col(item_age_group_col), pl.lit("Unknown")).alias(age_key),
        pl.coalesce(pl.col(item_category_col), pl.lit("Unknown")).alias(category_key),
    ]).unique()
    
    # Enrich transactions with item attributes
    enriched_hist = hist_transactions.join(item_attrs, on=item_id_col, how="left").with_columns([
        pl.col(brand_key).fill_null("Unknown"),
        pl.col(age_key).fill_null("Unknown"),
        pl.col(category_key).fill_null("Unknown"),
        pl.col(price_col).cast(pl.Float64).fill_null(0.0),
        pl.col(quantity_col).cast(pl.Float64).fill_null(1.0)
    ])

    # === 1. ORIGINAL CATEGORICAL FEATURES ===
    brand_counts = enriched_hist.group_by([customer_id_col, brand_key]).agg(pl.count().alias("brand_count"))
    age_counts = enriched_hist.group_by([customer_id_col, age_key]).agg(pl.count().alias("age_group_count"))
    category_counts = enriched_hist.group_by([customer_id_col, category_key]).agg(pl.count().alias("category_count"))
    
    # === 2. TEMPORAL FEATURES (RECENCY) ===
    enriched_recent_hist = hist_recent_window.join(item_attrs, on=item_id_col, how="left").with_columns([
        pl.col(brand_key).fill_null("Unknown"),
        pl.col(age_key).fill_null("Unknown"),
        pl.col(category_key).fill_null("Unknown"),
    ])
    
    brand_recent_counts = enriched_recent_hist.group_by([customer_id_col, brand_key]).agg(pl.count().alias("brand_recent_count"))
    age_recent_counts = enriched_recent_hist.group_by([customer_id_col, age_key]).agg(pl.count().alias("age_recent_count"))
    category_recent_counts = enriched_recent_hist.group_by([customer_id_col, category_key]).agg(pl.count().alias("category_recent_count"))
    
    # === 3. USER BEHAVIOR FEATURES ===
    user_stats = enriched_hist.group_by(customer_id_col).agg([
        pl.count().alias("user_total_purchases"),
        pl.col(item_id_col).n_unique().alias("user_unique_items"),
        pl.col(brand_key).n_unique().alias("user_unique_brands"),
        pl.col(category_key).n_unique().alias("user_unique_categories"),
        pl.col(price_col).mean().alias("user_avg_spend"),
        pl.col(quantity_col).sum().alias("user_total_quantity"),
        (pl.col(price_col) * pl.col(quantity_col)).sum().alias("user_lifetime_value")
    ])
    
    user_item_freq = enriched_hist.group_by([customer_id_col, item_id_col]).agg([
        pl.count().alias("user_item_purchase_count"),
        pl.col(quantity_col).sum().alias("user_item_total_qty"),
        (pl.col(price_col) * pl.col(quantity_col)).sum().alias("user_item_total_spend")
    ])
    
    # === 4. ITEM POPULARITY & ATTRIBUTES ===
    item_stats = enriched_hist.group_by(item_id_col).agg([
        pl.count().alias("item_global_purchases"),
        pl.col(customer_id_col).n_unique().alias("item_unique_customers"),
        pl.col(price_col).mean().alias("item_avg_price"),
        pl.col(quantity_col).sum().alias("item_total_volume")
    ])
    
    brand_popularity = enriched_hist.group_by(brand_key).agg(pl.count().alias("brand_global_popularity"))
    category_popularity = enriched_hist.group_by(category_key).agg(pl.count().alias("category_global_popularity"))
    
    # === 5. DIVERSITY FEATURES ===
    user_brand_diversity = enriched_hist.group_by([customer_id_col, brand_key]).agg(
        pl.col(item_id_col).n_unique().alias("user_brand_item_diversity")
    )
    user_category_diversity = enriched_hist.group_by([customer_id_col, category_key]).agg(
        pl.col(item_id_col).n_unique().alias("user_category_item_diversity")
    )
    
    # =========================================================================
    # === ASSEMBLE CANDIDATES (FIXED FOR LEAKAGE) ===
    # =========================================================================
    
    # 1. Positives: Users who actually bought items in history OR recently
    # (These are the rows we definitely need to score)
    positives = pl.concat([
        hist_transactions.select([customer_id_col, item_id_col]),
        recent_transactions.select([customer_id_col, item_id_col])
    ]).unique()
    
    # 2. Negatives/Discovery: "Popular Items" Strategy
    # We take active users and pair them with the Top N popular items.
    # This creates rows where user has NO history with item (X_11=0) and didn't buy it (Y=0).
    active_users = hist_transactions.select(customer_id_col).unique()
    
    # Get Top N items by volume in history
    top_popular_items = (
        hist_transactions
        .group_by(item_id_col)
        .count()
        .top_k(n_popular_candidates, by="count")
        .select(item_id_col)
    )
    
    # Cross join active users with popular items
    candidates_popular = active_users.join(top_popular_items, how="cross")
    
    # Combine Positives and Generated Negatives
    candidate_pairs = pl.concat([positives, candidates_popular]).unique()
    
    # 3. Filter by known users (if user_id/customer_id table provided)
    resolved_user_key = _resolve_user_key(users, customer_id_col, user_id_col, users_schema)
    if resolved_user_key:
        known_users = users.select(pl.col(resolved_user_key).alias(customer_id_col)).unique()
        candidate_pairs = candidate_pairs.join(known_users, on=customer_id_col, how="inner")

    # =========================================================================

    # Prepare final metadata for Candidates
    candidates = candidate_pairs.join(item_attrs, on=item_id_col, how="left").with_columns([
        pl.col(brand_key).fill_null("Unknown"),
        pl.col(age_key).fill_null("Unknown"),
        pl.col(category_key).fill_null("Unknown"),
    ])
    
    recent_labels = recent_transactions.select([customer_id_col, item_id_col]).unique().with_columns(pl.lit(1).alias("Y"))
    
    # === JOIN ALL FEATURES ===
    feature_table = (
        candidates
        # Original features
        .join(brand_counts, on=[customer_id_col, brand_key], how="left")
        .join(age_counts, on=[customer_id_col, age_key], how="left")
        .join(category_counts, on=[customer_id_col, category_key], how="left")
        # Recency
        .join(brand_recent_counts, on=[customer_id_col, brand_key], how="left")
        .join(age_recent_counts, on=[customer_id_col, age_key], how="left")
        .join(category_recent_counts, on=[customer_id_col, category_key], how="left")
        # User Stats
        .join(user_stats, on=customer_id_col, how="left")
        # User-Item Stats
        .join(user_item_freq, on=[customer_id_col, item_id_col], how="left")
        # Item Stats
        .join(item_stats, on=item_id_col, how="left")
        # Brand/Cat Stats
        .join(brand_popularity, on=brand_key, how="left")
        .join(category_popularity, on=category_key, how="left")
        # Diversity
        .join(user_brand_diversity, on=[customer_id_col, brand_key], how="left")
        .join(user_category_diversity, on=[customer_id_col, category_key], how="left")
        # Labels
        .join(recent_labels, on=[customer_id_col, item_id_col], how="left")
        .with_columns([
            # Fill Nulls for Categorical Counts
            pl.col("brand_count").fill_null(0).cast(pl.Int64).alias("X_1"),
            pl.col("age_group_count").fill_null(0).cast(pl.Int64).alias("X_2"),
            pl.col("category_count").fill_null(0).cast(pl.Int64).alias("X_3"),
            
            # Recency
            pl.col("brand_recent_count").fill_null(0).cast(pl.Int64).alias("X_4"),
            pl.col("age_recent_count").fill_null(0).cast(pl.Int64).alias("X_5"),
            pl.col("category_recent_count").fill_null(0).cast(pl.Int64).alias("X_6"),
            
            # User General
            pl.col("user_total_purchases").fill_null(0).cast(pl.Int64).alias("X_7"),
            pl.col("user_unique_items").fill_null(0).cast(pl.Int64).alias("X_8"),
            pl.col("user_unique_brands").fill_null(0).cast(pl.Int64).alias("X_9"),
            pl.col("user_unique_categories").fill_null(0).cast(pl.Int64).alias("X_10"),
            
            # User-Item Freq
            pl.col("user_item_purchase_count").fill_null(0).cast(pl.Int64).alias("X_11"),
            
            # Item Popularity
            pl.col("item_global_purchases").fill_null(0).cast(pl.Int64).alias("X_12"),
            pl.col("item_unique_customers").fill_null(0).cast(pl.Int64).alias("X_13"),
            pl.col("brand_global_popularity").fill_null(0).cast(pl.Int64).alias("X_14"),
            pl.col("category_global_popularity").fill_null(0).cast(pl.Int64).alias("X_15"),
            
            # Diversity
            pl.col("user_brand_item_diversity").fill_null(0).cast(pl.Int64).alias("X_16"),
            pl.col("user_category_item_diversity").fill_null(0).cast(pl.Int64).alias("X_17"),
            
            # --- RATIOS (Encoded and Normalized by Logic) ---
            (pl.col("brand_recent_count").fill_null(0) / (pl.col("brand_count").fill_null(0) + 1)).alias("X_18"),
            (pl.col("user_item_purchase_count").fill_null(0) / (pl.col("user_total_purchases").fill_null(0) + 1)).alias("X_19"),
            (pl.col("item_unique_customers").fill_null(0) / (pl.col("item_global_purchases").fill_null(0) + 1)).alias("X_20"),
            (pl.col("user_unique_items").fill_null(0) / (pl.col("user_total_purchases").fill_null(0) + 1)).alias("X_21"),
            
            # --- NEW PRICE & QUANTITY FEATURES (Log Normalized) ---
            pl.col("item_avg_price").fill_null(0).log1p().alias("X_22"),
            pl.col("item_total_volume").fill_null(0).log1p().alias("X_23"),
            pl.col("user_avg_spend").fill_null(0).log1p().alias("X_24"),
            pl.col("user_total_quantity").fill_null(0).log1p().alias("X_25"),
            pl.col("user_lifetime_value").fill_null(0).log1p().alias("X_26"),
            pl.col("user_item_total_spend").fill_null(0).log1p().alias("X_27"),
            pl.col("user_item_total_qty").fill_null(0).alias("X_28"),
            ((pl.col("item_avg_price").fill_null(0) + 1) / (pl.col("user_avg_spend").fill_null(0) + 1)).alias("X_29"),

            # Label
            pl.col("Y").fill_null(0).cast(pl.UInt8).alias("Y"),
        ])
        .select([
            pl.col(customer_id_col).alias("X-1"),
            pl.col(item_id_col).alias("X_0"),
            *[f"X_{i}" for i in range(1, 30)],
            "Y",
        ])
    )
    
    return feature_table