from __future__ import annotations
import argparse
from datetime import datetime, timedelta
from typing import Optional, Sequence, Dict, List, Any
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

def build_feature_label_table_with_gt(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    begin_hist: datetime,
    end_hist: datetime,
    gt_dict: Dict[Any, Dict[str, List[Any]]],
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
    n_popular_candidates: int = 50,
    include_gt_in_candidates: bool = True
) -> pl.LazyFrame:
    """
    Enhanced feature builder that uses pre-computed ground truth for labels.
    FIXED: Uses historical active users for candidates instead of future GT users.
    """
    _validate_window(begin_hist, end_hist, "Historical window")
    
    _ensure_columns_exist(
        transactions, 
        [transaction_time_col, customer_id_col, item_id_col, price_col, quantity_col], 
        "transactions"
    )
    _ensure_columns_exist(
        items, [item_id_col, item_brand_col, item_age_group_col, item_category_col], "items"
    )
    
    users_schema = list(users.collect_schema().keys())
    
    # Only use history window for features
    hist_transactions = _build_time_filter(transactions, transaction_time_col, begin_hist, end_hist)
    
    # Recency Split within history
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

    # === FIX: Calculate Global Average Price for Imputation ===
    # We use .collect().item() to get a scalar value to fill nulls later
    global_avg_price = enriched_hist.select(pl.col(price_col).mean()).collect().item()
    if global_avg_price is None: global_avg_price = 0.0
    
    # === BUILD ALL FEATURE AGGREGATIONS ===
    brand_counts = enriched_hist.group_by([customer_id_col, brand_key]).agg(pl.count().alias("brand_count"))
    age_counts = enriched_hist.group_by([customer_id_col, age_key]).agg(pl.count().alias("age_group_count"))
    category_counts = enriched_hist.group_by([customer_id_col, category_key]).agg(pl.count().alias("category_count"))
    
    enriched_recent_hist = hist_recent_window.join(item_attrs, on=item_id_col, how="left").with_columns([
        pl.col(brand_key).fill_null("Unknown"),
        pl.col(age_key).fill_null("Unknown"),
        pl.col(category_key).fill_null("Unknown"),
    ])
    
    brand_recent_counts = enriched_recent_hist.group_by([customer_id_col, brand_key]).agg(pl.count().alias("brand_recent_count"))
    age_recent_counts = enriched_recent_hist.group_by([customer_id_col, age_key]).agg(pl.count().alias("age_recent_count"))
    category_recent_counts = enriched_recent_hist.group_by([customer_id_col, category_key]).agg(pl.count().alias("category_recent_count"))
    
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
    
    item_stats = enriched_hist.group_by(item_id_col).agg([
        pl.count().alias("item_global_purchases"),
        pl.col(customer_id_col).n_unique().alias("item_unique_customers"),
        pl.col(price_col).mean().alias("item_avg_price"),
        pl.col(quantity_col).sum().alias("item_total_volume")
    ])
    
    brand_popularity = enriched_hist.group_by(brand_key).agg(pl.count().alias("brand_global_popularity"))
    category_popularity = enriched_hist.group_by(category_key).agg(pl.count().alias("category_global_popularity"))
    
    user_brand_diversity = enriched_hist.group_by([customer_id_col, brand_key]).agg(
        pl.col(item_id_col).n_unique().alias("user_brand_item_diversity")
    )
    user_category_diversity = enriched_hist.group_by([customer_id_col, category_key]).agg(
        pl.col(item_id_col).n_unique().alias("user_category_item_diversity")
    )
    
    # =========================================================================
    # === CANDIDATE GENERATION ===
    # =========================================================================
    
    transaction_schema = hist_transactions.collect_schema()
    customer_dtype = transaction_schema[customer_id_col]
    item_dtype = transaction_schema[item_id_col]
    
    # Base candidates: Historical purchases
    historical_purchases = hist_transactions.select([customer_id_col, item_id_col]).unique()
    
    # === FIX: Use HISTORICAL Active Users, not Future GT Users ===
    active_users_hist = hist_transactions.select(customer_id_col).unique()
    
    top_popular_items = (
        hist_transactions
        .group_by(item_id_col)
        .count()
        .top_k(n_popular_candidates, by="count")
        .select(item_id_col)
    )
    
    # Join popular items with Historically Active Users
    candidates_popular = active_users_hist.join(top_popular_items, how="cross")
    
    if include_gt_in_candidates:
        # TRAINING: Add true positives
        gt_positive_pairs = []
        for customer_id, gt_data in gt_dict.items():
            for item_id in gt_data['list_items']:
                gt_positive_pairs.append({customer_id_col: customer_id, item_id_col: item_id})
        
        positives_from_gt = (
            pl.DataFrame(gt_positive_pairs)
            .with_columns([
                pl.col(customer_id_col).cast(customer_dtype),
                pl.col(item_id_col).cast(item_dtype)
            ])
            .lazy()
        )
        candidate_pairs = pl.concat([historical_purchases, candidates_popular, positives_from_gt]).unique()
        print(f"  [Candidates] Included GT items for training")
    else:
        # TESTING: Realistic (History + Popular only)
        candidate_pairs = pl.concat([historical_purchases, candidates_popular]).unique()
        print(f"  [Candidates] Excluded GT items for realistic testing")
    
    # Filter by known users
    resolved_user_key = _resolve_user_key(users, customer_id_col, user_id_col, users_schema)
    if resolved_user_key:
        known_users = users.select(pl.col(resolved_user_key).alias(customer_id_col)).unique()
        candidate_pairs = candidate_pairs.join(known_users, on=customer_id_col, how="inner")
    
    # Add item attributes to candidates
    candidates = candidate_pairs.join(item_attrs, on=item_id_col, how="left").with_columns([
        pl.col(brand_key).fill_null("Unknown"),
        pl.col(age_key).fill_null("Unknown"),
        pl.col(category_key).fill_null("Unknown"),
    ])
    
    # =========================================================================
    # === CREATE LABELS FROM GT ===
    # =========================================================================
    gt_label_pairs = []
    for customer_id, gt_data in gt_dict.items():
        for item_id in gt_data['list_items']:
            gt_label_pairs.append({customer_id_col: customer_id, item_id_col: item_id, "Y": 1})
    
    gt_labels = (
        pl.DataFrame(gt_label_pairs)
        .with_columns([
            pl.col(customer_id_col).cast(customer_dtype),
            pl.col(item_id_col).cast(item_dtype),
            pl.col("Y").cast(pl.UInt8)
        ])
        .lazy()
    )
    
    # =========================================================================
    # === JOIN ALL FEATURES ===
    # =========================================================================
    feature_table = (
        candidates
        .join(brand_counts, on=[customer_id_col, brand_key], how="left")
        .join(age_counts, on=[customer_id_col, age_key], how="left")
        .join(category_counts, on=[customer_id_col, category_key], how="left")
        .join(brand_recent_counts, on=[customer_id_col, brand_key], how="left")
        .join(age_recent_counts, on=[customer_id_col, age_key], how="left")
        .join(category_recent_counts, on=[customer_id_col, category_key], how="left")
        .join(user_stats, on=customer_id_col, how="left")
        .join(user_item_freq, on=[customer_id_col, item_id_col], how="left")
        .join(item_stats, on=item_id_col, how="left")
        .join(brand_popularity, on=brand_key, how="left")
        .join(category_popularity, on=category_key, how="left")
        .join(user_brand_diversity, on=[customer_id_col, brand_key], how="left")
        .join(user_category_diversity, on=[customer_id_col, category_key], how="left")
        .join(gt_labels, on=[customer_id_col, item_id_col], how="left")
        .with_columns([
            pl.col("brand_count").fill_null(0).cast(pl.Int64).alias("X_1"),
            pl.col("age_group_count").fill_null(0).cast(pl.Int64).alias("X_2"),
            pl.col("category_count").fill_null(0).cast(pl.Int64).alias("X_3"),
            pl.col("brand_recent_count").fill_null(0).cast(pl.Int64).alias("X_4"),
            pl.col("age_recent_count").fill_null(0).cast(pl.Int64).alias("X_5"),
            pl.col("category_recent_count").fill_null(0).cast(pl.Int64).alias("X_6"),
            pl.col("user_total_purchases").fill_null(0).cast(pl.Int64).alias("X_7"),
            pl.col("user_unique_items").fill_null(0).cast(pl.Int64).alias("X_8"),
            pl.col("user_unique_brands").fill_null(0).cast(pl.Int64).alias("X_9"),
            pl.col("user_unique_categories").fill_null(0).cast(pl.Int64).alias("X_10"),
            pl.col("user_item_purchase_count").fill_null(0).cast(pl.Int64).alias("X_11"),
            pl.col("item_global_purchases").fill_null(0).cast(pl.Int64).alias("X_12"),
            pl.col("item_unique_customers").fill_null(0).cast(pl.Int64).alias("X_13"),
            pl.col("brand_global_popularity").fill_null(0).cast(pl.Int64).alias("X_14"),
            pl.col("category_global_popularity").fill_null(0).cast(pl.Int64).alias("X_15"),
            pl.col("user_brand_item_diversity").fill_null(0).cast(pl.Int64).alias("X_16"),
            pl.col("user_category_item_diversity").fill_null(0).cast(pl.Int64).alias("X_17"),
            (pl.col("brand_recent_count").fill_null(0) / (pl.col("brand_count").fill_null(0) + 1)).alias("X_18"),
            (pl.col("user_item_purchase_count").fill_null(0) / (pl.col("user_total_purchases").fill_null(0) + 1)).alias("X_19"),
            (pl.col("item_unique_customers").fill_null(0) / (pl.col("item_global_purchases").fill_null(0) + 1)).alias("X_20"),
            (pl.col("user_unique_items").fill_null(0) / (pl.col("user_total_purchases").fill_null(0) + 1)).alias("X_21"),
            # FIX: Fill with Global Avg Price, not 0
            pl.col("item_avg_price").fill_null(global_avg_price).log1p().alias("X_22"),
            pl.col("item_total_volume").fill_null(0).log1p().alias("X_23"),
            pl.col("user_avg_spend").fill_null(0).log1p().alias("X_24"),
            pl.col("user_total_quantity").fill_null(0).log1p().alias("X_25"),
            pl.col("user_lifetime_value").fill_null(0).log1p().alias("X_26"),
            pl.col("user_item_total_spend").fill_null(0).log1p().alias("X_27"),
            pl.col("user_item_total_qty").fill_null(0).alias("X_28"),
            # FIX: Fill with Global Avg Price, not 0
            ((pl.col("item_avg_price").fill_null(global_avg_price) + 1) / (pl.col("user_avg_spend").fill_null(0) + 1)).alias("X_29"),
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
    n_popular_candidates: int = 50
) -> pl.LazyFrame:
    """
    Original feature builder with recent window for labels (for Train/Val).
    FIXED: Removed 'recent_transactions' from Candidates to prevent label leakage.
    """
    _validate_window(begin_hist, end_hist, "Historical window")
    _validate_window(begin_recent, end_recent, "Recent window")
    
    _ensure_columns_exist(
        transactions, 
        [transaction_time_col, customer_id_col, item_id_col, price_col, quantity_col], 
        "transactions"
    )
    _ensure_columns_exist(
        items, [item_id_col, item_brand_col, item_age_group_col, item_category_col], "items"
    )
    
    users_schema = list(users.collect_schema().keys())
    
    hist_transactions = _build_time_filter(transactions, transaction_time_col, begin_hist, end_hist)
    recent_transactions = _build_time_filter(transactions, transaction_time_col, begin_recent, end_recent)
    
    hist_duration = (end_hist - begin_hist).days
    mid_hist = begin_hist + timedelta(days=hist_duration // 2)
    hist_recent_window = _build_time_filter(transactions, transaction_time_col, mid_hist, end_hist)
    
    brand_key = "_feature_brand"
    age_key = "_feature_age_group"
    category_key = "_feature_category"
    
    item_attrs = items.select([
        item_id_col,
        pl.coalesce(pl.col(item_brand_col), pl.lit("Unknown")).alias(brand_key),
        pl.coalesce(pl.col(item_age_group_col), pl.lit("Unknown")).alias(age_key),
        pl.coalesce(pl.col(item_category_col), pl.lit("Unknown")).alias(category_key),
    ]).unique()
    
    enriched_hist = hist_transactions.join(item_attrs, on=item_id_col, how="left").with_columns([
        pl.col(brand_key).fill_null("Unknown"),
        pl.col(age_key).fill_null("Unknown"),
        pl.col(category_key).fill_null("Unknown"),
        pl.col(price_col).cast(pl.Float64).fill_null(0.0),
        pl.col(quantity_col).cast(pl.Float64).fill_null(1.0)
    ])
    
    # === FIX: Calculate Global Average Price for Imputation ===
    global_avg_price = enriched_hist.select(pl.col(price_col).mean()).collect().item()
    if global_avg_price is None: global_avg_price = 0.0

    brand_counts = enriched_hist.group_by([customer_id_col, brand_key]).agg(pl.count().alias("brand_count"))
    age_counts = enriched_hist.group_by([customer_id_col, age_key]).agg(pl.count().alias("age_group_count"))
    category_counts = enriched_hist.group_by([customer_id_col, category_key]).agg(pl.count().alias("category_count"))
    
    enriched_recent_hist = hist_recent_window.join(item_attrs, on=item_id_col, how="left").with_columns([
        pl.col(brand_key).fill_null("Unknown"),
        pl.col(age_key).fill_null("Unknown"),
        pl.col(category_key).fill_null("Unknown"),
    ])
    
    brand_recent_counts = enriched_recent_hist.group_by([customer_id_col, brand_key]).agg(pl.count().alias("brand_recent_count"))
    age_recent_counts = enriched_recent_hist.group_by([customer_id_col, age_key]).agg(pl.count().alias("age_recent_count"))
    category_recent_counts = enriched_recent_hist.group_by([customer_id_col, category_key]).agg(pl.count().alias("category_recent_count"))
    
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
    
    item_stats = enriched_hist.group_by(item_id_col).agg([
        pl.count().alias("item_global_purchases"),
        pl.col(customer_id_col).n_unique().alias("item_unique_customers"),
        pl.col(price_col).mean().alias("item_avg_price"),
        pl.col(quantity_col).sum().alias("item_total_volume")
    ])
    
    brand_popularity = enriched_hist.group_by(brand_key).agg(pl.count().alias("brand_global_popularity"))
    category_popularity = enriched_hist.group_by(category_key).agg(pl.count().alias("category_global_popularity"))
    
    user_brand_diversity = enriched_hist.group_by([customer_id_col, brand_key]).agg(
        pl.col(item_id_col).n_unique().alias("user_brand_item_diversity")
    )
    user_category_diversity = enriched_hist.group_by([customer_id_col, category_key]).agg(
        pl.col(item_id_col).n_unique().alias("user_category_item_diversity")
    )
    
    # === CANDIDATE GENERATION ===
    # FIX: Only use HISTORICAL transactions for positives. 
    # Do NOT include 'recent_transactions' here. That is "Training on the Answer Key".
    positives = hist_transactions.select([customer_id_col, item_id_col]).unique()
    
    active_users = hist_transactions.select(customer_id_col).unique()
    
    top_popular_items = (
        hist_transactions
        .group_by(item_id_col)
        .count()
        .top_k(n_popular_candidates, by="count")
        .select(item_id_col)
    )
    
    candidates_popular = active_users.join(top_popular_items, how="cross")
    candidate_pairs = pl.concat([positives, candidates_popular]).unique()
    
    resolved_user_key = _resolve_user_key(users, customer_id_col, user_id_col, users_schema)
    if resolved_user_key:
        known_users = users.select(pl.col(resolved_user_key).alias(customer_id_col)).unique()
        candidate_pairs = candidate_pairs.join(known_users, on=customer_id_col, how="inner")
    
    candidates = candidate_pairs.join(item_attrs, on=item_id_col, how="left").with_columns([
        pl.col(brand_key).fill_null("Unknown"),
        pl.col(age_key).fill_null("Unknown"),
        pl.col(category_key).fill_null("Unknown"),
    ])
    
    recent_labels = recent_transactions.select([customer_id_col, item_id_col]).unique().with_columns(pl.lit(1).alias("Y"))
    
    feature_table = (
        candidates
        .join(brand_counts, on=[customer_id_col, brand_key], how="left")
        .join(age_counts, on=[customer_id_col, age_key], how="left")
        .join(category_counts, on=[customer_id_col, category_key], how="left")
        .join(brand_recent_counts, on=[customer_id_col, brand_key], how="left")
        .join(age_recent_counts, on=[customer_id_col, age_key], how="left")
        .join(category_recent_counts, on=[customer_id_col, category_key], how="left")
        .join(user_stats, on=customer_id_col, how="left")
        .join(user_item_freq, on=[customer_id_col, item_id_col], how="left")
        .join(item_stats, on=item_id_col, how="left")
        .join(brand_popularity, on=brand_key, how="left")
        .join(category_popularity, on=category_key, how="left")
        .join(user_brand_diversity, on=[customer_id_col, brand_key], how="left")
        .join(user_category_diversity, on=[customer_id_col, category_key], how="left")
        .join(recent_labels, on=[customer_id_col, item_id_col], how="left")
        .with_columns([
            pl.col("brand_count").fill_null(0).cast(pl.Int64).alias("X_1"),
            pl.col("age_group_count").fill_null(0).cast(pl.Int64).alias("X_2"),
            pl.col("category_count").fill_null(0).cast(pl.Int64).alias("X_3"),
            pl.col("brand_recent_count").fill_null(0).cast(pl.Int64).alias("X_4"),
            pl.col("age_recent_count").fill_null(0).cast(pl.Int64).alias("X_5"),
            pl.col("category_recent_count").fill_null(0).cast(pl.Int64).alias("X_6"),
            pl.col("user_total_purchases").fill_null(0).cast(pl.Int64).alias("X_7"),
            pl.col("user_unique_items").fill_null(0).cast(pl.Int64).alias("X_8"),
            pl.col("user_unique_brands").fill_null(0).cast(pl.Int64).alias("X_9"),
            pl.col("user_unique_categories").fill_null(0).cast(pl.Int64).alias("X_10"),
            pl.col("user_item_purchase_count").fill_null(0).cast(pl.Int64).alias("X_11"),
            pl.col("item_global_purchases").fill_null(0).cast(pl.Int64).alias("X_12"),
            pl.col("item_unique_customers").fill_null(0).cast(pl.Int64).alias("X_13"),
            pl.col("brand_global_popularity").fill_null(0).cast(pl.Int64).alias("X_14"),
            pl.col("category_global_popularity").fill_null(0).cast(pl.Int64).alias("X_15"),
            pl.col("user_brand_item_diversity").fill_null(0).cast(pl.Int64).alias("X_16"),
            pl.col("user_category_item_diversity").fill_null(0).cast(pl.Int64).alias("X_17"),
            (pl.col("brand_recent_count").fill_null(0) / (pl.col("brand_count").fill_null(0) + 1)).alias("X_18"),
            (pl.col("user_item_purchase_count").fill_null(0) / (pl.col("user_total_purchases").fill_null(0) + 1)).alias("X_19"),
            (pl.col("item_unique_customers").fill_null(0) / (pl.col("item_global_purchases").fill_null(0) + 1)).alias("X_20"),
            (pl.col("user_unique_items").fill_null(0) / (pl.col("user_total_purchases").fill_null(0) + 1)).alias("X_21"),
            # FIX: Fill with Global Avg Price
            pl.col("item_avg_price").fill_null(global_avg_price).log1p().alias("X_22"),
            pl.col("item_total_volume").fill_null(0).log1p().alias("X_23"),
            pl.col("user_avg_spend").fill_null(0).log1p().alias("X_24"),
            pl.col("user_total_quantity").fill_null(0).log1p().alias("X_25"),
            pl.col("user_lifetime_value").fill_null(0).log1p().alias("X_26"),
            pl.col("user_item_total_spend").fill_null(0).log1p().alias("X_27"),
            pl.col("user_item_total_qty").fill_null(0).alias("X_28"),
            # FIX: Fill with Global Avg Price
            ((pl.col("item_avg_price").fill_null(global_avg_price) + 1) / (pl.col("user_avg_spend").fill_null(0) + 1)).alias("X_29"),
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