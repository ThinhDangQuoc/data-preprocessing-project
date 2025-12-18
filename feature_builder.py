from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence
import polars as pl
import numpy as np

# ============================================================================
# HELPER: Fail Fast Validation
# ============================================================================
def _ensure_columns_exist(lf: pl.LazyFrame, columns: Sequence[str], source_name: str) -> None:
    try:
        schema = lf.schema
        missing = [c for c in columns if c not in schema]
        if missing:
            print(f"Warning: Missing columns in '{source_name}': {missing}. Found: {list(schema.keys())}")
    except Exception as e:
        print(f"Warning: Could not validate columns for {source_name}: {e}")

# ============================================================================
# STAGE 1: SCALABLE CANDIDATE GENERATION
# ============================================================================

def generate_candidates(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    begin_hist: datetime,
    end_hist: datetime,
    gt_dict: Dict[Any, Dict[str, List[Any]]] | None = None,
    cf_model: Any = None,
    *,
    transaction_time_col: str = "created_date",
    customer_id_col: str = "customer_id",
    item_id_col: str = "item_id",
    price_col: str = "price",
    # Configuration
    n_trending: int = 100,
    n_popular: int = 100,
    n_user_history: int = 50,
    n_similar_items: int = 20, 
    n_new_items: int = 50,
    n_price_band: int = 50,
    n_cf_candidates: int = 50,
) -> pl.LazyFrame:
    
    print(f"\n{'='*80}\nSTAGE 1: GENERATING CANDIDATES (FIXED)\n{'='*80}")

    # 1. Validation
    _ensure_columns_exist(transactions, [customer_id_col, item_id_col, transaction_time_col, price_col], "transactions")
    _ensure_columns_exist(items, [item_id_col], "items")
    _ensure_columns_exist(users, [customer_id_col], "users")

    # 2. Time Anchors
    end_hist_lit = pl.lit(end_hist, dtype=pl.Datetime)
    recent_window_start = end_hist - timedelta(days=7)
    recent_window_lit = pl.lit(recent_window_start, dtype=pl.Datetime)

    # 3. Filter History & DETECT SCHEMA
    hist_lf = transactions.filter(
        (pl.col(transaction_time_col) >= begin_hist) & 
        (pl.col(transaction_time_col) <= end_hist)
    )

    # --- CRITICAL FIX: Detect canonical types ---
    dummy_schema = hist_lf.schema
    cid_dtype = dummy_schema[customer_id_col]
    iid_dtype = dummy_schema[item_id_col]
    print(f"  > Detected Schema - User: {cid_dtype}, Item: {iid_dtype}")

    # Helper to enforce types on all candidate branches
    def enforce_types(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Ensure all candidate sources have consistent schema"""
        return lf.select([
            pl.col(customer_id_col).cast(cid_dtype),
            pl.col(item_id_col).cast(iid_dtype),
            pl.col("source").cast(pl.Categorical)
        ])

    # 4. Target Users
    if gt_dict:
        target_ids = list(gt_dict.keys())
        active_users = pl.DataFrame({customer_id_col: target_ids}).lazy()
        # Cast to match transaction schema
        active_users = active_users.with_columns(pl.col(customer_id_col).cast(cid_dtype))
    else:
        active_users = users.select(customer_id_col).unique()
        active_users = active_users.with_columns(pl.col(customer_id_col).cast(cid_dtype))

    # ==========================================
    # STRATEGY 1: POPULAR & TRENDING (Global)
    # ==========================================
    item_stats = (
        hist_lf
        .group_by(item_id_col)
        .agg([
            pl.len().alias("cnt"),
            pl.col(customer_id_col).n_unique().alias("uniq_users"),
            pl.col(transaction_time_col).filter(pl.col(transaction_time_col) >= recent_window_lit).len().alias("recent_cnt")
        ])
    )

    popular_items = (
        item_stats
        .sort("cnt", descending=True)
        .head(n_popular)
        .select(item_id_col)
        .with_columns(pl.lit("popular").cast(pl.Categorical).alias("source"))
    )

    trending_items = (
        item_stats
        .sort("recent_cnt", descending=True)
        .head(n_trending)
        .select(item_id_col)
        .with_columns(pl.lit("trending").cast(pl.Categorical).alias("source"))
    )

    cands_global = (
        active_users
        .join(pl.concat([popular_items, trending_items]), how="cross")
    )
    cands_global = enforce_types(cands_global)

    # ==========================================
    # STRATEGY 2: USER HISTORY
    # ==========================================
    cands_history = (
        hist_lf
        .join(active_users, on=customer_id_col)
        .group_by([customer_id_col, item_id_col])
        .agg([
            pl.len().alias("cnt"),
            pl.col(transaction_time_col).max().alias("last_ts")
        ])
        .with_columns(
            (pl.col("cnt") * (1.0 / (1.0 + (end_hist_lit - pl.col("last_ts")).dt.total_days()))).alias("score")
        )
        .filter(pl.col("score").rank(method="dense", descending=True).over(customer_id_col) <= n_user_history)
        .select([customer_id_col, item_id_col])
        .with_columns(pl.lit("history").cast(pl.Categorical).alias("source"))
    )
    cands_history = enforce_types(cands_history)

    # ==========================================
    # STRATEGY 3: CO-OCCURRENCE
    # ==========================================
    recent_transactions = hist_lf.filter(pl.col(transaction_time_col) >= (end_hist - timedelta(days=30)))
    
    user_baskets = (
        recent_transactions
        .sort(transaction_time_col, descending=True)
        .group_by(customer_id_col)
        .agg(pl.col(item_id_col).unique().head(20).alias("basket")) 
    )

    cands_cooc = (
        user_baskets
        .join(active_users, on=customer_id_col)
        .explode("basket")
        .rename({"basket": "item_id_A"})
        .join(
            recent_transactions.select([customer_id_col, item_id_col]),
            on=customer_id_col  # FIX: Use on= instead of left_on/right_on when keys match
        )
        .filter(pl.col("item_id_A") != pl.col(item_id_col))
        .group_by([customer_id_col, item_id_col]) 
        .agg(pl.len().alias("count"))
        .filter(pl.col("count").rank("dense", descending=True).over(customer_id_col) <= n_similar_items)
        .select([customer_id_col, item_id_col])
        .with_columns(pl.lit("similar_items").cast(pl.Categorical).alias("source"))
    )
    cands_cooc = enforce_types(cands_cooc)

    # ==========================================
    # STRATEGY 4: NEW ITEMS
    # ==========================================
    older_items = (
        hist_lf
        .filter(pl.col(transaction_time_col) < recent_window_lit)
        .select(item_id_col)
        .unique()
    )

    truly_new_items = (
        hist_lf
        .filter(pl.col(transaction_time_col) >= recent_window_lit)
        .select(item_id_col)
        .unique()
        .join(older_items, on=item_id_col, how="anti")
        .head(n_new_items)
    )

    cands_new = (
        active_users
        .join(truly_new_items, how="cross")
        .with_columns(pl.lit("new_arrival").cast(pl.Categorical).alias("source"))
    )
    cands_new = enforce_types(cands_new)

    # ==========================================
    # STRATEGY 5: PRICE BANDS
    # ==========================================
    user_spend = (
        hist_lf
        .group_by(customer_id_col)
        .agg(pl.col(price_col).mean().alias("mean_price"))
    )
    
    item_cost = (
        hist_lf
        .group_by(item_id_col)
        .agg(pl.col(price_col).mean().alias("mean_price"))
    )

    def add_price_bucket(lf):
        return lf.with_columns(
            (pl.col("mean_price").log1p() * 2).cast(pl.Int32).alias("price_bucket")
        )

    user_spend_binned = add_price_bucket(user_spend)
    item_cost_binned = add_price_bucket(item_cost)

    top_items_per_bucket = (
        item_cost_binned
        .filter(pl.col("price_bucket").is_not_null())
        .sort("mean_price")  # Sort for deterministic results
        .group_by("price_bucket", maintain_order=True)
        .head(n_price_band)
    )

    cands_price = (
        active_users
        .join(user_spend_binned, on=customer_id_col, how="inner")  # Only users with history
        .join(top_items_per_bucket, on="price_bucket", how="inner")
        .select([customer_id_col, item_id_col])
        .with_columns(pl.lit("price_match").cast(pl.Categorical).alias("source"))
    )
    cands_price = enforce_types(cands_price)

    # ==========================================
    # STRATEGY 6: CF (If available)
    # ==========================================
    # Initialize empty with CORRECT types
    cands_cf = pl.DataFrame(schema={
        customer_id_col: cid_dtype, 
        item_id_col: iid_dtype, 
        "source": pl.Categorical
    }).lazy()

    if cf_model is not None:
        try:
            target_ids_list = active_users.collect()[customer_id_col].to_list()
            if target_ids_list:
                print(f"  > Fetching CF candidates for {len(target_ids_list)} users...")
                cf_res = cf_model.recommend_batch(
                    target_ids_list, 
                    n_candidates=n_cf_candidates
                )
                
                # Check if column exists and rename
                if "candidate_source" in cf_res.collect_schema().names():
                    cf_res = cf_res.rename({"candidate_source": "source"})
                
                # Force types to match schema
                cands_cf = cf_res.select([
                    pl.col(customer_id_col).cast(cid_dtype),
                    pl.col(item_id_col).cast(iid_dtype),
                    pl.col("source").cast(pl.Categorical)
                ])
                print(f"  > CF generated {cands_cf.select(pl.len()).collect()[0,0]} candidate pairs")
        except Exception as e:
            print(f"  Warning: CF generation failed: {e}")
            import traceback
            traceback.print_exc()

    # ==========================================
    # MERGE & AGGREGATE
    # ==========================================
    
    print("  > Merging candidates...")
    all_cands = pl.concat([
        cands_global,
        cands_history,
        cands_cooc, 
        cands_new,
        cands_price,
        cands_cf
    ], how="diagonal")

    final_cands = (
        all_cands
        .group_by([customer_id_col, item_id_col])
        .agg(pl.col("source").unique().alias("candidate_sources"))
    )

    # ==========================================
    # FALLBACK (Cold Start)
    # ==========================================
    existing_users = final_cands.select(customer_id_col).unique()
    missing_users = active_users.join(existing_users, on=customer_id_col, how="anti")
    
    # Mix popular and trending for better diversity
    fallback_items = pl.concat([
        popular_items.head(15),
        trending_items.head(15)
    ]).unique(subset=[item_id_col])
    
    fallback_cands = (
        missing_users
        .join(fallback_items, how="cross")
        .with_columns(pl.lit(["cold_start_fallback"]).cast(pl.List(pl.Categorical)).alias("candidate_sources"))
        .select([
            pl.col(customer_id_col).cast(cid_dtype),
            pl.col(item_id_col).cast(iid_dtype),
            pl.col("candidate_sources")
        ])
    )

    final_result = pl.concat([final_cands, fallback_cands], how="diagonal")
    
    # Print stats
    stats = final_result.select([
        pl.col(customer_id_col).n_unique().alias("n_users"),
        pl.len().alias("n_pairs")
    ]).collect()
    print(f"  ✓ Generated {stats['n_pairs'][0]:,} candidates for {stats['n_users'][0]:,} users")
    
    return final_result


# ============================================================================
# STAGE 2: FEATURE BUILDING (FIXED)
# ============================================================================

def build_ranking_features(
    candidates: pl.LazyFrame,
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    end_hist: datetime,
    begin_recent: datetime | None = None,
    end_recent: datetime | None = None,
    gt_dict: Dict | None = None,
    *,
    customer_id_col: str = "customer_id",
    item_id_col: str = "item_id",
    price_col: str = "price",
    transaction_time_col: str = "created_date"
) -> pl.LazyFrame:
    
    print(f"\n{'='*80}\nSTAGE 2: FEATURE ENGINEERING (FIXED)\n{'='*80}")
    
    _ensure_columns_exist(candidates, [customer_id_col, item_id_col, "candidate_sources"], "candidates")

    end_hist_lit = pl.lit(end_hist, dtype=pl.Datetime)

    # 1. User Features (Global)
    user_feats = (
        transactions
        .filter(pl.col(transaction_time_col) <= end_hist_lit)
        .group_by(customer_id_col)
        .agg([
            pl.len().alias("user_cnt"),
            pl.col(item_id_col).n_unique().alias("user_uniq_items"),
            pl.col(price_col).mean().alias("user_avg_spend"),
            pl.col(price_col).std().alias("user_std_spend"),
            (end_hist_lit - pl.col(transaction_time_col).max()).dt.total_days().alias("user_days_since_active"),
            pl.col(transaction_time_col).min().alias("user_first_purchase")
        ])
        .with_columns([
            # User tenure in days
            (end_hist_lit - pl.col("user_first_purchase")).dt.total_days().alias("user_tenure_days")
        ])
    )

    # 2. Item Features (Global)
    item_feats = (
        transactions
        .filter(pl.col(transaction_time_col) <= end_hist_lit)
        .group_by(item_id_col)
        .agg([
            pl.len().alias("item_cnt"),
            pl.col(price_col).mean().alias("item_avg_price"),
            pl.col(price_col).std().alias("item_std_price"),
            pl.col(customer_id_col).n_unique().alias("item_uniq_users"),
            (end_hist_lit - pl.col(transaction_time_col).max()).dt.total_days().alias("item_days_since_sold")
        ])
    )

    # 3. User-Item Interaction Features
    user_item_feats = (
        transactions
        .filter(pl.col(transaction_time_col) <= end_hist_lit)
        .group_by([customer_id_col, item_id_col])
        .agg([
            pl.len().alias("ui_purchase_count"),
            pl.col(price_col).sum().alias("ui_total_spent"),
            (end_hist_lit - pl.col(transaction_time_col).max()).dt.total_days().alias("ui_days_since_last"),
            (pl.col(transaction_time_col).max() - pl.col(transaction_time_col).min()).dt.total_days().alias("ui_purchase_span")
        ])
    )

    # 4. Join all features to candidates
    df_features = (
        candidates
        .join(user_feats, on=customer_id_col, how="left")
        .join(item_feats, on=item_id_col, how="left")
        .join(user_item_feats, on=[customer_id_col, item_id_col], how="left")
    )

    # 5. Derived Features
    df_features = df_features.with_columns([
        # Price ratio (user's avg spend vs item's avg price)
        (pl.col("user_avg_spend") / (pl.col("item_avg_price") + 1e-6)).alias("price_ratio"),
        
        # Item popularity score
        (pl.col("item_cnt") / (pl.col("item_cnt").max().over([]) + 1e-6)).alias("item_popularity_norm"),
        
        # User activity level
        (pl.col("user_cnt") / (pl.col("user_tenure_days") + 1)).alias("user_activity_rate"),
        
        # Has purchased before (binary)
        pl.col("ui_purchase_count").is_not_null().cast(pl.UInt8).alias("has_purchased_before")
    ])

    # 6. Source Feature Encoding
    source_features = []
    for src in ["popular", "trending", "history", "similar_items", "new_arrival", "price_match", "cf_match"]:
        df_features = df_features.with_columns(
            pl.col("candidate_sources").list.contains(src).cast(pl.UInt8).alias(f"src_{src}")
        )
        source_features.append(f"src_{src}")

    # 7. Fill Nulls with appropriate defaults
    df_features = df_features.with_columns([
        # User features
        pl.col("user_cnt").fill_null(0),
        pl.col("user_uniq_items").fill_null(0),
        pl.col("user_avg_spend").fill_null(0),
        pl.col("user_std_spend").fill_null(0),
        pl.col("user_days_since_active").fill_null(999),
        pl.col("user_tenure_days").fill_null(1),
        
        # Item features
        pl.col("item_cnt").fill_null(0),
        pl.col("item_avg_price").fill_null(0),
        pl.col("item_std_price").fill_null(0),
        pl.col("item_uniq_users").fill_null(0),
        pl.col("item_days_since_sold").fill_null(999),
        
        # User-Item features
        pl.col("ui_purchase_count").fill_null(0),
        pl.col("ui_total_spent").fill_null(0),
        pl.col("ui_days_since_last").fill_null(999),
        pl.col("ui_purchase_span").fill_null(0),
        
        # Derived features
        pl.col("price_ratio").fill_null(1.0),
        pl.col("item_popularity_norm").fill_null(0),
        pl.col("user_activity_rate").fill_null(0),
    ])
    
    # 8. CRITICAL FIX: Complete Feature Mapping
    # Map ALL features to X_N format for XGBoost
    feature_mapping = [
        # IDs (for joining, not training)
        pl.col(customer_id_col).alias("X-1"),
        pl.col(item_id_col).alias("X_0"),
        
        # User features (X_1 to X_6)
        pl.col("user_cnt").cast(pl.Float32).alias("X_1"),
        pl.col("user_uniq_items").cast(pl.Float32).alias("X_2"),
        pl.col("user_avg_spend").cast(pl.Float32).alias("X_3"),
        pl.col("user_std_spend").cast(pl.Float32).alias("X_4"),
        pl.col("user_days_since_active").cast(pl.Float32).alias("X_5"),
        pl.col("user_tenure_days").cast(pl.Float32).alias("X_6"),
        
        # Item features (X_7 to X_11)
        pl.col("item_cnt").cast(pl.Float32).alias("X_7"),
        pl.col("item_avg_price").cast(pl.Float32).alias("X_8"),
        pl.col("item_std_price").cast(pl.Float32).alias("X_9"),
        pl.col("item_uniq_users").cast(pl.Float32).alias("X_10"),
        pl.col("item_days_since_sold").cast(pl.Float32).alias("X_11"),
        
        # User-Item interaction features (X_12 to X_15)
        pl.col("ui_purchase_count").cast(pl.Float32).alias("X_12"),
        pl.col("ui_total_spent").cast(pl.Float32).alias("X_13"),
        pl.col("ui_days_since_last").cast(pl.Float32).alias("X_14"),
        pl.col("ui_purchase_span").cast(pl.Float32).alias("X_15"),
        
        # Derived features (X_16 to X_19)
        pl.col("price_ratio").cast(pl.Float32).alias("X_16"),
        pl.col("item_popularity_norm").cast(pl.Float32).alias("X_17"),
        pl.col("user_activity_rate").cast(pl.Float32).alias("X_18"),
        pl.col("has_purchased_before").cast(pl.Float32).alias("X_19"),
        
        # Source features (X_100 to X_106)
        pl.col("src_popular").cast(pl.Float32).alias("X_100"),
        pl.col("src_trending").cast(pl.Float32).alias("X_101"),
        pl.col("src_history").cast(pl.Float32).alias("X_102"),
        pl.col("src_similar_items").cast(pl.Float32).alias("X_103"),
        pl.col("src_new_arrival").cast(pl.Float32).alias("X_104"),
        pl.col("src_price_match").cast(pl.Float32).alias("X_105"),
        pl.col("src_cf_match").cast(pl.Float32).alias("X_106"),
    ]
    
    result = df_features.with_columns(feature_mapping)
    
    # Print feature summary
    print(f"  ✓ Built {len([m for m in feature_mapping if m.meta.output_name().startswith('X_')])} features")
    
    return result