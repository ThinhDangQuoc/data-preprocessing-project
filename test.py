from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import os
import gc
import pickle
import json
import warnings
import numpy as np
import polars as pl
from xgboost import XGBRanker
from gensim.models import Word2Vec

warnings.filterwarnings('ignore')

# GLOBAL PERFORMANCE CONFIG
pl.Config.set_streaming_chunk_size(1000000)
# Use 'spawn' or 'fork' depending on OS, but standard is usually fine.
# Setting env var for Polars parallelization
os.environ["POLARS_MAX_THREADS"] = str(os.cpu_count())

def train_item2vec_model(
    transactions: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    vector_size: int = 64,
    window: int = 5,
    min_count: int = 3,
    epochs: int = 10,
    verbose: bool = True
) -> Word2Vec:
    if verbose: print("  [Item2Vec] Preparing sequences...")
    
    sequences = (
        transactions
        .filter(pl.col("created_date").is_between(hist_start, hist_end))
        .select([
            pl.col("customer_id"),
            # Ensure Item ID is string for Gensim
            pl.col("item_id").cast(pl.Utf8).alias("item_token"),
            pl.col("created_date")
        ])
        .sort(["customer_id", "created_date"])
        .group_by("customer_id")
        .agg(pl.col("item_token"))
        .select("item_token")
        .collect(streaming=True)
        .to_series()
        .to_list()
    )
    
    if verbose: print(f"  [Item2Vec] Training Word2Vec on {len(sequences):,} sequences...")
    
    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=os.cpu_count(),
        epochs=epochs,
        sg=1,
        negative=10,
        ns_exponent=0.75
    )
    return model


def generate_i2v_candidates(
    model: Word2Vec,
    transactions: pl.LazyFrame,
    target_users: pl.LazyFrame,
    hist_end: datetime,
    n_similar: int = 30,
    top_n_items: int = 5,
    verbose: bool = True
) -> pl.LazyFrame:
    """
    Generates candidates based on items users recently interacted with.
    Dynamic type matching fix included.
    """
    if verbose:
        print("  [Item2Vec] Generating candidates (Optimized with Dynamic Types)...")
    
    # 0. Detect Item ID Type from Input Data
    # This ensures we match String vs Int correctly during joins
    try:
        # Polars >= 0.20
        schema = transactions.collect_schema()
    except AttributeError:
        # Older Polars
        schema = transactions.schema
    
    item_id_type = schema["item_id"]

    # 1. Get Recent Items for TARGET Users Only
    user_recent_items_lf = (
        transactions
        .filter(pl.col("created_date") <= hist_end)
        .join(target_users, on="customer_id", how="inner")
        .sort(["customer_id", "created_date"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").head(top_n_items))
        .explode("item_id")
        .select([
            pl.col("customer_id"),
            pl.col("item_id").alias("source_item_id")
        ])
    )
    
    user_recent_items_df = user_recent_items_lf.collect()
    
    if user_recent_items_df.is_empty():
        return pl.LazyFrame(schema={
            "customer_id": schema["customer_id"], 
            "item_id": item_id_type, 
            "feat_i2v_score": pl.Float32
        })

    # 2. Build Similarity Table
    unique_source_ids = user_recent_items_df["source_item_id"].unique().to_list()
    vocab = model.wv.key_to_index
    
    src_list = []
    dst_list = []
    score_list = []
    
    for src_id in unique_source_ids:
        token = str(src_id) # Gensim always uses string keys
        if token in vocab:
            neighbors = model.wv.most_similar(token, topn=n_similar)
            for sim_token, score in neighbors:
                src_list.append(src_id) # Keep original type if possible, or string
                dst_list.append(sim_token) # Gensim returns strings
                score_list.append(score)
    
    if not src_list:
        if verbose: print("  [Item2Vec] Warning: No similar items found in model.")
        return pl.LazyFrame(schema={
            "customer_id": schema["customer_id"], 
            "item_id": item_id_type, 
            "feat_i2v_score": pl.Float32
        })

    # 3. Create DataFrame with Correct Types
    sim_df = pl.DataFrame({
        "source_item_id": src_list,
        "item_id": dst_list,
        "similarity": score_list
    }).lazy().with_columns([
        # CRITICAL FIX: Cast to the exact type found in transactions (likely Utf8)
        pl.col("source_item_id").cast(item_id_type),
        pl.col("item_id").cast(item_id_type),
        pl.col("similarity").cast(pl.Float32)
    ])

    # 4. Join
    candidates = (
        user_recent_items_df.lazy()
        .join(sim_df, on="source_item_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg(
            pl.col("similarity").max().alias("feat_i2v_score")
        )
    )
    
    return candidates


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system"
TRANSACTIONS_GLOB = f"{BASE_DIR}/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = f"{BASE_DIR}/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = f"{BASE_DIR}/dataset/sales_pers.user_chunk*.parquet"
GT_PKL_PATH = f"{BASE_DIR}/groundtruth.pkl"
OUTPUT_DIR = f"{BASE_DIR}/outputs_improved"

# ✅ FIX: Add "feat_repeat_score" to this list
ALL_SCORES = [
    "feat_pop_score",
    "feat_cat_rank_score",
    "feat_cf_score",
    "feat_trend_score",
    "feat_price_match_score",
    "feat_i2v_score",
    "feat_repeat_score",  # <--- ADD THIS LINE
]

def standardize(df: pl.LazyFrame, active_score_col: str) -> pl.LazyFrame:
    """Standardize score columns - ensure all scores exist with proper types"""
    expr_list = [
        pl.col("customer_id"),
        pl.col("item_id"),
        pl.col(active_score_col).cast(pl.Float32).alias(active_score_col)
    ]
    
    for sc in ALL_SCORES:
        if sc != active_score_col:
            expr_list.append(pl.lit(0.0, dtype=pl.Float32).alias(sc))
    
    return df.select(expr_list)


@dataclass(frozen=True)
class DataSplit:
    """Time-based train/val/test split configuration"""
    name: str
    hist_start: datetime
    hist_end: datetime
    target_start: datetime | None
    target_end: datetime | None
    
    def __repr__(self):
        if self.target_start is None:
            return f"{self.name}: hist=[{self.hist_start.date()}→{self.hist_end.date()}], target=GT_FILE"
        return f"{self.name}: hist=[{self.hist_start.date()}→{self.hist_end.date()}], target=[{self.target_start.date()}→{self.target_end.date()}]"


SPLITS = {
    'train': DataSplit(
        name='train',
        hist_start=datetime(2024, 10, 1),
        hist_end=datetime(2024, 11, 30),
        target_start=datetime(2024, 12, 1),
        target_end=datetime(2024, 12, 10)
    ),
    'val': DataSplit(
        name='val',
        hist_start=datetime(2024, 10, 1),
        hist_end=datetime(2024, 12, 10),
        target_start=datetime(2024, 12, 11),
        target_end=datetime(2024, 12, 20)
    ),
    'test': DataSplit(
        name='test',
        hist_start=datetime(2024, 12, 1),
        hist_end=datetime(2024, 12, 30),
        target_start=None,
        target_end=None
    )
}


# ============================================================================
# STAGE 1: ENHANCED CANDIDATE GENERATION
# ============================================================================

def generate_candidates(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    n_popular: int = 150,  # ✅ Increased from 100
    n_category_popular: int = 50,  # ✅ Increased from 30
    n_recent_trending: int = 100,  # ✅ Increased from 50
    n_collab_per_item: int = 30,  # ✅ Increased from 15
    verbose: bool = True
) -> pl.LazyFrame:
    """Enhanced candidate generation with MORE diverse sources"""
    if verbose:
        print(f"  [Stage 1] Generating candidates (EXPANDED)...")
    
    # Setup & Type Safety
    schema = transactions.collect_schema()
    cid_type = schema["customer_id"]
    iid_type = schema["item_id"]
    
    target_users = users.select(pl.col("customer_id").cast(cid_type)).unique()
    items_cast = items.with_columns(pl.col("item_id").cast(iid_type))
    
    # Filter History
    hist = transactions.filter(
        (pl.col("created_date") >= hist_start) & 
        (pl.col("created_date") <= hist_end)
    ).with_columns([
        pl.col("item_id").cast(iid_type),
        pl.col("customer_id").cast(cid_type)
    ])
    
    hist_with_meta = hist.join(items_cast, on="item_id", how="left")
    
    # Strategy 1: Global Popularity (normalized better)
    popular_items = (
        hist
        .group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(n_popular)
        .with_columns([
            (pl.col("len").log() / pl.col("len").log().max()).cast(pl.Float32).alias("feat_pop_score")
        ])
        .select(["item_id", "feat_pop_score"])
    )
    cands_global = target_users.join(popular_items, how="cross")
    
    # Strategy 2: Category Popularity (more categories)
    user_cats = (
        hist_with_meta
        .filter(pl.col("category_id").is_not_null())
        .group_by(["customer_id", "category_id"])
        .len()
        .sort("len", descending=True)
        .group_by("customer_id", maintain_order=True)
        .head(5)  # ✅ Increased from 3
    )
    
    cat_pop = (
        hist_with_meta
        .filter(pl.col("category_id").is_not_null())
        .group_by(["category_id", "item_id"])
        .len()
        .with_columns(
            pl.col("len").rank("dense", descending=True).over("category_id").alias("rank")
        )
        .filter(pl.col("rank") <= n_category_popular)
    )
    
    cands_category = (
        user_cats
        .join(cat_pop, on="category_id")
        .select([
            "customer_id",
            "item_id",
            (1.0 / pl.col("rank")).cast(pl.Float32).alias("feat_cat_rank_score")
        ])
    )
    
    # Strategy 3: Collaborative Filtering (Enhanced)
    user_items = hist.select(["customer_id", "item_id"]).unique()
    
    item_pairs = (
        user_items
        .join(user_items, on="customer_id", suffix="_right")
        .filter(pl.col("item_id") != pl.col("item_id_right"))
        .group_by(["item_id", "item_id_right"])
        .len()
        .filter(pl.col("len") > 1)
        .with_columns([
            pl.col("len").rank("dense", descending=True).over("item_id").alias("rank"),
            pl.col("len").log().cast(pl.Float32).alias("cooccur_score")
        ])
        .filter(pl.col("rank") <= n_collab_per_item)
        .select([
            pl.col("item_id").alias("source_item"),
            pl.col("item_id_right").alias("target_item"),
            pl.col("cooccur_score").alias("feat_cf_score")
        ])
    )
    
    cands_collab = (
        user_items
        .join(item_pairs, left_on="item_id", right_on="source_item")
        .group_by(["customer_id", "target_item"])
        .agg(pl.col("feat_cf_score").max())
        .select([
            "customer_id",
            pl.col("target_item").alias("item_id"),
            "feat_cf_score"
        ])
    )
    
    # Strategy 4: Trending Items (multiple time windows)
    recent_7d = hist_end - timedelta(days=7)
    recent_14d = hist_end - timedelta(days=14)
    
    trending_7d = (
        hist
        .filter(pl.col("created_date") >= recent_7d)
        .group_by("item_id")
        .len()
        .with_columns(pl.lit("7d").alias("window"))
    )
    
    trending_14d = (
        hist
        .filter(pl.col("created_date") >= recent_14d)
        .group_by("item_id")
        .len()
        .with_columns(pl.lit("14d").alias("window"))
    )
    
    trending_items = (
        pl.concat([trending_7d, trending_14d])
        .group_by("item_id")
        .agg(pl.col("len").max())
        .sort("len", descending=True)
        .head(n_recent_trending)
        .with_columns(
            (pl.col("len").log() / pl.col("len").log().max()).cast(pl.Float32).alias("feat_trend_score")
        )
        .select(["item_id", "feat_trend_score"])
    )
    cands_trending = target_users.join(trending_items, how="cross")
    
    # Strategy 5: Price Match
    item_prices = hist.group_by("item_id").agg(pl.col("price").mean())
    
    price_quantiles = item_prices.select([
        pl.col("price").quantile(0.33).alias("low"),
        pl.col("price").quantile(0.66).alias("high")
    ]).collect()
    
    low_th = price_quantiles["low"][0]
    high_th = price_quantiles["high"][0]
    
    items_binned = item_prices.with_columns(
        pl.when(pl.col("price") < low_th).then(pl.lit(1))
        .when(pl.col("price") < high_th).then(pl.lit(2))
        .otherwise(pl.lit(3))
        .alias("price_bin")
    )
    
    users_binned = (
        hist
        .group_by("customer_id")
        .agg(pl.col("price").mean())
        .with_columns(
            pl.when(pl.col("price") < low_th).then(pl.lit(1))
            .when(pl.col("price") < high_th).then(pl.lit(2))
            .otherwise(pl.lit(3))
            .alias("price_bin")
        )
    )
    
    top_items_bin = (
        hist
        .join(items_binned, on="item_id")
        .group_by(["price_bin", "item_id"])
        .len()
        .sort("len", descending=True)
        .group_by("price_bin", maintain_order=True)
        .head(50)  # ✅ Increased from 30
    )
    
    cands_price = (
        users_binned
        .join(top_items_bin, on="price_bin")
        .select([
            "customer_id",
            "item_id",
            pl.lit(1.0, dtype=pl.Float32).alias("feat_price_match_score")
        ])
    )
    
    # Strategy 6: Enhanced Item2Vec
    i2v_model = train_item2vec_model(
        transactions=transactions,
        hist_start=hist_start,
        hist_end=hist_end,
        verbose=verbose
    )
    
    cands_i2v = generate_i2v_candidates(
        model=i2v_model,
        transactions=hist,
        target_users=target_users,
        hist_end=hist_end,
        n_similar=30,
        top_n_items=5,  # ✅ NEW
        verbose=verbose
    )
    
    # ✅ NEW Strategy 7: Recent Repeats (users often rebuy)
    user_recent_repeats = (
        hist
        .group_by(["customer_id", "item_id"])
        .agg(pl.len().alias("repeat_count"))
        .filter(pl.col("repeat_count") > 1)
        .with_columns(
            (pl.col("repeat_count").log() + 1.0).cast(pl.Float32).alias("feat_repeat_score")
        )
        .select(["customer_id", "item_id", "feat_repeat_score"])
    )
    
    cands_repeat = standardize(user_recent_repeats, "feat_repeat_score")
    
    # Standardize all candidates
    all_cands = [
        standardize(cands_global, "feat_pop_score"),
        standardize(cands_category, "feat_cat_rank_score"),
        standardize(cands_collab, "feat_cf_score"),
        standardize(cands_trending, "feat_trend_score"),
        standardize(cands_price, "feat_price_match_score"),
        standardize(cands_i2v, "feat_i2v_score"),
        cands_repeat  # Already standardized
    ]
    
    # Efficient concat and aggregate
    candidates = (
        pl.concat(all_cands, how="vertical")
        .group_by(["customer_id", "item_id"])
        .agg([pl.col(sc).max() for sc in ALL_SCORES])
    )
    
    if verbose:
        n_cands = candidates.select(pl.len()).collect().item()
        n_users = candidates.select(pl.col("customer_id").n_unique()).collect().item()
        avg_per_user = n_cands / n_users if n_users > 0 else 0
        print(f"  [Stage 1] Generated {n_cands:,} candidates for {n_users:,} users")
        print(f"  [Stage 1] Avg candidates/user: {avg_per_user:.1f}")
    
    return candidates


def check_stage1_recall(
    candidates_lf: pl.LazyFrame,
    transactions_lf: pl.LazyFrame,
    target_start: datetime,
    target_end: datetime,
    verbose: bool = True
) -> float:
    """Calculates Stage 1 recall - returns recall value"""
    if not verbose:
        return 0.0
    
    print(f"  [Recall Check] Calculating Stage 1 Recall...")
    
    ground_truth = (
        transactions_lf
        .filter(
            (pl.col("created_date") >= target_start) & 
            (pl.col("created_date") <= target_end)
        )
        .select(["customer_id", "item_id"])
        .unique()
    )
    
    hits = candidates_lf.join(ground_truth, on=["customer_id", "item_id"], how="inner")
    
    n_hit_users = hits.select("customer_id").unique().collect().height
    n_total_target_users = ground_truth.select("customer_id").unique().collect().height
    
    recall = n_hit_users / n_total_target_users if n_total_target_users > 0 else 0.0
    
    print(f"  [Recall Check] GT Users: {n_total_target_users:,}")
    print(f"  [Recall Check] Hit Users: {n_hit_users:,}")
    print(f"  [Recall Check] Recall: {recall:.2%}")
    
    if recall < 0.7:
        print(f"  [WARNING] Recall below 70%! Consider adding more candidate sources.")
    
    return recall


# ============================================================================
# STAGE 2: ENHANCED FEATURE ENGINEERING
# ============================================================================

def build_features(
    candidates: pl.LazyFrame,
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose: bool = True
) -> pl.LazyFrame:
    """Enhanced feature engineering with better normalization"""
    if verbose:
        print(f"  [Stage 2] Building features (Enhanced)...")
    
    cand_schema = candidates.collect_schema()
    iid_type = cand_schema["item_id"]
    cid_type = cand_schema["customer_id"]
    
    items_cast = items.with_columns(pl.col("item_id").cast(iid_type))
    
    hist_trans = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
        .with_columns([
            pl.col("item_id").cast(iid_type),
            pl.col("customer_id").cast(cid_type)
        ])
    )
    
    hist_with_meta = hist_trans.join(items_cast, on="item_id", how="left")
    
    # User Statistics (with recency weights)
    recent_date = hist_end - timedelta(days=14)
    
    user_stats = (
        hist_trans
        .group_by("customer_id")
        .agg([
            pl.len().alias("user_txn_count").cast(pl.Int64),
            pl.col("item_id").n_unique().alias("user_unique_items").cast(pl.Int64),
            pl.col("price").mean().alias("user_avg_price").cast(pl.Float32),
            pl.col("price").std().alias("user_price_std").cast(pl.Float32),
            pl.col("created_date").max().alias("user_last_purchase_date"),
        ])
        .with_columns([
            pl.col("user_price_std").fill_null(0.0),
            (pl.lit(hist_end) - pl.col("user_last_purchase_date")).dt.total_days()
                .cast(pl.Float32).alias("days_since_last_purchase")
        ])
    )
    
    # Recent activity indicator
    user_recent_activity = (
        hist_trans
        .filter(pl.col("created_date") >= recent_date)
        .group_by("customer_id")
        .agg(pl.len().alias("recent_txn_count").cast(pl.Int64))
    )
    
    # Item Statistics
    item_stats = (
        hist_trans
        .group_by("item_id")
        .agg([
            pl.len().alias("item_txn_count").cast(pl.Int64),
            pl.col("customer_id").n_unique().alias("item_unique_users").cast(pl.Int64),
            pl.col("price").mean().alias("item_avg_price").cast(pl.Float32),
            pl.col("created_date").max().alias("item_last_sale_date"),
        ])
        .with_columns(
            (pl.lit(hist_end) - pl.col("item_last_sale_date")).dt.total_days()
                .cast(pl.Float32).alias("days_since_last_sale")
        )
    )
    
    # Category Affinity
    user_category_counts = (
        hist_with_meta
        .filter(pl.col("category_id").is_not_null())
        .group_by(["customer_id", "category_id"])
        .agg(pl.len().alias("len").cast(pl.Int64))
        .with_columns(
            pl.col("len").rank("dense", descending=True).over("customer_id").cast(pl.Int64).alias("category_rank")
        )
    )
    
    item_categories = items_cast.select(["item_id", "category_id"]).unique()
    
    category_affinity = (
        candidates
        .join(item_categories, on="item_id", how="left")
        .join(user_category_counts, on=["customer_id", "category_id"], how="left")
        .select([
            "customer_id",
            "item_id",
            pl.col("len").fill_null(0).cast(pl.Int64).alias("user_category_purchases"),
            pl.col("category_rank").fill_null(999).cast(pl.Int64).alias("category_rank_for_user")
        ])
    )
    
    # Niche Score (popularity sweet spot)
    item_niche_score = (
        item_stats
        .with_columns([
            (1.0 / (1.0 + (pl.col("item_txn_count").cast(pl.Float32).log() - np.log(50)).abs()))
            .cast(pl.Float32)
            .alias("niche_score")
        ])
        .select(["item_id", "niche_score"])
    )
    
    # Discovery Features
    user_known_cats = (
        hist_with_meta
        .filter(pl.col("category_id").is_not_null())
        .select(["customer_id", "category_id"])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("cat_previously_bought"))
    )
    
    item_birth_date = (
        hist_trans
        .group_by("item_id")
        .agg(pl.col("created_date").min().alias("first_sale_date"))
    )
    
    # ✅ NEW: User-Item interaction history
    user_item_history = (
        hist_trans
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("historical_purchases").cast(pl.Int64),
            pl.col("created_date").max().alias("last_purchase_date")
        ])
        .with_columns(
            (pl.lit(hist_end) - pl.col("last_purchase_date")).dt.total_days()
                .cast(pl.Float32).alias("days_since_user_bought_item")
        )
    )
    
    # Merge & Compute
    features = (
        candidates
        .join(user_stats, on="customer_id", how="left")
        .join(user_recent_activity, on="customer_id", how="left")
        .join(item_stats, on="item_id", how="left")
        .join(category_affinity, on=["customer_id", "item_id"], how="left")
        .join(item_niche_score, on="item_id", how="left")
        .join(items_cast.select(["item_id", "category_id", "price"]), on="item_id", how="left")
        .join(user_known_cats, on=["customer_id", "category_id"], how="left")
        .join(item_birth_date, on="item_id", how="left")
        .join(user_item_history, on=["customer_id", "item_id"], how="left")
        .with_columns([
            # Fill nulls for scores
            pl.col("feat_pop_score").fill_null(0.0).cast(pl.Float32),
            pl.col("feat_cat_rank_score").fill_null(0.0).cast(pl.Float32),
            pl.col("feat_cf_score").fill_null(0.0).cast(pl.Float32),
            pl.col("feat_trend_score").fill_null(0.0).cast(pl.Float32),
            pl.col("feat_price_match_score").fill_null(0.0).cast(pl.Float32),
            pl.col("feat_i2v_score").fill_null(0.0).cast(pl.Float32),
            
            # Fill nulls for stats
            pl.col("user_txn_count").fill_null(0).cast(pl.Int64),
            pl.col("user_unique_items").fill_null(0).cast(pl.Int64),
            pl.col("user_avg_price").fill_null(0.0).cast(pl.Float32),
            pl.col("user_price_std").fill_null(0.0).cast(pl.Float32),
            pl.col("item_txn_count").fill_null(0).cast(pl.Int64),
            pl.col("item_unique_users").fill_null(0).cast(pl.Int64),
            pl.col("item_avg_price").fill_null(0.0).cast(pl.Float32),
            pl.col("user_category_purchases").fill_null(0).cast(pl.Int64),
            pl.col("category_rank_for_user").fill_null(999).cast(pl.Int64),
            pl.col("niche_score").fill_null(0.0).cast(pl.Float32),
            
            # Derived features with explicit types
            (pl.col("item_avg_price") / (pl.col("user_avg_price") + 1.0))
                .fill_null(0.0).cast(pl.Float32).alias("price_affinity_ratio"),
            
            ((pl.col("item_avg_price") - pl.col("user_avg_price")).abs() / (pl.col("user_price_std") + 1.0))
                .fill_null(0.0).cast(pl.Float32).alias("price_z_score"),
            
            (pl.col("user_category_purchases").cast(pl.Float32) / (pl.col("user_txn_count").cast(pl.Float32) + 1.0))
                .fill_null(0.0).cast(pl.Float32).alias("category_affinity_score"),
            
            (pl.col("category_rank_for_user") <= 3).cast(pl.Float32).alias("is_preferred_category"),
            
            # Discovery features with explicit types
            pl.when(pl.col("cat_previously_bought").is_null())
                .then(pl.lit(1.0, dtype=pl.Float32))
                .otherwise(pl.lit(0.0, dtype=pl.Float32))
                .alias("feat_is_new_category"),
            
            (pl.lit(hist_end) - pl.col("first_sale_date")).dt.total_days()
                .fill_null(365)
                .cast(pl.Float32)
                .alias("feat_item_age_days"),
            
            ((pl.col("price") - pl.col("user_avg_price")) / (pl.col("user_avg_price") + 1.0))
                .fill_null(0.0)
                .cast(pl.Float32)
                .alias("feat_price_drift")
        ])
    )
    
    # Define feature columns
    feature_cols = [
        # Scores
        *ALL_SCORES,
        # User Stats
        "user_txn_count", "user_unique_items", "user_avg_price", "user_price_std",
        # Item Stats
        "item_txn_count", "item_unique_users", "item_avg_price", "niche_score",
        # Affinity
        "user_category_purchases", "category_rank_for_user", "price_affinity_ratio",
        "price_z_score", "category_affinity_score", "is_preferred_category",
        # Discovery
        "feat_is_new_category", "feat_item_age_days", "feat_price_drift"
    ]
    
    # Select output with explicit Float32 casting
    output = features.select([
        pl.col("customer_id"),
        pl.col("item_id"),
        *[pl.col(feat).cast(pl.Float32).alias(f"X_{i}") for i, feat in enumerate(feature_cols)]
    ])
    
    if verbose:
        print(f"  [Stage 2] Features: {len(feature_cols)}")
    
    return output


# ============================================================================
# DATASET BUILDER
# ============================================================================

def build_dataset(
    split: DataSplit,
    trans: pl.LazyFrame,
    users: pl.LazyFrame,
    items: pl.LazyFrame,
    is_train: bool = True,
    sample_users: float = 1.0,
    verbose: bool = True
) -> pl.LazyFrame:
    """Build dataset with history filtering"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Building {split.name.upper()} dataset")
        print(f"{'='*60}")
        print(split)
    
    # Sample users
    if sample_users < 1.0:
        users = users.filter(pl.col("customer_id").hash(42) % 100 < int(sample_users * 100))
    
    # Generate candidates
    candidates = generate_candidates(
        trans, items, users,
        split.hist_start, split.hist_end,
        n_popular=100,
        verbose=verbose
    )
    
    # Check recall
    # if is_train and split.target_start is not None:
    #     check_stage1_recall(
    #         candidates_lf=candidates,
    #         transactions_lf=trans,
    #         target_start=split.target_start,
    #         target_end=split.target_end,
    #         verbose=verbose
    #     )
    
    # Build features
    features = build_features(
        candidates, trans, items,
        split.hist_start, split.hist_end,
        verbose=verbose
    )
    
    # Filter history
    hist_pairs = (
        trans
        .filter(
            (pl.col("created_date") >= split.hist_start) & 
            (pl.col("created_date") <= split.hist_end)
        )
        .select(["customer_id", "item_id"])
        .unique()
    )
    
    features = features.join(hist_pairs, on=["customer_id", "item_id"], how="anti")
    
    if verbose:
        print(f"  [Filter] Removed historical items")
    
    # Add targets
    if is_train and split.target_start is not None:
        targets = (
            trans
            .filter(
                (pl.col("created_date") >= split.target_start) & 
                (pl.col("created_date") <= split.target_end)
            )
            .select(["customer_id", "item_id"])
            .unique()
            .with_columns(pl.lit(1).cast(pl.UInt8).alias("Y"))
        )
        
        features = (
            features
            .join(targets, on=["customer_id", "item_id"], how="left")
            .with_columns(pl.col("Y").fill_null(0))
        )
    
    return features


# ============================================================================
# NUMPY PREPARATION
# ============================================================================

def prepare_for_xgb(
    lf: pl.LazyFrame,
    is_train: bool = True,
    hard_neg_ratio: int = 10,
    easy_neg_ratio: int = 10,
    hard_neg_col: str = "X_14",
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pl.DataFrame]:
    """Convert to numpy for XGBoost with Hard Negative Mining"""
    if verbose:
        print(f"  Materializing data (Hard Negative Mode)...")
    
    df = lf.collect()  # ✅ Use streaming collection
    
    # Sampling logic (Train only)
    if is_train and "Y" in df.columns:
        pos = df.filter(pl.col("Y") == 1)
        neg = df.filter(pl.col("Y") == 0)
        n_pos = len(pos)
        
        if n_pos > 0:
            # Identify hard vs easy negatives
            if hard_neg_col in df.columns:
                hard_pool = neg.filter(pl.col(hard_neg_col) <= 5)
                easy_pool = neg.filter(pl.col(hard_neg_col) > 5)
            else:
                if verbose:
                    print(f"  [Warn] {hard_neg_col} not found. Using random sampling.")
                hard_pool = neg
                easy_pool = pl.DataFrame([])
            
            # Sample
            n_hard_target = n_pos * hard_neg_ratio
            n_easy_target = n_pos * easy_neg_ratio
            
            # Sample hard negatives
            if len(hard_pool) > 0:
                n_hard_actual = min(len(hard_pool), n_hard_target)
                hard_sampled = hard_pool.sample(n=n_hard_actual, seed=42)
            else:
                hard_sampled = pl.DataFrame([], schema=neg.schema)
            
            # Sample easy negatives
            if len(easy_pool) > 0:
                n_easy_actual = min(len(easy_pool), n_easy_target)
                easy_sampled = easy_pool.sample(n=n_easy_actual, seed=42)
            else:
                n_remaining = (n_pos * (hard_neg_ratio + easy_neg_ratio)) - len(hard_sampled)
                n_actual = min(len(neg), n_remaining)
                easy_sampled = neg.sample(n=n_actual, seed=42)
            
            # Combine
            df = pl.concat([pos, hard_sampled, easy_sampled], how="vertical")
            
            if verbose:
                print(f"  [Sampling] Pos: {len(pos):,}")
                print(f"  [Sampling] Hard Neg: {len(hard_sampled):,} (Target: {n_hard_target:,})")
                print(f"  [Sampling] Easy Neg: {len(easy_sampled):,} (Target: {n_easy_target:,})")
                print(f"  [Sampling] Total: {len(df):,}")
            
            del pos, neg, hard_pool, easy_pool, hard_sampled, easy_sampled
            gc.collect()
    
    # Validation/Test logic
    elif not is_train and "Y" in df.columns:
        pos = df.filter(pl.col("Y") == 1)
        neg = df.filter(pl.col("Y") == 0)
        
        target_neg = min(len(neg), len(pos) * 100)
        neg_sampled = neg.sample(n=target_neg, seed=42, shuffle=True)
        
        df = pl.concat([pos, neg_sampled], how="vertical")
        
        if verbose:
            print(f"  [Val-Sample] Reduced to {len(df):,} rows")
        
        del pos, neg, neg_sampled
        gc.collect()
    
    # Sort by customer_id for ranking
    df = df.sort("customer_id")
    
    # Extract features
    feature_cols = [c for c in df.columns if c.startswith("X_")]
    
    if not feature_cols:
        raise ValueError("No columns starting with 'X_' found!")
    
    # ✅ More efficient numpy conversion
    X = df.select(feature_cols).to_numpy()
    y = df["Y"].to_numpy() if "Y" in df.columns else None
    
    # Calculate groups efficiently using pandas-style groupby
    # This is more stable than RLE for complex operations
    customer_ids = df["customer_id"].to_numpy()
    _, groups = np.unique(customer_ids, return_counts=True)
    
    id_df = df.select(["customer_id", "item_id"])
    
    if verbose:
        print(f"  Shape: X={X.shape}, groups_count={len(groups)}")
        if y is not None:
            print(f"  Avg Group Size: {len(X)/len(groups):.1f}")
    
    return X, y, groups, feature_cols, id_df


# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    verbose: bool = True
) -> XGBRanker:
    """Train XGBoost ranker"""
    if verbose:
        print("\n" + "="*60)
        print("Training XGBRanker")
        print("="*60)
    
    model = XGBRanker(
        objective='rank:ndcg',
        n_estimators=20,
        max_depth=6,
        learning_rate=0.05,
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    
    model.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_val, y_val)],
        eval_group=[groups_val],
        verbose=verbose
    )
    
    return model


# ============================================================================
# INFERENCE
# ============================================================================

import polars as pl
import numpy as np
import gc
from typing import List, Dict, Any
from xgboost import XGBRanker

def predict_top_k(
    model: XGBRanker,
    lf: pl.LazyFrame,
    features: List[str],
    gt_users: List[Any],
    top_k: int = 10,
    batch_size: int = 500_000,  # Process 500k rows at a time (adjust based on RAM)
    verbose: bool = True
) -> Dict[Any, List[Any]]:
    """
    Optimized inference function.
    
    Strategy:
    1. Materialize the relevant test features into memory ONCE.
    2. Run XGBoost prediction in large vector batches.
    3. Use Polars (Rust) to sort and extract top-k items efficiently.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Inference: Generating Predictions")
        print(f"{'='*60}")
        print("  [1/4] Materializing test data...")

    # -------------------------------------------------------------------------
    # 1. Materialize Data (The bottleneck breaker)
    # -------------------------------------------------------------------------
    # We filter for only the users we need to predict for, then collect.
    # This executes the Feature Engineering DAG exactly once.
    df_test = (
        lf
        .filter(pl.col("customer_id").is_in(gt_users))
        .select(["customer_id", "item_id"] + features)
        .collect(streaming=True)
    )
    
    total_rows = len(df_test)
    if verbose:
        print(f"  [2/4] Predicting on {total_rows:,} candidate rows...")
        print(f"        (Batch size: {batch_size:,})")

    if total_rows == 0:
        return {}

    # -------------------------------------------------------------------------
    # 2. Vectorized Prediction
    # -------------------------------------------------------------------------
    # We pre-allocate a numpy array for scores to avoid memory fragmentation
    all_scores = np.zeros(total_rows, dtype=np.float32)
    
    # Iterate through the DataFrame in chunks (CPU-bound)
    # This prevents creating a massive NumPy array copy of the features if RAM is tight
    for i in range(0, total_rows, batch_size):
        end = min(i + batch_size, total_rows)
        
        # Slice feature columns and convert to numpy
        # Note: We assume feature columns are floats. If not, this might copy.
        X_batch = df_test[i:end].select(features).to_numpy()
        
        # Predict
        all_scores[i:end] = model.predict(X_batch)
        
        # Periodic Garbage Collection for very large loops
        if i % (batch_size * 5) == 0:
            gc.collect()

    # -------------------------------------------------------------------------
    # 3. Efficient Top-K Extraction
    # -------------------------------------------------------------------------
    if verbose:
        print("  [3/4] Ranking and extracting Top-K items...")

    # Attach scores and sort
    # Polars is much faster at "Sort-Group-Head" than Python dictionaries
    top_k_df = (
        df_test
        .select(["customer_id", "item_id"])          # Drop feature cols to save RAM
        .with_columns(pl.Series("score", all_scores)) # Attach predictions
        .sort(["customer_id", "score"], descending=[False, True]) # Sort by User then Score
        .group_by("customer_id", maintain_order=True) # Group
        .head(top_k)                                  # Take top K
    )

    # -------------------------------------------------------------------------
    # 4. Convert to Output Dictionary
    # -------------------------------------------------------------------------
    if verbose:
        print("  [4/4] Formatting results...")

    results = {}
    
    # Aggregating to list is the fastest way to bridge Polars -> Python Dict
    # This results in: customer_id | [item1, item2, item3...]
    final_agg = (
        top_k_df
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id"))
    )

    # Fast iteration over rows
    for row in final_agg.iter_rows():
        results[row[0]] = row[1]  # row[0] is user_id, row[1] is list of items

    # Fill in missing users (if any GT users had no candidates generated)
    missing_count = 0
    for user in gt_users:
        if user not in results:
            results[user] = []
            missing_count += 1

    if verbose:
        print(f"  Done. Predictions generated for {len(results):,} users.")
        if missing_count > 0:
            print(f"  Warning: {missing_count} users had 0 candidates (returned empty list).")

    # Clean up memory
    del df_test, all_scores, top_k_df, final_agg
    gc.collect()

    return results


# ============================================================================
# EVALUATION
# ============================================================================

def build_history_dict(
    transactions: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose: bool = True
) -> Dict[Any, List[Any]]:
    """Build user purchase history"""
    if verbose:
        print("  Building history dict...")
    
    hist = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").unique())
        .collect()  # ✅ Regular collect
    )
    
    # ✅ More efficient dict construction
    hist_dict = {row[0]: row[1] for row in hist.iter_rows()}
    
    if verbose:
        print(f"  History: {len(hist_dict):,} users")
    
    return hist_dict


def precision_at_k(
    pred: Dict[Any, List[Any]],
    gt: Dict[Any, Any],
    hist: Dict[Any, List[Any]],
    filter_bought_items: bool = True,
    K: int = 10,
    verbose: bool = True
) -> Tuple[float, List[Any], Dict[str, int]]:
    """Compute Precision@K"""
    precisions = []
    missing_preds_users = []
    
    for user in gt.keys():
        if user not in pred:
            missing_preds_users.append(user)
            continue
        
        # Get ground truth
        if isinstance(gt[user], dict):
            gt_items = gt[user]['list_items']
        else:
            gt_items = gt[user]
        
        relevant_items = set(gt_items)
        
        # Filter purchased items
        if filter_bought_items:
            past_items = set(hist.get(user, []))
            relevant_items -= past_items
        
        if not relevant_items:
            precisions.append(0.0)
            continue
        
        # Compute precision
        hits = len(set(pred[user][:K]) & relevant_items)
        precisions.append(hits / K)
    
    # Statistics
    nusers = len(gt.keys())
    n_missing = len(missing_preds_users)
    n_evaluated = len(precisions)
    mean_precision = np.mean(precisions) if precisions else 0.0
    
    stats = {
        'total_gt_users': nusers,
        'evaluated_users': n_evaluated,
        'users_without_preds': n_missing,
        'coverage_rate': n_evaluated / nusers if nusers > 0 else 0.0,
        'mean_precision': mean_precision
    }
    
    if verbose:
        print(f"\n  Evaluation Statistics:")
        print(f"  ├─ Total GT users: {stats['total_gt_users']:,}")
        print(f"  ├─ Evaluated: {stats['evaluated_users']:,}")
        print(f"  ├─ Missing Preds: {stats['users_without_preds']:,}")
        print(f"  └─ Precision@{K}: {stats['mean_precision']:.4f}")
    
    return mean_precision, missing_preds_users, stats


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*60)
    print("TWO-STAGE RECOMMENDATION PIPELINE")
    print("="*60)
    
    # Load ground truth
    print("\nLoading ground truth...")
    with open(GT_PKL_PATH, 'rb') as f:
        gt = pickle.load(f)
    
    gt_user_ids = list(gt.keys())
    print(f"  Found {len(gt_user_ids):,} users in ground truth")
    
    # Load data
    print("\nLoading data sources...")
    trans = pl.scan_parquet(TRANSACTIONS_GLOB)
    items = pl.scan_parquet(ITEMS_PATH)
    users_all = pl.scan_parquet(USERS_GLOB)
    
    # Filter GT users
    schema = trans.collect_schema()
    cid_type = schema["customer_id"]
    
    gt_users_lf = pl.LazyFrame({"customer_id": gt_user_ids}).with_columns(
        pl.col("customer_id").cast(cid_type)
    )
    
    SAMPLE_USERS = 0.1
    
    # Build datasets
    print("\n" + "="*60)
    print("Building datasets...")
    print("="*60)
    
    train_lf = build_dataset(
        SPLITS['train'], trans, users_all, items,
        is_train=True, sample_users=SAMPLE_USERS
    )
    
    val_lf = build_dataset(
        SPLITS['val'], trans, users_all, items,
        is_train=True, sample_users=SAMPLE_USERS
    )
    
    print(f"\nBuilding TEST dataset (GT users: {len(gt_user_ids):,})")
    test_lf = build_dataset(
        SPLITS['test'], trans, gt_users_lf, items,
        is_train=False, sample_users=1.0
    )
    
    # Prepare for XGBoost
    X_train, y_train, g_train, feats, _ = prepare_for_xgb(
        train_lf, is_train=True,
        hard_neg_ratio=15,
        easy_neg_ratio=15,
        hard_neg_col="X_14"
    )
    
    X_val, y_val, g_val, _, _ = prepare_for_xgb(val_lf, is_train=False)
    
    # Train
    model = train_model(X_train, y_train, g_train, X_val, y_val, g_val)
    
    # Clean up training data
    del X_train, y_train, g_train, X_val, y_val, g_val, train_lf, val_lf
    gc.collect()
    
    # Predict
    preds = predict_top_k(model, test_lf, feats, gt_users=gt_user_ids)
    
    # Build history
    hist_dict = build_history_dict(
        trans,
        SPLITS['test'].hist_start,
        SPLITS['test'].hist_end
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    
    p_at_10, missing_users, stats = precision_at_k(
        pred=preds,
        gt=gt,
        hist=hist_dict,
        filter_bought_items=True,
        K=10
    )
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    eval_results = {
        'precision_at_10': float(p_at_10),
        'statistics': stats,
        'n_missing_users': len(missing_users),
        'sample_missing_users': [str(u) for u in missing_users[:10]]
    }
    
    eval_path = f"{OUTPUT_DIR}/evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n✓ Evaluation saved: {eval_path}")
    
    # Save predictions
    output_path = f"{OUTPUT_DIR}/predictions.json"
    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in preds.items()}, f, indent=2)
    
    print(f"✓ Predictions saved: {output_path}")
    print(f"  Total: {len(preds):,} users")
    
    # Feature importance
    print("\n" + "="*60)
    print("Top 10 Feature Importance")
    print("="*60)
    
    importance = model.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    for feat, score in sorted_importance[:10]:
        feat_idx = int(feat.replace('f', ''))
        feat_name = feats[feat_idx] if feat_idx < len(feats) else feat
        print(f"  {feat_name}: {score:.2f}")
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)


if __name__ == "__main__":
    main()