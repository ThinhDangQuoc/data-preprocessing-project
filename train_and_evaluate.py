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
os.environ["POLARS_MAX_THREADS"] = str(os.cpu_count())

# ============================================================================
# CONFIG & SAMPLING
# ============================================================================

# ✅ CHANGE: Set this to 0.1 to run everything on 10% of users
SAMPLING_RATE = 0.01 

BASE_DIR = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system"
TRANSACTIONS_GLOB = f"{BASE_DIR}/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = f"{BASE_DIR}/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = f"{BASE_DIR}/dataset/sales_pers.user_chunk*.parquet"
GT_PKL_PATH = f"{BASE_DIR}/groundtruth.pkl"
OUTPUT_DIR = f"{BASE_DIR}/outputs_fast_0.1" # Changed output dir to reflect sampling

ALL_SCORES = [
    "feat_pop_score",
    "feat_cat_rank_score",
    "feat_cf_score",
    "feat_trend_score",
    "feat_price_match_score",
    "feat_i2v_score",
    "feat_repeat_score",
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
# LOGIC: ITEM2VEC & CANDIDATES
# ============================================================================

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
    if verbose:
        print("  [Item2Vec] Generating candidates...")
    
    try:
        schema = transactions.collect_schema()
    except AttributeError:
        schema = transactions.schema
    
    item_id_type = schema["item_id"]

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

    unique_source_ids = user_recent_items_df["source_item_id"].unique().to_list()
    vocab = model.wv.key_to_index
    
    src_list = []
    dst_list = []
    score_list = []
    
    for src_id in unique_source_ids:
        token = str(src_id) 
        if token in vocab:
            neighbors = model.wv.most_similar(token, topn=n_similar)
            for sim_token, score in neighbors:
                src_list.append(src_id) 
                dst_list.append(sim_token) 
                score_list.append(score)
    
    if not src_list:
        return pl.LazyFrame(schema={
            "customer_id": schema["customer_id"], 
            "item_id": item_id_type, 
            "feat_i2v_score": pl.Float32
        })

    sim_df = pl.DataFrame({
        "source_item_id": src_list,
        "item_id": dst_list,
        "similarity": score_list
    }).lazy().with_columns([
        pl.col("source_item_id").cast(item_id_type),
        pl.col("item_id").cast(item_id_type),
        pl.col("similarity").cast(pl.Float32)
    ])

    candidates = (
        user_recent_items_df.lazy()
        .join(sim_df, on="source_item_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg(pl.col("similarity").max().alias("feat_i2v_score"))
    )
    
    return candidates

# ============================================================================
# STAGE 1: CANDIDATE GENERATION
# ============================================================================

def generate_candidates(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    n_popular: int = 150,
    n_category_popular: int = 50,
    n_recent_trending: int = 100,
    n_collab_per_item: int = 30,
    verbose: bool = True
) -> pl.LazyFrame:
    if verbose:
        print(f"  [Stage 1] Generating candidates...")
    
    schema = transactions.collect_schema()
    cid_type = schema["customer_id"]
    iid_type = schema["item_id"]
    
    target_users = users.select(pl.col("customer_id").cast(cid_type)).unique()
    items_cast = items.with_columns(pl.col("item_id").cast(iid_type))
    
    hist = transactions.filter(
        (pl.col("created_date") >= hist_start) & 
        (pl.col("created_date") <= hist_end)
    ).with_columns([
        pl.col("item_id").cast(iid_type),
        pl.col("customer_id").cast(cid_type)
    ])
    
    hist_with_meta = hist.join(items_cast, on="item_id", how="left")
    
    # 1. Global Popularity
    popular_items = (
        hist.group_by("item_id").len()
        .sort("len", descending=True).head(n_popular)
        .with_columns([(pl.col("len").log() / pl.col("len").log().max()).cast(pl.Float32).alias("feat_pop_score")])
        .select(["item_id", "feat_pop_score"])
    )
    cands_global = target_users.join(popular_items, how="cross")
    
    # 2. Category Popularity
    user_cats = (
        hist_with_meta.filter(pl.col("category_id").is_not_null())
        .group_by(["customer_id", "category_id"]).len()
        .sort("len", descending=True)
        .group_by("customer_id", maintain_order=True).head(5)
    )
    
    cat_pop = (
        hist_with_meta.filter(pl.col("category_id").is_not_null())
        .group_by(["category_id", "item_id"]).len()
        .with_columns(pl.col("len").rank("dense", descending=True).over("category_id").alias("rank"))
        .filter(pl.col("rank") <= n_category_popular)
    )
    
    cands_category = (
        user_cats.join(cat_pop, on="category_id")
        .select(["customer_id", "item_id", (1.0 / pl.col("rank")).cast(pl.Float32).alias("feat_cat_rank_score")])
    )
    
    # 3. Collaborative Filtering
    user_items = hist.select(["customer_id", "item_id"]).unique()
    item_pairs = (
        user_items.join(user_items, on="customer_id", suffix="_right")
        .filter(pl.col("item_id") != pl.col("item_id_right"))
        .group_by(["item_id", "item_id_right"]).len()
        .filter(pl.col("len") > 1)
        .with_columns([
            pl.col("len").rank("dense", descending=True).over("item_id").alias("rank"),
            pl.col("len").log().cast(pl.Float32).alias("cooccur_score")
        ])
        .filter(pl.col("rank") <= n_collab_per_item)
        .select([pl.col("item_id").alias("source_item"), pl.col("item_id_right").alias("target_item"), pl.col("cooccur_score").alias("feat_cf_score")])
    )
    
    cands_collab = (
        user_items.join(item_pairs, left_on="item_id", right_on="source_item")
        .group_by(["customer_id", "target_item"]).agg(pl.col("feat_cf_score").max())
        .select(["customer_id", pl.col("target_item").alias("item_id"), "feat_cf_score"])
    )
    
    # 4. Trending
    recent_7d = hist_end - timedelta(days=7)
    recent_14d = hist_end - timedelta(days=14)
    trending_items = (
        pl.concat([
            hist.filter(pl.col("created_date") >= recent_7d).group_by("item_id").len(),
            hist.filter(pl.col("created_date") >= recent_14d).group_by("item_id").len()
        ])
        .group_by("item_id").agg(pl.col("len").max())
        .sort("len", descending=True).head(n_recent_trending)
        .with_columns((pl.col("len").log() / pl.col("len").log().max()).cast(pl.Float32).alias("feat_trend_score"))
        .select(["item_id", "feat_trend_score"])
    )
    cands_trending = target_users.join(trending_items, how="cross")
    
    # 5. Price Match
    item_prices = hist.group_by("item_id").agg(pl.col("price").mean())
    pq = item_prices.select([pl.col("price").quantile(0.33).alias("low"), pl.col("price").quantile(0.66).alias("high")]).collect()
    low_th, high_th = pq["low"][0], pq["high"][0]
    
    items_binned = item_prices.with_columns(
        pl.when(pl.col("price") < low_th).then(pl.lit(1))
        .when(pl.col("price") < high_th).then(pl.lit(2)).otherwise(pl.lit(3)).alias("price_bin")
    )
    users_binned = hist.group_by("customer_id").agg(pl.col("price").mean()).with_columns(
        pl.when(pl.col("price") < low_th).then(pl.lit(1))
        .when(pl.col("price") < high_th).then(pl.lit(2)).otherwise(pl.lit(3)).alias("price_bin")
    )
    top_items_bin = hist.join(items_binned, on="item_id").group_by(["price_bin", "item_id"]).len().sort("len", descending=True).group_by("price_bin").head(50)
    
    cands_price = users_binned.join(top_items_bin, on="price_bin").select(["customer_id", "item_id", pl.lit(1.0, dtype=pl.Float32).alias("feat_price_match_score")])
    
    # 6. Item2Vec
    i2v_model = train_item2vec_model(transactions, hist_start, hist_end, verbose=verbose)
    cands_i2v = generate_i2v_candidates(i2v_model, hist, target_users, hist_end, n_similar=30, verbose=verbose)
    
    # 7. Recent Repeats
    cands_repeat = (
        hist.group_by(["customer_id", "item_id"]).agg(pl.len().alias("repeat_count"))
        .filter(pl.col("repeat_count") > 1)
        .with_columns((pl.col("repeat_count").log() + 1.0).cast(pl.Float32).alias("feat_repeat_score"))
        .select(["customer_id", "item_id", "feat_repeat_score"])
    )
    cands_repeat = standardize(cands_repeat, "feat_repeat_score")
    
    all_cands = [
        standardize(cands_global, "feat_pop_score"),
        standardize(cands_category, "feat_cat_rank_score"),
        standardize(cands_collab, "feat_cf_score"),
        standardize(cands_trending, "feat_trend_score"),
        standardize(cands_price, "feat_price_match_score"),
        standardize(cands_i2v, "feat_i2v_score"),
        cands_repeat
    ]
    
    candidates = pl.concat(all_cands, how="vertical").group_by(["customer_id", "item_id"]).agg([pl.col(sc).max() for sc in ALL_SCORES])
    
    if verbose:
        n_cands = candidates.select(pl.len()).collect().item()
        print(f"  [Stage 1] Generated {n_cands:,} candidates")
    
    return candidates


# ============================================================================
# STAGE 2: FEATURE ENGINEERING
# ============================================================================

def build_features(
    candidates: pl.LazyFrame,
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose: bool = True
) -> pl.LazyFrame:
    if verbose: print(f"  [Stage 2] Building features...")
    
    cand_schema = candidates.collect_schema()
    iid_type, cid_type = cand_schema["item_id"], cand_schema["customer_id"]
    items_cast = items.with_columns(pl.col("item_id").cast(iid_type))
    
    hist_trans = transactions.filter(
        (pl.col("created_date") >= hist_start) & (pl.col("created_date") <= hist_end)
    ).with_columns([pl.col("item_id").cast(iid_type), pl.col("customer_id").cast(cid_type)])
    
    hist_with_meta = hist_trans.join(items_cast, on="item_id", how="left")
    recent_date = hist_end - timedelta(days=14)
    
    user_stats = hist_trans.group_by("customer_id").agg([
        pl.len().alias("user_txn_count").cast(pl.Int64),
        pl.col("item_id").n_unique().alias("user_unique_items").cast(pl.Int64),
        pl.col("price").mean().alias("user_avg_price").cast(pl.Float32),
        pl.col("price").std().alias("user_price_std").cast(pl.Float32),
        pl.col("created_date").max().alias("user_last_purchase_date"),
    ]).with_columns([
        pl.col("user_price_std").fill_null(0.0),
        (pl.lit(hist_end) - pl.col("user_last_purchase_date")).dt.total_days().cast(pl.Float32).alias("days_since_last_purchase")
    ])
    
    user_recent_activity = hist_trans.filter(pl.col("created_date") >= recent_date).group_by("customer_id").agg(pl.len().alias("recent_txn_count").cast(pl.Int64))
    
    item_stats = hist_trans.group_by("item_id").agg([
        pl.len().alias("item_txn_count").cast(pl.Int64),
        pl.col("customer_id").n_unique().alias("item_unique_users").cast(pl.Int64),
        pl.col("price").mean().alias("item_avg_price").cast(pl.Float32),
        pl.col("created_date").max().alias("item_last_sale_date"),
    ]).with_columns((pl.lit(hist_end) - pl.col("item_last_sale_date")).dt.total_days().cast(pl.Float32).alias("days_since_last_sale"))
    
    user_category_counts = hist_with_meta.filter(pl.col("category_id").is_not_null()).group_by(["customer_id", "category_id"]).agg(pl.len().alias("len").cast(pl.Int64)).with_columns(
        pl.col("len").rank("dense", descending=True).over("customer_id").cast(pl.Int64).alias("category_rank")
    )
    
    category_affinity = candidates.join(items_cast.select(["item_id", "category_id"]).unique(), on="item_id", how="left").join(user_category_counts, on=["customer_id", "category_id"], how="left").select([
        "customer_id", "item_id", pl.col("len").fill_null(0).cast(pl.Int64).alias("user_category_purchases"), pl.col("category_rank").fill_null(999).cast(pl.Int64).alias("category_rank_for_user")
    ])
    
    item_niche_score = item_stats.with_columns([(1.0 / (1.0 + (pl.col("item_txn_count").cast(pl.Float32).log() - np.log(50)).abs())).cast(pl.Float32).alias("niche_score")]).select(["item_id", "niche_score"])
    user_known_cats = hist_with_meta.filter(pl.col("category_id").is_not_null()).select(["customer_id", "category_id"]).unique().with_columns(pl.lit(1, dtype=pl.Int8).alias("cat_previously_bought"))
    item_birth_date = hist_trans.group_by("item_id").agg(pl.col("created_date").min().alias("first_sale_date"))
    
    user_item_history = hist_trans.group_by(["customer_id", "item_id"]).agg([
        pl.len().alias("historical_purchases").cast(pl.Int64), pl.col("created_date").max().alias("last_purchase_date")
    ]).with_columns((pl.lit(hist_end) - pl.col("last_purchase_date")).dt.total_days().cast(pl.Float32).alias("days_since_user_bought_item"))
    
    features = (
        candidates.join(user_stats, on="customer_id", how="left").join(user_recent_activity, on="customer_id", how="left")
        .join(item_stats, on="item_id", how="left").join(category_affinity, on=["customer_id", "item_id"], how="left")
        .join(item_niche_score, on="item_id", how="left").join(items_cast.select(["item_id", "category_id", "price"]), on="item_id", how="left")
        .join(user_known_cats, on=["customer_id", "category_id"], how="left").join(item_birth_date, on="item_id", how="left")
        .join(user_item_history, on=["customer_id", "item_id"], how="left")
        .with_columns([
            pl.col(sc).fill_null(0.0).cast(pl.Float32) for sc in ALL_SCORES
        ]).with_columns([
            pl.col("user_txn_count").fill_null(0), pl.col("user_unique_items").fill_null(0),
            pl.col("user_avg_price").fill_null(0.0), pl.col("user_price_std").fill_null(0.0),
            pl.col("item_txn_count").fill_null(0), pl.col("item_unique_users").fill_null(0),
            pl.col("item_avg_price").fill_null(0.0), pl.col("user_category_purchases").fill_null(0),
            pl.col("category_rank_for_user").fill_null(999), pl.col("niche_score").fill_null(0.0),
            (pl.col("item_avg_price") / (pl.col("user_avg_price") + 1.0)).fill_null(0.0).cast(pl.Float32).alias("price_affinity_ratio"),
            ((pl.col("item_avg_price") - pl.col("user_avg_price")).abs() / (pl.col("user_price_std") + 1.0)).fill_null(0.0).cast(pl.Float32).alias("price_z_score"),
            (pl.col("user_category_purchases").cast(pl.Float32) / (pl.col("user_txn_count").cast(pl.Float32) + 1.0)).fill_null(0.0).cast(pl.Float32).alias("category_affinity_score"),
            (pl.col("category_rank_for_user") <= 3).cast(pl.Float32).alias("is_preferred_category"),
            pl.when(pl.col("cat_previously_bought").is_null()).then(pl.lit(1.0, dtype=pl.Float32)).otherwise(pl.lit(0.0, dtype=pl.Float32)).alias("feat_is_new_category"),
            (pl.lit(hist_end) - pl.col("first_sale_date")).dt.total_days().fill_null(365).cast(pl.Float32).alias("feat_item_age_days"),
            ((pl.col("price") - pl.col("user_avg_price")) / (pl.col("user_avg_price") + 1.0)).fill_null(0.0).cast(pl.Float32).alias("feat_price_drift")
        ])
    )
    
    feature_cols = [
        *ALL_SCORES, "user_txn_count", "user_unique_items", "user_avg_price", "user_price_std",
        "item_txn_count", "item_unique_users", "item_avg_price", "niche_score",
        "user_category_purchases", "category_rank_for_user", "price_affinity_ratio",
        "price_z_score", "category_affinity_score", "is_preferred_category",
        "feat_is_new_category", "feat_item_age_days", "feat_price_drift"
    ]
    
    return features.select([pl.col("customer_id"), pl.col("item_id"), *[pl.col(feat).cast(pl.Float32).alias(f"X_{i}") for i, feat in enumerate(feature_cols)]])


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
        print(f"Building {split.name.upper()} dataset (Sampling: {sample_users*100:.1f}%)")
        print(f"{'='*60}")
    
    # ✅ Apply sampling deterministically
    if sample_users < 1.0:
        users = users.filter(pl.col("customer_id").hash(42) % 100 < int(sample_users * 100))
    
    candidates = generate_candidates(
        trans, items, users,
        split.hist_start, split.hist_end,
        n_popular=100,
        verbose=verbose
    )
    
    features = build_features(
        candidates, trans, items,
        split.hist_start, split.hist_end,
        verbose=verbose
    )
    
    hist_pairs = (
        trans.filter((pl.col("created_date") >= split.hist_start) & (pl.col("created_date") <= split.hist_end))
        .select(["customer_id", "item_id"]).unique()
    )
    
    features = features.join(hist_pairs, on=["customer_id", "item_id"], how="anti")
    
    if is_train and split.target_start is not None:
        targets = (
            trans.filter((pl.col("created_date") >= split.target_start) & (pl.col("created_date") <= split.target_end))
            .select(["customer_id", "item_id"]).unique()
            .with_columns(pl.lit(1).cast(pl.UInt8).alias("Y"))
        )
        features = features.join(targets, on=["customer_id", "item_id"], how="left").with_columns(pl.col("Y").fill_null(0))
    
    return features

# ============================================================================
# PREPARATION & TRAINING
# ============================================================================

def prepare_for_xgb(
    lf: pl.LazyFrame,
    is_train: bool = True,
    hard_neg_ratio: int = 10,
    easy_neg_ratio: int = 10,
    hard_neg_col: str = "X_14",
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pl.DataFrame]:
    if verbose: print(f"  Materializing data...")
    df = lf.collect()
    
    if is_train and "Y" in df.columns:
        pos = df.filter(pl.col("Y") == 1)
        neg = df.filter(pl.col("Y") == 0)
        n_pos = len(pos)
        
        if n_pos > 0:
            if hard_neg_col in df.columns:
                hard_pool = neg.filter(pl.col(hard_neg_col) <= 5)
                easy_pool = neg.filter(pl.col(hard_neg_col) > 5)
            else:
                hard_pool = neg
                easy_pool = pl.DataFrame([])
            
            n_hard_target = n_pos * hard_neg_ratio
            n_easy_target = n_pos * easy_neg_ratio
            
            hard_sampled = hard_pool.sample(n=min(len(hard_pool), n_hard_target), seed=42) if len(hard_pool) > 0 else pl.DataFrame([], schema=neg.schema)
            easy_sampled = easy_pool.sample(n=min(len(easy_pool), n_easy_target), seed=42) if len(easy_pool) > 0 else neg.sample(n=min(len(neg), n_pos*20), seed=42)
            
            df = pl.concat([pos, hard_sampled, easy_sampled], how="vertical")
            del pos, neg, hard_pool, easy_pool
            gc.collect()
    
    elif not is_train and "Y" in df.columns:
        pos = df.filter(pl.col("Y") == 1)
        neg = df.filter(pl.col("Y") == 0)
        target_neg = min(len(neg), len(pos) * 100)
        neg_sampled = neg.sample(n=target_neg, seed=42, shuffle=True)
        df = pl.concat([pos, neg_sampled], how="vertical")
        del pos, neg, neg_sampled
        gc.collect()
    
    df = df.sort("customer_id")
    feature_cols = [c for c in df.columns if c.startswith("X_")]
    X = df.select(feature_cols).to_numpy()
    y = df["Y"].to_numpy() if "Y" in df.columns else None
    customer_ids = df["customer_id"].to_numpy()
    _, groups = np.unique(customer_ids, return_counts=True)
    
    return X, y, groups, feature_cols, df.select(["customer_id", "item_id"])


def train_model(X_train, y_train, g_train, X_val, y_val, g_val, verbose=True) -> XGBRanker:
    if verbose: print("\nTraining XGBRanker...")
    model = XGBRanker(
        objective='rank:ndcg',
        n_estimators=20,
        max_depth=6,
        learning_rate=0.05,
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train, group=g_train, eval_set=[(X_val, y_val)], eval_group=[g_val], verbose=verbose)
    return model

# ============================================================================
# INFERENCE & EVALUATION
# ============================================================================

def predict_top_k(
    model: XGBRanker,
    lf: pl.LazyFrame,
    features: List[str],
    gt_users: List[Any],
    top_k: int = 10,
    batch_size: int = 500_000,
    verbose: bool = True
) -> Dict[Any, List[Any]]:
    if verbose: print(f"  [Inference] Materializing test data for {len(gt_users):,} users...")
    
    df_test = lf.filter(pl.col("customer_id").is_in(gt_users)).select(["customer_id", "item_id"] + features).collect(streaming=True)
    total_rows = len(df_test)
    if total_rows == 0: return {}

    if verbose: print(f"  [Inference] Predicting on {total_rows:,} rows...")
    all_scores = np.zeros(total_rows, dtype=np.float32)
    
    for i in range(0, total_rows, batch_size):
        end = min(i + batch_size, total_rows)
        X_batch = df_test[i:end].select(features).to_numpy()
        all_scores[i:end] = model.predict(X_batch)
        if i % (batch_size * 5) == 0: gc.collect()

    top_k_df = (
        df_test.select(["customer_id", "item_id"])
        .with_columns(pl.Series("score", all_scores))
        .sort(["customer_id", "score"], descending=[False, True])
        .group_by("customer_id", maintain_order=True).head(top_k)
    )

    results = {}
    final_agg = top_k_df.group_by("customer_id", maintain_order=True).agg(pl.col("item_id"))
    for row in final_agg.iter_rows():
        results[row[0]] = row[1]
    
    del df_test, all_scores, top_k_df
    gc.collect()
    return results


def precision_at_k(pred, gt, hist, filter_bought_items=True, K=10, verbose=True):
    precisions = []
    missing = []
    
    for user in gt.keys():
        if user not in pred:
            missing.append(user)
            continue
        
        gt_items = gt[user]['list_items'] if isinstance(gt[user], dict) else gt[user]
        relevant = set(gt_items)
        if filter_bought_items:
            relevant -= set(hist.get(user, []))
        
        if not relevant:
            precisions.append(0.0)
            continue
            
        hits = len(set(pred[user][:K]) & relevant)
        precisions.append(hits / K)
    
    mean_precision = np.mean(precisions) if precisions else 0.0
    stats = {'total_gt_users': len(gt), 'evaluated': len(precisions), 'mean_precision': mean_precision}
    if verbose:
        print(f"  Evaluated: {stats['evaluated']} / {stats['total_gt_users']} (P@{K}: {mean_precision:.4f})")
    return mean_precision, missing, stats

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print(f"PIPELINE START (Sampling Rate: {SAMPLING_RATE * 100:.0f}%)")
    print("="*60)
    
    # 1. Load Ground Truth
    with open(GT_PKL_PATH, 'rb') as f:
        gt = pickle.load(f)
    print(f"  Loaded GT: {len(gt):,} users total")

    # 2. Load LazyFrames
    trans = pl.scan_parquet(TRANSACTIONS_GLOB)
    items = pl.scan_parquet(ITEMS_PATH)
    users_all = pl.scan_parquet(USERS_GLOB)
    
    # Filter GT users for polars
    schema = trans.collect_schema()
    gt_users_lf = pl.LazyFrame({"customer_id": list(gt.keys())}).with_columns(pl.col("customer_id").cast(schema["customer_id"]))
    
    # 3. Build Datasets (Train/Val/Test ALL sampled)
    # ✅ Apply sampling to all splits for speed
    train_lf = build_dataset(SPLITS['train'], trans, users_all, items, is_train=True, sample_users=SAMPLING_RATE)
    val_lf = build_dataset(SPLITS['val'], trans, users_all, items, is_train=True, sample_users=SAMPLING_RATE)
    test_lf = build_dataset(SPLITS['test'], trans, gt_users_lf, items, is_train=False, sample_users=SAMPLING_RATE)
    
    # 4. Train
    X_train, y_train, g_train, feats, _ = prepare_for_xgb(train_lf, is_train=True, hard_neg_ratio=15, easy_neg_ratio=15)
    X_val, y_val, g_val, _, _ = prepare_for_xgb(val_lf, is_train=False)
    
    model = train_model(X_train, y_train, g_train, X_val, y_val, g_val)
    
    del X_train, y_train, g_train, X_val, y_val, g_val, train_lf, val_lf
    gc.collect()
    
    # 5. Inference
    # ✅ CRITICAL: Extract the actual test users that survived sampling
    # Since we sampled 10% of users in `build_dataset` (via hash), we need to know WHICH 10% 
    # so we only evaluate those.
    print(f"\nIdentifying sampled test users...")
    sampled_test_users = test_lf.select("customer_id").unique().collect()["customer_id"].to_list()
    print(f"  Test set contains {len(sampled_test_users):,} users (approx {SAMPLING_RATE*100}%)")
    
    # Predict only for sampled users
    preds = predict_top_k(model, test_lf, feats, gt_users=sampled_test_users)
    
    # 6. Evaluation
    # ✅ CRITICAL: Filter GT dictionary to only match the sampled users
    # Otherwise, metrics will think we missed 90% of the users.
    gt_sampled = {k: v for k, v in gt.items() if k in sampled_test_users}
    
    hist_dict = (
        trans.filter((pl.col("created_date") >= SPLITS['test'].hist_start) & (pl.col("created_date") <= SPLITS['test'].hist_end))
        .group_by("customer_id").agg(pl.col("item_id").unique()).collect()
    )
    hist_dict = {row[0]: row[1] for row in hist_dict.iter_rows()}
    
    p_at_10, _, stats = precision_at_k(preds, gt_sampled, hist_dict, K=10)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/evaluation.json", "w") as f: json.dump({'precision_at_10': p_at_10, 'stats': stats}, f, indent=2)
    with open(f"{OUTPUT_DIR}/predictions.json", "w") as f: json.dump({str(k): v for k, v in preds.items()}, f, indent=2)
    
    print(f"\n✓ Saved to {OUTPUT_DIR}")
    print("Pipeline Complete!")

if __name__ == "__main__":
    main()