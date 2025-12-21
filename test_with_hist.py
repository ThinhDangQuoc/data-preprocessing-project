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
from preprocess import *
from utils import *
from feature import *
warnings.filterwarnings('ignore')
import os
import polars as pl

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# Best: set environment variable before Polars does heavy work
os.environ["POLARS_MAX_THREADS"] = str(MAX_WORKERS)



# GLOBAL PERFORMANCE CONFIG
pl.Config.set_streaming_chunk_size(1000000)
# Use 'spawn' or 'fork' depending on OS, but standard is usually fine.
# Setting env var for Polars parallelization
os.environ["POLARS_MAX_THREADS"] = str(os.cpu_count())

"""
CRITICAL FIXES FOR CANDIDATE GENERATION
========================================

Problems Fixed:
1. Low Recall@80 (53%) - Missing too many GT items
2. Weak collaborative filtering (only 3 co-occurrences)
3. Item2Vec too restrictive (only 10 neighbors)
4. Missing repeat purchase patterns
"""

import polars as pl
from datetime import datetime
from gensim.models import Word2Vec
import os




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
    """Item2Vec training (unchanged)"""
    if verbose: print("  [Item2Vec] Training...")
    
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
        .collect()
        .to_series()
        .to_list()
    )
    
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
    n_similar: int = 50,
    top_n_items: int = 5,
    verbose: bool = True
) -> pl.LazyFrame:
    """Item2Vec candidate generation (unchanged from your version)"""
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
        if verbose: print("  [Item2Vec] Warning: No similar items found")
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

# Line 89 - UPDATE THIS
ALL_SCORES = [
    "feat_pop_score",
    "feat_cat_rank_score",
    "feat_cf_score",
    "feat_trend_score",
    "feat_i2v_score",
    "feat_repurchase_score",  # ADDED - was "feat_repeat_score"
]

def standardize(df: pl.LazyFrame, active_score_col: str) -> pl.LazyFrame:
    # Always output: customer_id, item_id, then ALL_SCORES in fixed order
    exprs = [
        pl.col("customer_id"),
        pl.col("item_id"),
    ]

    for sc in ALL_SCORES:
        if sc == active_score_col:
            exprs.append(pl.col(active_score_col).cast(pl.Float32).alias(sc))
        else:
            exprs.append(pl.lit(0.0, dtype=pl.Float32).alias(sc))

    return df.select(exprs)



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


# SPLITS = {
#     'train': DataSplit(
#         name='train',
#         hist_start=datetime(2024, 10, 1),
#         hist_end=datetime(2024, 11, 30),
#         target_start=datetime(2024, 12, 1),
#         target_end=datetime(2024, 12, 10)
#     ),
#     'val': DataSplit(
#         name='val',
#         hist_start=datetime(2024, 10, 1),
#         hist_end=datetime(2024, 12, 10),
#         target_start=datetime(2024, 12, 11),
#         target_end=datetime(2024, 12, 20)
#     ),
#     'test': DataSplit(
#         name='test',
#         hist_start=datetime(2024, 12, 1),
#         hist_end=datetime(2024, 12, 30),
#         target_start=None,
#         target_end=None
#     )
# }


#  ============================================================================
# SOLUTION 1: REDUCE CANDIDATE EXPLOSION
# ============================================================================

def generate_candidates(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    max_candidates_per_user: int = 100,
    include_repurchase_candidates: bool = True,  # NEW PARAMETER
    verbose: bool = True
) -> pl.LazyFrame:
    """
    MODIFIED: Generate candidates WITH repurchase items as a dedicated strategy
    """
    if verbose:
        print(f"\n[Candidate Gen V3 - REPURCHASE] Target: {max_candidates_per_user} candidates/user")
        print(f"  Repurchase mode: {'ENABLED' if include_repurchase_candidates else 'DISABLED'}")
    
    # Setup (same as before)
    try:
        schema = transactions.collect_schema()
    except (AttributeError, NotImplementedError):
        sample = transactions.head(1).collect()
        schema = sample.schema
    
    cid_type = schema.get("customer_id", pl.Utf8)
    iid_type = schema.get("item_id", pl.Utf8)
    
    target_users = (
        users.select(pl.col("customer_id"))
        .unique()
        .with_columns(pl.col("customer_id").cast(cid_type))
    )
    
    items_cast = items.with_columns(pl.col("item_id").cast(iid_type))
    
    hist = (
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
    
    hist_with_meta = hist.join(items_cast, on="item_id", how="left")
    
    # ========================================================================
    # 1. POPULAR ITEMS (Baseline)
    # ========================================================================
    popular_items = (
        hist
        .group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(30)
        .with_columns([
            (pl.col("len").log1p() / pl.col("len").log1p().max())
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("feat_pop_score")
        ])
        .select(["item_id", "feat_pop_score"])
    )
    
    cands_global = target_users.join(popular_items, how="cross")
    
    # ========================================================================
    # 2. CATEGORY-BASED
    # ========================================================================
    user_top_cats = (
        hist_with_meta
        .filter(pl.col("category_id").is_not_null())
        .group_by(["customer_id", "category_id"])
        .len()
        .sort("len", descending=True)
        .group_by("customer_id", maintain_order=True)
        .head(5)
        .select(["customer_id", "category_id"])
    )
    
    cat_pop = (
        hist_with_meta
        .filter(pl.col("category_id").is_not_null())
        .group_by(["category_id", "item_id"])
        .len()
        .with_columns(
            pl.col("len")
            .rank("dense", descending=True)
            .over("category_id")
            .alias("rank")
        )
        .filter(pl.col("rank") <= 30)
    )
    
    cands_category = (
        user_top_cats
        .join(cat_pop, on="category_id", how="inner")
        .select([
            "customer_id",
            "item_id",
            (1.0 / pl.col("rank")).cast(pl.Float32).alias("feat_cat_rank_score")
        ])
    )
    
    # ========================================================================
    # 3. COLLABORATIVE FILTERING
    # ========================================================================
    user_recent_items = (
        hist
        .sort(["customer_id", "created_date"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").head(10))
        .explode("item_id")
    )
    
    item_pairs = (
        hist
        .select(["customer_id", "item_id"])
        .unique()
        .join(
            hist.select(["customer_id", "item_id"]).unique(), 
            on="customer_id", 
            suffix="_right"
        )
        .filter(pl.col("item_id") != pl.col("item_id_right"))
        .group_by(["item_id", "item_id_right"])
        .len()
        .filter(pl.col("len") >= 1)
        .with_columns([
            pl.col("len")
            .rank("dense", descending=True)
            .over("item_id")
            .alias("rank"),
            pl.col("len").log1p().cast(pl.Float32).alias("cooccur_score")
        ])
        .filter(pl.col("rank") <= 25)
        .select([
            pl.col("item_id").alias("source_item"),
            pl.col("item_id_right").alias("target_item"),
            pl.col("cooccur_score").alias("feat_cf_score")
        ])
    )
    
    cands_collab = (
        user_recent_items
        .join(item_pairs, left_on="item_id", right_on="source_item", how="inner")
        .group_by(["customer_id", "target_item"])
        .agg(pl.col("feat_cf_score").max())
        .select([
            "customer_id",
            pl.col("target_item").alias("item_id"),
            "feat_cf_score"
        ])
    )
    
    # ========================================================================
    # 4. NEW/MODIFIED: REPURCHASE CANDIDATES (High-Value Strategy!)
    # ========================================================================
    if include_repurchase_candidates:
        if verbose:
            print("  [NEW] Generating Repurchase Candidates...")
        
        # Strategy: Items user bought before, ranked by repurchase likelihood
        user_repurchase_items = (
            hist
            .group_by(["customer_id", "item_id"])
            .agg([
                pl.len().alias("purchase_count"),
                pl.col("created_date").max().alias("last_purchase"),
                pl.col("created_date").min().alias("first_purchase"),
                pl.col("price").mean().alias("avg_price")
            ])
            .with_columns([
                # Recency: More recent = higher score
                (hist_end - pl.col("last_purchase")).dt.total_days().alias("days_since_last"),
                
                # Purchase frequency
                (pl.col("purchase_count").log1p() + 1.0).alias("frequency_score"),
                
                # Regularity: Time between purchases
                ((pl.col("last_purchase") - pl.col("first_purchase")).dt.total_days() / 
                 (pl.col("purchase_count").cast(pl.Float32) + 1.0)).alias("avg_days_between")
            ])
            .with_columns([
                # Repurchase score combines:
                # 1. Frequency (how often they bought it)
                # 2. Recency (how recent was last purchase)
                # 3. Regularity (do they buy it regularly?)
                (
                    pl.col("frequency_score") * 2.0 +  # Frequency is important
                    pl.when(pl.col("days_since_last") < 30)
                    .then(3.0)  # Very recent = very high score
                    .when(pl.col("days_since_last") < 60)
                    .then(2.0)
                    .when(pl.col("days_since_last") < 90)
                    .then(1.5)
                    .otherwise(1.0) +  # Recency boost
                    pl.when(pl.col("avg_days_between") < 60)
                    .then(1.5)  # Regular purchases = boost
                    .otherwise(1.0)
                )
                .cast(pl.Float32)
                .alias("feat_repurchase_score")
            ])
            .select([
                "customer_id",
                "item_id",
                "feat_repurchase_score",
                "purchase_count",
                "days_since_last"
            ])
        )
        
        cands_repurchase = user_repurchase_items.select([
            "customer_id",
            "item_id",
            "feat_repurchase_score"
        ])
        
        if verbose:
            stats = cands_repurchase.select([
                pl.len().alias("total"),
                pl.col("customer_id").n_unique().alias("users")
            ]).collect()
            print(f"    Generated {stats['total'][0]:,} repurchase candidates for {stats['users'][0]:,} users")
    else:
        # Create empty repurchase candidates if disabled
        cands_repurchase = pl.LazyFrame({
            "customer_id": [], "item_id": [], "feat_repurchase_score": []
        }, schema={
            "customer_id": cid_type, "item_id": iid_type, 
            "feat_repurchase_score": pl.Float32
        })
    
    # ========================================================================
    # 5. ITEM2VEC (Keep but optimize)
    # ========================================================================
    try:
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
            n_similar=50,
            top_n_items=10,
            verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"  [WARNING] Item2Vec failed: {e}")
        cands_i2v = pl.LazyFrame({
            "customer_id": [], "item_id": [], "feat_i2v_score": []
        }, schema={
            "customer_id": cid_type, "item_id": iid_type, 
            "feat_i2v_score": pl.Float32
        })
    
    # ========================================================================
    # 6. TRENDING ITEMS
    # ========================================================================
    recent_date = hist_end - pl.duration(days=14)
    
    trending_items = (
        hist
        .filter(pl.col("created_date") >= recent_date)
        .group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(50)
        .with_columns([
            (pl.col("len").log1p() / pl.col("len").log1p().max())
            .cast(pl.Float32)
            .alias("feat_trend_score")
        ])
        .select(["item_id", "feat_trend_score"])
    )
    
    cands_trending = target_users.join(trending_items, how="cross")
    
    # ========================================================================
    # UPDATE: Include repurchase score in ALL_SCORES
    # ========================================================================
    ALL_SCORES_WITH_REPURCHASE = [
        "feat_pop_score",
        "feat_cat_rank_score",
        "feat_cf_score",
        "feat_trend_score",
        "feat_i2v_score",
        "feat_repurchase_score",  # ADDED
    ]
    
    def standardize(df: pl.LazyFrame, active_score: str) -> pl.LazyFrame:
        exprs = [pl.col("customer_id"), pl.col("item_id")]
        for sc in ALL_SCORES_WITH_REPURCHASE:
            if sc == active_score:
                exprs.append(pl.col(active_score).cast(pl.Float32).alias(sc))
            else:
                exprs.append(pl.lit(0.0, dtype=pl.Float32).alias(sc))
        return df.select(exprs)
    
    # ========================================================================
    # COMBINE WITH ENHANCED REPURCHASE WEIGHTING
    # ========================================================================
    all_cands = [
        standardize(cands_global, "feat_pop_score"),
        standardize(cands_category, "feat_cat_rank_score"),
        standardize(cands_collab, "feat_cf_score"),
        standardize(cands_trending, "feat_trend_score"),
        standardize(cands_i2v, "feat_i2v_score"),
        standardize(cands_repurchase, "feat_repurchase_score"),  # ADDED
    ]
    
    candidates = (
        pl.concat(all_cands, how="vertical")
        .group_by(["customer_id", "item_id"])
        .agg([pl.col(sc).max().fill_null(0.0).alias(sc) for sc in ALL_SCORES_WITH_REPURCHASE])
    )
    
    # ========================================================================
    # SMART RANKING: Prioritize Repurchase > CF > Category > Rest
    # ========================================================================
    candidates = (
        candidates
        .with_columns([
            (
                pl.col("feat_repurchase_score") * 4.0 +  # HIGHEST: Repurchase intent
                pl.col("feat_cf_score") * 2.0 +          # Collaborative
                pl.col("feat_cat_rank_score") * 1.5 +    # Category preference
                pl.col("feat_i2v_score") * 1.0 +         # Semantic similarity
                pl.col("feat_trend_score") * 0.5 +       # Trending items
                pl.col("feat_pop_score") * 0.3           # Popularity baseline
            ).alias("_combined_score")
        ])
        .sort(["customer_id", "_combined_score"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .head(max_candidates_per_user)
        .drop("_combined_score")
    )
    
    if verbose:
        stats = candidates.select([
            pl.len().alias("total_cands"),
            pl.col("customer_id").n_unique().alias("n_users")
        ]).collect()
        
        n_cands = stats["total_cands"][0]
        n_users = stats["n_users"][0]
        avg_per_user = n_cands / n_users if n_users > 0 else 0
        
        print(f"  Generated {n_cands:,} candidates for {n_users:,} users")
        print(f"  Avg candidates/user: {avg_per_user:.1f}")
    
    return candidates


def recall_at_k_candidates(candidates_lf: pl.LazyFrame, trans_lf: pl.LazyFrame,
                           target_start: datetime, target_end: datetime,
                           K: int = 80, verbose=True) -> float:
    # GT pairs in target window
    gt_pairs = (
        trans_lf
        .filter((pl.col("created_date") >= target_start) & (pl.col("created_date") <= target_end))
        .select(["customer_id","item_id"])
        .unique()
    )

    # take top-K candidates per user (if not already limited)
    cand_topk = (
        candidates_lf
        .group_by("customer_id", maintain_order=True)
        .head(K)
        .select(["customer_id","item_id"])
    )

    hit_users = cand_topk.join(gt_pairs, on=["customer_id","item_id"], how="inner") \
                         .select("customer_id").unique()

    n_hit = hit_users.collect().height
    n_gt_users = gt_pairs.select("customer_id").unique().collect().height
    recall = n_hit / n_gt_users if n_gt_users else 0.0

    if verbose:
        print(f"[Recall@{K}] GT users={n_gt_users:,} hit users={n_hit:,} recall={recall:.2%}")
    return recall


# ============================================================================
# STAGE 2: ENHANCED FEATURE ENGINEERING
# ============================================================================

# ============================================================================
# FIXED STAGE 2: FEATURE ENGINEERING
# ============================================================================

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
        "past_cnt", "days_since",
        "purchase_regularity_score", "repurchase_composite_score"
    ]
    
    output_cols = [pl.col("customer_id"), pl.col("item_id")]
    for col in final_cols:
        output_cols.append(
            pl.col(col).fill_null(0.0).cast(pl.Float32).alias(col)
        )
    
    return features.select(output_cols)


# ============================================================================
# DATASET BUILDER
# ============================================================================

# ============================================================================
# FIXED: DATASET BUILDER
# ============================================================================

def build_dataset_v2(
    split_name: str,
    split_config: dict,
    trans_clean: pl.LazyFrame,
    items_clean: pl.LazyFrame,
    users_clean: pl.LazyFrame,
    is_train: bool = True,
    verbose: bool = True,
    *,
    max_train_items_per_user: int = 50,
    seed: int = 42,
    do_recall_check: bool = False,
    recall_k: int = 100,
    allow_repurchase: bool = True,
) -> pl.LazyFrame:
    
    if verbose:
        print(f"\n{'='*60}\nBuilding Dataset V3 FIXED: {split_name}\n{'='*60}")
    
    # 1. Generate Candidates (Lazy)
    candidates_lazy = generate_candidates(
        trans_clean, items_clean, users_clean,
        split_config["hist_start"], split_config["hist_end"],
        verbose=verbose
    )
    
    # -----------------------------------------------------------------------
    # CRITICAL OPTIMIZATION: Break Lineage
    # We collect candidates here to prevent the complex generation graph 
    # from being duplicated inside every feature function.
    # -----------------------------------------------------------------------
    if verbose: print(f"  [Optimization] Materializing candidates to break graph lineage...")
    
    # Collect to memory (it's small: users * 100 rows)
    candidates_df = candidates_lazy.collect()
    
    # Convert back to Lazy for feature engineering, but now it's a leaf node!
    candidates = candidates_df.lazy()
    
    if verbose: print(f"  [Optimization] Candidates materialized: {candidates_df.height:,} rows.")
    # -----------------------------------------------------------------------

    # 2. Recall check (optional)
    if do_recall_check and split_config.get("target_start"):
        recall_at_k_candidates(candidates, trans_clean,
            split_config["target_start"], split_config["target_end"], K=recall_k, verbose=True)
    
    # 3. Filter history if needed
    if not allow_repurchase:
        if verbose: print("  [Filtering] Removing previously purchased items...")
        hist_pairs = (
            trans_clean
            .filter(pl.col("created_date") <= split_config["hist_end"])
            .select(["customer_id", "item_id"]).unique()
        )
        candidates = candidates.join(hist_pairs, on=["customer_id", "item_id"], how="anti")
    
    # 4. Build features (Now safe because candidates is simple)
    features = build_features_robust(
        candidates=candidates,
        transactions=trans_clean,
        items=items_clean,
        hist_start=split_config["hist_start"],
        hist_end=split_config["hist_end"],
        verbose=verbose
    )
    
    # 5. Train/Val Logic (Same as before)
    if is_train:
        return sample_train_pairs_v3_FIXED(
            candidates_with_features=features,
            trans_clean=trans_clean,
            target_start=split_config["target_start"],
            target_end=split_config["target_end"],
            max_items_per_user=max_train_items_per_user,
            seed=seed,
            verbose=verbose,
        )
    
    # Val/Test: Add labels
    if split_config.get("target_start") is not None:
        targets = (
            trans_clean
            .filter(pl.col("created_date").is_between(split_config["target_start"], split_config["target_end"]))
            .select(["customer_id", "item_id"]).unique()
            .with_columns(pl.lit(1).cast(pl.UInt8).alias("Y"))
        )
        features = features.join(targets, on=["customer_id", "item_id"], how="left").with_columns(pl.col("Y").fill_null(0))
    
    return features


# ============================================================================
# NUMPY PREPARATION
# ============================================================================

# ============================================================================
# FIXED: NUMPY PREPARATION
# ============================================================================

import polars as pl
import numpy as np
def prepare_for_xgb_v2(
    lf: pl.LazyFrame,
    is_train: bool = True,
    verbose: bool = True,
    *,
    filter_no_positive_users: bool = False,  # ← CHANGED: Don't filter in train
    rand_seed: int = 42,
):
    """
    FIXED: Don't filter users in training (already sampled properly)
    """
    if verbose:
        print(f"\n[XGB Prep V3 FIXED] is_train={is_train}")
    
    exclude_cols = ["customer_id", "item_id", "Y", "created_date", "item_token"]
    
    df = lf.collect()
    
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt8]
    ]
    
    if verbose:
        print(f"  Features: {len(feature_cols)}")
    
    # ⭐ ONLY filter no-positive users in VAL/TEST, not TRAIN
    if "Y" in df.columns and filter_no_positive_users and not is_train:
        pos_users = (
            df.filter(pl.col("Y") == 1)
            .select("customer_id")
            .unique()
        )
        before = df.height
        df = df.join(pos_users, on="customer_id", how="inner")
        if verbose and before > df.height:
            print(f"  Filtered no-positive users: {before:,} -> {df.height:,} rows")
    
    # Sort by user
    df_final = df.with_columns(pl.lit(np.random.rand(len(df))).alias("shuffle_key"))
    df_final = df_final.sort(["customer_id", "shuffle_key"])
    
    # Build arrays
    X = df_final.select(feature_cols).to_numpy()
    y = df_final["Y"].to_numpy() if "Y" in df_final.columns else None
    query_ids = df_final["customer_id"].to_numpy()
    _, groups = np.unique(query_ids, return_counts=True)
    
    if verbose:
        n_samples = df_final.height
        n_groups = len(groups)
        avg_group = n_samples / n_groups if n_groups > 0 else 0
        if y is not None:
            pos_count = int(y.sum())
            pos_rate = pos_count / n_samples if n_samples > 0 else 0
            print(f"  Output: {n_samples:,} rows, {n_groups:,} groups")
            print(f"  Positive: {pos_count:,} ({pos_rate:.2%})")
            print(f"  Avg items/group: {avg_group:.1f}")
        else:
            print(f"  Output: {n_samples:,} rows, {n_groups:,} groups (no labels)")
    
    return X, y, groups, feature_cols, df_final.select(["customer_id", "item_id"])



# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    X_train, y_train, groups_train,
    X_val, y_val, groups_val,
    verbose: bool = True
) -> XGBRanker:
    """
    FIXED: Prevent overfitting with proper regularization
    """
    
    print("\n[Training XGBRanker V3 - Anti-Overfitting]")
    
    model = XGBRanker(
        objective='rank:ndcg',
        n_estimators=2000,       # Increased (we'll use early stopping)
        learning_rate=0.05,      # REDUCED (was 0.1) - slower = better generalization
        max_depth=4,             # REDUCED (was 5) - prevent memorization
        subsample=0.7,           # REDUCED (was 0.8) - more randomness
        colsample_bytree=0.7,    # REDUCED (was 0.8) - feature sampling
        colsample_bylevel=0.7,   # NEW - per-level feature sampling
        reg_lambda=5.0,          # INCREASED (was 2.0) - stronger L2
        reg_alpha=1.0,           # INCREASED (was 0.5) - stronger L1
        min_child_weight=10,     # INCREASED (was 5) - larger leaf nodes
        gamma=0.5,               # NEW - minimum loss reduction for split
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
        eval_metric='ndcg@10',
        early_stopping_rounds=20,  # NEW - stop if no improvement for 30 rounds
    )
    
    model.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_group=[groups_train, groups_val],
        verbose=10,
    )
    
    # Report best iteration
    best_iteration = model.best_iteration
    best_score = model.best_score
    print(f"\n✓ Best iteration: {best_iteration}")
    print(f"✓ Best validation NDCG@10: {best_score:.4f}")
    
    return model


# ============================================================================
# FIX 4: COLD-START FALLBACK STRATEGY
# ============================================================================

def create_cold_start_recommendations(
    items: pl.LazyFrame,
    transactions: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    top_k: int = 100,
    verbose: bool = True
) -> List[Any]:
    """
    Generate fallback recommendations for cold-start users
    
    Strategy:
    1. Trending items (recent popularity spike)
    2. New items (launched recently)
    3. Popular items (safety net)
    """
    
    recent_date = hist_end - timedelta(days=14)
    
    # 1. Trending items (high recent activity)
    trending = (
        transactions
        .filter(
            (pl.col("created_date") >= recent_date) &
            (pl.col("created_date") <= hist_end)
        )
        .group_by("item_id")
        .agg([
            pl.len().alias("recent_sales"),
            pl.col("customer_id").n_unique().alias("recent_buyers")
        ])
        .with_columns([
            (pl.col("recent_sales") * pl.col("recent_buyers").log1p())
            .alias("trend_score")
        ])
        .sort("trend_score", descending=True)
        .head(40)
        .select("item_id")
        .collect()
    )
    
    # 2. New items (launched in last 30 days)
    new_items = (
        transactions
        .filter(pl.col("created_date") <= hist_end)
        .group_by("item_id")
        .agg([
            pl.col("created_date").min().alias("first_sale")
        ])
        .filter(
            (hist_end - pl.col("first_sale")).dt.total_days() <= 30
        )
        .join(
            transactions
            .filter(
                (pl.col("created_date") >= recent_date) &
                (pl.col("created_date") <= hist_end)
            )
            .group_by("item_id")
            .len()
            .rename({"len": "sales"}),
            on="item_id",
            how="inner"
        )
        .sort("sales", descending=True)
        .head(30)
        .select("item_id")
        .collect()
    )
    
    # 3. Overall popular items
    popular = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) &
            (pl.col("created_date") <= hist_end)
        )
        .group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(30)
        .select("item_id")
        .collect()
    )
    
    # Combine: trending > new > popular
    cold_start_items = (
        pl.concat([trending, new_items, popular], how="vertical")
        .unique()
        .head(top_k)
        .to_series()
        .to_list()
    )
    
    if verbose:
        print(f"\n[Cold-Start Fallback] Created {len(cold_start_items)} recommendations")
    
    return cold_start_items
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
        .collect()
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

import numpy as np

def precision_at_k_customer(
    pred,
    gt,
    hist,
    filter_bought_items: bool = False,  # CHANGED DEFAULT to False
    K: int = 10,
    *,
    return_stats: bool = True,
):
    """
    MODIFIED: Customer-compatible Precision@K with repurchase support
    """
    precisions = []
    cold_start_users = []
    nusers = len(gt.keys())

    n_missing_hist = 0
    n_missing_pred = 0
    n_empty_relevant_after_filter = 0

    for user in gt.keys():
        missing_hist = (user not in hist)
        missing_pred = (user not in pred)

        if missing_hist or missing_pred:
            cold_start_users.append(user)
            if missing_hist:
                n_missing_hist += 1
            if missing_pred:
                n_missing_pred += 1
            continue

        val = gt[user]
        gt_items = val["list_items"] if isinstance(val, dict) else val
        relevant_items = set(gt_items)

        # MODIFIED: Only filter if explicitly requested
        if filter_bought_items and user in hist:
            relevant_items -= set(hist[user])

        if not relevant_items:
            n_empty_relevant_after_filter += 1

        hits = len(set(pred[user][:K]) & relevant_items)
        precisions.append(hits / K)

    mean_precision = float(np.mean(precisions)) if precisions else 0.0

    if not return_stats:
        return mean_precision, cold_start_users

    evaluated = len(precisions)
    stats = {
        "total_gt_users": nusers,
        "evaluated_users": evaluated,
        "cold_start_users": len(cold_start_users),
        "missing_hist": n_missing_hist,
        "missing_pred": n_missing_pred,
        "coverage_rate": (evaluated / nusers) if nusers else 0.0,
        "empty_relevant_after_filter": n_empty_relevant_after_filter,
        "precision_at_k": mean_precision,
        "K": K,
        "filter_bought_items": filter_bought_items,
    }

    return mean_precision, cold_start_users, stats

def precision_at_k(
    pred: Dict[Any, List[Any]],
    gt: Dict[Any, Any],
    hist: Dict[Any, List[Any]],
    filter_bought_items: bool = False,  # CHANGED DEFAULT to False
    K: int = 10,
    verbose: bool = True
) -> Tuple[float, List[Any], Dict[str, int]]:
    """
    MODIFIED: Default to NOT filtering bought items (allow repurchase)
    """
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
        
        # MODIFIED: Only filter if explicitly requested
        if filter_bought_items and user in hist:
            past_items = set(hist.get(user, []))
            relevant_items -= past_items
            if verbose and len(relevant_items) == 0:
                print(f"  Warning: User {user} has no new items after filtering")
        
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
        'mean_precision': mean_precision,
        'filter_bought_items': filter_bought_items
    }
    
    if verbose:
        print(f"\n  Evaluation Statistics:")
        print(f"  ├─ Total GT users: {stats['total_gt_users']:,}")
        print(f"  ├─ Evaluated: {stats['evaluated_users']:,}")
        print(f"  ├─ Missing Preds: {stats['users_without_preds']:,}")
        print(f"  ├─ Filter Bought: {filter_bought_items}")
        print(f"  └─ Precision@{K}: {stats['mean_precision']:.4f}")
    
    return mean_precision, missing_preds_users, stats

def predict_with_cold_start_fallback(
    model: XGBRanker,
    lf: pl.LazyFrame,
    features: List[str],
    gt_users: List[Any],
    hist_dict: Dict[Any, List[Any]],
    cold_start_items: List[Any],
    top_k: int = 10,
    batch_size: int = 500_000,
    verbose: bool = True
) -> Dict[Any, List[Any]]:
    """
    Predict with fallback for cold-start users
    """
    
    # Regular prediction (same as before)
    results = predict_top_k(
        model, lf, features, gt_users,
        top_k=top_k, batch_size=batch_size, verbose=verbose
    )
    
    # Fill in cold-start users
    cold_start_count = 0
    for user in gt_users:
        if user not in hist_dict or len(hist_dict[user]) == 0:
            # Cold-start user - use fallback
            if user not in results or len(results[user]) == 0:
                results[user] = cold_start_items[:top_k]
                cold_start_count += 1
    
    if verbose:
        print(f"\n[Cold-Start Fallback] Applied to {cold_start_count:,} users")
    
    return results

# ============================================================================
# MAIN PIPELINE
# ============================================================================

# ============================================================================
# FIXED: MAIN PIPELINE
# ============================================================================

def main():
    """
    FIXED: Complete pipeline with all bug fixes
    """
    # SETTINGS
    RECREATE_PREPROCESSED = True
    PROCESSED_DIR = f"{BASE_DIR}/processed_v1"
    TRAIN_SAMPLE_RATE = 0.1
    VAL_SAMPLE_RATE = 0.1
    TEST_SAMPLE_RATE = 1.0  # Always 1.0 for test
    
    print("="*60)
    print("RECOMMENDATION SYSTEM PIPELINE")
    print("="*60)
    
    # ========================================================================
    # 1. Load Ground Truth
    # ========================================================================
    print("\n[1/8] Loading Ground Truth...")
    with open(GT_PKL_PATH, 'rb') as f:
        gt = pickle.load(f)
    gt_user_ids = list(gt.keys())
    print(f"  GT Users: {len(gt_user_ids):,}")
    
    # ========================================================================
    # 2. Initialize Raw Data
    # ========================================================================
    print("\n[2/8] Loading Raw Data...")
    raw_trans = pl.scan_parquet(TRANSACTIONS_GLOB)
    raw_items = pl.scan_parquet(ITEMS_PATH)
    raw_users = pl.scan_parquet(USERS_GLOB)
    print("  ✓ Data loaded (lazy)")
    
    # ========================================================================
    # 3. Create Splits (FIXED: Now function exists)
    # ========================================================================
    print("\n[3/8] Creating Time Splits...")
    ROBUST_SPLITS = create_realistic_splits(raw_trans)
    
    for split_name, config in ROBUST_SPLITS.items():
        print(f"  {split_name}:")
        print(f"    History: {config['hist_start'].date()} → {config['hist_end'].date()}")
        if config['target_start']:
            print(f"    Target:  {config['target_start'].date()} → {config['target_end'].date()}")
    
    # ========================================================================
    # 4. Process TRAIN Split
    # ========================================================================
    print("\n[4/8] Processing TRAIN Split...")
    t_trans, t_items, t_users = get_preprocessed_data(
        "train", ROBUST_SPLITS['train'], 
        raw_trans, raw_items, raw_users,
        output_dir=PROCESSED_DIR, 
        recreate=RECREATE_PREPROCESSED,
        sample_rate=TRAIN_SAMPLE_RATE
    )
    
    train_ds = build_dataset_v2(
        "train",
        ROBUST_SPLITS["train"],
        t_trans, t_items, t_users,
        is_train=True,
        verbose=True,
        seed=42,
        do_recall_check=True,
        recall_k=80,
        allow_repurchase=True  # NEW: Allow repurchases
    )
    
    # ========================================================================
    # 5. Process VAL Split (FIXED: Now gets labels)
    # ========================================================================
    print("\n[5/8] Processing VAL Split...")
    v_trans, v_items, v_users = get_preprocessed_data(
        "val", ROBUST_SPLITS['val'], 
        raw_trans, raw_items, raw_users,
        output_dir=PROCESSED_DIR, 
        recreate=RECREATE_PREPROCESSED,
        sample_rate=VAL_SAMPLE_RATE
    )
    
    val_ds = build_dataset_v2(
        "val",
        ROBUST_SPLITS["val"],
        v_trans, v_items, v_users,
        is_train=False,
        verbose=True,
        do_recall_check=True,
        recall_k=80,
        allow_repurchase=True  # NEW: Allow repurchases
    )
    val_ds = sample_validation_data(
        val_ds, 
        target_positive_rate=0.10,
        verbose=True
    )
    # ========================================================================
    # 6. Process TEST Split
    # ========================================================================
    print("\n[6/8] Processing TEST Split...")
    
    # Filter to GT users only
    gt_users_only = raw_users.filter(pl.col("customer_id").is_in(gt_user_ids))
    
    te_trans, te_items, te_users = get_preprocessed_data(
        "test", ROBUST_SPLITS['test'], 
        raw_trans, raw_items, gt_users_only,
        output_dir=PROCESSED_DIR, 
        recreate=RECREATE_PREPROCESSED,
        sample_rate=TEST_SAMPLE_RATE  # Always 1.0
    )
    
    test_ds = build_dataset_v2(
        "test", 
        ROBUST_SPLITS['test'], 
        te_trans, te_items, te_users, 
        is_train=False,
        verbose=True,
        allow_repurchase=True  # NEW: Allow repurchases
    )
    
    
    # ========================================================================
    # 7. Train Model
    # ========================================================================
    print("\n[7/8] Training Model...")
    
    # Prepare training data
    X_train, y_train, g_train, feats, _ = prepare_for_xgb_v2(
        train_ds, 
        is_train=True,
        verbose=True
    )
    
    # Prepare validation data (FIXED: Now has labels)
    X_val, y_val, g_val, _, _ = prepare_for_xgb_v2(
        val_ds, 
        is_train=False,  # Keep all data, no sampling
        verbose=True,
        filter_no_positive_users=True,   # <- important

    )
    
    # Train
    model = train_model(
        X_train, y_train, g_train, 
        X_val, y_val, g_val,
        verbose=True
    )
    model.save_model(f"{OUTPUT_DIR}/xgb_ranker.json")
    # Clean up training data
    del X_train, y_train, g_train, X_val, y_val, g_val, train_ds, val_ds
    gc.collect()
    
    # ========================================================================
    # 8. Predict & Evaluate
    # ========================================================================
    print("\n[8/8] Generating Predictions...")
    # Create cold-start fallback
    cold_start_items = create_cold_start_recommendations(
        te_items, te_trans,
        ROBUST_SPLITS['test']['hist_start'],
        ROBUST_SPLITS['test']['hist_end'],
        verbose=True
    )

    
    # Build history for filtering
    hist_dict = build_history_dict(
        te_trans,
        ROBUST_SPLITS['test']['hist_start'],
        ROBUST_SPLITS['test']['hist_end'],
        verbose=True
    )
    preds = predict_with_cold_start_fallback(
        model, test_ds, feats, gt_user_ids,
        hist_dict, cold_start_items,
        top_k=10, verbose=True
    )
   
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    p_at_10, missing_users, stats = precision_at_k(
        pred=preds,
        gt=gt,
        hist=hist_dict,
        filter_bought_items=False,  # CHANGED: Allow repurchases
        K=10,
        verbose=True
    )
    
    p_at_10_customer, cold_start_users, stats_customer = precision_at_k_customer(
        pred=preds,
        gt=gt,
        hist=hist_dict,
        filter_bought_items=False,  # CHANGED: Allow repurchases
        K=10,
        return_stats=True
    )

    print("\n  Evaluation Statistics (Customer Metric):")
    print(f"  ├─ Total GT users: {stats['total_gt_users']:,}")
    print(f"  ├─ Evaluated: {stats['evaluated_users']:,}")
    print(f"  │   ├─ Missing hist: {stats['missing_hist']:,}")
    print(f"  │   └─ Missing pred: {stats['missing_pred']:,}")
    print(f"  ├─ Coverage rate: {stats['coverage_rate']:.2%}")
    print(f"  ├─ Empty relevant after filter: {stats['empty_relevant_after_filter']:,}")
    print(f"  └─ Precision@{stats['K']}: {stats['precision_at_k']:.4f}")

    # ========================================================================
    # 9. Save Results
    # ========================================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save evaluation
    eval_results = {
        'precision_at_10': float(p_at_10),
        'statistics': stats,
        'n_missing_users': len(missing_users),
        'sample_missing_users': [str(u) for u in missing_users[:10]]
    }
    
    eval_path = f"{OUTPUT_DIR}/evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"✓ Evaluation saved: {eval_path}")
    
    # Save predictions
    output_path = f"{OUTPUT_DIR}/predictions.json"
    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in preds.items()}, f, indent=2)
    print(f"✓ Predictions saved: {output_path}")
    print(f"  Total: {len(preds):,} users")
    
    # Feature importance
    print("\n" + "="*60)
    print("TOP 10 FEATURE IMPORTANCE")
    print("="*60)
    
    importance = model.get_booster().get_score(importance_type='gain')
    sorted_importance = sorted(
        importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for feat, score in sorted_importance[:10]:
        feat_idx = int(feat.replace('f', ''))
        feat_name = feats[feat_idx] if feat_idx < len(feats) else feat
        print(f"  {feat_name}: {score:.2f}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Final Precision@10: {p_at_10:.4f}")


if __name__ == "__main__":
    main()