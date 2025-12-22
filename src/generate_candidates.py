from __future__ import annotations
import polars as pl
from datetime import datetime
from gensim.models import Word2Vec
import os
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


def train_item2vec_model(
    transactions: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    vector_size: int = 64,
    window: int = 5,
    min_count: int = 3,
    epochs: int = 20,
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



def generate_i2v_candidates_optimized(
    model: Word2Vec,
    transactions: pl.LazyFrame,
    target_users: pl.LazyFrame,
    hist_end: datetime,
    n_similar: int = 50,
    top_n_items: int = 10,
    max_candidates_per_user: int = 50,  # ← NEW: Hard limit per user
    verbose: bool = True
) -> pl.LazyFrame:
    """
    MEMORY-OPTIMIZED Item2Vec candidate generation
    
    Key Optimizations:
    1. Limit candidates per user EARLY
    2. Batch similarity lookups
    3. Filter low-scoring candidates
    """
    
    if verbose:
        print("  [Item2Vec] Generating candidates (optimized)...")
    
    try:
        schema = transactions.collect_schema()
    except AttributeError:
        schema = transactions.schema
    
    item_id_type = schema["item_id"]
    
    # Get user recent items
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
    
    # ========================================================================
    # OPTIMIZATION 1: Build similarity lookup efficiently
    # ========================================================================
    unique_source_ids = user_recent_items_df["source_item_id"].unique().to_list()
    vocab = model.wv.key_to_index
    
    if verbose:
        print(f"  [Item2Vec] Computing similarities for {len(unique_source_ids):,} source items...")
    
    # Pre-compute all similarities (faster than per-user)
    similarity_dict = {}
    valid_sources = 0
    
    for src_id in unique_source_ids:
        token = str(src_id)
        if token in vocab:
            try:
                # Get top N similar items
                neighbors = model.wv.most_similar(token, topn=n_similar)
                similarity_dict[src_id] = neighbors
                valid_sources += 1
            except KeyError:
                continue
    
    if not similarity_dict:
        if verbose:
            print("  [Item2Vec] Warning: No valid similarities found")
        return pl.LazyFrame(schema={
            "customer_id": schema["customer_id"], 
            "item_id": item_id_type, 
            "feat_i2v_score": pl.Float32
        })
    
    if verbose:
        print(f"  [Item2Vec] Found similarities for {valid_sources:,}/{len(unique_source_ids):,} items")
    
    # ========================================================================
    # OPTIMIZATION 2: Build similarity DataFrame (vectorized)
    # ========================================================================
    src_list = []
    dst_list = []
    score_list = []
    
    for src_id, neighbors in similarity_dict.items():
        for sim_token, score in neighbors:
            # FILTER: Only keep scores > 0.3 (saves memory)
            if score > 0.3:
                src_list.append(src_id)
                dst_list.append(sim_token)
                score_list.append(score)
    
    if not src_list:
        if verbose:
            print("  [Item2Vec] Warning: No high-quality similarities (score > 0.3)")
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
    
    # ========================================================================
    # OPTIMIZATION 3: Join and aggregate with hard limit
    # ========================================================================
    candidates = (
        user_recent_items_df.lazy()
        .join(sim_df, on="source_item_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg(
            pl.col("similarity").max().alias("feat_i2v_score")
        )
        # CRITICAL: Limit per user BEFORE returning
        .sort(["customer_id", "feat_i2v_score"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .head(max_candidates_per_user)  # ← Hard cap per user
    )
    
    if verbose:
        stats = candidates.select([
            pl.len().alias("total"),
            pl.col("customer_id").n_unique().alias("users"),
            pl.col("feat_i2v_score").mean().alias("avg_score")
        ]).collect()
        print(f"  [Item2Vec] Generated {stats['total'][0]:,} candidates")
        print(f"  [Item2Vec] {stats['users'][0]:,} users, avg score: {stats['avg_score'][0]:.3f}")
    
    return candidates


# ============================================================================
# MAIN INTEGRATION FUNCTION
# ============================================================================


def generate_candidates(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    max_candidates_per_user: int = 150,  # ← INCREASED
    verbose: bool = True
) -> pl.LazyFrame:
    """
    FIXED: Use OLD function architecture (no intent filtering) + MORE candidates
    
    Key Changes:
    1. Keep ALL strategies separate (no premature filtering)
    2. Increase candidate pool sizes
    3. NO anti-join (let repurchases compete naturally)
    4. Let ranker decide what's best
    """
    
    if verbose:
        print(f"\n[Candidate Gen - HIGH RECALL] Target: {max_candidates_per_user}/user")
    
    # Setup
    try:
        schema = transactions.collect_schema()
    except (AttributeError, NotImplementedError):
        sample = transactions.head(1).collect()
        schema = sample.schema
    
    cid_type = schema.get("customer_id", pl.Utf8)
    iid_type = schema.get("item_id", pl.Utf8)
    
    target_users = users.select(pl.col("customer_id")).unique()
    items_cast = items.with_columns(pl.col("item_id").cast(iid_type))
    
    hist = (
        transactions
        .filter(pl.col("created_date").is_between(hist_start, hist_end))
        .with_columns([
            pl.col("item_id").cast(iid_type),
            pl.col("customer_id").cast(cid_type)
        ])
    )
    
    hist_with_meta = hist.join(items_cast, on="item_id", how="left")
    
    # ========================================================================
    # STRATEGY 1: REPURCHASE (KEEP ALL HISTORY - NO FILTERING!)
    # ========================================================================
    if verbose:
        print("  [1/7] Repurchase candidates...")
    
    cands_repurchase = (
        hist
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("purchase_count"),
            pl.col("created_date").max().alias("last_purchase")
        ])
        # FIX: Calculate days_since FIRST
        .with_columns([
            (hist_end - pl.col("last_purchase")).dt.total_days().alias("days_since")
        ])
        # FIX: Then calculate score in a SUBSEQUENT step
        .with_columns([
            (
                pl.col("purchase_count").log1p() * 2.0 +
                pl.when(pl.col("days_since") < 30).then(3.0)
                .when(pl.col("days_since") < 60).then(2.0)
                .otherwise(1.0)
            )
            .cast(pl.Float32)
            .alias("feat_repurchase_score")
        ])
        .select(["customer_id", "item_id", "feat_repurchase_score"])
    )
    
    # ========================================================================
    # STRATEGY 2: POPULAR (INCREASED)
    # ========================================================================
    if verbose:
        print("  [2/7] Popular items...")
    
    popular_items = (
        hist
        .group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(100)  # ← INCREASED from 30
        .with_columns([
            (pl.col("len").log1p() / pl.col("len").log1p().max())
            .cast(pl.Float32)
            .alias("feat_pop_score")
        ])
        .select(["item_id", "feat_pop_score"])
        .collect()  # Small, collect now
    )
    
    cands_global = target_users.join(popular_items.lazy(), how="cross")
    
    # ========================================================================
    # STRATEGY 3: CATEGORY-BASED (INCREASED)
    # ========================================================================
    if verbose:
        print("  [3/7] Category affinity...")
    
    user_top_cats = (
        hist_with_meta
        .filter(pl.col("category_id").is_not_null())
        .group_by(["customer_id", "category_id"])
        .len()
        .sort("len", descending=True)
        .group_by("customer_id", maintain_order=True)
        .head(8)  # ← INCREASED from 5
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
        .filter(pl.col("rank") <= 50)  # ← INCREASED from 30
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
    # STRATEGY 4: COLLABORATIVE FILTERING (INCREASED)
    # ========================================================================
    if verbose:
        print("  [4/7] Collaborative filtering...")
    
    user_recent_items = (
        hist
        .sort(["customer_id", "created_date"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").head(15))  # ← INCREASED from 10
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
        .filter(pl.col("len") >= 1)  # ← Keep threshold at 1
        .with_columns([
            pl.col("len")
            .rank("dense", descending=True)
            .over("item_id")
            .alias("rank"),
            pl.col("len").log1p().cast(pl.Float32).alias("cooccur_score")
        ])
        .filter(pl.col("rank") <= 50)  # ← INCREASED from 25
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
    # STRATEGY 5: TRENDING (INCREASED)
    # ========================================================================
    if verbose:
        print("  [5/7] Trending items...")
    
    recent_date = hist_end - timedelta(days=14)
    trending_items = (
        hist
        .filter(pl.col("created_date") >= recent_date)
        .group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(100)  # ← INCREASED from 50
        .with_columns([
            (pl.col("len").log1p() / pl.col("len").log1p().max())
            .cast(pl.Float32)
            .alias("feat_trend_score")
        ])
        .select(["item_id", "feat_trend_score"])
        .collect()
    )
    
    cands_trending = target_users.join(trending_items.lazy(), how="cross")
    
    # ========================================================================
    # STRATEGY 6: NEW ITEMS (Discovery boost)
    # ========================================================================
    if verbose:
        print("  [6/7] New items...")
    
    new_items = (
        hist
        .group_by("item_id")
        .agg(pl.col("created_date").min().alias("first_sale"))
        .filter((hist_end - pl.col("first_sale")).dt.total_days() <= 45)  # Last 45 days
        .join(
            hist.group_by("item_id").len().rename({"len": "sales"}),
            on="item_id"
        )
        .filter(pl.col("sales") >= 5)  # At least 5 sales (not too niche)
        .sort("sales", descending=True)
        .head(100)
        .with_columns([
            pl.lit(2.0).cast(pl.Float32).alias("feat_new_score")
        ])
        .select(["item_id", "feat_new_score"])
        .collect()
    )
    
    cands_new = target_users.join(new_items.lazy(), how="cross")
    
    # ========================================================================
    # STRATEGY 7: ITEM2VEC (Keep but don't fail pipeline)
    # ========================================================================
    if verbose:
        print("  [7/7] Item2Vec embeddings...")
    
    try:
        i2v_model = train_item2vec_model(
            transactions=transactions,
            hist_start=hist_start,
            hist_end=hist_end,
            verbose=False
        )
        
        cands_i2v = generate_i2v_candidates_optimized(
            model=i2v_model,
            transactions=hist,
            target_users=target_users,
            hist_end=hist_end,
            n_similar=100,  # ← INCREASED from 50
            top_n_items=15,  # ← INCREASED from 10
            max_candidates_per_user=100,
            verbose=False
        )
        
        del i2v_model
        gc.collect()
        
    except Exception as e:
        if verbose:
            print(f"    Item2Vec failed: {e}")
        cands_i2v = pl.LazyFrame({
            "customer_id": [], "item_id": [], "feat_i2v_score": []
        }, schema={
            "customer_id": cid_type, "item_id": iid_type, 
            "feat_i2v_score": pl.Float32
        })
    
    # ========================================================================
    # COMBINE ALL STRATEGIES (NO FILTERING!)
    # ========================================================================
    
    ALL_SCORES = [
        "feat_pop_score",
        "feat_cat_rank_score",
        "feat_cf_score",
        "feat_trend_score",
        "feat_i2v_score",
        "feat_repurchase_score",
        "feat_new_score"
    ]
    
    def standardize(df: pl.LazyFrame, active_score: str) -> pl.LazyFrame:
        exprs = [pl.col("customer_id"), pl.col("item_id")]
        for sc in ALL_SCORES:
            if sc == active_score:
                exprs.append(pl.col(active_score).cast(pl.Float32).alias(sc))
            else:
                exprs.append(pl.lit(0.0, dtype=pl.Float32).alias(sc))
        return df.select(exprs)
    
    all_cands = [
        standardize(cands_global, "feat_pop_score"),
        standardize(cands_category, "feat_cat_rank_score"),
        standardize(cands_collab, "feat_cf_score"),
        standardize(cands_trending, "feat_trend_score"),
        standardize(cands_i2v, "feat_i2v_score"),
        standardize(cands_repurchase, "feat_repurchase_score"),
        standardize(cands_new, "feat_new_score")
    ]
    
    # ========================================================================
    # AGGREGATE AND RANK (Let ranker decide, don't filter by intent!)
    # ========================================================================
    
    candidates = (
        pl.concat(all_cands, how="vertical")
        .group_by(["customer_id", "item_id"])
        .agg([pl.col(sc).max().fill_null(0.0).alias(sc) for sc in ALL_SCORES])
        .with_columns([
            # Simple weighted sum (model will learn better weights)
            (
                pl.col("feat_repurchase_score") * 2.0 +
                pl.col("feat_cf_score") * 2.0 +
                pl.col("feat_cat_rank_score") * 1.5 +
                pl.col("feat_i2v_score") * 2.0 +
                pl.col("feat_new_score") * 1.0 +
                pl.col("feat_trend_score") * 1 +
                pl.col("feat_pop_score") * 1
            )
            .alias("_combined_score")
        ])
        .sort(["customer_id", "_combined_score"], descending=[False, True])
        .group_by("customer_id", maintain_order=True)
        .head(max_candidates_per_user)
        .drop("_combined_score")
    )
    
    if verbose:
        stats = candidates.select([
            pl.len().alias("total"),
            pl.col("customer_id").n_unique().alias("users")
        ]).collect()
        
        total = stats["total"][0]
        users = stats["users"][0]
        
        print(f"\n  ✓ Generated {total:,} candidates for {users:,} users")
        print(f"  ✓ Avg per user: {total/users:.1f}")
    
    return candidates
