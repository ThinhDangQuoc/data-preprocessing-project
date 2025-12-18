from __future__ import annotations
from datetime import datetime, timedelta
import polars as pl
import numpy as np
from gensim.models import Word2Vec
import warnings

# Import configuration
from config import ALL_SCORES, DataSplit

warnings.filterwarnings('ignore')

# ============================================================================
# ITEM2VEC FUNCTIONS
# ============================================================================

def train_item2vec_model(
    transactions: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    vector_size: int = 48,
    window: int = 3,
    min_count: int = 2,
    epochs: int = 5,
    verbose: bool = True
) -> Word2Vec:
    """Trains a Word2Vec model on user purchase sequences."""
    if verbose:
        print("  [Item2Vec] Preparing sequences...")
    
    sequences = (
        transactions
        .filter(
            (pl.col("created_date") >= hist_start) & 
            (pl.col("created_date") <= hist_end)
        )
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
    
    if verbose:
        print(f"  [Item2Vec] Training Word2Vec on {len(sequences):,} user sequences...")
    
    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        sg=1
    )
    
    if verbose:
        print(f"  [Item2Vec] Vocab size: {len(model.wv.index_to_key):,}")
    
    return model


def generate_i2v_candidates(
    model: Word2Vec,
    transactions: pl.LazyFrame,
    target_users: pl.LazyFrame,
    hist_end: datetime,
    n_similar: int = 20,
    verbose: bool = True
) -> pl.LazyFrame:
    """Generates candidates based on the user's LAST purchased item."""
    if verbose:
        print("  [Item2Vec] Generating candidates...")
    
    vocab = model.wv.index_to_key
    sim_data = []
    
    for item_token in vocab:
        try:
            src_id = int(item_token)
            neighbors = model.wv.most_similar(item_token, topn=n_similar)
            for sim_token, score in neighbors:
                sim_data.append({
                    "last_item_id": src_id,
                    "item_id": int(sim_token),
                    "feat_i2v_score": float(score)
                })
        except (ValueError, KeyError):
            continue
    
    sim_df = pl.DataFrame(sim_data).lazy()
    
    user_last_items = (
        transactions
        .filter(pl.col("created_date") <= hist_end)
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").sort_by("created_date").last())
        .rename({"item_id": "last_item_id"})
    )
    
    target_last_items = target_users.join(user_last_items, on="customer_id", how="inner")
    
    candidates = (
        target_last_items
        .join(sim_df, on="last_item_id", how="inner")
        .select([
            "customer_id",
            "item_id",
            pl.col("feat_i2v_score").cast(pl.Float32)
        ])
    )
    
    return candidates

# ============================================================================
# CANDIDATE GENERATION HELPER
# ============================================================================

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


def check_stage1_recall(
    candidates_lf: pl.LazyFrame,
    transactions_lf: pl.LazyFrame,
    target_start: datetime,
    target_end: datetime,
    verbose: bool = True
) -> None:
    """Calculates Stage 1 recall"""
    if not verbose:
        return
    
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
    
    if recall < 0.2:
        print(f"  [WARNING] Recall is very low! Add more candidate sources.")


def generate_candidates(
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    users: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    n_popular: int = 100,
    n_category_popular: int = 30,
    n_recent_trending: int = 50,
    verbose: bool = True
) -> pl.LazyFrame:
    """Enhanced candidate generation with scoring"""
    if verbose:
        print(f"  [Stage 1] Generating candidates with scores...")
    
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
    
    # Strategy 1: Global Popularity
    popular_items = (
        hist
        .group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(n_popular)
        .select([
            pl.col("item_id"),
            (pl.col("len").log() + 1.0).cast(pl.Float32).alias("feat_pop_score")
        ])
    )
    cands_global = target_users.join(popular_items, how="cross")
    
    # Strategy 2: Category Popularity
    user_cats = (
        hist_with_meta
        .filter(pl.col("category_id").is_not_null())
        .group_by(["customer_id", "category_id"])
        .len()
        .sort("len", descending=True)
        .group_by("customer_id", maintain_order=True)
        .head(3)
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
    
    # Strategy 3: Collaborative Filtering
    user_items = hist.select(["customer_id", "item_id"]).unique()
    
    item_pairs = (
        user_items
        .join(user_items, on="customer_id", suffix="_right")
        .filter(pl.col("item_id") != pl.col("item_id_right"))
        .group_by(["item_id", "item_id_right"])
        .len()
        .filter(pl.col("len") > 1)
        .with_columns(
            pl.col("len").rank("dense", descending=True).over("item_id").alias("rank")
        )
        .filter(pl.col("rank") <= 15)
        .select([
            pl.col("item_id").alias("source_item"),
            pl.col("item_id_right").alias("target_item"),
            pl.col("len").cast(pl.Float32).alias("feat_cf_score")
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
    
    # Strategy 4: Trending Items
    recent_date = hist_end - timedelta(days=7)
    trending_items = (
        hist
        .filter(pl.col("created_date") >= recent_date)
        .group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(n_recent_trending)
        .select([
            "item_id",
            pl.col("len").cast(pl.Float32).alias("feat_trend_score")
        ])
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
        .head(30)
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
    
    # Strategy 6: Item2Vec
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
        verbose=verbose
    )
    
    all_cands = [
        standardize(cands_global, "feat_pop_score"),
        standardize(cands_category, "feat_cat_rank_score"),
        standardize(cands_collab, "feat_cf_score"),
        standardize(cands_trending, "feat_trend_score"),
        standardize(cands_price, "feat_price_match_score"),
        standardize(cands_i2v, "feat_i2v_score")
    ]
    
    candidates = (
        pl.concat(all_cands, how="vertical")
        .group_by(["customer_id", "item_id"])
        .agg([pl.col(sc).max() for sc in ALL_SCORES])
    )
    
    if verbose:
        n_cands = candidates.select(pl.len()).collect().item()
        print(f"  [Stage 1] Generated {n_cands:,} candidates (incl. Item2Vec)")
    
    return candidates

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features(
    candidates: pl.LazyFrame,
    transactions: pl.LazyFrame,
    items: pl.LazyFrame,
    hist_start: datetime,
    hist_end: datetime,
    verbose: bool = True
) -> pl.LazyFrame:
    """Enhanced feature engineering including discovery features"""
    if verbose:
        print(f"  [Stage 2] Building features (with Discovery)...")
    
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
    
    # User Statistics
    user_stats = (
        hist_trans
        .group_by("customer_id")
        .agg([
            pl.len().alias("user_txn_count"),
            pl.col("item_id").n_unique().alias("user_unique_items"),
            pl.col("price").mean().alias("user_avg_price"),
            pl.col("price").std().alias("user_price_std"),
        ])
        .with_columns(pl.col("user_price_std").fill_null(0))
    )
    
    # Item Statistics
    item_stats = (
        hist_trans
        .group_by("item_id")
        .agg([
            pl.len().alias("item_txn_count"),
            pl.col("customer_id").n_unique().alias("item_unique_users"),
            pl.col("price").mean().alias("item_avg_price"),
        ])
    )
    
    # Category Affinity
    user_category_counts = (
        hist_with_meta
        .filter(pl.col("category_id").is_not_null())
        .group_by(["customer_id", "category_id"])
        .len()
        .with_columns(
            pl.col("len").rank("dense", descending=True).over("customer_id").alias("category_rank")
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
            pl.col("len").fill_null(0).alias("user_category_purchases"),
            pl.col("category_rank").fill_null(999).alias("category_rank_for_user")
        ])
    )
    
    # Niche Score
    optimal_popularity = 50
    item_niche_score = (
        item_stats
        .with_columns([
            (1.0 / (1.0 + (pl.col("item_txn_count").log() - np.log(optimal_popularity)).abs()))
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
        .with_columns(pl.lit(1).alias("cat_previously_bought"))
    )
    
    item_birth_date = (
        hist_trans
        .group_by("item_id")
        .agg(pl.col("created_date").min().alias("first_sale_date"))
    )
    
    # Merge & Compute
    features = (
        candidates
        .join(user_stats, on="customer_id", how="left")
        .join(item_stats, on="item_id", how="left")
        .join(category_affinity, on=["customer_id", "item_id"], how="left")
        .join(item_niche_score, on="item_id", how="left")
        .join(items_cast.select(["item_id", "category_id", "price"]), on="item_id", how="left")
        .join(user_known_cats, on=["customer_id", "category_id"], how="left")
        .join(item_birth_date, on="item_id", how="left")
        .with_columns([
            *[pl.col(sc).fill_null(0) for sc in ALL_SCORES],
            pl.col("user_txn_count").fill_null(0),
            pl.col("user_unique_items").fill_null(0),
            pl.col("user_avg_price").fill_null(0),
            pl.col("user_price_std").fill_null(0),
            pl.col("item_txn_count").fill_null(0),
            pl.col("item_unique_users").fill_null(0),
            pl.col("item_avg_price").fill_null(0),
            pl.col("user_category_purchases").fill_null(0),
            pl.col("category_rank_for_user").fill_null(999),
            pl.col("niche_score").fill_null(0),
            
            (pl.col("item_avg_price") / (pl.col("user_avg_price") + 1))
                .fill_null(0).cast(pl.Float32).alias("price_affinity_ratio"),
            
            ((pl.col("item_avg_price") - pl.col("user_avg_price")).abs() / (pl.col("user_price_std") + 1))
                .fill_null(0).cast(pl.Float32).alias("price_z_score"),
            
            (pl.col("user_category_purchases") / (pl.col("user_txn_count") + 1))
                .fill_null(0).cast(pl.Float32).alias("category_affinity_score"),
            
            (pl.col("category_rank_for_user") <= 3).cast(pl.Float32).alias("is_preferred_category"),
            
            pl.when(pl.col("cat_previously_bought").is_null())
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
                .cast(pl.Float32)
                .alias("feat_is_new_category"),
            
            (pl.lit(hist_end) - pl.col("first_sale_date")).dt.total_days()
                .fill_null(365)
                .cast(pl.Float32)
                .alias("feat_item_age_days"),
            
            ((pl.col("price") - pl.col("user_avg_price")) / (pl.col("user_avg_price") + 1.0))
                .fill_null(0)
                .cast(pl.Float32)
                .alias("feat_price_drift")
        ])
    )
    
    feature_cols = [
        *ALL_SCORES,
        "user_txn_count", "user_unique_items", "user_avg_price", "user_price_std",
        "item_txn_count", "item_unique_users", "item_avg_price", "niche_score",
        "user_category_purchases", "category_rank_for_user", "price_affinity_ratio",
        "price_z_score", "category_affinity_score", "is_preferred_category",
        "feat_is_new_category", "feat_item_age_days", "feat_price_drift"
    ]
    
    output = features.select([
        "customer_id",
        "item_id",
        *[pl.col(feat).cast(pl.Float32).alias(f"X_{i}") for i, feat in enumerate(feature_cols)]
    ])
    
    if verbose:
        print(f"  [Stage 2] Features: {len(feature_cols)}")
    
    return output


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
    
    if sample_users < 1.0:
        users = users.filter(pl.col("customer_id").hash(42) % 100 < int(sample_users * 100))
    
    candidates = generate_candidates(
        trans, items, users,
        split.hist_start, split.hist_end,
        n_popular=100,
        verbose=verbose
    )
    
    if is_train and split.target_start is not None:
        check_stage1_recall(
            candidates_lf=candidates,
            transactions_lf=trans,
            target_start=split.target_start,
            target_end=split.target_end,
            verbose=verbose
        )
    
    features = build_features(
        candidates, trans, items,
        split.hist_start, split.hist_end,
        verbose=verbose
    )
    
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