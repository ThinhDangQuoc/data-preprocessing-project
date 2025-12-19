import os
import polars as pl
from pathlib import Path
from preprocess import preprocess_pipeline
from datetime import datetime, timedelta

def get_preprocessed_data(
    split_name: str,
    split_config: dict,
    raw_trans: pl.LazyFrame,
    raw_items: pl.LazyFrame,
    raw_users: pl.LazyFrame,
    output_dir: str = "processed_data",
    recreate: bool = False,
    sample_rate: float = 1.0,
    verbose: bool = True
):
    """
    Checks for existing preprocessed files. 
    If they don't exist (or recreate=True), runs the pipeline and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define file paths based on split name
    trans_path = Path(output_dir) / f"trans_{split_name}.parquet"
    items_path = Path(output_dir) / f"items_{split_name}.parquet"
    users_path = Path(output_dir) / f"users_{split_name}.parquet"
    
    paths_exist = trans_path.exists() and items_path.exists() and users_path.exists()
    
    if not recreate and paths_exist:
        if verbose: print(f"  [Cache] Loading preprocessed {split_name} data from {output_dir}...")
        return (
            pl.scan_parquet(str(trans_path)),
            pl.scan_parquet(str(items_path)),
            pl.scan_parquet(str(users_path))
        )

    if verbose: print(f"  [Pipeline] Recreating {split_name} preprocessed data...")
    
    # Run the user's provided logic
    trans_clean, items_quality, users_sampled = preprocess_pipeline(
        raw_trans, raw_items, raw_users,
        split_config, sample_rate, verbose
    )
    
    # Materialize and Save
    if verbose: print(f"  [IO] Saving processed files to {output_dir}...")
    
    # We collect and write here to ensure the "Preprocessing" stage is finished
    trans_clean.collect().write_parquet(trans_path)
    items_quality.collect().write_parquet(items_path)
    users_sampled.collect().write_parquet(users_path)
    
    return (
        pl.scan_parquet(str(trans_path)),
        pl.scan_parquet(str(items_path)),
        pl.scan_parquet(str(users_path))
    )
    
def sample_train_pairs_lazy(
    candidates: pl.LazyFrame,
    trans_clean: pl.LazyFrame,
    *,
    target_start: datetime,
    target_end: datetime,
    n_neg_per_user: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> pl.LazyFrame:
    """
    Input: candidates after anti-join, may already contain stage-1 score cols.
    Output: small LF with customer_id, item_id, Y (pos + sampled neg per user)
    """

    # Targets (positives)
    targets = (
        trans_clean
        .filter((pl.col("created_date") >= target_start) & (pl.col("created_date") <= target_end))
        .select(["customer_id", "item_id"])
        .unique()
        .with_columns(pl.lit(1).cast(pl.UInt8).alias("Y"))
    )

    # Label candidates
    cand_labeled = (
        candidates
        .join(targets, on=["customer_id", "item_id"], how="left")
        .with_columns(pl.col("Y").fill_null(0).cast(pl.UInt8))
    )

    # Keep only users with >=1 positive
    pos_users = (
        cand_labeled
        .filter(pl.col("Y") == 1)
        .select("customer_id")
        .unique()
    )
    cand_labeled = cand_labeled.join(pos_users, on="customer_id", how="inner")

    pos_df = cand_labeled.filter(pl.col("Y") == 1)
    neg_df = cand_labeled.filter(pl.col("Y") == 0)

    # Random per-user negative sampling (Polars-compatible)
    neg_sampled = (
        neg_df
        .with_columns(pl.arange(0, pl.len()).shuffle(seed=seed).alias("__rnd"))
        .sort("__rnd")
        .group_by("customer_id", maintain_order=True)
        .head(n_neg_per_user)
        .drop("__rnd")
    )

    sampled = (
        pl.concat([pos_df, neg_sampled], how="vertical")
        .sort(["customer_id", "Y"], descending=[False, True])
    )

    if verbose:
        s = sampled.select([
            pl.len().alias("rows"),
            pl.col("customer_id").n_unique().alias("users"),
            pl.col("Y").sum().alias("pos")
        ]).collect()
        rows = int(s["rows"][0]); users = int(s["users"][0]); pos = int(s["pos"][0])
        print(f"  [Train Fast] Sampled users={users:,} rows={rows:,} pos={pos:,} avg/group={rows/users:.2f}")

    return sampled
