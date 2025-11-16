from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import polars as pl


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
) -> pl.LazyFrame:
    """
    Build the feature-label table as described in the project spec.

    Parameters
    ----------
    transactions:
        lazyframe where each row is a transaction. Must contain customer, item and timestamp columns.
    items:
        item catalog that provides brand / age_group / category lookups.
    users:
        user catalog (used to drop customers that are not modeled).
    begin_hist:
        inclusive start of the historical window used for counting item attributes.
    end_hist:
        inclusive end of the historical window used for counting item attributes.
    begin_recent:
        inclusive start of the recent window used for the label.
    end_recent:
        inclusive end of the recent window used for the label.
    transaction_time_col:
        name of the timestamp column inside ``transactions``.
    customer_id_col:
        name of the customer identifier column shared across tables.
    item_id_col:
        name of the item identifier column shared across tables.
    item_brand_col:
        name of the brand column inside ``items``.
    item_age_group_col:
        name of the age_group column inside ``items``.
    item_category_col:
        name of the category column inside ``items``.
    user_id_col:
        optional override for the column that identifies the customer inside ``users``.
        When omitted the function auto-detects ``customer_id`` or ``user_id``.

    Returns
    -------
    pl.LazyFrame
        Columns: X_-1 (customer_id), X_0 (item_id), X_1 (brand count),
        X_2 (age_group count), X_3 (category count), Y (label).
    """

    _validate_window(begin_hist, end_hist, "Historical window")
    _validate_window(begin_recent, end_recent, "Recent window")

    _ensure_columns_exist(
        transactions,
        [transaction_time_col, customer_id_col, item_id_col],
        "transactions",
    )
    _ensure_columns_exist(items, [item_id_col, item_brand_col, item_age_group_col, item_category_col], "items")
    users_schema = list(users.collect_schema().keys())

    hist_transactions = _build_time_filter(
        transactions, transaction_time_col, begin_hist, end_hist
    )
    recent_transactions = _build_time_filter(
        transactions, transaction_time_col, begin_recent, end_recent
    )

    hist_pairs = (
        hist_transactions
        .select([customer_id_col, item_id_col])
        .unique()
    )
    recent_pairs = (
        recent_transactions
        .select([customer_id_col, item_id_col])
        .unique()
    )
    candidate_pairs = pl.concat([hist_pairs, recent_pairs]).unique()

    resolved_user_key = _resolve_user_key(users, customer_id_col, user_id_col, users_schema)
    if resolved_user_key:
        known_users = (
            users.select(pl.col(resolved_user_key).alias(customer_id_col))
            .unique()
        )
        candidate_pairs = candidate_pairs.join(
            known_users, on=customer_id_col, how="inner"
        )

    brand_key = "_feature_brand"
    age_key = "_feature_age_group"
    category_key = "_feature_category"

    item_attrs = (
        items.select(
            [
                item_id_col,
                pl.coalesce(pl.col(item_brand_col), pl.lit("Unknown")).alias(brand_key),
                pl.coalesce(pl.col(item_age_group_col), pl.lit("Unknown")).alias(age_key),
                pl.coalesce(pl.col(item_category_col), pl.lit("Unknown")).alias(category_key),
            ]
        )
        .unique()
    )

    enriched_hist = (
        hist_transactions.join(item_attrs, on=item_id_col, how="left")
        .with_columns(
            [
                pl.col(brand_key).fill_null("Unknown"),
                pl.col(age_key).fill_null("Unknown"),
                pl.col(category_key).fill_null("Unknown"),
            ]
        )
    )

    brand_counts = (
        enriched_hist.group_by([customer_id_col, brand_key])
        .agg(pl.count().alias("brand_count"))
    )
    age_counts = (
        enriched_hist.group_by([customer_id_col, age_key])
        .agg(pl.count().alias("age_group_count"))
    )
    category_counts = (
        enriched_hist.group_by([customer_id_col, category_key])
        .agg(pl.count().alias("category_count"))
    )

    candidates = (
        candidate_pairs.join(item_attrs, on=item_id_col, how="left")
        .with_columns(
            [
                pl.col(brand_key).fill_null("Unknown"),
                pl.col(age_key).fill_null("Unknown"),
                pl.col(category_key).fill_null("Unknown"),
            ]
        )
    )

    recent_labels = (
        recent_pairs.with_columns(pl.lit(1).alias("Y"))
    )

    feature_table = (
        candidates.join(
            brand_counts,
            left_on=[customer_id_col, brand_key],
            right_on=[customer_id_col, brand_key],
            how="left",
        )
        .join(
            age_counts,
            left_on=[customer_id_col, age_key],
            right_on=[customer_id_col, age_key],
            how="left",
        )
        .join(
            category_counts,
            left_on=[customer_id_col, category_key],
            right_on=[customer_id_col, category_key],
            how="left",
        )
        .join(
            recent_labels,
            on=[customer_id_col, item_id_col],
            how="left",
        )
        .with_columns(
            [
                pl.col("brand_count").fill_null(0).cast(pl.Int64).alias("X_1"),
                pl.col("age_group_count").fill_null(0).cast(pl.Int64).alias("X_2"),
                pl.col("category_count").fill_null(0).cast(pl.Int64).alias("X_3"),
                pl.col("Y").fill_null(0).cast(pl.UInt8).alias("Y"),
            ]
        )
        .select(
            [
                pl.col(customer_id_col).alias("X_-1"),
                pl.col(item_id_col).alias("X_0"),
                "X_1",
                "X_2",
                "X_3",
                "Y",
            ]
        )
    )

    return feature_table


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Timestamp must be ISO-like (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature/label table from lazy datasets.")
    parser.add_argument("--transactions", required=True, help="Polars-friendly glob for transaction parquet.")
    parser.add_argument("--items", required=True, help="Parquet path for items metadata.")
    parser.add_argument("--users", required=True, help="Parquet path for user catalog.")
    parser.add_argument("--begin-hist", required=True, type=_parse_datetime, help="Inclusive start of historical window.")
    parser.add_argument("--end-hist", required=True, type=_parse_datetime, help="Inclusive end of historical window.")
    parser.add_argument("--begin-recent", required=True, type=_parse_datetime, help="Inclusive start of recent window.")
    parser.add_argument("--end-recent", required=True, type=_parse_datetime, help="Inclusive end of recent window.")
    parser.add_argument("--transaction-col", default="timestamp", help="Name of the transaction timestamp column.")
    parser.add_argument("--customer-col", default="customer_id", help="Name of the customer identifier column.")
    parser.add_argument("--item-col", default="item_id", help="Name of the item identifier column.")
    parser.add_argument("--brand-col", default="brand", help="Item brand column.")
    parser.add_argument("--age-col", default="age_group", help="Item age_group column.")
    parser.add_argument("--category-col", default="category", help="Item category column.")
    parser.add_argument("--user-col", default=None, help="Optional override for user id column in users.json.")
    parser.add_argument("--out", help="Path to write the resulting feature table (parquet or csv).")
    parser.add_argument("--preview", type=int, default=20, help="Number of rows to print when no --out provided.")

    args = parser.parse_args()

    transactions = pl.scan_parquet(args.transactions)
    items = pl.scan_parquet(args.items)
    users = pl.scan_parquet(args.users)

    table = build_feature_label_table(
        transactions,
        items,
        users,
        args.begin_hist,
        args.end_hist,
        args.begin_recent,
        args.end_recent,
        transaction_time_col=args.transaction_col,
        customer_id_col=args.customer_col,
        item_id_col=args.item_col,
        item_brand_col=args.brand_col,
        item_age_group_col=args.age_col,
        item_category_col=args.category_col,
        user_id_col=args.user_col,
    ).collect()

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".csv":
            table.write_csv(path)
        else:
            table.write_parquet(path)
        print(f"Feature table written to {path}")
    else:
        print(table.head(args.preview))


if __name__ == "__main__":
    main()

print(pl.read_parquet('features.parquet').head())


'''python feature_builder.py \
  --transactions "recommendation dataset/sales_pers.purchase_history_daily_chunk_*.parquet" \
  --items "recommendation dataset/sales_pers.item_chunk_0.parquet" \
  --users "recommendation dataset/sales_pers.user_chunk_0.parquet" \
  --begin-hist 2000-01-01 --end-hist 2100-01-01 \   
  --begin-recent 2024-04-01 --end-recent 2024-04-30 \
  --transaction-col created_date \
  --customer-col user_id \
  --out features.parquet
'''
