from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Tuple

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from feature_builder import build_feature_label_table

TRANSACTIONS_GLOB = "recommendation dataset/sales_pers.purchase_history_daily_chunk_*.parquet"
ITEMS_PATH = "recommendation dataset/sales_pers.item_chunk_0.parquet"
USERS_PATH = "recommendation dataset/sales_pers.user_chunk_0.parquet"


@dataclass(frozen=True)
class FeatureWindow:
    history_start: datetime
    history_end: datetime
    recent_start: datetime
    recent_end: datetime


TRAIN_WINDOW = FeatureWindow(
    history_start=datetime(2024, 11, 1),
    history_end=datetime(2024, 12, 5),
    recent_start=datetime(2024, 12, 6),
    recent_end=datetime(2024, 12, 12),
)

TEST_WINDOW = FeatureWindow(
    history_start=datetime(2024, 11, 8),
    history_end=datetime(2024, 12, 12),
    recent_start=datetime(2024, 12, 13),
    recent_end=datetime(2024, 12, 19),
)

RANK_KS = (5, 10, 20)


def _scan_sources() -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    transactions = pl.scan_parquet(TRANSACTIONS_GLOB)
    items = pl.scan_parquet(ITEMS_PATH)
    users = pl.scan_parquet(USERS_PATH)
    return transactions, items, users


def _build_table(window: FeatureWindow, *, transactions: pl.LazyFrame, items: pl.LazyFrame, users: pl.LazyFrame) -> pl.DataFrame:
    return build_feature_label_table(
        transactions,
        items,
        users,
        window.history_start,
        window.history_end,
        window.recent_start,
        window.recent_end,
        transaction_time_col="created_date",
        customer_id_col="user_id",
    ).collect()


def _extract_features(table: pl.DataFrame) -> np.ndarray:
    return table.select(["X_1", "X_2", "X_3"]).to_numpy().astype(np.float64)


def _train_model(table: pl.DataFrame) -> LogisticRegression:
    X = _extract_features(table)
    y = table["Y"].to_numpy()
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model


def _score_table(model: LogisticRegression, table: pl.DataFrame) -> pl.DataFrame:
    scores = model.predict_proba(_extract_features(table))[:, 1]
    return table.with_columns(pl.Series("score", scores))


def _topk_per_user(scored: pl.DataFrame, k: int) -> pl.DataFrame:
    return (
        scored.sort(["X_-1", "score"], descending=[False, True])
        .group_by("X_-1")
        .head(k)
        .with_columns(
            (pl.int_range(0, pl.len()).over("X_-1") + 1).alias("rank")
        )
    )


def _precision_at_k(topk: pl.DataFrame) -> float:
    per_user = (
        topk.group_by("X_-1")
        .agg(
            pl.len().alias("retrieved"),
            pl.col("Y").sum().alias("hits"),
        )
        .with_columns(
            (pl.col("hits") / pl.col("retrieved")).alias("precision")
        )
    )
    return float(per_user["precision"].mean())


def _ndcg_at_k(scored: pl.DataFrame, topk: pl.DataFrame, k: int) -> float:
    dcg = (
        topk.with_columns(
            (
                pl.col("Y")
                / (
                    (pl.col("rank") + 1)
                    .cast(pl.Float64)
                    .log()
                    / np.log(2.0)
                )
            ).alias("dcg_contrib")
        )
        .group_by("X_-1")
        .agg(pl.sum("dcg_contrib").alias("dcg"))
    )

    positives_per_user = (
        scored.filter(pl.col("Y") == 1)
        .group_by("X_-1")
        .agg(pl.len().alias("pos_count"))
    )

    metrics = (
        dcg.join(positives_per_user, on="X_-1", how="left")
        .with_columns(pl.col("pos_count").fill_null(0).alias("pos_count"))
        .with_columns(
            pl.min_horizontal(pl.col("pos_count"), pl.lit(k)).alias("ideal_hits")
        )
    )

    ideal_hits = list(range(0, k + 1))
    idcg_values = [0.0]
    acc = 0.0
    for i in range(1, k + 1):
        acc += 1.0 / np.log2(i + 1)
        idcg_values.append(acc)
    idcg_table = pl.DataFrame({"ideal_hits": ideal_hits, "idcg": idcg_values})

    metrics = metrics.join(idcg_table, on="ideal_hits", how="left")
    scored_users = metrics.filter(pl.col("idcg") > 0).with_columns(
        (pl.col("dcg") / pl.col("idcg")).alias("ndcg")
    )
    if scored_users.height == 0:
        return 0.0
    return float(scored_users["ndcg"].mean())


def evaluate_ranking(scored: pl.DataFrame, ks: Iterable[int]) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    for k in ks:
        topk = _topk_per_user(scored, k)
        results[k] = {
            "precision": _precision_at_k(topk),
            "ndcg": _ndcg_at_k(scored, topk, k),
        }
    return results


def main() -> None:
    transactions, items, users = _scan_sources()

    train_table = _build_table(
        TRAIN_WINDOW, transactions=transactions, items=items, users=users
    )
    test_table = _build_table(
        TEST_WINDOW, transactions=transactions, items=items, users=users
    )

    print(f"Train table: {train_table.shape}, positives={int(train_table['Y'].sum())}")
    print(f"Validation table: {test_table.shape}, positives={int(test_table['Y'].sum())}")

    model = _train_model(train_table)
    scored_test = _score_table(model, test_table)

    print("\nTop global predictions (sample):")
    print(
        scored_test.sort("score", descending=True)
        .select(["X_-1", "X_0", "score", "Y"])
        .head(10)
    )

    metrics = evaluate_ranking(scored_test, RANK_KS)
    print("\nRanking metrics against validation labels:")
    for k in RANK_KS:
        vals = metrics[k]
        print(f"  K={k:>2}: precision@K={vals['precision']:.4f}, NDCG@K={vals['ndcg']:.4f}")


if __name__ == "__main__":
    main()
