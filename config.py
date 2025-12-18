from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

BASE_DIR = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system"
TRANSACTIONS_GLOB = f"{BASE_DIR}/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = f"{BASE_DIR}/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = f"{BASE_DIR}/dataset/sales_pers.user_chunk*.parquet"
GT_PKL_PATH = f"{BASE_DIR}/groundtruth.pkl"
OUTPUT_DIR = f"{BASE_DIR}/outputs_simple"

ALL_SCORES = [
    "feat_pop_score",
    "feat_cat_rank_score",
    "feat_cf_score",
    "feat_trend_score",
    "feat_price_match_score",
    "feat_i2v_score",
]

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