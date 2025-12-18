from __future__ import annotations
from typing import List, Any, Tuple, Dict
from datetime import datetime
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
import warnings
import gc

warnings.filterwarnings('ignore')

class CollaborativeFilter:
    def __init__(
        self,
        n_factors: int = 50,
        min_support: int = 5,  # Increased slightly to reduce noise
        random_state: int = 42,
        enable_nmf: bool = False,  # Default False for speed
        enable_item_metadata: bool = False,
        diversity_alpha: float = 0.1, 
    ):
        self.n_factors = n_factors
        self.min_support = min_support
        self.random_state = random_state
        self.enable_nmf = enable_nmf
        self.enable_item_metadata = enable_item_metadata
        self.diversity_alpha = diversity_alpha
        
        # Model Artifacts
        self.user_factors: np.ndarray | None = None  # Combined SVD/NMF
        self.item_factors: np.ndarray | None = None  # Combined SVD/NMF
        
        # Mappings
        self.user_map: Dict[Any, int] = {}
        self.item_map: Dict[Any, int] = {}
        self.reverse_item_map: Dict[int, Any] = {}
        
        # History for masking (CSR format for fast row slicing)
        self.user_item_matrix: csr_matrix | None = None
        
        # Metadata
        self.global_mean = 0.0
        self.item_popularity: np.ndarray | None = None
        
    def fit(
        self,
        transactions: pl.LazyFrame,
        items: pl.LazyFrame | None = None,
        begin_date: datetime | None = None,
        end_date: datetime | None = None,
        customer_id_col: str = "customer_id",
        item_id_col: str = "item_id",
        transaction_time_col: str = "created_date",
        quantity_col: str = "quantity"
    ) -> None:
        print(f"\n{'='*80}\n[CF] Training High-Performance Matrix Factorization\n{'='*80}")
        
        # 1. Filter & Weighting (Polars)
        filter_expr = pl.lit(True)
        if begin_date and end_date:
            filter_expr = (
                (pl.col(transaction_time_col) >= begin_date) & 
                (pl.col(transaction_time_col) <= end_date)
            )

        print("  > Aggregating interactions...")
        df_interactions = (
            transactions
            .filter(filter_expr)
            .group_by([customer_id_col, item_id_col])
            .agg([
                pl.count().alias("count"),
                pl.col(quantity_col).sum().alias("qty")
            ])
            # Filter low-support noise
            .filter(pl.col("count") >= 1) 
            .with_columns([
                # Implicit Feedback Formula: 1 + log(counts) + log(qty)
                (1 + pl.col("count").log1p() + pl.col("qty").fill_null(1).log1p()).alias("rating")
            ])
            .collect()
        )
        
        if df_interactions.height == 0:
            print("⚠️ No data found in window.")
            return

        # 2. Optimized Mapping (using Categorical codes)
        print("  > Building sparse matrix...")
        # Sort to ensure deterministic ID mapping
        df_interactions = df_interactions.sort([customer_id_col, item_id_col])
        
        # Extract unique IDs efficiently
        unique_users = df_interactions.select(customer_id_col).unique(maintain_order=True)
        unique_items = df_interactions.select(item_id_col).unique(maintain_order=True)
        
        # Create Maps
        user_ids = unique_users[customer_id_col].to_list()
        item_ids = unique_items[item_id_col].to_list()
        
        self.user_map = {uid: i for i, uid in enumerate(user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(item_ids)}
        self.reverse_item_map = {i: iid for i, iid in enumerate(item_ids)}
        
        n_users = len(user_ids)
        n_items = len(item_ids)
        
        # Map DataFrame to Indices
        # Using join is safer than map_dict for large data
        u_map_df = pl.DataFrame({customer_id_col: user_ids, "u_idx": np.arange(n_users, dtype=np.int32)})
        i_map_df = pl.DataFrame({item_id_col: item_ids, "i_idx": np.arange(n_items, dtype=np.int32)})
        
        mapped_df = (
            df_interactions
            .join(u_map_df, on=customer_id_col, how="inner")
            .join(i_map_df, on=item_id_col, how="inner")
            .select(["u_idx", "i_idx", "rating"])
        )
        
        # Construct CSR Matrix
        row_idx = mapped_df["u_idx"].to_numpy()
        col_idx = mapped_df["i_idx"].to_numpy()
        data = mapped_df["rating"].to_numpy().astype(np.float32)
        
        self.user_item_matrix = csr_matrix((data, (row_idx, col_idx)), shape=(n_users, n_items))
        
        # 3. Discovery Metrics
        self.item_popularity = np.array(self.user_item_matrix.sum(axis=0)).flatten()
        # Normalize popularity for later penalties
        self.item_popularity = self.item_popularity / (self.item_popularity.max() + 1e-6)
        
        del df_interactions, mapped_df, row_idx, col_idx, data
        gc.collect()

        # 4. SVD Factorization (Randomized for speed)
        print(f"  > Decomposing Matrix ({n_users}x{n_items}) with SVD...")
        svd = TruncatedSVD(
            n_components=self.n_factors,
            algorithm='randomized',
            random_state=self.random_state,
            n_iter=5
        )
        self.user_factors = svd.fit_transform(self.user_item_matrix).astype(np.float32)
        self.item_factors = svd.components_.T.astype(np.float32)
        
        print(f"  ✓ Explained Variance: {svd.explained_variance_ratio_.sum():.2%}")
        
        # 5. Optional NMF (Only if small item space & explicitly enabled)
        if self.enable_nmf and n_items < 20000:
            print("  > Adding NMF factors for discovery...")
            try:
                nmf = NMF(
                    n_components=self.n_factors // 2,
                    init='nndsvda',
                    random_state=self.random_state
                )
                W = nmf.fit_transform(self.user_item_matrix).astype(np.float32)
                H = nmf.components_.T.astype(np.float32)
                
                # Concatenate Factors (Normalize first to keep scales aligned)
                self.user_factors = np.hstack([normalize(self.user_factors), normalize(W)])
                self.item_factors = np.hstack([normalize(self.item_factors), normalize(H)])
            except Exception as e:
                print(f"  ⚠️ NMF Failed: {e}")

        # Final cleanup
        # We KEEP user_item_matrix for history masking during inference
        print("✅ CF Model Trained.")

    def recommend_batch(
        self, 
        user_ids: List[Any], 
        n_candidates: int = 50,
        batch_size: int = 1000
    ) -> pl.LazyFrame:
        """
        High-performance vectorized batch recommendation.
        Returns a LazyFrame to fit into the pipeline.
        """
        if self.user_factors is None:
            return pl.DataFrame([], schema={"customer_id": pl.Utf8, "item_id": pl.Int32, "candidate_source": pl.Categorical}).lazy()

        # 1. Filter valid users
        valid_uids = [u for u in user_ids if u in self.user_map]
        if not valid_uids:
            return pl.DataFrame([], schema={"customer_id": pl.Utf8, "item_id": pl.Int32, "candidate_source": pl.Categorical}).lazy()

        # Generator to yield chunks of results
        def process_chunk(chunk_uids):
            chunk_indices = [self.user_map[u] for u in chunk_uids]
            
            # A. Compute Scores: (Batch_Users x Factors) @ (Factors x Items) = (Batch x Items)
            # This is dense matrix multiplication - extremely fast in BLAS
            batch_factors = self.user_factors[chunk_indices]
            scores = np.dot(batch_factors, self.item_factors.T)
            
            # B. Discovery Logic: Penalize extremely popular items slightly
            if self.diversity_alpha > 0:
                # Broadcast subtraction
                scores -= (self.item_popularity * self.diversity_alpha)

            # C. Mask History (Crucial)
            # Slicing CSR matrix rows gives us what they bought
            history_batch = self.user_item_matrix[chunk_indices]
            
            # Set bought items to negative infinity efficiently
            # We iterate through the sparse structure
            for i in range(len(chunk_indices)):
                # Get indices of non-zero elements for this user row
                start_ptr = history_batch.indptr[i]
                end_ptr = history_batch.indptr[i+1]
                cols = history_batch.indices[start_ptr:end_ptr]
                scores[i, cols] = -np.inf

            # D. Vectorized Top-K
            # argpartition puts the top K items at the end of the array (unsorted)
            # This is O(N) instead of O(N log N) for full sort
            top_k_unsorted = np.argpartition(scores, -n_candidates, axis=1)[:, -n_candidates:]
            
            # Map back to IDs
            result_rows = []
            for i, uid in enumerate(chunk_uids):
                # Retrieve item indices for this user
                item_indices = top_k_unsorted[i]
                
                # Optional: Sort them by score if strict ranking matters here
                # (XGBoost will re-rank anyway, so exact order here is less critical, but good for debugging)
                user_scores = scores[i, item_indices]
                sorted_local_idx = np.argsort(user_scores)[::-1]
                sorted_global_idx = item_indices[sorted_local_idx]
                
                for item_idx in sorted_global_idx:
                    # Logic check: if score is -inf, don't recommend
                    if scores[i, item_idx] == -np.inf:
                        continue
                    result_rows.append((uid, self.reverse_item_map[item_idx], "cf_match"))
            
            return result_rows

        # Run batching
        all_results = []
        for i in range(0, len(valid_uids), batch_size):
            chunk = valid_uids[i : i + batch_size]
            all_results.extend(process_chunk(chunk))

        # Return as LazyFrame
        schema = {
            "customer_id": pl.Utf8, # Adjust type if needed
            "item_id": pl.Int32,    # Adjust type if needed
            "candidate_source": pl.Categorical
        }
        
        # Infer types from first row if possible to avoid schema errors
        if all_results:
             # Just strict casting usually works best
            return pl.DataFrame(all_results, schema=["customer_id", "item_id", "candidate_source"], orient="row").lazy().with_columns(pl.col("candidate_source").cast(pl.Categorical))
        else:
            return pl.DataFrame([], schema=schema).lazy()

    def get_similar_items(self, item_id: Any, n_similar: int = 20) -> List[Any]:
        """
        Fast Item-Item similarity using Latent Factors (Dot Product).
        Replaces the O(N^2) memory bomb.
        """
        if self.item_factors is None or item_id not in self.item_map:
            return []
            
        target_idx = self.item_map[item_id]
        target_vector = self.item_factors[target_idx]
        
        # Dot product of target item vs all other items
        # shape: (n_items,)
        scores = np.dot(self.item_factors, target_vector)
        
        # Mask self
        scores[target_idx] = -np.inf
        
        # Top-K
        top_indices = np.argpartition(scores, -n_similar)[-n_similar:]
        
        # Sort top K
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        return [self.reverse_item_map[i] for i in top_indices]