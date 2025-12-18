import streamlit as st
import pickle
import json
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Configuration
BASE_DIR = "/workingspace_aiclub/WorkingSpace/Personal/quannh/Project/recommend_system"
TRANSACTIONS_GLOB = f"{BASE_DIR}/dataset/sales_pers.purchase_history_daily_chunk*.parquet"
ITEMS_PATH = f"{BASE_DIR}/dataset/sales_pers.item_chunk_0.parquet"
USERS_GLOB = f"{BASE_DIR}/dataset/sales_pers.user_chunk*.parquet"
GT_PKL_PATH = f"{BASE_DIR}/groundtruth.pkl"
SUBMISSION_PATH = f"{BASE_DIR}/outputs/submission_enhanced_discovery.json"
MODEL_PATH = f"{BASE_DIR}/outputs/xgbranker_enhanced_discovery.pkl"

st.set_page_config(page_title="RecSys Debug Dashboard", layout="wide")

# Cache data loading
@st.cache_data
def load_ground_truth():
    """Load ground truth data"""
    try:
        with open(GT_PKL_PATH, 'rb') as f:
            gt_raw = pickle.load(f)
        return {str(k): v for k, v in gt_raw.items()}
    except Exception as e:
        st.error(f"Error loading ground truth: {e}")
        return {}

@st.cache_data
def load_predictions():
    """Load model predictions"""
    try:
        with open(SUBMISSION_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return {}

@st.cache_data
def load_transactions():
    """Load transaction data"""
    try:
        df = pl.scan_parquet(TRANSACTIONS_GLOB).collect()
        return df
    except Exception as e:
        st.error(f"Error loading transactions: {e}")
        return pl.DataFrame()

@st.cache_data
def load_items():
    """Load item catalog"""
    try:
        df = pl.scan_parquet(ITEMS_PATH).collect()
        return df
    except Exception as e:
        st.error(f"Error loading items: {e}")
        return pl.DataFrame()

def get_user_history(customer_id, transactions, end_date, n_recent=20):
    """Get user's recent purchase history"""
    history = (
        transactions
        .filter(pl.col("customer_id") == customer_id)
        .filter(pl.col("created_date") <= end_date)
        .sort("created_date", descending=True)
        .head(n_recent)
    )
    return history

def get_item_features(item_ids, transactions, items, end_date):
    """Calculate item features for display"""
    if len(item_ids) == 0:
        return pl.DataFrame()
    
    # Global stats
    item_stats = (
        transactions
        .filter(pl.col("created_date") <= end_date)
        .filter(pl.col("item_id").is_in(item_ids))
        .group_by("item_id")
        .agg([
            pl.len().alias("total_purchases"),
            pl.col("customer_id").n_unique().alias("unique_buyers"),
            pl.col("price").mean().alias("avg_price"),
            pl.col("created_date").max().alias("last_purchase_date"),
        ])
    )
    
    # Join with item catalog if available
    if not items.is_empty() and "item_id" in items.columns:
        item_stats = item_stats.join(items, on="item_id", how="left")
    
    return item_stats

def calculate_metrics(predictions, ground_truth, k_values=[5, 10]):
    """Calculate precision@K metrics"""
    metrics = {}
    
    for k in k_values:
        precisions = []
        for user_id, gt_items in ground_truth.items():
            if user_id not in predictions:
                continue
            
            # Get ground truth items
            if isinstance(gt_items, dict):
                gt_set = set(gt_items.get('list_items', []))
            else:
                gt_set = set(gt_items)
            
            if len(gt_set) == 0:
                continue
            
            # Get predictions
            pred_items = predictions[user_id][:k]
            
            # Calculate precision
            hits = len(set(pred_items) & gt_set)
            precision = hits / k
            precisions.append(precision)
        
        metrics[f'P@{k}'] = np.mean(precisions) if precisions else 0.0
        metrics[f'Coverage@{k}'] = len([p for p in precisions if p > 0]) / len(precisions) if precisions else 0.0
    
    return metrics

# Main UI
st.title("🎯 Recommendation System Debug Dashboard")
st.markdown("---")

# Load data
with st.spinner("Loading data..."):
    gt_dict = load_ground_truth()
    predictions = load_predictions()
    transactions = load_transactions()
    items = load_items()

if not gt_dict:
    st.error("Could not load ground truth data. Please check the file path.")
    st.stop()

# Sidebar - User Selection
st.sidebar.header("🔍 User Selection")

# Get list of users
user_ids = list(gt_dict.keys())
st.sidebar.info(f"Total users in GT: {len(user_ids)}")

# Random user button
if st.sidebar.button("🎲 Select Random User", use_container_width=True):
    st.session_state.selected_user = np.random.choice(user_ids)

# Manual user selection
selected_user = st.sidebar.selectbox(
    "Or select a user:",
    options=[""] + user_ids,
    index=0 if "selected_user" not in st.session_state else user_ids.index(st.session_state.selected_user) + 1,
    key="user_selector"
)

if selected_user:
    st.session_state.selected_user = selected_user

# Display user analysis if selected
if "selected_user" in st.session_state and st.session_state.selected_user:
    user_id = st.session_state.selected_user
    
    st.header(f"📊 Analysis for Customer: `{user_id}`")
    
    # Define date windows
    test_history_end = datetime(2024, 12, 30)
    test_history_start = datetime(2024, 12, 1)
    
    # Get user data
    user_history = get_user_history(user_id, transactions, test_history_end, n_recent=50)
    
    # Get ground truth
    gt_items = gt_dict.get(user_id, {})
    if isinstance(gt_items, dict):
        gt_items_list = gt_items.get('list_items', [])
    else:
        gt_items_list = gt_items
    
    # Get predictions
    pred_items = predictions.get(user_id, [])
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Recent Purchases", len(user_history))
    with col2:
        st.metric("Ground Truth Items", len(gt_items_list))
    with col3:
        st.metric("Predicted Items", len(pred_items))
    
    st.markdown("---")
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛒 Purchase History", 
        "🎯 Model Predictions", 
        "✅ Ground Truth", 
        "📈 Performance Analysis"
    ])
    
    with tab1:
        st.subheader("Recent Purchase History")
        
        if not user_history.is_empty():
            # Get item features for history
            history_items = user_history["item_id"].to_list()
            history_features = get_item_features(history_items, transactions, items, test_history_end)
            
            # Merge with user history
            display_history = user_history.join(history_features, on="item_id", how="left")
            
            # Display table
            st.dataframe(
                display_history.select([
                    "created_date", "item_id", "quantity", "price",
                    "total_purchases", "unique_buyers", "avg_price"
                ]).sort("created_date", descending=True),
                use_container_width=True,
                height=400
            )
            
            # Summary stats
            st.markdown("### 📊 Purchase Patterns")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_spent = user_history["price"].sum()
                st.metric("Total Spent", f"${total_spent:,.2f}")
            
            with col2:
                avg_price = user_history["price"].mean()
                st.metric("Avg. Price", f"${avg_price:,.2f}")
            
            with col3:
                unique_items = user_history["item_id"].n_unique()
                st.metric("Unique Items", unique_items)
        else:
            st.warning("No purchase history found for this user in the analysis window.")
    
    with tab2:
        st.subheader("Top 10 Model Predictions")
        
        if pred_items:
            # Get features for predicted items
            pred_features = get_item_features(pred_items[:10], transactions, items, test_history_end)
            
            # Check which predictions are in ground truth
            pred_df = pl.DataFrame({
                "rank": range(1, len(pred_items[:10]) + 1),
                "item_id": pred_items[:10]
            })
            
            # Join with features
            pred_display = pred_df.join(pred_features, on="item_id", how="left")
            
            # Add ground truth indicator
            pred_display = pred_display.with_columns([
                pl.col("item_id").is_in(gt_items_list).alias("in_ground_truth")
            ])
            
            # Add user history indicator
            if not user_history.is_empty():
                hist_items = user_history["item_id"].to_list()
                pred_display = pred_display.with_columns([
                    pl.col("item_id").is_in(hist_items).alias("already_purchased")
                ])
            
            # Display with highlighting
            st.dataframe(
                pred_display,
                use_container_width=True,
                height=400,
                column_config={
                    "in_ground_truth": st.column_config.CheckboxColumn(
                        "✓ In GT",
                        help="Item is in ground truth",
                    ),
                    "already_purchased": st.column_config.CheckboxColumn(
                        "🔁 Repurchase",
                        help="User already purchased this item",
                    ),
                }
            )
            
            # Show matches
            matches = set(pred_items[:10]) & set(gt_items_list)
            if matches:
                st.success(f"✅ {len(matches)} predictions match ground truth: {matches}")
            else:
                st.error("❌ No predictions match ground truth in top 10")
            
            # Check for repurchases
            if not user_history.is_empty():
                repurchases = set(pred_items[:10]) & set(hist_items)
                if repurchases:
                    st.info(f"🔁 {len(repurchases)} predictions are repurchases: {repurchases}")
        else:
            st.warning("No predictions found for this user.")
    
    with tab3:
        st.subheader("Ground Truth Items")
        
        if gt_items_list:
            # Get features for GT items
            gt_features = get_item_features(gt_items_list, transactions, items, test_history_end)
            
            # Check which GT items were predicted
            gt_df = pl.DataFrame({"item_id": gt_items_list})
            gt_display = gt_df.join(gt_features, on="item_id", how="left")
            
            # Add prediction indicator
            gt_display = gt_display.with_columns([
                pl.col("item_id").is_in(pred_items[:10]).alias("predicted_top10"),
                pl.col("item_id").is_in(pred_items[:20]).alias("predicted_top20")
            ])
            
            st.dataframe(
                gt_display,
                use_container_width=True,
                height=400,
                column_config={
                    "predicted_top10": st.column_config.CheckboxColumn(
                        "✓ Top 10",
                        help="Item was predicted in top 10",
                    ),
                    "predicted_top20": st.column_config.CheckboxColumn(
                        "✓ Top 20",
                        help="Item was predicted in top 20",
                    ),
                }
            )
            
            # Summary
            in_top10 = sum([1 for item in gt_items_list if item in pred_items[:10]])
            in_top20 = sum([1 for item in gt_items_list if item in pred_items[:20]])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total GT Items", len(gt_items_list))
            with col2:
                st.metric("Found in Top 10", in_top10)
            with col3:
                st.metric("Found in Top 20", in_top20)
        else:
            st.warning("No ground truth items for this user.")
    
    with tab4:
        st.subheader("Performance Analysis")
        
        # Calculate precision for this user
        def calc_user_precision(pred, gt, k):
            if not pred or not gt:
                return 0.0
            return len(set(pred[:k]) & set(gt)) / k
        
        p_at_5 = calc_user_precision(pred_items, gt_items_list, 5)
        p_at_10 = calc_user_precision(pred_items, gt_items_list, 10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precision@5", f"{p_at_5:.2%}")
        with col2:
            st.metric("Precision@10", f"{p_at_10:.2%}")
        
        # Show prediction quality
        st.markdown("### 🎯 Prediction Analysis")
        
        if pred_items and gt_items_list:
            # Find where GT items appear in predictions
            positions = []
            for gt_item in gt_items_list:
                if gt_item in pred_items:
                    positions.append(pred_items.index(gt_item) + 1)
            
            if positions:
                st.success(f"Ground truth items found at positions: {sorted(positions)}")
                st.metric("Best Rank", min(positions))
                st.metric("Worst Rank", max(positions))
            else:
                st.error("None of the ground truth items were predicted.")
        
        # Show item overlap with history
        st.markdown("### 🔄 History vs Predictions")
        
        if not user_history.is_empty() and pred_items:
            hist_items = set(user_history["item_id"].to_list())
            pred_set = set(pred_items[:20])
            
            overlap = len(hist_items & pred_set)
            new_items = len(pred_set - hist_items)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Repurchase Recs", overlap, 
                         help="Items in predictions that user already bought")
            with col2:
                st.metric("Discovery Recs", new_items,
                         help="Items in predictions that are new to user")

else:
    st.info("👈 Please select a user from the sidebar to begin analysis")

# Global metrics in sidebar
st.sidebar.markdown("---")
st.sidebar.header("📊 Overall Metrics")

if predictions and gt_dict:
    metrics = calculate_metrics(predictions, gt_dict, k_values=[5, 10])
    
    st.sidebar.metric("Overall P@5", f"{metrics['P@5']:.2%}")
    st.sidebar.metric("Overall P@10", f"{metrics['P@10']:.2%}")
    st.sidebar.metric("Coverage@5", f"{metrics['Coverage@5']:.2%}")
    st.sidebar.metric("Coverage@10", f"{metrics['Coverage@10']:.2%}")