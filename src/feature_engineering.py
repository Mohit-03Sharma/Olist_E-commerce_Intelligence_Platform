"""
feature_engineering.py — Transform master_orders into ML-ready features.

This module handles:
  1. Filtering to delivered orders with reviews
  2. Creating seller-level aggregated features
  3. Building all feature groups (delivery, order, payment, product, geo, time)
  4. Encoding categoricals (one-hot + target encoding)
  5. Scaling numerical features
  6. Handling class imbalance (SMOTE)
  7. Train/test split with stratification

Usage:
    from src.feature_engineering import build_feature_matrix
    X_train, X_test, y_train, y_test, feature_names = build_feature_matrix(
        "data/processed/master_orders.csv"
    )
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


# ── 1. SELLER HISTORICAL FEATURES ──────────────────────────────────────────

def build_seller_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate seller-level stats from historical data.
    These represent the seller's track record AT THE TIME of the order.
    
    In production you'd compute these with a time-aware rolling window;
    here we use the full dataset as a proxy.
    """
    seller_stats = (
        df.groupby("seller_id")
        .agg(
            seller_avg_review=("review_score", "mean"),
            seller_order_count=("order_id", "count"),
            seller_avg_delivery_time=("delivery_time_days", "mean"),
            seller_negative_pct=("review_score", lambda x: (x <= 2).mean()),
        )
        .round(4)
    )
    return seller_stats


# ── 2. TARGET ENCODING ─────────────────────────────────────────────────────

def target_encode(train: pd.DataFrame, test: pd.DataFrame, col: str,
                  target: str, smoothing: int = 10) -> tuple:
    """
    Target-encode a categorical column with smoothing to prevent overfitting.
    
    Formula: encoded = (count * cat_mean + smoothing * global_mean) / (count + smoothing)
    """
    global_mean = train[target].mean()
    agg = train.groupby(col)[target].agg(["mean", "count"])
    smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    train_encoded = train[col].map(smooth).fillna(global_mean)
    test_encoded = test[col].map(smooth).fillna(global_mean)
    return train_encoded, test_encoded


# ── 3. MAIN FEATURE BUILDER ────────────────────────────────────────────────

def build_feature_matrix(
    csv_path: str = "data/processed/master_orders.csv",
    test_size: float = 0.2,
    apply_smote: bool = True,
    random_state: int = 42,
):
    """
    Build the complete feature matrix from the master table.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print("🔧 Building feature matrix...\n")

    # ── Load & filter ──
    df = pd.read_csv(csv_path, parse_dates=[
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ])

    # Keep only delivered orders with a review score
    df = df[(df["order_status"] == "delivered") & (df["review_score"].notna())].copy()
    print(f"  Filtered to {len(df):,} delivered orders with reviews")

    # ── Target variable ──
    df["is_negative_review"] = (df["review_score"] <= 2).astype(int)
    neg_rate = df["is_negative_review"].mean()
    print(f"  Target: is_negative_review  |  Positive rate: {neg_rate:.1%}")

    # ── Seller features ──
    seller_stats = build_seller_features(df)
    df = df.merge(seller_stats, on="seller_id", how="left")

    # ── Time features ──
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["is_weekend"] = df["purchase_dayofweek"].isin([5, 6]).astype(int)

    # ── Define feature columns ──
    delivery_feats = [
        "delivery_time_days", "estimated_delivery_days",
        "delivery_delta_days", "is_late",
    ]
    order_feats = [
        "order_item_count", "total_price", "total_freight",
        "freight_ratio", "avg_item_price",
    ]
    payment_feats = [
        "payment_installments", "payment_methods_count",
    ]
    product_feats = [
        "product_weight_g", "product_length_cm", "product_height_cm",
        "product_width_cm", "product_photos_qty", "product_description_length",
    ]
    seller_feats = [
        "seller_avg_review", "seller_order_count",
        "seller_avg_delivery_time", "seller_negative_pct",
    ]
    geo_feats = [
        "seller_customer_distance_km", "same_state",
    ]
    time_feats = [
        "purchase_hour", "purchase_dayofweek", "purchase_month", "is_weekend",
    ]

    numerical_feats = (
        delivery_feats + order_feats + payment_feats +
        product_feats + seller_feats + geo_feats + time_feats
    )

    # Categoricals for one-hot encoding (low cardinality)
    onehot_cats = ["payment_type"]

    # Categoricals for target encoding (high cardinality)
    target_enc_cats = ["product_category", "customer_state", "seller_state"]

    # ── Fill missing numericals ──
    for col in numerical_feats:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ── Fill missing categoricals ──
    for col in onehot_cats + target_enc_cats:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # ── Train/test split (stratified) BEFORE encoding to prevent leakage ──
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df["is_negative_review"],
    )
    print(f"\n  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # ── One-hot encoding ──
    onehot_cols = []
    for col in onehot_cats:
        dummies_train = pd.get_dummies(train_df[col], prefix=col, drop_first=True)
        dummies_test = pd.get_dummies(test_df[col], prefix=col, drop_first=True)
        # Align columns
        for c in dummies_train.columns:
            if c not in dummies_test.columns:
                dummies_test[c] = 0
        dummies_test = dummies_test[dummies_train.columns]
        train_df = pd.concat([train_df, dummies_train], axis=1)
        test_df = pd.concat([test_df, dummies_test], axis=1)
        onehot_cols.extend(dummies_train.columns.tolist())

    # ── Target encoding ──
    target_enc_cols = []
    for col in target_enc_cats:
        enc_name = f"{col}_encoded"
        train_df[enc_name], test_df[enc_name] = target_encode(
            train_df, test_df, col, "is_negative_review"
        )
        target_enc_cols.append(enc_name)

    # ── Assemble feature matrix ──
    feature_names = numerical_feats + onehot_cols + target_enc_cols
    # Remove any features not in DataFrame
    feature_names = [f for f in feature_names if f in train_df.columns]

    X_train = train_df[feature_names].values
    X_test = test_df[feature_names].values
    y_train = train_df["is_negative_review"].values
    y_test = test_df["is_negative_review"].values

    # ── Scale numerical features ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n  Features: {len(feature_names)}")
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"  y_train balance: {y_train.mean():.1%} positive")

    # ── SMOTE for class imbalance ──
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"  After SMOTE: {X_train.shape[0]:,} samples  |  {y_train.mean():.1%} positive")

    print(f"\n✅ Feature matrix ready!")
    return X_train, X_test, y_train, y_test, feature_names


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feat_names = build_feature_matrix()
    print(f"\nFeature list ({len(feat_names)}):")
    for i, f in enumerate(feat_names, 1):
        print(f"  {i:2d}. {f}")