"""
data_loader.py — Load, join, and prepare the Olist E-Commerce dataset.

This module handles:
  1. Loading all 8 raw CSV files
  2. Cleaning & deduplicating geolocation data
  3. Aggregating order_items and order_payments to order-level
  4. Joining all tables into a single master_orders DataFrame
  5. Saving the result to data/processed/master_orders.csv

Usage:
    from src.data_loader import load_all_tables, build_master_table
    tables = load_all_tables("data/raw")
    master = build_master_table(tables)
"""

import os
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt


# ── FILE NAMES ──────────────────────────────────────────────────────────────

FILE_MAP = {
    "customers":  "olist_customers_dataset.csv",
    "orders":     "olist_orders_dataset.csv",
    "items":      "olist_order_items_dataset.csv",
    "payments":   "olist_order_payments_dataset.csv",
    "reviews":    "olist_order_reviews_dataset.csv",
    "products":   "olist_products_dataset.csv",
    "sellers":    "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


# ── 1. LOAD RAW TABLES ─────────────────────────────────────────────────────

def load_all_tables(raw_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """Load all 8 Olist CSV files into a dict of DataFrames."""
    tables = {}
    for name, fname in FILE_MAP.items():
        path = os.path.join(raw_dir, fname)
        df = pd.read_csv(path)
        print(f"  Loaded {name:15s} → {df.shape[0]:>7,} rows × {df.shape[1]} cols")
        tables[name] = df
    print(f"\n✅ All {len(tables)} tables loaded.")
    return tables


def explore_table(df: pd.DataFrame, name: str = "Table") -> None:
    """Print a quick profile of a DataFrame (for notebook use)."""
    print(f"\n{'='*60}")
    print(f"  {name}  —  {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"{'='*60}")
    print(f"\nDtypes:\n{df.dtypes}\n")
    print(f"Nulls:\n{df.isnull().sum()}\n")
    print(f"Sample (3 rows):")
    print(df.head(3).to_string())


# ── 2. CLEAN & DEDUPLICATE GEOLOCATION ──────────────────────────────────────

def clean_geolocation(geo: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate geolocation by zip_code_prefix.
    Strategy: take mean lat/lng per zip code to get a single representative point.
    Rationale: multiple entries per zip exist with slightly different coordinates;
    averaging gives a stable centroid.
    """
    geo_dedup = (
        geo.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            geolocation_lat=("geolocation_lat", "mean"),
            geolocation_lng=("geolocation_lng", "mean"),
            geolocation_city=("geolocation_city", "first"),
            geolocation_state=("geolocation_state", "first"),
        )
    )
    print(f"  Geolocation: {geo.shape[0]:,} → {geo_dedup.shape[0]:,} (deduplicated by zip)")
    return geo_dedup


# ── 3. AGGREGATE ORDER ITEMS TO ORDER-LEVEL ─────────────────────────────────

def aggregate_order_items(items: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate order_items to one row per order.
    
    Decisions:
      - price: SUM (total order value)
      - freight_value: SUM (total shipping cost)
      - order_item_id: MAX (count of items, since item_id is sequential 1,2,3…)
      - product_id: FIRST (primary product — used for category lookup)
      - seller_id: FIRST (primary seller)
    """
    agg = (
        items.groupby("order_id", as_index=False)
        .agg(
            order_item_count=("order_item_id", "max"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            avg_item_price=("price", "mean"),
            product_id=("product_id", "first"),   # primary product
            seller_id=("seller_id", "first"),      # primary seller
        )
    )
    agg["freight_ratio"] = (agg["total_freight"] / agg["total_price"]).replace(
        [np.inf, -np.inf], np.nan
    )
    print(f"  Order items: {items.shape[0]:,} → {agg.shape[0]:,} (aggregated to order-level)")
    return agg


# ── 4. AGGREGATE PAYMENTS TO ORDER-LEVEL ────────────────────────────────────

def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate order_payments to one row per order.
    
    Decisions:
      - payment_value: SUM (total paid)
      - payment_installments: MAX (highest installment plan used)
      - payment_type: FIRST (primary payment method)
      - payment_sequential: MAX → renamed to payment_methods_count
    """
    agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            total_payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=("payment_type", "first"),
            payment_methods_count=("payment_sequential", "max"),
        )
    )
    print(f"  Payments: {payments.shape[0]:,} → {agg.shape[0]:,} (aggregated to order-level)")
    return agg


# ── 5. PARSE DATETIME COLUMNS ──────────────────────────────────────────────

DATETIME_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

def parse_order_dates(orders: pd.DataFrame) -> pd.DataFrame:
    """Convert all timestamp columns to datetime."""
    orders = orders.copy()
    for col in DATETIME_COLS:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")
    return orders


# ── 6. HAVERSINE DISTANCE ──────────────────────────────────────────────────

def haversine(lat1, lng1, lat2, lng2):
    """Vectorized haversine distance in km between two coordinate arrays."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


# ── 7. BUILD MASTER TABLE ──────────────────────────────────────────────────

def build_master_table(
    tables: dict[str, pd.DataFrame],
    save_path: str = "data/processed/master_orders.csv",
) -> pd.DataFrame:
    """
    Join all 8 tables into a single master DataFrame at order-level.
    
    Join chain:
      orders
        ← customers    (on customer_id)
        ← items_agg    (on order_id)
        ← payments_agg (on order_id)
        ← reviews       (on order_id)
        ← products      (via items → product_id)
        ← sellers       (via items → seller_id)
        ← geo_customer  (customer zip → lat/lng)
        ← geo_seller    (seller zip → lat/lng)
    """
    print("\n🔧 Building master table...\n")

    # Prepare components
    orders = parse_order_dates(tables["orders"])
    customers = tables["customers"]
    items_agg = aggregate_order_items(tables["items"])
    payments_agg = aggregate_payments(tables["payments"])
    reviews = tables["reviews"][["order_id", "review_score", "review_comment_message"]]
    products = tables["products"]
    sellers = tables["sellers"]
    geo = clean_geolocation(tables["geolocation"])

    # ── Start joining ──

    # 1. Orders ← Customers
    master = orders.merge(customers, on="customer_id", how="left")
    print(f"  + customers  → {master.shape[0]:,} rows")

    # 2. ← Aggregated items
    master = master.merge(items_agg, on="order_id", how="left")
    print(f"  + items_agg  → {master.shape[0]:,} rows")

    # 3. ← Aggregated payments
    master = master.merge(payments_agg, on="order_id", how="left")
    print(f"  + payments   → {master.shape[0]:,} rows")

    # 4. ← Reviews (left join — some orders have no review)
    master = master.merge(reviews, on="order_id", how="left")
    print(f"  + reviews    → {master.shape[0]:,} rows  (nulls = no review yet)")

    # 5. ← Products (via product_id from items aggregation)
    product_cols = [
        "product_id", "product_category_name",
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
        "product_photos_qty", "product_description_length",
        "product_name_lenght",  # note: typo in original dataset
    ]
    product_cols = [c for c in product_cols if c in products.columns]
    master = master.merge(products[product_cols], on="product_id", how="left")
    print(f"  + products   → {master.shape[0]:,} rows")

    # 5b. ← Category translation (Portuguese → English)
    cat_trans = tables["category_translation"]
    master = master.merge(cat_trans, on="product_category_name", how="left")
    master["product_category_name_english"] = (
        master["product_category_name_english"]
        .fillna(master["product_category_name"])  # keep Portuguese if no translation
        .str.replace("_", " ")
        .str.title()
    )
    master.drop(columns=["product_category_name"], inplace=True)
    master.rename(columns={"product_category_name_english": "product_category"}, inplace=True)
    print(f"  + cat_trans  → categories translated to English")

    # 6. ← Sellers
    seller_cols = ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]
    master = master.merge(sellers[seller_cols], on="seller_id", how="left")
    print(f"  + sellers    → {master.shape[0]:,} rows")

    # 7. ← Geolocation for customers
    master = master.merge(
        geo.rename(columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
        })[["customer_zip_code_prefix", "customer_lat", "customer_lng"]],
        on="customer_zip_code_prefix",
        how="left",
    )
    print(f"  + geo_cust   → {master.shape[0]:,} rows")

    # 8. ← Geolocation for sellers
    master = master.merge(
        geo.rename(columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng",
        })[["seller_zip_code_prefix", "seller_lat", "seller_lng"]],
        on="seller_zip_code_prefix",
        how="left",
    )
    print(f"  + geo_seller → {master.shape[0]:,} rows")

    # ── Derived columns ──

    # Delivery time features
    master["delivery_time_days"] = (
        master["order_delivered_customer_date"] - master["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    master["estimated_delivery_days"] = (
        master["order_estimated_delivery_date"] - master["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    master["delivery_delta_days"] = (
        master["order_delivered_customer_date"] - master["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    master["is_late"] = (master["delivery_delta_days"] > 0).astype(int)

    # Seller-customer distance
    mask = master[["customer_lat", "customer_lng", "seller_lat", "seller_lng"]].notna().all(axis=1)
    master.loc[mask, "seller_customer_distance_km"] = haversine(
        master.loc[mask, "customer_lat"],
        master.loc[mask, "customer_lng"],
        master.loc[mask, "seller_lat"],
        master.loc[mask, "seller_lng"],
    )

    # Same state flag
    master["same_state"] = (master["customer_state"] == master["seller_state"]).astype(int)

    # ── Save ──
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    master.to_csv(save_path, index=False)
    print(f"\n✅ Master table saved → {save_path}")
    print(f"   Shape: {master.shape[0]:,} rows × {master.shape[1]} cols")

    return master


# ── CLI ENTRY POINT ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tables = load_all_tables("data/raw")
    master = build_master_table(tables)
    print("\nColumn list:")
    for i, col in enumerate(master.columns, 1):
        print(f"  {i:2d}. {col}")