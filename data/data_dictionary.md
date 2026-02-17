# Data Dictionary — Master Orders Table

## Source
Built by joining all 8 Olist dataset tables into a single order-level table.
See `src/data_loader.py` for implementation and `sql/02_master_join.sql` for SQL equivalent.

## Grain
**One row per order.** Multi-item orders are aggregated (sum of prices, max item count, first product/seller).

---

## Column Reference

### Order Identifiers
| Column | Type | Description |
|--------|------|-------------|
| `order_id` | str | Unique order identifier |
| `order_status` | str | Order status (delivered, shipped, canceled, etc.) |

### Timestamps
| Column | Type | Description |
|--------|------|-------------|
| `order_purchase_timestamp` | datetime | When the customer placed the order |
| `order_approved_at` | datetime | When payment was approved |
| `order_delivered_carrier_date` | datetime | When order was handed to carrier |
| `order_delivered_customer_date` | datetime | When customer received the order |
| `order_estimated_delivery_date` | datetime | Estimated delivery date shown to customer |

### Customer Info
| Column | Type | Description |
|--------|------|-------------|
| `customer_unique_id` | str | Unique customer (deduplicated across orders) |
| `customer_zip_code_prefix` | int | First 5 digits of customer zip code |
| `customer_city` | str | Customer city name |
| `customer_state` | str | Customer state (2-letter code) |
| `customer_lat` | float | Customer latitude (mean of zip centroid) |
| `customer_lng` | float | Customer longitude (mean of zip centroid) |

### Order Items (Aggregated)
| Column | Type | Description |
|--------|------|-------------|
| `order_item_count` | int | Number of items in this order |
| `total_price` | float | Sum of item prices (BRL) |
| `total_freight` | float | Sum of freight/shipping costs (BRL) |
| `avg_item_price` | float | Mean price per item |
| `freight_ratio` | float | total_freight / total_price (shipping burden) |
| `product_id` | str | Product ID of the primary (first) item |
| `seller_id` | str | Seller ID of the primary (first) item |

### Payment (Aggregated)
| Column | Type | Description |
|--------|------|-------------|
| `total_payment_value` | float | Total amount paid (BRL) |
| `payment_installments` | int | Max installments used |
| `payment_type` | str | Primary payment method (credit_card, boleto, voucher, debit_card) |
| `payment_methods_count` | int | Number of distinct payment methods used |

### Review
| Column | Type | Description |
|--------|------|-------------|
| `review_score` | float | Customer rating 1–5 (NaN if no review submitted) |
| `review_comment_message` | str | Free-text review comment (often in Portuguese, many NaNs) |

### Product Info (Primary Product)
| Column | Type | Description |
|--------|------|-------------|
| `product_category_name` | str | Product category (in Portuguese) |
| `product_weight_g` | float | Product weight in grams |
| `product_length_cm` | float | Product length in cm |
| `product_height_cm` | float | Product height in cm |
| `product_width_cm` | float | Product width in cm |
| `product_photos_qty` | float | Number of photos in listing |
| `product_description_length` | float | Character count of product description |
| `product_name_lenght` | float | Character count of product name (typo in original data) |

### Seller Info
| Column | Type | Description |
|--------|------|-------------|
| `seller_zip_code_prefix` | int | First 5 digits of seller zip code |
| `seller_city` | str | Seller city |
| `seller_state` | str | Seller state (2-letter code) |
| `seller_lat` | float | Seller latitude |
| `seller_lng` | float | Seller longitude |

### Derived Features
| Column | Type | Description |
|--------|------|-------------|
| `delivery_time_days` | float | Actual delivery duration (delivered − purchased) |
| `estimated_delivery_days` | float | Promised delivery window (estimated − purchased) |
| `delivery_delta_days` | float | Late/early indicator (delivered − estimated). Negative = early, Positive = late |
| `is_late` | int | Binary flag: 1 if delivered after estimated date |
| `seller_customer_distance_km` | float | Haversine distance between seller and customer (km) |
| `same_state` | int | 1 if seller and customer are in the same state |

---

## Key Decisions

1. **Multi-item orders**: Aggregated to order-level. Prices summed, item count from max(order_item_id), first product/seller kept as "primary."
2. **Multi-payment orders**: Total payment summed, max installments taken, first payment type kept.
3. **Geolocation deduplication**: Mean lat/lng per zip code prefix (multiple rows exist per zip with slight variations).
4. **Missing reviews**: Left join preserves orders without reviews (review_score = NaN). These are excluded from modeling but kept for completeness.
5. **Distance calculation**: Haversine formula on deduplicated geo coordinates. NaN where either lat/lng is missing.
