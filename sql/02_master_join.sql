-- ============================================================================
-- 02_master_join.sql
-- Master 8-table join: produces one row per order with all relevant info
-- ============================================================================

-- Step 1: Aggregate order_items to order-level
WITH items_agg AS (
    SELECT
        order_id,
        MAX(order_item_id)                AS order_item_count,
        SUM(price)                        AS total_price,
        SUM(freight_value)                AS total_freight,
        AVG(price)                        AS avg_item_price,
        SUM(freight_value) / NULLIF(SUM(price), 0) AS freight_ratio,
        -- Take first product/seller as "primary"
        MIN(product_id)                   AS product_id,
        MIN(seller_id)                    AS seller_id
    FROM order_items
    GROUP BY order_id
),

-- Step 2: Aggregate payments to order-level
payments_agg AS (
    SELECT
        order_id,
        SUM(payment_value)                AS total_payment_value,
        MAX(payment_installments)         AS payment_installments,
        MIN(payment_type)                 AS payment_type,
        MAX(payment_sequential)           AS payment_methods_count
    FROM order_payments
    GROUP BY order_id
),

-- Step 3: Deduplicate geolocation (mean lat/lng per zip)
geo_dedup AS (
    SELECT
        geolocation_zip_code_prefix,
        AVG(geolocation_lat)              AS geo_lat,
        AVG(geolocation_lng)              AS geo_lng,
        MIN(geolocation_city)             AS geo_city,
        MIN(geolocation_state)            AS geo_state
    FROM geolocation
    GROUP BY geolocation_zip_code_prefix
)

-- Step 4: Master join
SELECT
    -- Order info
    o.order_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,

    -- Customer info
    c.customer_unique_id,
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state,

    -- Items (aggregated)
    ia.order_item_count,
    ia.total_price,
    ia.total_freight,
    ia.avg_item_price,
    ia.freight_ratio,

    -- Payments (aggregated)
    pa.total_payment_value,
    pa.payment_installments,
    pa.payment_type,
    pa.payment_methods_count,

    -- Reviews
    r.review_score,
    r.review_comment_message,

    -- Product info (primary product)
    p.product_category_name,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,
    p.product_photos_qty,
    p.product_description_length,

    -- Seller info
    s.seller_zip_code_prefix,
    s.seller_city,
    s.seller_state,

    -- Customer geolocation
    gc.geo_lat                           AS customer_lat,
    gc.geo_lng                           AS customer_lng,

    -- Seller geolocation
    gs.geo_lat                           AS seller_lat,
    gs.geo_lng                           AS seller_lng,

    -- Derived: same state?
    CASE WHEN c.customer_state = s.seller_state THEN 1 ELSE 0 END AS same_state,

    -- Derived: delivery time (days)
    JULIANDAY(o.order_delivered_customer_date) - JULIANDAY(o.order_purchase_timestamp)
        AS delivery_time_days,

    -- Derived: delivery delta (positive = late)
    JULIANDAY(o.order_delivered_customer_date) - JULIANDAY(o.order_estimated_delivery_date)
        AS delivery_delta_days,

    -- Derived: is late?
    CASE
        WHEN JULIANDAY(o.order_delivered_customer_date) > JULIANDAY(o.order_estimated_delivery_date)
        THEN 1 ELSE 0
    END AS is_late

FROM orders o

LEFT JOIN customers c
    ON o.customer_id = c.customer_id

LEFT JOIN items_agg ia
    ON o.order_id = ia.order_id

LEFT JOIN payments_agg pa
    ON o.order_id = pa.order_id

LEFT JOIN order_reviews r
    ON o.order_id = r.order_id

LEFT JOIN products p
    ON ia.product_id = p.product_id

LEFT JOIN sellers s
    ON ia.seller_id = s.seller_id

LEFT JOIN geo_dedup gc
    ON c.customer_zip_code_prefix = gc.geolocation_zip_code_prefix

LEFT JOIN geo_dedup gs
    ON s.seller_zip_code_prefix = gs.geolocation_zip_code_prefix;