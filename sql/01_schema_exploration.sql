-- ============================================================================
-- 01_schema_exploration.sql
-- Explore each of the 8 Olist tables: row counts, nulls, key distributions
-- ============================================================================

-- 1. Row counts for all tables
SELECT 'customers'  AS tbl, COUNT(*) AS rows FROM customers
UNION ALL
SELECT 'orders',     COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL
SELECT 'order_payments', COUNT(*) FROM order_payments
UNION ALL
SELECT 'order_reviews', COUNT(*) FROM order_reviews
UNION ALL
SELECT 'products',   COUNT(*) FROM products
UNION ALL
SELECT 'sellers',    COUNT(*) FROM sellers
UNION ALL
SELECT 'geolocation', COUNT(*) FROM geolocation;


-- 2. Check uniqueness of primary keys
-- customers: should be 1 row per customer_id
SELECT 'customers' AS tbl,
       COUNT(*) AS total,
       COUNT(DISTINCT customer_id) AS unique_keys,
       CASE WHEN COUNT(*) = COUNT(DISTINCT customer_id) THEN 'UNIQUE' ELSE 'DUPLICATES' END AS status
FROM customers;

-- orders: should be 1 row per order_id
SELECT 'orders' AS tbl,
       COUNT(*) AS total,
       COUNT(DISTINCT order_id) AS unique_keys,
       CASE WHEN COUNT(*) = COUNT(DISTINCT order_id) THEN 'UNIQUE' ELSE 'DUPLICATES' END AS status
FROM orders;


-- 3. Null analysis on orders table
SELECT
    SUM(CASE WHEN order_approved_at IS NULL THEN 1 ELSE 0 END) AS null_approved,
    SUM(CASE WHEN order_delivered_carrier_date IS NULL THEN 1 ELSE 0 END) AS null_carrier,
    SUM(CASE WHEN order_delivered_customer_date IS NULL THEN 1 ELSE 0 END) AS null_delivered,
    COUNT(*) AS total_rows
FROM orders;


-- 4. Orders per customer distribution
SELECT orders_per_customer, COUNT(*) AS num_customers
FROM (
    SELECT customer_id, COUNT(*) AS orders_per_customer
    FROM orders
    GROUP BY customer_id
) sub
GROUP BY orders_per_customer
ORDER BY orders_per_customer;


-- 5. Items per order distribution
SELECT items_per_order, COUNT(*) AS num_orders
FROM (
    SELECT order_id, COUNT(*) AS items_per_order
    FROM order_items
    GROUP BY order_id
) sub
GROUP BY items_per_order
ORDER BY items_per_order;


-- 6. Payment methods per order
SELECT methods_per_order, COUNT(*) AS num_orders
FROM (
    SELECT order_id, COUNT(DISTINCT payment_type) AS methods_per_order
    FROM order_payments
    GROUP BY order_id
) sub
GROUP BY methods_per_order
ORDER BY methods_per_order;


-- 7. Review score distribution
SELECT review_score, COUNT(*) AS cnt,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct
FROM order_reviews
GROUP BY review_score
ORDER BY review_score;


-- 8. Geolocation duplicates per zip
SELECT dups, COUNT(*) AS num_zips
FROM (
    SELECT geolocation_zip_code_prefix, COUNT(*) AS dups
    FROM geolocation
    GROUP BY geolocation_zip_code_prefix
) sub
GROUP BY dups
ORDER BY dups
LIMIT 10;