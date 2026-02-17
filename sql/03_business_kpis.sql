-- ============================================================================
-- 03_business_kpis.sql
-- Key business metrics queries for EDA
-- ============================================================================

-- 1. Overall KPIs
SELECT
    COUNT(DISTINCT order_id)                          AS total_orders,
    COUNT(DISTINCT customer_unique_id)                AS unique_customers,
    ROUND(SUM(total_price), 2)                        AS total_revenue,
    ROUND(AVG(total_price), 2)                        AS avg_order_value,
    ROUND(AVG(review_score), 2)                       AS avg_review_score,
    ROUND(AVG(delivery_time_days), 1)                 AS avg_delivery_days
FROM master_orders
WHERE order_status = 'delivered';


-- 2. Monthly revenue and order trend
SELECT
    strftime('%Y-%m', order_purchase_timestamp)       AS month,
    COUNT(*)                                          AS orders,
    ROUND(SUM(total_price), 2)                        AS revenue,
    ROUND(AVG(total_price), 2)                        AS aov,
    ROUND(AVG(review_score), 2)                       AS avg_review
FROM master_orders
WHERE order_status = 'delivered'
GROUP BY month
ORDER BY month;


-- 3. Review score vs delivery performance
SELECT
    review_score,
    COUNT(*)                                          AS order_count,
    ROUND(AVG(delivery_time_days), 1)                 AS avg_delivery_days,
    ROUND(AVG(delivery_delta_days), 1)                AS avg_delta_days,
    ROUND(100.0 * SUM(is_late) / COUNT(*), 1)        AS pct_late
FROM master_orders
WHERE order_status = 'delivered' AND review_score IS NOT NULL
GROUP BY review_score
ORDER BY review_score;


-- 4. Top 10 states by order volume
SELECT
    customer_state,
    COUNT(*)                                          AS orders,
    ROUND(SUM(total_price), 2)                        AS revenue,
    ROUND(AVG(review_score), 2)                       AS avg_review
FROM master_orders
WHERE order_status = 'delivered'
GROUP BY customer_state
ORDER BY orders DESC
LIMIT 10;


-- 5. Top 10 product categories by revenue
SELECT
    product_category_name,
    COUNT(*)                                          AS orders,
    ROUND(SUM(total_price), 2)                        AS revenue,
    ROUND(AVG(review_score), 2)                       AS avg_review
FROM master_orders
WHERE order_status = 'delivered' AND product_category_name IS NOT NULL
GROUP BY product_category_name
ORDER BY revenue DESC
LIMIT 10;


-- 6. Late delivery impact — negative review rate by lateness bucket
SELECT
    CASE
        WHEN delivery_delta_days <= -10 THEN '10+ days early'
        WHEN delivery_delta_days <= -5  THEN '5-10 days early'
        WHEN delivery_delta_days <= 0   THEN '0-5 days early'
        WHEN delivery_delta_days <= 5   THEN '0-5 days late'
        WHEN delivery_delta_days <= 10  THEN '5-10 days late'
        ELSE '10+ days late'
    END AS delivery_bucket,
    COUNT(*) AS orders,
    ROUND(AVG(review_score), 2) AS avg_review,
    ROUND(100.0 * SUM(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END) / COUNT(*), 1)
        AS pct_negative
FROM master_orders
WHERE order_status = 'delivered'
  AND delivery_delta_days IS NOT NULL
  AND review_score IS NOT NULL
GROUP BY delivery_bucket
ORDER BY MIN(delivery_delta_days);


-- 7. Payment method performance
SELECT
    payment_type,
    COUNT(*)                     AS orders,
    ROUND(AVG(total_price), 2)   AS avg_order_value,
    ROUND(AVG(review_score), 2)  AS avg_review
FROM master_orders
WHERE order_status = 'delivered' AND review_score IS NOT NULL
GROUP BY payment_type
ORDER BY orders DESC;


-- 8. Same-state vs cross-state delivery performance
SELECT
    CASE WHEN same_state = 1 THEN 'Same State' ELSE 'Cross State' END AS delivery_type,
    COUNT(*)                                          AS orders,
    ROUND(AVG(delivery_time_days), 1)                 AS avg_delivery_days,
    ROUND(AVG(review_score), 2)                       AS avg_review,
    ROUND(100.0 * SUM(CASE WHEN review_score <= 2 THEN 1 ELSE 0 END) / COUNT(*), 1)
        AS pct_negative
FROM master_orders
WHERE order_status = 'delivered' AND review_score IS NOT NULL
GROUP BY same_state;