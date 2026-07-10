-- 1. Top 10 neighbourhoods by median price (active listings, n>=10)
SELECT 
    neighbourhood_name,
    COUNT(*) AS listing_count,
    ROUND(MEDIAN(price), 2) AS median_price,
    ROUND(AVG(price), 2) AS avg_price
FROM fact_listing_performance f
JOIN dim_listing l ON f.listing_id = l.listing_id
JOIN v_listing_demand v ON f.listing_id = v.listing_id
WHERE v.demand_segment = 'active' AND l.price_is_valid
GROUP BY neighbourhood_name
HAVING COUNT(*) >= 10
ORDER BY median_price DESC
LIMIT 10;

-- 2. Host concentration analysis
SELECT 
    CASE 
        WHEN listing_count = 1 THEN '1 listing'
        WHEN listing_count BETWEEN 2 AND 5 THEN '2-5 listings'
        WHEN listing_count BETWEEN 6 AND 20 THEN '6-20 listings'
        ELSE '20+ listings'
    END AS host_tier,
    COUNT(*) AS host_count,
    SUM(listing_count) AS total_listings,
    ROUND(SUM(listing_count) * 100.0 / 15293, 1) AS pct_of_market
FROM (
    SELECT host_id, COUNT(*) AS listing_count
    FROM dim_listing
    GROUP BY host_id
) host_counts
GROUP BY host_tier
ORDER BY total_listings DESC;

-- 3. Superhost performance comparison
SELECT 
    h.host_is_superhost,
    COUNT(*) AS listings,
    ROUND(AVG(f.price), 2) AS avg_price,
    ROUND(AVG(f.review_scores_rating), 3) AS avg_rating,
    ROUND(AVG(f.number_of_reviews), 1) AS avg_reviews
FROM dim_host h
JOIN dim_listing l ON h.host_id = l.host_id
JOIN fact_listing_performance f ON l.listing_id = f.listing_id
WHERE h.host_is_superhost IN ('t', 'f')
GROUP BY h.host_is_superhost;

-- 4. Demand segmentation breakdown
SELECT 
    demand_segment,
    COUNT(*) AS listing_count,
    ROUND(COUNT(*) * 100.0 / 15293, 1) AS pct_of_total
FROM v_listing_demand
GROUP BY demand_segment
ORDER BY listing_count DESC;

-- 5. Seasonal availability by month
SELECT 
    EXTRACT(month FROM date) AS month,
    ROUND(AVG(CASE WHEN available THEN 1.0 ELSE 0.0 END) * 100, 2) AS pct_available,
    COUNT(DISTINCT listing_id) AS active_listings
FROM fact_calendar
GROUP BY month
ORDER BY month;