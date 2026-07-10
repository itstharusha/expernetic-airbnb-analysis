import duckdb
import pandas as pd

def build_warehouse(db_path="data/warehouse.duckdb"):
    con = duckdb.connect(db_path)

    print("Loading cleaned parquet files into DuckDB...")
    listings = pd.read_parquet("data/processed/listings_clean.parquet")
    calendar = pd.read_parquet("data/processed/calendar_clean.parquet")
    reviews = pd.read_parquet("data/processed/reviews_clean.parquet")
    neighbourhoods = pd.read_csv("data/raw/neighbourhoods.csv")

    con.execute("CREATE OR REPLACE TABLE stg_listings AS SELECT * FROM listings")
    con.execute("CREATE OR REPLACE TABLE stg_calendar AS SELECT * FROM calendar")
    con.execute("CREATE OR REPLACE TABLE stg_reviews AS SELECT * FROM reviews")
    con.execute("CREATE OR REPLACE TABLE stg_neighbourhoods AS SELECT * FROM neighbourhoods")

    print("Building dimension tables...")

    # DIM_HOST — one row per host
    con.execute("""
        CREATE OR REPLACE TABLE dim_host AS
        SELECT DISTINCT
            host_id,
            host_name,
            host_is_superhost,
            host_identity_verified,
            host_has_profile_pic,
            host_listings_count,
            hosts_time_as_host_years,
            hosts_time_as_user_years,
            host_location
        FROM stg_listings
    """)

    # DIM_NEIGHBOURHOOD
    con.execute("""
        CREATE OR REPLACE TABLE dim_neighbourhood AS
        SELECT DISTINCT
            neighbourhood_cleansed AS neighbourhood_name,
            neighbourhood_group_cleansed AS neighbourhood_group
        FROM stg_listings
        WHERE neighbourhood_cleansed IS NOT NULL
    """)

    # DIM_LISTING — descriptive attributes only
    con.execute("""
        CREATE OR REPLACE TABLE dim_listing AS
        SELECT
            id AS listing_id,
            host_id,
            neighbourhood_cleansed AS neighbourhood_name,
            property_type,
            room_type,
            accommodates,
            bathrooms,
            bedrooms,
            beds,
            latitude,
            longitude,
            has_reviews,
            price_is_valid,
            location_is_valid
        FROM stg_listings
    """)

    # DIM_DATE — generated from calendar's date range
    con.execute("""
        CREATE OR REPLACE TABLE dim_date AS
        SELECT DISTINCT
            date,
            EXTRACT(year FROM date) AS year,
            EXTRACT(month FROM date) AS month,
            EXTRACT(dow FROM date) AS day_of_week,
            CASE WHEN EXTRACT(dow FROM date) IN (0, 6) THEN TRUE ELSE FALSE END AS is_weekend
        FROM stg_calendar
    """)

    print("Building fact tables...")

    # FACT_LISTING_PERFORMANCE — one row per listing (grain: listing)
    con.execute("""
        CREATE OR REPLACE TABLE fact_listing_performance AS
        SELECT
            id AS listing_id,
            price,
            number_of_reviews,
            number_of_reviews_ltm,
            reviews_per_month,
            review_scores_rating,
            review_scores_cleanliness,
            review_scores_location,
            review_scores_value,
            estimated_occupancy_l365d,
            estimated_revenue_l365d,
            availability_365,
            minimum_nights,
            maximum_nights,
            calculated_host_listings_count
        FROM stg_listings
    """)
    # DEMAND SEGMENTATION VIEW (Finding 8)
    con.execute("""
        CREATE OR REPLACE VIEW v_listing_demand AS
        SELECT
            listing_id,
            number_of_reviews,
            estimated_occupancy_l365d,
            CASE
                WHEN number_of_reviews = 0 AND estimated_occupancy_l365d = 0 THEN 'never_active'
                WHEN number_of_reviews > 0 AND estimated_occupancy_l365d = 0 THEN 'dormant_with_history'
                ELSE 'active'
            END AS demand_segment
        FROM fact_listing_performance
    """)

    # FACT_CALENDAR — one row per listing per date (grain: listing x date)
    con.execute("""
        CREATE OR REPLACE TABLE fact_calendar AS
        SELECT
            c.listing_id,
            c.date,
            c.available,
            c.minimum_nights,
            c.maximum_nights
        FROM stg_calendar c
        INNER JOIN dim_listing l ON c.listing_id = l.listing_id
    """)

    # FACT_REVIEWS — one row per review (grain: review)
    con.execute("""
        CREATE OR REPLACE TABLE fact_reviews AS
        SELECT
            r.id AS review_id,
            r.listing_id,
            r.date,
            r.reviewer_id,
            r.comment_length
        FROM stg_reviews r
        INNER JOIN dim_listing l ON r.listing_id = l.listing_id
    """)

    print("\n--- Table row counts ---")
    for table in ['dim_host', 'dim_neighbourhood', 'dim_listing', 'dim_date',
                  'fact_listing_performance', 'fact_calendar', 'fact_reviews']:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count:,} rows")

    con.close()
    print(f"\nWarehouse saved to {db_path}")

if __name__ == "__main__":
    build_warehouse()