import duckdb
import json

con = duckdb.connect('data/warehouse.duckdb', read_only=True)

# List all tables
tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name").fetchall()
print("=== TABLES ===")
for t in tables:
    print(t[0])

# Show stg_listings columns
print("\n=== stg_listings columns ===")
cols = con.execute("PRAGMA table_info(stg_listings)").fetchall()
for c in cols:
    print(c[1], '->', c[2])

# Check amenities column sample
print("\n=== amenities sample (5 rows) ===")
try:
    rows = con.execute("SELECT amenities FROM stg_listings LIMIT 5").fetchall()
    for r in rows:
        print(repr(r[0][:200]) if r[0] else None)
except Exception as e:
    print(f"Error: {e}")

# Check dim_listing columns  
print("\n=== dim_listing columns ===")
cols2 = con.execute("PRAGMA table_info(dim_listing)").fetchall()
for c in cols2:
    print(c[1], '->', c[2])

# Check fact_listing_performance columns
print("\n=== fact_listing_performance columns ===")
cols3 = con.execute("PRAGMA table_info(fact_listing_performance)").fetchall()
for c in cols3:
    print(c[1], '->', c[2])

# Sample row counts
print("\n=== Row counts ===")
for t_name in ['dim_host', 'dim_listing', 'dim_neighbourhood', 'dim_date', 
               'fact_listing_performance', 'fact_calendar', 'fact_reviews', 'v_listing_demand']:
    try:
        count = con.execute(f"SELECT COUNT(*) FROM {t_name}").fetchone()[0]
        print(f"{t_name}: {count:,}")
    except Exception as e:
        print(f"{t_name}: ERROR - {e}")

# Sample listing data
print("\n=== Sample fact_listing_performance (3 rows) ===")
rows = con.execute("""
    SELECT flp.listing_id, dl.room_type, dl.neighbourhood_name, 
           flp.price, flp.review_scores_rating, flp.estimated_occupancy_l365d
    FROM fact_listing_performance flp 
    JOIN dim_listing dl ON flp.listing_id = dl.listing_id
    WHERE flp.price > 0
    LIMIT 3
""").fetchall()
for r in rows:
    print(r)

con.close()
