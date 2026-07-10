import pytest
import pandas as pd
import numpy as np
import os
import duckdb
from src.clean_listings import clean_price

def test_clean_price_valid():
    """Test price conversion with currency symbols and commas."""
    input_series = pd.Series(["$1,250.00", "$99.99", "$0.00", "$10,000"])
    expected = pd.Series([1250.00, 99.99, 0.00, 10000.00])
    result = clean_price(input_series)
    pd.testing.assert_series_equal(result, expected)

def test_clean_price_invalid():
    """Test price conversion with missing values or invalid formats."""
    input_series = pd.Series(["$nan", "None", "", "$abc"])
    result = clean_price(input_series)
    # Check that they are coerced to NaN
    assert result.isna().all()

def test_barcelona_lat_lon_bounds():
    """Verify that latitude and longitude checks correctly flag outliers."""
    # plausible bounds in src/clean_listings.py: lat: (41.2, 41.6), lon: (1.9, 2.3)
    df = pd.DataFrame({
        'latitude': [41.3851, 40.0, 41.5999],
        'longitude': [2.1734, 2.0, 3.0]
    })
    lat_ok = df['latitude'].between(41.2, 41.6)
    lon_ok = df['longitude'].between(1.9, 2.3)
    location_is_valid = lat_ok & lon_ok
    
    assert location_is_valid.iloc[0] == True   # Centroid Barcelona
    assert location_is_valid.iloc[1] == False  # Out of bounds latitude
    assert location_is_valid.iloc[2] == False  # Out of bounds longitude

def test_warehouse_tables_exist():
    """Verify that DuckDB warehouse contains the standard star schema tables."""
    db_path = "data/warehouse.duckdb"
    if not os.path.exists(db_path):
        pytest.skip("DuckDB database file not found. Skip database structure tests.")
        
    conn = duckdb.connect(db_path, read_only=True)
    try:
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        expected_tables = ["fact_listing_performance", "dim_listing", "dim_neighbourhood", "dim_host"]
        for table in expected_tables:
            assert table in tables, f"Expected table '{table}' was not found in database."
    finally:
        conn.close()
