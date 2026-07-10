import pandas as pd
import numpy as np

DEAD_COLUMNS = [
    'neighborhood_overview', 'host_since', 'host_response_time',
    'host_thumbnail_url', 'host_acceptance_rate', 'host_response_rate',
    'host_verifications', 'neighbourhood', 'host_total_listings_count',
    'host_neighbourhood', 'calendar_updated', 'instant_bookable'
]

def clean_price(series):
    """Convert '$1,234.00' style strings to float."""
    cleaned = (series.astype(str)
               .str.replace(r'[\$,]', '', regex=True)
               .str.strip()
               .replace(['nan', 'None', ''], np.nan))
    return pd.to_numeric(cleaned, errors='coerce')

def clean_listings(input_path, output_path):
    print("Loading raw listings...")
    df = pd.read_csv(input_path, compression='gzip', low_memory=False)
    original_shape = df.shape
    print(f"Original shape: {original_shape}")

    # 1. Drop dead columns
    df = df.drop(columns=[c for c in DEAD_COLUMNS if c in df.columns])
    print(f"Dropped {len(DEAD_COLUMNS)} dead columns")

    # 2. Clean price fields
    price_cols = ['price', 'price_quote_price_per_night', 'price_quote_total_price']
    for col in price_cols:
        if col in df.columns:
            df[col] = clean_price(df[col])

    # 3. Validate price is non-negative
    invalid_price = df[df['price'] < 0]
    print(f"Rows with negative price (will flag, not drop): {len(invalid_price)}")
    df['price_is_valid'] = df['price'] >= 0

    # 4. Validate lat/long are within Barcelona's plausible bounds
    lat_ok = df['latitude'].between(41.2, 41.6)
    lon_ok = df['longitude'].between(1.9, 2.3)
    df['location_is_valid'] = lat_ok & lon_ok
    print(f"Rows with invalid lat/long: {(~df['location_is_valid']).sum()}")

    # 5. Standardize room_type and property_type casing/whitespace
    for col in ['room_type', 'property_type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 6. Parse date fields
    date_cols = ['last_scraped', 'first_review', 'last_review', 'calendar_last_scraped']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 7. Create has_reviews flag (per Finding 2 in decision log)
    df['has_reviews'] = df['number_of_reviews'] > 0

    # 8. Deduplicate on id
    before = len(df)
    df = df.drop_duplicates(subset='id')
    print(f"Removed {before - len(df)} duplicate listing IDs")

    print(f"Final shape: {df.shape}")
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_listings("data/raw/listings.csv.gz", "data/processed/listings_clean.parquet")