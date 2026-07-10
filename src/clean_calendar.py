import pandas as pd

def clean_calendar(input_path, output_path):
    print("Loading raw calendar...")
    df = pd.read_csv(input_path, compression='gzip', low_memory=False)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # 1. Parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 2. Standardize 'available' to boolean
    if df['available'].dtype == object:
        df['available'] = df['available'].map({'t': True, 'f': False})

    # 3. Validate min/max nights are sane
    print(f"Negative minimum_nights: {(df['minimum_nights'] < 0).sum()}")
    print(f"Negative maximum_nights: {(df['maximum_nights'] < 0).sum()}")

    # 4. Foreign key sanity
    print(f"Unique listing_ids in calendar: {df['listing_id'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # 5. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates(subset=['listing_id', 'date'])
    print(f"Removed {before - len(df)} duplicate (listing_id, date) rows")

    print(f"Final shape: {df.shape}")
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_calendar("data/raw/calendar.csv.gz", "data/processed/calendar_clean.parquet")