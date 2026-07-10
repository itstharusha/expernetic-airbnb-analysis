import pandas as pd

def clean_reviews(input_path, output_path, listings_path):
    print("Loading raw reviews...")
    df = pd.read_csv(input_path, compression='gzip', low_memory=False)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # 1. Parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 2. Drop rows with null comments (can't analyze empty review text)
    before = len(df)
    df = df.dropna(subset=['comments'])
    print(f"Dropped {before - len(df)} rows with null review text")

    # 3. Drop exact duplicate reviews
    before = len(df)
    df = df.drop_duplicates(subset='id')
    print(f"Removed {before - len(df)} duplicate review IDs")

    # 4. Check foreign key integrity against listings
    listings = pd.read_parquet(listings_path)
    listing_ids = set(listings['id'])
    review_listing_ids = set(df['listing_id'])
    orphans = review_listing_ids - listing_ids
    print(f"Orphan listing_ids in reviews (no matching listing): {len(orphans)}")

    # 5. Basic text quality flag: very short reviews (likely low-value)
    df['comment_length'] = df['comments'].str.len()
    print(f"Reviews under 10 characters: {(df['comment_length'] < 10).sum()}")

    print(f"Final shape: {df.shape}")
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_reviews(
        "data/raw/reviews.csv.gz",
        "data/processed/reviews_clean.parquet",
        "data/processed/listings_clean.parquet"
    )