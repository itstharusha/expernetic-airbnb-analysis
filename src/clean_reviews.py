import os

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


def clean_reviews(
    input_path: str, output_path: str, listings_path: str, force: bool = False
) -> pd.DataFrame:
    """
    Cleans raw reviews data and saves to parquet.

    Args:
        input_path: Path to the raw reviews CSV.
        output_path: Path to save the processed parquet file.
        listings_path: Path to the clean listings parquet (for FK checking).
        force: If True, re-run even if the output file exists.

    Returns:
        Cleaned Pandas DataFrame.
    """
    if not force and os.path.exists(output_path):
        logger.info(f"Output file {output_path} already exists. Skipping clean_reviews.")
        return pd.read_parquet(output_path)

    logger.info(f"Loading raw reviews from {input_path}...")
    try:
        df = pd.read_csv(input_path, compression="gzip", low_memory=False)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        raise

    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # 1. Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 2. Drop rows with null comments (can't analyze empty review text)
    before = len(df)
    df = df.dropna(subset=["comments"])
    logger.info(f"Dropped {before - len(df)} rows with null review text")

    # 3. Drop exact duplicate reviews
    before = len(df)
    df = df.drop_duplicates(subset="id")
    logger.info(f"Removed {before - len(df)} duplicate review IDs")

    # 4. Check foreign key integrity against listings
    if os.path.exists(listings_path):
        listings = pd.read_parquet(listings_path)
        listing_ids = set(listings["id"])
        review_listing_ids = set(df["listing_id"])
        orphans = review_listing_ids - listing_ids
        logger.warning(f"Orphan listing_ids in reviews (no matching listing): {len(orphans)}")
    else:
        logger.warning(f"Listings file not found at {listings_path}, skipping FK check.")

    # 5. Basic text quality flag: very short reviews (likely low-value)
    df["comment_length"] = df["comments"].str.len()
    logger.info(f"Reviews under 10 characters: {(df['comment_length'] < 10).sum()}")

    logger.info(f"Final shape: {df.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    return df


if __name__ == "__main__":
    clean_reviews(
        "data/raw/reviews.csv.gz",
        "data/processed/reviews_clean.parquet",
        "data/processed/listings_clean.parquet",
    )
