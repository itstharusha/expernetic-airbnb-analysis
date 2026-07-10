import os

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


def clean_calendar(input_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Cleans raw calendar data and saves to parquet.

    Args:
        input_path: Path to the raw calendar CSV.
        output_path: Path to save the processed parquet file.
        force: If True, re-run even if the output file exists.

    Returns:
        Cleaned Pandas DataFrame.
    """
    if not force and os.path.exists(output_path):
        logger.info(f"Output file {output_path} already exists. Skipping clean_calendar.")
        return pd.read_parquet(output_path)

    logger.info(f"Loading raw calendar from {input_path}...")
    try:
        df = pd.read_csv(input_path, compression="gzip", low_memory=False)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        raise

    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # 1. Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 2. Standardize 'available' to boolean
    df['available'] = df['available'].replace({'t': True, 'f': False})

    # 3. Validate min/max nights are sane
    logger.warning(f"Negative minimum_nights: {(df['minimum_nights'] < 0).sum()}")
    logger.warning(f"Negative maximum_nights: {(df['maximum_nights'] < 0).sum()}")

    # 4. Foreign key sanity
    logger.info(f"Unique listing_ids in calendar: {df['listing_id'].nunique()}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # 5. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates(subset=["listing_id", "date"])
    logger.info(f"Removed {before - len(df)} duplicate (listing_id, date) rows")

    logger.info(f"Final shape: {df.shape}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    return df


if __name__ == "__main__":
    clean_calendar("data/raw/calendar.csv.gz", "data/processed/calendar_clean.parquet")
