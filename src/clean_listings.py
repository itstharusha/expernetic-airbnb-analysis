import os

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

DEAD_COLUMNS = [
    "neighborhood_overview",
    "host_since",
    "host_response_time",
    "host_thumbnail_url",
    "host_acceptance_rate",
    "host_response_rate",
    "host_verifications",
    "neighbourhood",
    "host_total_listings_count",
    "host_neighbourhood",
    "calendar_updated",
    "instant_bookable",
]


def clean_price(series: pd.Series) -> pd.Series:
    """
    Convert '$1,234.00' style strings to float.

    Args:
        series: Pandas Series containing price strings.

    Returns:
        Pandas Series with parsed float values.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .replace(["nan", "None", ""], np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_listings(input_path: str, output_path: str, force: bool = False) -> pd.DataFrame:
    """
    Cleans raw listings data and saves to parquet.

    Args:
        input_path: Path to the raw listings CSV.
        output_path: Path to save the processed parquet file.
        force: If True, re-run even if the output file exists.

    Returns:
        Cleaned Pandas DataFrame.
    """
    if not force and os.path.exists(output_path):
        logger.info(f"Output file {output_path} already exists. Skipping clean_listings.")
        return pd.read_parquet(output_path)

    logger.info(f"Loading raw listings from {input_path}...")
    try:
        df = pd.read_csv(input_path, compression="gzip", low_memory=False)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        raise

    original_shape = df.shape
    logger.info(f"Original shape: {original_shape}")

    # 1. Drop dead columns
    cols_to_drop = [c for c in DEAD_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} dead columns")

    # 2. Clean price fields
    price_cols = ["price", "price_quote_price_per_night", "price_quote_total_price"]
    for col in price_cols:
        if col in df.columns:
            df[col] = clean_price(df[col])

    # 3. Validate price is non-negative
    invalid_price = df[df["price"] < 0]
    logger.warning(f"Rows with negative price (will flag, not drop): {len(invalid_price)}")
    df["price_is_valid"] = df["price"] >= 0

    # 4. Validate lat/long are within Barcelona's plausible bounds
    lat_ok = df["latitude"].between(41.2, 41.6)
    lon_ok = df["longitude"].between(1.9, 2.3)
    df["location_is_valid"] = lat_ok & lon_ok
    invalid_locs = (~df["location_is_valid"]).sum()
    logger.warning(f"Rows with invalid lat/long: {invalid_locs}")

    # 5. Standardize room_type and property_type casing/whitespace
    for col in ["room_type", "property_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 6. Parse date fields
    date_cols = ["last_scraped", "first_review", "last_review", "calendar_last_scraped"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 7. Create has_reviews flag (per Finding 2 in decision log)
    df["has_reviews"] = df["number_of_reviews"] > 0

    # 8. Deduplicate on id
    before = len(df)
    df = df.drop_duplicates(subset="id")
    duplicates_removed = before - len(df)
    logger.info(f"Removed {duplicates_removed} duplicate listing IDs")

    logger.info(f"Final shape: {df.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    clean_listings("data/raw/listings.csv.gz", "data/processed/listings_clean.parquet")
