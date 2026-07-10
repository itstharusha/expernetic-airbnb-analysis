from typing import Optional

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def profile_file(
    path: str, name: str, nrows: Optional[int] = None, compression: str = "infer"
) -> pd.DataFrame:
    """
    Profiles a given CSV file by loading it and displaying basic statistics.

    Args:
        path: Path to the CSV file.
        name: Name of the dataset for logging purposes.
        nrows: Number of rows to read (useful for large files).
        compression: Compression type of the file.

    Returns:
        The loaded Pandas DataFrame.
    """
    logger.info(f"\n{'=' * 80}\nPROFILE: {name}\n{'=' * 80}")
    try:
        df = pd.read_csv(path, compression=compression, nrows=nrows, low_memory=False)
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        raise

    logger.info(f"Shape: {df.shape}")
    logger.info(f"\nColumn dtypes:\n{df.dtypes}")
    logger.info(
        f"\nNull counts (top 20):\n{df.isnull().sum().sort_values(ascending=False).head(20)}"
    )
    logger.info(f"\nSample rows:\n{df.head(2)}")
    return df


if __name__ == "__main__":
    listings = profile_file("data/raw/listings.csv.gz", "LISTINGS (detailed)")
    calendar = profile_file("data/raw/calendar.csv.gz", "CALENDAR", nrows=100000)
    reviews = profile_file("data/raw/reviews.csv.gz", "REVIEWS (detailed)", nrows=100000)
    neighbourhoods = profile_file("data/raw/neighbourhoods.csv", "NEIGHBOURHOODS")

    logger.info(f"\n{'=' * 80}\nINVESTIGATING THE 15293 NULL PATTERN\n{'=' * 80}")
    logger.info(f"Total rows in listings: {len(listings)}")
    logger.info(f"Rows missing host_since: {listings['host_since'].isnull().sum()}")

    # Check if these nulls always co-occur
    suspect_cols = ["host_since", "host_response_time", "instant_bookable", "calendar_updated"]
    null_mask = listings[suspect_cols].isnull()
    logger.info("\nDo all 4 columns go null together?")
    logger.info(f"{null_mask.all(axis=1).sum()} rows have ALL 4 null")
    logger.info(f"{null_mask.any(axis=1).sum()} rows have AT LEAST 1 null")

    # Check the 'source' column - often explains this
    if "source" in listings.columns:
        logger.info(f"\nSource value counts overall:\n{listings['source'].value_counts()}")
        logger.info(
            "\nSource value counts WHERE host_since is null:\n"
            f"{listings[listings['host_since'].isnull()]['source'].value_counts()}"
        )
        logger.info(f"\n{'=' * 80}")

    logger.info("FULL NULL-RATE AUDIT — LISTINGS")
    logger.info(f"{'=' * 80}")
    null_pct = (listings.isnull().sum() / len(listings) * 100).sort_values(ascending=False)
    logger.info(f"\n{null_pct.to_string()}")
