import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_warehouse import build_warehouse
from src.clean_calendar import clean_calendar
from src.clean_listings import clean_listings
from src.clean_reviews import clean_reviews
from src.logging_config import get_logger

logger = get_logger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EXPERNETIC ETL pipeline.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if outputs exist.")
    parser.add_argument(
        "--db-path", default="data/warehouse.duckdb", help="Path to DuckDB warehouse."
    )
    args = parser.parse_args()

    start_time = time.time()
    logger.info("Starting EXPERNETIC ETL Pipeline")
    
    try:
        logger.info("--- Step 1: Cleaning Listings ---")
        clean_listings(
            input_path="data/raw/listings.csv.gz",
            output_path="data/processed/listings_clean.parquet",
            force=args.force,
        )

        logger.info("--- Step 2: Cleaning Calendar ---")
        clean_calendar(
            input_path="data/raw/calendar.csv.gz",
            output_path="data/processed/calendar_clean.parquet",
            force=args.force,
        )

        logger.info("--- Step 3: Cleaning Reviews ---")
        clean_reviews(
            input_path="data/raw/reviews.csv.gz",
            output_path="data/processed/reviews_clean.parquet",
            listings_path="data/processed/listings_clean.parquet",
            force=args.force,
        )

        logger.info("--- Step 4: Building Warehouse ---")
        build_warehouse(db_path=args.db_path)

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed:.2f} seconds.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
