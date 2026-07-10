import argparse
import json
import os
import sys
import time
from datetime import datetime

import duckdb
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.logging_config import get_logger

logger = get_logger(__name__)

MODELS_DIR = "models"
EXPERIMENTS_DIR = "experiments"


def get_data(db_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from warehouse at {db_path}...")
    try:
        con = duckdb.connect(db_path, read_only=True)
        raw = con.execute("""
            SELECT f.listing_id, f.price,
                   l.room_type, l.property_type, l.accommodates,
                   l.bathrooms, l.bedrooms, l.beds,
                   l.neighbourhood_name, n.neighbourhood_group,
                   h.host_is_superhost, h.host_listings_count, h.hosts_time_as_host_years,
                   f.number_of_reviews, f.review_scores_rating,
                   f.review_scores_cleanliness, f.review_scores_location,
                   f.review_scores_value, f.availability_365,
                   f.minimum_nights, f.estimated_occupancy_l365d,
                   v.demand_segment
            FROM fact_listing_performance f
            JOIN dim_listing      l ON f.listing_id = l.listing_id
            JOIN dim_neighbourhood n ON l.neighbourhood_name = n.neighbourhood_name
            JOIN dim_host         h ON l.host_id = h.host_id
            JOIN v_listing_demand v ON f.listing_id = v.listing_id
            WHERE f.price IS NOT NULL AND l.price_is_valid AND v.demand_segment = 'active'
            """).df()
        con.close()
        return raw
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def train_model(args: argparse.Namespace) -> None:
    start_time = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    raw = get_data(args.db_path)
    logger.info(f"Raw shape: {raw.shape}")

    df = raw.copy()
    df["log_price"] = np.log1p(df["price"])
    df["is_superhost"] = (df["host_is_superhost"] == "t").astype(int)
    df["has_rating"] = df["review_scores_rating"].notna().astype(int)

    for col in [
        "review_scores_rating",
        "review_scores_cleanliness",
        "review_scores_location",
        "review_scores_value",
    ]:
        df[col] = df[col].fillna(0)

    IMPUTE_COLS = [
        "bathrooms",
        "bedrooms",
        "beds",
        "host_listings_count",
        "hosts_time_as_host_years",
        "minimum_nights",
    ]
    medians = {col: float(df[col].median()) for col in IMPUTE_COLS}
    for col, med in medians.items():
        df[col] = df[col].fillna(med)

    df["beds_per_person"] = df["beds"] / df["accommodates"].clip(lower=1)
    df["room_type_enc"] = df["room_type"].map(
        {"Entire home/apt": 3, "Private room": 2, "Hotel room": 1, "Shared room": 0}
    )

    neighbourhood_dummies = pd.get_dummies(df["neighbourhood_group"], prefix="ng", drop_first=True)
    df = pd.concat([df, neighbourhood_dummies], axis=1)

    feature_cols = [
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "room_type_enc",
        "minimum_nights",
        "availability_365",
        "number_of_reviews",
        "review_scores_rating",
        "review_scores_cleanliness",
        "review_scores_location",
        "review_scores_value",
        "estimated_occupancy_l365d",
        "host_listings_count",
        "hosts_time_as_host_years",
        "is_superhost",
        "has_rating",
        "beds_per_person",
    ] + list(neighbourhood_dummies.columns)

    X = df[feature_cols].astype(float)
    y = df["log_price"]

    logger.info(
        f"Training XGBoost (seed={args.seed}, estimators={args.n_estimators}, depth={args.max_depth})..."
    )
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)

    y_pred = np.expm1(model.predict(X))
    mae = mean_absolute_error(df["price"].values, y_pred)
    rmse = np.sqrt(mean_squared_error(df["price"].values, y_pred))
    logger.info(f"Train MAE=€{mae:.2f}  RMSE=€{rmse:.2f}")

    logger.info("Building SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    meta = {
        "feature_cols": feature_cols,
        "ng_dummy_cols": list(neighbourhood_dummies.columns),
        "medians": medians,
        "room_type_enc": {
            "Entire home/apt": 3,
            "Private room": 2,
            "Hotel room": 1,
            "Shared room": 0,
        },
    }

    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.joblib"))
    joblib.dump(meta, os.path.join(MODELS_DIR, "model_meta.joblib"))

    logger.info("Saved model artefacts.")

    elapsed = time.time() - start_time

    # Save experiment metadata
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": elapsed,
        "dataset_shape": raw.shape,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "metrics": {"train_mae": mae, "train_rmse": rmse},
    }

    exp_file = os.path.join(EXPERIMENTS_DIR, f"run_{run_id}.json")
    with open(exp_file, "w") as f:
        json.dump(experiment, f, indent=4)

    logger.info(f"Experiment metadata saved to {exp_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model and save artifacts.")
    parser.add_argument(
        "--db-path", default="data/warehouse.duckdb", help="Path to DuckDB warehouse."
    )
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    try:
        train_model(args)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
