"""
bootstrap_models.py — runs the same feature engineering as notebook 02 and saves
model artefacts to models/ so the Streamlit dashboard works immediately.

This is equivalent to running the 'Save Model Artefacts' cell in notebook 02.
"""

import os

import duckdb
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading data from warehouse …")
con = duckdb.connect("data/warehouse.duckdb", read_only=True)
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
print(f"  Raw shape: {raw.shape}")

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

print("Training XGBoost …")
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X, y)

y_pred = np.expm1(model.predict(X))
mae = mean_absolute_error(df["price"].values, y_pred)
rmse = np.sqrt(mean_squared_error(df["price"].values, y_pred))
print(f"  Train MAE=€{mae:.2f}  RMSE=€{rmse:.2f}")

print("Building SHAP explainer …")
explainer = shap.TreeExplainer(model)

meta = {
    "feature_cols": feature_cols,
    "ng_dummy_cols": list(neighbourhood_dummies.columns),
    "medians": medians,
    "room_type_enc": {"Entire home/apt": 3, "Private room": 2, "Hotel room": 1, "Shared room": 0},
}

joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.joblib"))
joblib.dump(meta, os.path.join(MODELS_DIR, "model_meta.joblib"))

print("\n✅  Saved model artefacts:")
for f in sorted(os.listdir(MODELS_DIR)):
    kb = os.path.getsize(os.path.join(MODELS_DIR, f)) // 1024
    print(f"   {f:<40} {kb:>5} KB")
