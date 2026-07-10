# Expernetic Data Engineering Internship - Assumptions & Decisions Log

This document logs the critical assumptions made and engineering decisions taken throughout the lifecycle of the Barcelona Airbnb project.

## 1. Data Quality & Ingestion

### Missing Values in Listings
- **Finding**: Some numeric features like `bathrooms`, `bedrooms`, and `beds` had missing values.
- **Decision**: Imputed missing values with the median of the respective feature to maintain robust summary statistics without dropping valuable listings.

### Amenities JSON-like Format
- **Finding**: The `amenities` column in the raw CSV is represented as a JSON-like string of an array (e.g., `["WiFi", "AC"]`).
- **Decision**: Used `ast.literal_eval` coupled with Regex token cleaning to robustly parse these strings into Python lists, which were then one-hot encoded using `MultiLabelBinarizer` for the recommender system.

### Erroneous Price Values
- **Finding**: Certain listings had a price of `0` or unrealistically low/high values (e.g., €1M/night).
- **Decision**: Filtered out listings where `price <= 0`. Target variable was log-transformed (`log1p(price)`) for the predictive model to handle extreme right skewness.

## 2. Schema Design (DuckDB)

### Star Schema Choice
- **Decision**: Implemented a dimensional modeling Star Schema approach (Fact: `fact_listing_performance`, Dimensions: `dim_listing`, `dim_neighbourhood`, `dim_host`).
- **Rationale**: Ensures optimal read performance for analytical queries and Streamlit dashboards while maintaining a clean, understandable structure.

### Demand Segmentation View
- **Decision**: Created `v_listing_demand` to dynamically classify listings into `active`, `dormant_with_history`, and `never_active`.
- **Rationale**: Helps to segment listings based on availability and review activity for targeted business insights.

## 3. Machine Learning (XGBoost Pricing Model)

### Feature Selection & Leakage
- **Decision**: Excluded target-leaking features from the XGBoost model such as future booking potentials or attributes directly derived from price. Features like `minimum_nights`, `room_type`, and `accommodates` were prioritized.

### Algorithmic Bias Identification
- **Finding**: The model systematically underpredicts prices in certain segments, notably Nou Barris (-15.5%) and shared rooms (-14.1%).
- **Decision**: Logged this in the Responsible AI framework. To mitigate business impact in the Streamlit app, we contextualize predictions with percentile ranks and neighborhood medians.

## 4. NLP & Recommender Systems

### Collaborative vs. Content-Based Filtering
- **Finding**: Collaborative filtering is partially applicable given the 1M+ reviews but suffers from a cold-start problem for 23% of "never-active" listings.
- **Decision**: Built a content-based recommender using Cosine Similarity on amenity vectors to ensure robust recommendations regardless of booking history.

## 5. Generative AI

### LLM Selection and Tooling
- **Decision**: Selected `llama-3.3-70b-versatile` via Groq for high-speed, cost-effective inference.
- **Integration**: The LLM is supplied with SHAP values representing the top 5 driving factors for the XGBoost model's prediction, ensuring the generative AI provides highly interpretable, data-driven advice.
