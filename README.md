# Barcelona Airbnb Market Intelligence Platform

### Enterprise Data Pipeline, Machine Learning, and Generative AI Analytics Engine

*Expernetic Data Engineering Technical Assignment*

---

## Executive Overview

This repository hosts a production-grade, end-to-end data platform built to ingest, clean, store, model, and analyze short-term rental data from the Inside Airbnb Barcelona June 2026 snapshot.

The system implements a classic ELT pipeline leveraging DuckDB as a high-performance, serverless analytical warehouse. It integrates an XGBoost regression model for predictive pricing, a cosine-similarity-based content recommender to address cold-start recommendations, and a Generative AI Dynamic Pricing Advisor powered by Groq to translate model explainability metrics into business-facing strategy. The entire workflow is surfaced through a responsive Streamlit dashboard.

## Submission Readiness Summary

This repository is organized as a submission package for the Expernetic technical assignment. The core deliverables are present and wired together:

- end-to-end ETL pipeline and data warehouse build scripts
- analysis reports and assumptions log in the reports folder
- interactive dashboard and optional AI pricing advisor flow
- automated tests for core cleaning and warehouse expectations

### Verified workflow

```powershell
python -m venv venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
python scripts/run_pipeline.py --force
streamlit run app/streamlit_dashboard.py
```

### Submission checklist

- Source code: present under src/, scripts/, and app/
- Reproducibility: Python environment and dependency instructions included
- Report artifacts: available in reports/
- AI disclosure: covered in the report appendix and submission summary
- Incomplete/optional work: documented honestly rather than overstated

---

## System Architecture and Data Flow

```mermaid
flowchart TD
    subgraph Ingestion and Cleaning
        Raw[Raw CSV gzip files] -->|profile_data.py| Profile[Data Quality Report]
        Raw -->|clean_listings.py| ListingsClean[listings_clean.parquet]
        Raw -->|clean_calendar.py| CalendarClean[calendar_clean.parquet]
        Raw -->|clean_reviews.py| ReviewsClean[reviews_clean.parquet]
    end

    subgraph Analytical Warehouse
        ListingsClean & CalendarClean & ReviewsClean -->|build_warehouse.py| DuckDB[(data/warehouse.duckdb)]
        DuckDB --> StarSchema[Star Schema: Fact and Dimensions]
    end

    subgraph Modeling and ML
        StarSchema -->|02_price_prediction.ipynb| XGBoost[XGBoost Price Regressor]
        XGBoost -->|SHAP Explainer| SHAP[SHAP Feature Impact]
        StarSchema -->|04_recommender.ipynb| Recommender[Cosine Similarity Amenity Recommender]
    end

    subgraph Generative AI
        SHAP & StarSchema -->|05_generative_ai.ipynb| PromptEngine[Prompt and Context Builder]
        PromptEngine -->|Groq API| GroqLLM[Llama-3.3-70b-versatile]
        GroqLLM -->|JSON/Text| AIAdvisor[AI Pricing and Host Strategy]
    end

    subgraph Presentation Layer
        StarSchema & XGBoost & Recommender & AIAdvisor -->|streamlit run| Streamlit[Interactive Streamlit Dashboard]
    end
```

---

## Repository Structure

```text
EXPERNETIC/
├── app/
│   └── streamlit_dashboard.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── warehouse.duckdb
├── experiments/
├── models/
│   ├── xgboost_model.joblib
│   ├── model_meta.joblib
│   └── shap_explainer.joblib
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_price_prediction.ipynb
│   ├── 03_nlp_reviews.ipynb
│   ├── 04_recommender.ipynb
│   └── 05_generative_ai.ipynb
├── reports/
│   ├── figures/
│   ├── Final_Report_Extended.md
│   ├── assumptions_and_decisions_log.md
│   ├── data_profile_raw_output.txt
│   └── market_intelligence_briefings.txt
├── scripts/
│   ├── run_pipeline.py
│   └── train_model.py
├── src/
│   ├── clean_listings.py
│   ├── clean_calendar.py
│   ├── clean_reviews.py
│   ├── build_warehouse.py
│   ├── profile_data.py
│   └── logging_config.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick Start and Setup

This repository is designed to run in a localized Python environment on Windows with Visual Studio Code.

### 1. Environment activation and dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the environment template and add the required values:

```bash
cp .env.example .env
```

If you plan to use the generative AI features, set your Groq API key in the environment:

```env
GROQ_API_KEY=your_key_here
```

### 3. Run the data pipeline

```bash
python scripts/run_pipeline.py
```

This runs the profiling, cleaning, and warehouse build steps in sequence.

### 4. Train the model

```bash
python scripts/train_model.py
```

This trains the pricing model and writes experiment metadata and model artifacts to the models and experiments folders.

### 5. Launch the dashboard

```bash
streamlit run app/streamlit_dashboard.py
```

---

## 🖥️ Streamlit Dashboard Walkthrough

The Streamlit dashboard acts as the primary visualization layer. Below are placeholders for tab-specific screenshots to capture once deployed:

### 1. Market Overview Tab
High-level KPIs, price distribution histograms, and interactive filter controls (neighbourhood, room type, capacity).
<img width="1919" height="917" alt="image" src="https://github.com/user-attachments/assets/f266e8be-2b76-4bcd-b0a4-b9e6933f7209" />


### 2. Geographic & Spatial Analysis Tab
Geographic price tier mappings, spatial distributions, and average pricing gradients across Barcelona.
<img width="1919" height="901" alt="image" src="https://github.com/user-attachments/assets/3fd171a8-ec04-4c8c-8ce0-f14b9816a12b" />
<img width="1916" height="910" alt="image" src="https://github.com/user-attachments/assets/785c727a-ddf6-4337-bbed-426b1bb8cdf6" />


### 3. Host Intelligence Tab
Insights on market concentration, portfolio sizes, professionalization rates, and the Superhost review premium.
<img width="1919" height="911" alt="image" src="https://github.com/user-attachments/assets/29526959-5b24-40bf-a5e0-8e3df05f73fb" />


### 4. AI Pricing Advisor Tab
Our flagship predictive and generative tab. Input listing parameters to get an XGBoost price estimate, local percentiles, SHAP feature impact, and dynamic strategy from Llama-3.
<img width="1919" height="916" alt="image" src="https://github.com/user-attachments/assets/e8bfb13f-3a60-4415-8e03-7a29f4738b2e" />


### 5. AI Market Briefings Tab
<img width="1919" height="890" alt="image" src="https://github.com/user-attachments/assets/550bbead-c27a-4af8-a1eb-3be72a3d9ae3" />

---

## 📊 Analytics and Database Schema

The platform implements a star schema model in DuckDB designed to minimize analytical query latency.

### Fact Table
* `fact_listing_performance`: Captures nightly prices, availability, review aggregate scores, occupancy proxies, and derived revenue estimates.

### Dimension Tables
* `dim_listing`: Property capacities, room types, clean parsed amenity arrays, and compliance night constraints.
* `dim_neighbourhood`: Centroid spatial coordinates and administrative district groupings.
* `dim_host`: Superhost indicators, total listings counts, response rates, and tenure metrics.


---

## Analytics and Database Schema

The platform implements a star-schema-style model in DuckDB designed to support analytical queries efficiently.

### Fact table

* fact_listing_performance: captures nightly price, availability, review aggregate scores, occupancy proxies, and derived performance indicators.

### Dimension tables

* dim_listing: contains property capacities, room types, and listing-level attributes.
* dim_neighbourhood: stores neighbourhood names and groupings.
* dim_host: stores host-level characteristics such as superhost status and portfolio size.

---

## Modeling and AI

### Price prediction with XGBoost

* Goal: predict log-transformed price values to handle right-skewness and improve model stability.
* Accuracy: R-squared and MAE are reported in the analysis notebooks and final report.
* Explainability: SHAP values are used to interpret the most influential predictors.

### Recommendation engine

* Strategy: a cosine-similarity-based recommender operates on amenity vectors and provides cold-start-friendly recommendations.

### Generative AI advisor

* Strategy: the notebook-based workflow uses Groq LLM responses to translate explainability outputs into executive and host-facing recommendations.

---

## Key Project Reports

* Comprehensive Final Report: a detailed written report covering methodology, statistical testing, modeling, and business interpretation.
* Assumptions and Decisions Log: documents critical engineering decisions and their trade-offs.
* Data profile outputs and market intelligence briefings are also included in the reports directory.

---

## Quality and Infrastructure

The project includes several improvements to support maintainability and reproducibility:

* centralized configuration in pyproject.toml for Black, Ruff, mypy, and pytest,
* structured logging in the ETL modules instead of ad hoc print-based output,
* type hints for core pipeline functions,
* idempotent processing behavior with optional force re-runs,
* automated tests and a GitHub Actions workflow for linting, typing, tests, and Docker builds.

---

## Testing

Run the test suite locally with:

```bash
pytest
```

Additional quality checks:

```bash
ruff check .
mypy src/
```
