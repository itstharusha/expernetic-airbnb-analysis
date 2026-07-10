# Barcelona Airbnb Market Intelligence Platform

## Overview

This repository contains an end-to-end data engineering and analytics project for the Barcelona Airbnb market. It combines data ingestion, cleaning, warehouse modeling, statistical analysis, machine learning, and generative AI to support market intelligence workflows.

The current implementation includes:
- an ELT pipeline for raw Inside Airbnb data,
- a DuckDB analytical warehouse with a star-schema-style model,
- notebooks for exploratory analysis, price prediction, NLP review analysis, and recommender work,
- a Streamlit dashboard for interactive exploration,
- automated quality checks, typing, logging, CI, and reproducible scripts.

## Architecture

The project is organized around a simple analytical workflow:
1. Raw source files are profiled and cleaned.
2. Cleaned datasets are stored as processed parquet files and loaded into the DuckDB warehouse.
3. Analytical and modeling notebooks consume the warehouse and produce reports, models, and dashboard assets.
4. A Streamlit application presents the results for interactive exploration.

## Repository Structure

```text
EXPERNETIC/
├── app/                      # Streamlit dashboard application
├── data/                     # Raw, processed, and warehouse data
├── experiments/              # Experiment metadata and run artifacts
├── models/                   # Serialized model and explainability artifacts
├── notebooks/                # Jupyter notebooks for analysis and modeling
├── reports/                  # Reports, briefs, figures, and logs
├── scripts/                  # CLI entrypoints for pipeline and model training
├── src/                      # Core ETL and data processing modules
├── tests/                    # Unit and integration tests
├── pyproject.toml            # Formatting, linting, typing, and test configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview and usage guide
```

## Setup

### 1. Create and activate a Python environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example environment file and add the required values:

```bash
cp .env.example .env
```

If you plan to use the generative AI features, set your Groq API key in the environment:

```env
GROQ_API_KEY=your_key_here
```

## Running the Pipeline

The repository now includes CLI entrypoints for the core workflow.

### Run the full ETL pipeline

```bash
python scripts/run_pipeline.py
```

### Train the model

```bash
python scripts/train_model.py
```

### Launch the dashboard

```bash
streamlit run app/streamlit_dashboard.py
```

## Quality and Infrastructure

The project includes several improvements to support maintainability and reproducibility:
- centralized tooling configuration in pyproject.toml for Black, Ruff, mypy, and pytest,
- structured logging in the ETL modules instead of ad hoc print-based output,
- type hints for core pipeline functions,
- idempotent processing behavior with optional force re-runs,
- automated tests and a GitHub Actions workflow for linting, typing, tests, and Docker builds.

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

## Reports and Outputs

Key deliverables are located in the reports directory, including:
- the main analytical report,
- the assumptions and decisions log,
- data profiling outputs,
- market intelligence briefings,
- generated figures for analysis and presentation.

## Notes

The repository is designed to be practical and reproducible, with a balance of notebook-based exploration and production-oriented scripts. It is suitable for both local development and presentation-oriented evaluation workflows.
