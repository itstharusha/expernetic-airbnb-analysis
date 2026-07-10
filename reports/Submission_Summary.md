# Expernetic Technical Assignment — Submission Summary
### Summary of Completed and Incomplete Work & Prioritization Decisions

---

## 1. Summary of Completed Work

The following components have been fully implemented to a professional, production-ready standard:

| Component / Deliverable | Implementation Standard & Highlights |
| :--- | :--- |
| **Data Ingestion & Cleaning** | Standardized currency strings, handled missing values via robust median imputation, removed extreme outliers (>€10k/night), and validated coordinate boundaries within Barcelona. Output persisted in Parquet. |
| **DuckDB Data Warehouse** | Built a Star Schema (1 Fact, 3 Dimensions) optimized for OLAP aggregations. Implemented dynamic views for host portfolios and listing economics (San Francisco occupancy model). |
| **Statistical Analysis** | Conducted non-parametric hypothesis testing (Mann-Whitney U, Kruskal-Wallis H) across 5 core hypotheses (H1 to H5), verifying assumptions and computing effect sizes (Cohen's d, CLES, η²). |
| **Predictive Pricing Model** | Trained an XGBoost Regressor achieving an **R² of 0.857** and **MAE of €50.36**. Integrated SHAP TreeExplainer for full model explainability. Saved serialized model artifacts in the `/models` directory. |
| **Content-Based Recommender** | Engineered a Cosine-Similarity based recommendation system using one-hot encoded amenity vectors, solving the cold-start problem for never-booked listings. |
| **Generative AI Integration** | Connected Groq's Llama-3.3-70b-versatile model to interpret XGBoost predictions and positive/negative SHAP drivers, returning personalized, written strategic advice to hosts. |
| **Interactive Streamlit Dashboard** | Created a responsive 5-tab dashboard displaying Market Overview, Geographic Analysis, Host Concentration, the AI Price Advisor tool, and AI-generated briefings. |
| **Automated Unit Tests** | Implemented `pytest` suite verifying data cleaning functions, coordinates, and database schema integrity. |
| **Docker Containerization** | Provided a `Dockerfile` and `docker-compose.yml` to package and run the application with single-command reproducibility. |

---

## 2. Summary of Incomplete Work & Prioritization Decisions

In accordance with Section 11 of the assignment, the following items were intentionally left incomplete or scoped out. This is a conscious engineering trade-off to prioritize quality and depth over superficial completion.

### 2.1 Cross-City Comparisons (Section 12 of Final Report)
* **What was left out**: Ingesting and modeling multiple cities (e.g., London or Paris).
* **Prioritization Rationale**: The evaluation rubric places heavy weight on problem-solving depth, statistical rigor, and clear communication. Spreading development effort across multiple cities within a one-week timeline would have resulted in a generic, surface-level analysis. Instead, we chose to execute a deep-dive on the Barcelona market, delivering comprehensive statistical testing, advanced ML explainability, a custom recommender system, and GenAI pricing advisors.
* **Mitigation**: The code is structured architecturally to accept new city datasets with zero changes via environment variables.

### 2.2 dbt Model Integration
* **What was left out**: Creating a standalone dbt project to manage warehouse transformations.
* **Prioritization Rationale**: For a single-city analytical dataset of this size (~15,000 listings), introducing dbt adds significant deployment, orchestration, and compilation overhead with no runtime performance benefit.
* **Mitigation**: Database migrations, table definitions, and star-schema views are natively managed in Python using raw SQL strings via DuckDB connections.

### 2.3 Continuous Integration (CI/CD) and Automated Re-runs
* **What was left out**: Deployed Apache Airflow DAGs and automated drift monitoring (e.g., Evidently AI).
* **Prioritization Rationale**: These are production-ready MLOps components that require active cloud environments to run. Building mock infrastructure locally would take development time away from visual storytelling and analytical depth.
* **Mitigation**: A complete MLOps roadmap (including Airflow design sketches and Evidently AI drift-detection triggers) is detailed in Section 14 of the main report.

---

## 3. Assumptions & Decisions Log Summary
All engineering decisions (imputations, log-transformations, DuckDB schema choice, and Responsible AI mitigation for Nou Barris pricing bias) are fully documented in:
* **[`reports/assumptions_and_decisions_log.md`](file:///C:/Users/ASUS/Downloads/EXPERNETIC/reports/assumptions_and_decisions_log.md)**

---

## 4. AI Usage Disclosure
A full log of AI tools used, prompts applied, output validation methods, and suggestions that were explicitly rejected is documented in:
* **[Appendix A of the Final Report](file:///C:/Users/ASUS/Downloads/EXPERNETIC/reports/Final_Report_Extended.md#16-appendix-a--ai-usage-disclosure)**
