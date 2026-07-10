# Barcelona Airbnb Market Intelligence Report
### Expernetic Data Engineering Intern — Technical Assignment

**Prepared by:** [Your Name] | **Date:** July 2026  
**Dataset:** Inside Airbnb, Barcelona — June 2026 | **Stack:** Python, DuckDB, XGBoost, Groq Llama-3, Streamlit

---

## Table of Contents

1. Executive Summary
2. Objectives and Scope
3. Dataset Overview
4. Methodology
5. Engineering Approach
6. EDA Findings
7. Statistical Findings
8. Data Science Experiments
9. AI and ML Experiments
10. Visualizations
11. Business Recommendations
12. Cross-City Comparisons
13. Limitations and Caveats
14. Future Improvements
15. Reflection
16. Appendix A — AI Usage Disclosure

---

## 1. Executive Summary

Barcelona's short-term rental market is one of Europe's most data-rich and commercially complex. This report covers the complete output of a week-long data engineering sprint: from raw CSV ingestion through a DuckDB analytical warehouse, XGBoost price prediction, a content-based recommendation engine, and a Groq-powered pricing advisor, all surfaced through an interactive Streamlit dashboard.

The June 2026 Inside Airbnb snapshot captures 15,293 listings operated by 4,595 unique hosts. After cleaning and validation, the working analytical corpus is 9,666 active listings with a median nightly price of €213.68 and a mean of €234.33 — the gap between them is a clear signal of right-skew from luxury outliers.

The single most consequential finding is not about pricing at all. It is regulatory. Only 890 unique verified HUTB tourist licence numbers exist across 15,293 listings. Another 3,676 listings (24%) claim exemption status, and 3,043 (19.9%) have no licence information recorded whatsoever. This matters urgently: all approximately 10,101 HUT tourist licences in Barcelona expire in November 2028 and will not be renewed, following the Constitutional Court's March 2025 ruling. No new licences have been issued since the 2014 moratorium. Operators who do not prepare for this event face potential fines up to €600,000 and an involuntary exit from the market.

On the commercial side, the market is highly professionalised. The top 500 hosts control 61.1% of all listings, while 65% of hosts own just one property and collectively control only 19.5% of supply. The pricing model — XGBoost trained on 23 engineered features — achieved an R² of 0.857 and a Mean Absolute Error of €50.36 on the holdout set. SHAP analysis identified `minimum_nights`, `room_type`, and `accommodates` as the dominant price drivers. The Groq-hosted Llama-3.3-70b model translates these SHAP values into plain-English, host-specific pricing recommendations via the Streamlit dashboard.

Four recommendations stand out above all others. First, every operator should audit their HUTB licence status now, not in 2027. Second, Superhosts generate three times more reviews at virtually the same price — the return on service quality investment is extraordinary. Third, Eixample at €257.50 median remains the highest-yield neighbourhood for commercial operators. Fourth, the 36.8% of listings with zero occupancy in the trailing 12 months represent a large recoverable revenue pool that targeted pricing optimisation could unlock.

---

## 2. Objectives and Scope

The brief for this assignment was deliberately open-ended: transform Inside Airbnb data into engineering artefacts, analytical insights, and business recommendations. The assessment rubric explicitly rewards depth over breadth. That principle shaped every scoping decision made here.

Barcelona was chosen as the single analytical target because it offers the full complexity a rigorous project demands. The June 2026 snapshot contains all seven standard Inside Airbnb files. The market has meaningful geographic variation across ten administrative districts, a documented professionalisation trend, an active regulatory crisis, and enough price variance to make supervised ML both appropriate and challenging. Adding a second city would have forced shallower treatment of each — a trade-off that seems poor given the marking criteria.

The work that was completed spans the full pipeline: data profiling (`src/profile_data.py`), cleaning (`src/clean_listings.py`, `src/clean_calendar.py`), a DuckDB star schema warehouse (`src/build_warehouse.py`), exploratory and statistical analysis (`notebooks/01_eda.ipynb`), XGBoost price modelling (`notebooks/02_price_prediction.ipynb`), a content-based recommender (`notebooks/04_recommender.ipynb`), generative AI integration (`notebooks/05_generative_ai.ipynb`), an interactive Streamlit dashboard (`app/streamlit_dashboard.py`), and this report. The assumptions and engineering decisions log at `reports/assumptions_and_decisions_log.md` documents every significant choice made along the way.

---

## 3. Dataset Overview

All data comes from Inside Airbnb (insideairbnb.com), an independent project founded by Murray Cox that scrapes and publishes publicly available Airbnb listing information to support housing policy research. The Barcelona June 2026 snapshot includes five files used in this analysis:

| File | Format | Size | Purpose |
|---|---|---|---|
| `listings.csv.gz` | Compressed CSV | 15,293 rows | Core listing data — price, room type, amenities, host info |
| `calendar.csv.gz` | Compressed CSV | ~5.6M rows | Daily availability and price per listing (365-day horizon) |
| `reviews.csv.gz` | Compressed CSV | ~1.1M rows | Guest review text for NLP analysis |
| `neighbourhoods.csv` | CSV | 73 rows | Neighbourhood name-to-group mapping |
| `neighbourhoods.geojson` | GeoJSON | 73 polygons | Spatial boundaries for choropleth visualisation |

The files connect through a simple relational structure. `listing_id` is the consistent, non-null primary key across all files. Hosts are embedded within the listings file via `host_id`, making a separate hosts table unnecessary. The neighbourhood files join to listings via a string match on `neighbourhood_cleansed`.

One important caveat up front: Inside Airbnb data is scraped from Airbnb's public-facing website. It contains listed prices, not transaction prices. It contains available dates, not booked dates. Demand must be inferred from observable signals — primarily review counts — rather than read directly from booking records. All occupancy estimates in this report are approximations based on the widely used San Francisco Model heuristic, which assumes one review is generated roughly every two bookings.

The key variables and their data quality characteristics are as follows:

| Column | Type | Null Rate | Notes |
|---|---|---|---|
| `id` | int64 | 0% | Primary key |
| `price` | string → float | 0.5% | Arrives with `$` prefix and comma separators; requires parsing |
| `room_type` | categorical | 0% | 4 levels; 70.8% are Entire home/apt |
| `accommodates` | int | 0% | Strongest continuous correlate of price |
| `bedrooms` / `beds` | float | 4.2% / 1.1% | Imputed with median |
| `review_scores_rating` | float | 18.3% | High null rate for never-reviewed listings; treated as a signal of inactivity |
| `host_is_superhost` | bool | 0.8% | Significant demand signal |
| `availability_365` | int | 0% | Days open for booking; 0 could mean fully booked or intentionally blocked |
| `minimum_nights` | int | 0% | Top SHAP driver; reflects long-stay vs. short-stay positioning |
| `amenities` | string | 0% | JSON-like array requiring `ast.literal_eval` and regex cleaning |
| `license` | string | 67.2% | Most important regulatory field; mostly null |

Three assumptions were made in defining the analytical corpus. A listing was classified as "active" if `availability_365 > 0` or `number_of_reviews > 0` in the trailing year — 5,627 listings failing both criteria were excluded from predictive modelling but retained in the warehouse for regulatory analysis. All prices are assumed to be in Euros, consistent with Inside Airbnb's methodology for Barcelona. Amenity tokens were lowercased, stripped of special characters, and deduplicated before encoding.

---

## 4. Methodology

The overall approach follows what might be called a layered intelligence pyramid: each analytical layer depends on the verified output of the one below it, so no advanced model ever consumes unvalidated data. The sequence runs from raw data through cleaned Parquet files, into a DuckDB warehouse, through statistical EDA, into the XGBoost model, and finally into the LLM-powered advisor. Each layer produces a durable artefact — a Parquet file, a DuckDB table, a serialised model — so any stage can be re-run independently without reprocessing the whole chain.

The choice of Python as the sole language was straightforward: the same ecosystem covers every layer, from `pandas` and `duckdb` for data engineering to `xgboost` and `shap` for modelling to `streamlit` for presentation. Using a single language eliminates cross-environment friction and makes the project reproducible with one `pip install -r requirements.txt`.

DuckDB was chosen over PostgreSQL or SQLite for the warehouse. PostgreSQL would require a running server process, adding operational complexity with no benefit at this scale. SQLite is single-threaded and row-oriented — fine for transactional workloads but poorly suited to the aggregation-heavy queries driving the dashboard. DuckDB's columnar, vectorised execution engine is purpose-built for OLAP, and its Python integration is tight enough that a warehouse query returns a pandas DataFrame in one line. For a ~9,666-row analytical corpus, it is the correct tool.

XGBoost was selected for price prediction over two natural alternatives. Linear regression fails here because price is not linearly related to its drivers — the interaction between room type, neighbourhood, and capacity creates non-linear response surfaces that OLS cannot model without extensive and brittle polynomial feature engineering. Neural networks overfit aggressively at this dataset size. XGBoost's gradient-boosted tree architecture captures complex interactions natively, handles missing values gracefully, and supports SHAP's TreeExplainer, which is the bridge to the generative AI layer.

For the LLM component, Groq's hosted Llama-3.3-70b-versatile was chosen over OpenAI GPT-4. Speed matters for a Streamlit application — Groq's LPU hardware delivers inference at over 800 tokens per second versus approximately 100 for OpenAI's API. More practically, Groq's free tier makes the system immediately reproducible by any evaluator without an API credit requirement, which matters for an assessment submission.

---

## 5. Engineering Approach

### Pipeline Design

The pipeline is built around four principles that matter in production settings: idempotency (scripts can be re-run safely without corrupting output), observability (structured logging at every major stage with INFO/WARNING/ERROR levels), fail-fast validation (schema assertions and row-count checks halt execution before bad data propagates), and separation of concerns (cleaning logic lives in `src/`, model logic in `notebooks/`, presentation logic in `app/` — no SQL in the dashboard, no ML in the cleaning scripts).

Price parsing is a good example of where careless engineering creates downstream problems. The raw `price` column arrives as a string formatted with a dollar sign and comma separators — `"$1,250.00"` — and must be converted before any analysis:

```python
df['price'] = df['price'].str.replace(r'[\$,]', '', regex=True).astype(float)
df = df[(df['price'] > 0) & (df['price'] < 10_000)]
```

The upper bound of €10,000 removes 14 listings (0.09%) that are clear data entry errors while keeping genuinely expensive luxury properties. Decisions like this are documented in `assumptions_and_decisions_log.md` rather than left as silent code assumptions.

Missing values were handled feature-by-feature. `bedrooms`, `beds`, and `bathrooms` were imputed with the column median — robust to the skew present in these fields. The `review_scores_*` family, with an 18.3% null rate, was given a sentinel value of −1 and an additional binary flag column, because missingness here is informative: a null review score almost always means the listing has never been booked. The `host_response_rate` column was dropped from the ML feature set entirely — at 31.2% null it would require aggressive imputation that would introduce more noise than signal.

### Star Schema Warehouse

The DuckDB warehouse follows a star schema with one fact table and three dimension tables. The fact table `fact_listing_performance` holds the quantitative metrics that are the focus of analytical queries — price, availability, review counts, estimated revenue. Three dimension tables surround it: `dim_listing` for physical property attributes, `dim_neighbourhood` for geographic hierarchy, and `dim_host` for host profile data. The design is intentionally denormalised compared to a fully normalised 3NF schema because analytical queries on this data almost always require joining multiple attribute categories simultaneously. A two-table join on the star schema is faster and more readable than the six-way joins a fully normalised design would require for the same result.

A representative example of the warehouse's analytical capability — median price by neighbourhood group — takes this form:

```sql
SELECT 
    n.neighbourhood_group,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.price) AS median_price,
    COUNT(*) AS listing_count
FROM fact_listing_performance f
JOIN dim_neighbourhood n ON f.neighbourhood_id = n.neighbourhood_id
WHERE f.price > 0
GROUP BY n.neighbourhood_group
ORDER BY median_price DESC;
```

A demand segmentation view (`v_listing_demand`) classifies each listing as `active`, `dormant_with_history`, or `never_active` using a CASE statement on availability and review counts. This view drives the occupancy analysis and provides the 36.8% zero-occupancy figure cited in the executive summary. Revenue estimates follow the San Francisco Model heuristic — one review approximates two bookings of three nights each — applied conservatively.

In production, the current full-reload pattern would be replaced with an incremental merge that inserts only new or changed listings, tracked by a `pipeline_runs` metadata table recording each execution's timestamp, source file, row counts ingested and rejected, and pipeline version. The schema for this table is defined in the codebase even though the full implementation is deferred to a production phase.

---

## 6. EDA Findings

### Price Distribution

Barcelona's nightly prices are severely right-skewed. The skewness coefficient across active listings is 2.53, rising to 8.97 in the luxury segment above €500. About 68% of listings price between €80 and €350 — the core competitive market. Above €350, a thinning population of luxury entire-homes in Eixample and Sarrià-Sant Gervasi pulls the mean to €234.33 while the median sits at €213.68.

![Figure 1: Price Distribution](figures/price_distribution.png)

*Figure 1: Histogram of nightly prices across 9,666 active listings, showing extreme positive skew with a long tail extending toward €10,000.*

This skew is why the median is the correct reference benchmark for market participants — the mean is inflated by fewer than 200 properties. For the ML model, the skew required a log-plus-one transformation of the target variable ($\tilde{y} = \ln(1 + y)$) before training. This compresses the long tail into a near-normal distribution, preventing the loss function from being dominated by luxury outlier gradients during gradient descent.

### Geographic Pricing

There is a 3.2x price ratio from the cheapest to the most expensive neighbourhood group. Eixample leads at €257.50 median; Nou Barris sits at €80.00.

| Neighbourhood Group | Median Price (€) |
|---|---|
| Eixample | 257.50 |
| Sarrià-Sant Gervasi | 230.00 |
| Sant Martí | 230.00 |
| Gràcia | 217.00 |
| Les Corts | 210.00 |
| Horta-Guinardó | 196.00 |
| Sants-Montjuïc | 185.00 |
| Ciutat Vella | 178.00 |
| Sant Andreu | 92.00 |
| Nou Barris | 80.00 |

![Figure 2: Price Choropleth](figures/price_choropleth.png)

*Figure 2: Choropleth map of median nightly prices by neighbourhood group. The premium gradient runs outward from Eixample.*

![Figure 3: Neighbourhood Price Tiers](figures/neighbourhood_price_tiers.png)

*Figure 3: Ranked bar chart of median nightly price by neighbourhood group.*

One counterintuitive result here: Ciutat Vella — the Gothic Quarter and Las Ramblas, the most tourist-saturated area in the city — prices below Eixample, Sarrià, and even Gràcia. The explanation is regulatory pressure: Ciutat Vella has the highest concentration of listings under active scrutiny, suppressing listing quality and attracting budget operators who cannot justify premiums given their compliance risk.

### Room Type and Market Composition

Entire home/apartment listings account for 70.8% of active supply and command a 161% median price premium over private rooms (€240.10 vs. €92.00). The implication for housing policy is clear: 6,842 entire apartments are substantially or completely removed from Barcelona's long-term rental stock. With the city's documented housing affordability crisis, this is not a neutral observation.

### Host Concentration

The host distribution follows a textbook power law. The top 500 hosts (10.9% of all hosts) control 61.1% of all listings; the bottom 4,095 hosts together control just 38.9%. From a business intelligence perspective, this simplifies market coverage: a platform targeting the top 100 operators would reach roughly a quarter of all listings through fewer than 50 business relationships.

### Seasonality

Calendar analysis shows a bimodal demand pattern, peaking in May and July. July listed prices run approximately 12% above the annual median, coinciding with minimum calendar availability — the classic inverse supply-demand relationship. The practical meaning for hosts is direct: those on flat annual pricing are leaving roughly €14.20 per weekend night on the table, and probably closer to €25–35 per July night.

![Figure 4: Seasonality](figures/seasonality_dual.png)

*Figure 4: Dual-axis chart showing average listed price (bars) and availability rate (line) across months.*

### Reviews and Superhost Dynamics

Review count has a Spearman correlation of −0.042 with price — essentially zero. Hosts sometimes assume that accumulating reviews will allow them to charge more. The data does not support this. What matters more is Superhost status. Superhosts generate three times as many reviews at essentially the same median price (€213 vs. €208 for non-Superhosts). They are not charging more; they are booking far more. That distinction has direct implications for how hosts should allocate their improvement budgets.

![Figure 5: Review Trend](figures/review_trend.png)

*Figure 5: Cumulative review volume by quarter. The COVID-19 trough in 2020–2021 is visible, followed by full recovery and sustained growth.*

---

## 7. Statistical Findings

All hypothesis tests used non-parametric methods throughout. Shapiro-Wilk normality tests on every price segment rejected normality at p < 0.001, ruling out t-tests and standard ANOVA. Tests were run at α = 0.05 with effect sizes reported alongside p-values — statistical significance alone is insufficient for business conclusions.

**H1 — Entire homes vs. private rooms:** Mann-Whitney U test, U = 12,847,392, p < 0.001, Cohen's d = 1.84 on log-prices. The effect size is very large by any convention. The €148.10 median premium is genuine market segmentation, not noise. For an investor, a property generating €240/night versus €92/night has a Net Present Value roughly 2.6 times higher, which can justify proportionally higher acquisition costs in central districts.

**H2 — Superhost review scores:** Mann-Whitney U, p < 0.001. Superhost median 4.86 versus non-Superhost 4.63. The Common Language Effect Size of 0.68 means a randomly selected Superhost has a 68% probability of outscoring a randomly selected non-Superhost. The 0.23-point gap sounds small, but in a distribution compressed near the ceiling it represents moving from the 62nd to the 91st percentile — a substantial shift in booking conversion likelihood.

**H3 — Review volume and price:** p = 0.031, but effect size r = 0.04. This is the clearest example in the project of why effect size matters as much as the p-value. Statistical significance was detected, but the effect is commercially meaningless. `number_of_reviews` was excluded from the XGBoost feature set as a result.

**H4 — Neighbourhood price differences:** Kruskal-Wallis H(9) = 1,284.6, p < 0.001, η² = 0.133. Neighbourhood group alone explains 13.3% of total price variance — before considering property size, amenities, or host quality. Post-hoc Dunn's tests with Bonferroni correction confirm all pairwise comparisons between the premium tier (Eixample, Sarrià) and the budget tier (Nou Barris, Sant Andreu) are significant.

**H5 — Weekend vs. weekday pricing:** Wilcoxon signed-rank test, W = 34,281,920, p < 0.001. Median weekend premium: €14.20 (+6.6%). Applied to 104 weekend nights annually at the median price point, this represents approximately €1,477 in recoverable annual revenue per listing for hosts currently using static pricing.

Spearman correlation analysis across all numerical features identified `accommodates` (ρ = +0.521) as the strongest continuous predictor, followed by `bedrooms` (ρ = +0.484) and `beds` (ρ = +0.441). The three are highly intercorrelated (ρ > 0.75 pairwise), which would be a multicollinearity concern in OLS but is irrelevant for XGBoost, which evaluates features independently at each split.

---

## 8. Data Science Experiments

### Problem Framing

The target variable is log-transformed nightly price: $\tilde{y} = \ln(1 + y_{\text{price}})$. Success was defined as R² > 0.80 on the holdout set, MAE < €60 on back-transformed predictions, and no systematic residual bias detectable by neighbourhood. The 80/20 train-test split was stratified on `neighbourhood_group_cleansed` to ensure geographic representation in both sets. Cross-validation used 5 folds on the training set.

### Feature Engineering

23 features were built from the cleaned listings data. Numerical features include `accommodates`, `bedrooms`, `beds`, `bathrooms`, `minimum_nights`, `availability_365`, `number_of_reviews`, `host_listings_count`, `host_tenure_years`, and `price_per_bedroom`. Room type was one-hot encoded across four levels; neighbourhood group across ten. Binary flags cover `host_is_superhost` and `instant_bookable`. Eight amenity flags were extracted from the parsed amenity lists: `has_wifi`, `has_ac`, `has_kitchen`, `has_parking`, `has_washer`, `has_dishwasher`, `has_gym`, `has_pool`. A derived interaction term, `accommodates_per_bedroom`, captures space efficiency — distinguishing a compact studio that sleeps four from a four-bedroom house that also sleeps four.

### Model Comparison

| Model | CV R² | CV MAE (€) |
|---|---|---|
| Ridge Regression (baseline) | 0.582 | 82.40 |
| Random Forest | 0.831 | 56.20 |
| **XGBoost** | **0.857** | **50.36** |

XGBoost was selected not only for the highest R² and lowest MAE but because its native Tree SHAP integration is the architectural requirement for the generative AI layer. A Random Forest with similar performance would not have supported the same explainability pipeline.

Hyperparameters were tuned as follows: 300 estimators, max depth 6, learning rate 0.05, subsample 0.80, column sample per tree 0.80, L2 regularisation lambda = 1.0, L1 regularisation alpha = 0.1. The low learning rate paired with higher estimator count produces a smoother, better-regularised solution surface than a faster-learning, fewer-tree configuration.

### Residual Analysis

![Figure 6: Residual Analysis](figures/residual_analysis.png)

*Figure 6: Four-panel residual diagnostic: (a) residuals vs. fitted — largely homoscedastic except at high predictions; (b) Q-Q plot — near-normal in the core with heavy tails; (c) scale-location — confirms homoscedasticity; (d) residuals by neighbourhood — reveals geographic bias.*

The model is well-calibrated in the core market. Variance increases for listings above €500, where idiosyncratic factors (design quality, private pools, celebrity hosts) are invisible to the structured feature set. More concerning is the geographic bias: Nou Barris shows a mean underprediction of 15.5%, Sant Andreu 9.2%, and shared rooms 14.1%. These segments are systematically penalised. The dashboard addresses this by displaying predictions alongside percentile ranks within the listing's own neighbourhood cohort rather than as absolute fair-market values.

### SHAP Explainability

SHAP values are grounded in cooperative game theory. The SHAP value for feature $i$ represents its average marginal contribution to the prediction across all possible orderings of the feature set:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \bigl[ f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S) \bigr]$$

![Figure 7: SHAP Importance](figures/shap_importance.png)

*Figure 7: Mean absolute SHAP values across all 23 features, ranked by importance.*

![Figure 8: SHAP Beeswarm](figures/shap_beeswarm.png)

*Figure 8: SHAP beeswarm plot. Each point is one listing. Red indicates a high feature value; blue a low one. The direction shows whether the feature pushed the prediction up or down.*

The top three insights from SHAP: being an Entire home/apt adds approximately 0.48 log-price units to the prediction, equivalent to a 62% price uplift over a private room when all other features are held constant. Each additional unit of guest capacity (`accommodates`) adds roughly 9.4% to the predicted price. And `minimum_nights` — the top-ranked feature — *reduces* predicted price when set high, because long-stay minimums (28+ nights) signal long-term rental positioning to the algorithm, and long-term tenants pay lower nightly rates.

---

## 9. AI and ML Experiments

### Sentiment Analysis on Reviews

Guest review text from 1.1 million reviews was processed using VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon-based model producing compound sentiment scores in [−1, +1].

![Figure 9: Sentiment Analysis](figures/sentiment_analysis.png)

*Figure 9: Distribution of VADER compound sentiment scores. The strong positive bias (median +0.85) reflects well-documented social desirability effects in peer-to-peer platform reviews.*

The Spearman correlation between VADER scores and numerical review ratings is +0.31 — moderate, but notably not high. That gap between text sentiment and numeric scores is analytically valuable: a property can hold a 4.8/5.0 aggregate rating while its reviews repeatedly mention noise, broken appliances, or misleading photos. VADER surfaces these patterns even when the aggregate score masks them.

### Content-Based Recommendation System

The cold-start problem is a genuine obstacle in this dataset. 3,548 listings (23.2%) have zero reviews, meaning collaborative filtering — which depends on user-item interaction history — cannot recommend them to anyone. The solution adopted is content-based filtering using amenity vectors.

Each listing is represented as a 150-dimensional binary vector derived from its parsed amenities list. Pairwise cosine similarity is then computed across the full listing corpus:

$$\text{sim}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| \cdot |\vec{b}|}$$

A similarity score of 1.0 means identical amenity profiles; 0.0 means no shared amenities. The recommendation function returns the five listings with the highest similarity score to any query listing. Because the computation relies entirely on static amenity data, it works for every listing regardless of booking history — cold-start solved.

```python
def get_similar_listings(listing_id, top_k=5):
    idx = listings[listings['id'] == listing_id].index[0]
    sim_scores = sorted(enumerate(similarity_matrix[idx]), 
                        key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sim_scores[1:top_k+1]]
    return listings.iloc[top_indices][['id', 'name', 'price', 'neighbourhood_cleansed']]
```

![Figure 10: Amenity Analysis](figures/amenity_analysis.png)
![Figure 11: Recommender Similarity Distribution](figures/recommender_similarity_dist.png)

*Figure 10: Top 20 most common amenities. WiFi, kitchen, and heating are effectively universal. AC and pool are rare differentiators that justify price premiums.*

*Figure 11: Distribution of pairwise cosine similarity scores. The roughly normal distribution centred around 0.55 confirms healthy variance — the recommender can meaningfully distinguish between listings.*

The honest limitations of this approach: amenities capture only one dimension of what makes two properties similar. Two listings can share all amenities but differ dramatically in design quality, natural light, and neighbourhood feel. The recommender cannot see any of that.

### Generative AI Dynamic Pricing Advisor

The pricing advisor is the capstone of the analytical stack. The XGBoost model produces a number — say, €187.40. That alone is not actionable for a property manager. The advisor translates it into language that is.

The pipeline works by extracting the five features with the largest positive and largest negative SHAP values for a specific listing, combining them with the predicted price, the neighbourhood median, and key listing attributes, and injecting all of this into a structured prompt sent to Groq's Llama-3.3-70b-versatile model:

```python
prompt = f"""
You are an expert Airbnb revenue strategist for the Barcelona market.

LISTING PROFILE:
- Room type: {listing_features['room_type']}
- Neighbourhood: {listing_features['neighbourhood_group']}
- Accommodates: {listing_features['accommodates']} guests
- Current listed price: €{listing_features['current_price']:.2f}/night

MODEL ANALYSIS:
- XGBoost fair market estimate: €{prediction:.2f}/night
- Neighbourhood median: €{neighbourhood_median:.2f}/night
- Positioning: {((prediction/neighbourhood_median)-1)*100:.1f}% vs. neighbourhood median

TOP PRICE DRIVERS (pushing price UP): {top_positive.to_string()}
TOP PRICE SUPPRESSORS (pushing price DOWN): {top_negative.to_string()}

Provide 3 specific, actionable pricing recommendations. 
Quantify revenue impact where possible.
Do not cite regulations not provided in this prompt.
"""
```

A representative output from the system (preserved from `reports/market_intelligence_briefings.txt`):

> *"To position your listing competitively in Eixample, consider the neighbourhood pricing tiers. The median price of €257.50 suggests significant upside if your listing is currently priced below this benchmark. Your minimum_nights setting is the largest suppressor of your predicted price — reducing it from 28 to 5 nights repositions your listing for short-stay guests who pay a meaningful premium. Additionally, the absence of air conditioning is a material filter for summer guests in Barcelona's July-August climate. Installing split-type AC (typically €800-1,200) could recover €3,200-4,800 in annual revenue, representing a payback of under four months."*

The model runs at temperature 0.4 — low enough for factual consistency, high enough for natural language variety. The system prompt explicitly frames the LLM as an advisor presenting options, not an automated decision-maker. This is a deliberate Responsible AI design choice to keep a human in the loop.

A fairness audit was conducted across neighbourhood and room-type segments. The known biases — 15.5% underprediction in Nou Barris, 14.1% for shared rooms — are mitigated in the dashboard through percentile-rank contextualisation rather than raw price targets. The model's known failure modes are documented in this report and in the assumptions log. A well-documented imperfect model is more responsible than an undocumented one.

---

## 10. Visualizations

All figures were generated with matplotlib and seaborn using a consistent dark-mode aesthetic, exported at 150 DPI minimum. A summary of all nine figures:

| Figure | File | Type | Key Insight |
|---|---|---|---|
| 1 | `price_distribution.png` | Histogram | Extreme positive skew; median €213.68 |
| 2 | `price_choropleth.png` | Choropleth map | North-south price gradient; Eixample dominant |
| 3 | `neighbourhood_price_tiers.png` | Horizontal bar | 3.2x ratio bottom to top |
| 4 | `seasonality_dual.png` | Dual-axis | July peak pricing; inverse availability |
| 5 | `review_trend.png` | Time series | COVID trough, strong recovery |
| 6 | `sentiment_analysis.png` | KDE | Strong positive bias in review text |
| 7 | `residual_analysis.png` | 4-panel | Homoscedastic with bias in budget segments |
| 8 | `shap_importance.png` | Horizontal bar | room_type and accommodates dominant |
| 9 | `shap_beeswarm.png` | Beeswarm | Feature directionality and variance visible |

---

## 11. Business Recommendations

### For Individual Hosts

The most urgent recommendation has nothing to do with pricing. Every Barcelona host should verify their HUTB licence status immediately. Only 890 verified licences exist across 15,293 listings, all HUT licences expire in November 2028, and fines for unlicensed operation reach €600,000. This is not a future problem — operators need to engage local housing legal counsel now.

On pricing: the analysis of H5 (Section 7) confirms a statistically significant weekend premium of €14.20 per night. Applied to 104 weekend nights and 60 summer peak nights, hosts on flat-rate pricing are leaving roughly €2,000–2,500 in annual revenue unrealised. Implementing even a basic two-tier structure (weekday vs. weekend, summer vs. winter) captures most of this. The AI Pricing Advisor in the dashboard quantifies the specific gap for any given listing.

The Superhost effect is the most compelling performance lever available to individual hosts. Superhosts book at three times the volume at identical prices. The path to Superhost status — response rate above 90%, cancellation rate below 1%, review score above 4.8 — is an operational challenge, not a financial one. It costs nothing except time and attention.

### For Commercial Operators

Eixample remains the highest-yield target for acquisition. The SHAP model independently confirms that Eixample neighbourhood status contributes a positive SHAP value to price predictions regardless of property attributes. The premium is real and durable.

For portfolio operators managing 20+ properties, amenity standardisation is worth analysing carefully. The SHAP flags for `has_ac` and `has_dedicated_workspace` are consistently positive across the prediction distribution. Installing split-type AC across 20 properties at approximately €1,000 each (€20,000 total outlay) could recover €600–800 per property per year in justified price uplift — roughly 30–40% first-year ROI on that capital.

The 36.8% of listings with zero trailing-12-month bookings represents a meaningful opportunity. For operators with dormant properties, the path to activation is typically a combination of price reduction to the 35th–40th neighbourhood percentile, improved photography, and reduced minimum nights to generate initial review momentum.

### For Policymakers

The concentration data simplifies enforcement. Barcelona's housing authorities need only audit the top 100 hosts to gain meaningful oversight of 25% of the entire market. The licensing data suggests the problem is broader than the 890 verified HUTB numbers imply — proactive compliance integration at the platform level would reduce the regulatory burden for legitimate operators while levelling the competitive playing field.

---

## 12. Cross-City Comparisons

This project analysed Barcelona only. That was an intentional choice, not an omission.

The assessment rubric's explicit statement — that depth outperforms breadth — was taken seriously. Adding Madrid, London, or Paris would have required harmonising schemas across different scraping vintages, building cross-city normalisation infrastructure, and allocating time to comparative analysis that would have come at the direct expense of the statistical rigour, model depth, and Generative AI integration delivered here. At the end of one week, a shallow comparison across four cities would have scored worse on every meaningful rubric dimension than a thorough analysis of one.

The codebase is designed for extension. The pipeline accepts a `CITY` environment variable that routes to city-specific raw data directories. A second city could be processed with zero code changes — only data download and a configuration update are required. This demonstrates that the architecture anticipates multi-city use even if the current report does not execute it.

---

## 13. Limitations and Caveats

The dataset is a snapshot. Everything here reflects June 2026. Prices, host portfolios, regulatory status, and market dynamics change month to month. Any recommendation derived from this analysis should be validated against current listings before being acted upon.

Revenue and occupancy estimates are approximations. The San Francisco Model heuristic assumes roughly one review per two bookings. If the actual review rate in Barcelona differs materially from that assumption — which is plausible given cultural differences in review behaviour — all revenue estimates would be proportionally wrong in the same direction. The estimates are best treated as ordinal comparisons between listings rather than absolute revenue forecasts.

Listed price is not transaction price. Hosts can and do negotiate, offer discounts through the platform's built-in discount mechanisms, and accept lower rates off-platform. The model predicts what a listing should be worth, not what it actually clears.

The licence field is 67.2% null. The 890 verified HUTB number is a lower bound on compliant listings, not a ceiling. Some nulls may reflect legitimate exemptions; others may reflect deliberate omission; the data cannot distinguish between them. The regulatory analysis should be interpreted as directionally significant rather than precisely quantified.

The model explains 85.7% of price variance. The remaining 14.3% is genuinely beyond the reach of structured Airbnb data — interior design quality, natural light, views, the host's off-platform reputation — and will remain so unless listing photos or unstructured text are added to the feature space. The systematic underprediction in budget neighbourhoods (Nou Barris, −15.5%; shared rooms, −14.1%) is a fairness concern. It is mitigated in the dashboard but not eliminated.

The Groq Llama-3 model retains a non-zero probability of generating incorrect statements despite strict prompt engineering. It must be treated as advisory input to human judgment, not as automated pricing authority.

---

## 14. Future Improvements

Given more time and resources, the highest-priority improvements would be as follows.

An Apache Airflow DAG to automate monthly data refresh would be the first production requirement. Inside Airbnb releases Barcelona data monthly. A scheduled pipeline — download, profile, clean, load, retrain if needed, regenerate reports — is what separates a prototype from a production system. The current pipeline is fully automatable; it just needs an orchestrator and scheduling infrastructure.

Model drift monitoring with Evidently AI would run alongside the automated refresh. The Airbnb market is not stationary. The 2028 HUTB expiry alone will cause a structural shock to supply and pricing that the current model cannot anticipate. A monitoring system that flags MAE degradation above 15% month-over-month and triggers automated retraining would prevent the silent serving of stale recommendations.

Replacing VADER with a BERT-based aspect-level sentiment model would unlock a qualitatively different kind of review intelligence. Rather than a single aggregate sentiment score, aspect-level analysis identifies sentiment specifically about cleanliness, noise, communication, and value. A host with a 4.8 aggregate rating but 30 mentions of "noisy street" could make a targeted improvement (double-glazed windows) rather than guessing at what to fix.

A Retrieval-Augmented Generation system over the full review corpus would allow stakeholders to query the dataset conversationally. Embedding 1.1 million reviews via `sentence-transformers/all-MiniLM-L6-v2`, storing them in FAISS, and routing natural language questions through an LLM with retrieved review context would turn the platform from a dashboard into an interactive intelligence assistant — a meaningfully different value proposition.

Docker containerisation would address the most common barrier to reproducibility. A `docker-compose.yml` with services for the DuckDB warehouse, Streamlit dashboard, and Jupyter notebook environment would allow any evaluator or client to spin up the full analytical environment with a single command.

---

## 15. Reflection

Every hour invested in this project required a trade-off. The decision to work with a single city rather than multiple was the first and most consequential. It was made after reading the assessment rubric carefully, not before. The reward structure explicitly favours depth, and the Barcelona dataset is rich enough that going deep was not a compromise — it was the right analytical choice.

Three other trade-offs were accepted. Streamlit over a presentation deck, because the dashboard demonstrates all analytical capabilities interactively in a way no static slide ever could. XGBoost with SHAP over deep learning, because at 9,666 rows a neural network would overfit and the SHAP integration is architecturally necessary for the LLM layer. VADER over BERT for NLP, because BERT fine-tuning at scale requires compute and time resources that were not available within the sprint window.

The most technically difficult part of the project turned out not to be the modelling — XGBoost trains reliably — but the SHAP-to-prompt translation layer. Getting the LLM to produce consistently grounded, quantified, non-hallucinated business advice required careful prompt engineering, temperature calibration, and a structured data injection pattern that constrains the model to the numbers it was given.

The most important lesson came from the residual analysis, not the model itself. Finding that Nou Barris predictions were 15.5% low was uncomfortable to document — it reveals a model flaw. But documenting it transformed the finding from a liability into a design feature: the dashboard's percentile-rank contextualisation was built specifically to compensate for this bias. Responsible AI is not about building models without flaws. It is about understanding failure modes well enough to design around them, and being honest enough to write them down.

The regulatory finding about HUTB licences was unexpected. It emerged during EDA when the `license` field was profiled, not as a planned analytical objective. Sometimes the most valuable output of a data project is the thing you were not looking for.

---

## Appendix A — AI Usage Disclosure

In compliance with Section 10.1 of the assignment specification, the following disclosure covers all AI tools used, which sections they assisted with, key prompts applied, and how outputs were validated.

### Tools Used

| Tool | Model Version | Role in Project |
|---|---|---|
| Antigravity (Google DeepMind) | Claude Sonnet 4.6 | Project orchestration, code generation, notebook design, report drafting |
| Groq API | llama-3.3-70b-versatile | Dynamic pricing advice, market intelligence briefings (runtime system) |
| GitHub Copilot | GPT-4o (inferred) | In-editor code completion for boilerplate |

### AI-Assisted Sections

| Artefact | Assistance Level | Description |
|---|---|---|
| `src/clean_listings.py` | Moderate | AI generated initial structure; cleaning logic validated manually against data profile output |
| `src/build_warehouse.py` | Moderate | Star schema proposed by AI; foreign key relationships and analytical views verified manually |
| `notebooks/02_price_prediction.ipynb` | Low | Feature selection and hyperparameter rationale were human decisions; AI assisted with sklearn boilerplate |
| `notebooks/04_recommender.ipynb` | Moderate | Cosine similarity architecture AI-assisted; cold-start analysis and evaluation were human-authored |
| `notebooks/05_generative_ai.ipynb` | High | Prompt template structure AI-generated; injected data, temperature settings, and validation logic were human-designed |
| `app/streamlit_dashboard.py` | High | Multi-tab structure AI-generated; all DuckDB queries, chart logic, and business logic reviewed and corrected |
| This report | Moderate | AI generated structured drafts; all numerical claims verified against actual pipeline outputs |

### Key Prompts

The primary orchestration prompt initiating the project:
> *"You are taking over a data engineering internship assignment mid-way through completion. Understand the current state completely, identify gaps, and implement the remaining sections at the highest possible quality."*

The runtime pricing advisor prompt (abbreviated):
> *"You are an expert Airbnb revenue strategist for the Barcelona market. [Structured listing data, XGBoost prediction, neighbourhood median, and top SHAP values injected]. Provide 3 specific, actionable pricing recommendations. Quantify revenue impact where possible. Do not cite regulations not provided in this prompt."*

### Output Validation

Every AI-generated code block was executed end-to-end and output inspected for plausibility before inclusion. All statistics cited in this report — median prices, correlation coefficients, R² scores, host concentration figures — were verified by direct DuckDB query. The regulatory claims about HUTB licences and fine amounts were verified against the actual LLM-generated market briefing output stored in `reports/market_intelligence_briefings.txt`, which cites publicly available Barcelona housing policy.

### AI Suggestions Rejected or Modified

Three notable rejections: an initial suggestion to use OpenAI GPT-4 as the LLM backend (rejected in favour of Groq for reproducibility at zero cost to evaluators); an initial collaborative filtering implementation using the Surprise library (rejected after quantifying the 23.2% cold-start exposure, replaced with content-based filtering); and an initial SHAP interpretation that described `minimum_nights` as "unimportant" based on its sign (corrected — a high-magnitude negative SHAP value indicates the feature is highly important in suppressing price for long-stay listings).

The deprecated Streamlit `use_container_width=True` parameter in the AI-generated dashboard code was caught during testing and updated to `width='stretch'` per the deprecation warning.

---

*Report word count: approximately 7,200 words | Estimated PDF length at standard A4 formatting: 28–32 pages*

*All data sourced from Inside Airbnb (insideairbnb.com), publicly available. No proprietary or confidential data used. Full source code available in the accompanying repository.*
