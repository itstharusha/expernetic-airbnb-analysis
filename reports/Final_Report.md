# Barcelona Airbnb Market Intelligence Report
## Expernetic Data Engineering Intern — Technical Assignment

**Prepared by:** [Your Name]
**Date:** July 2026
**Dataset:** Inside Airbnb — Barcelona, June 2026 Snapshot
**City analyzed:** Barcelona, Spain (single city, depth-first approach)

---

## Table of Contents
1. Executive Summary
2. Objectives & Scope
3. Dataset Overview
4. Methodology
5. Engineering Approach
6. EDA Findings
7. Statistical Findings
8. Data Science Experiments
9. AI/ML Experiments
10. Visualizations
11. Business Recommendations
12. Limitations & Caveats
13. Future Improvements
14. Reflection
Appendix A: AI Usage Disclosure

---

## 1. Executive Summary

Barcelona's Airbnb market (15,293 listings, June 2026) is a mature,
commercially-dominated short-term rental market operating under an
accelerating regulatory crisis. Despite surface-level diversity — 4,595
unique hosts, 69 neighbourhoods, price range €11–€4,463/night — market
supply is heavily concentrated: the top 10.9% of hosts control 61.1% of
all listings, and a single operator manages 588 properties (3.8% of the
entire city market).

The market's structural and regulatory risk is significant. Only 890
verified HUT tourist licenses were identified across 15,293 listings,
while all ~10,101 existing city-wide licenses expire in November 2028
and will not be renewed — a Constitutional Court-upheld policy that
effectively phases out Barcelona's entire short-term rental apartment
market within 28 months of this analysis.

Analytically, this study produced five statistically confirmed hypotheses,
a machine learning price model (XGBoost, R²=0.857, MAE=€50.36), VADER
sentiment analysis on 5,885 English reviews, and LLM-generated market
intelligence briefings. Key actionable findings: entire-home listings
command a 161% price premium over private rooms (large effect, p<0.001);
Eixample is the sole standalone premium neighbourhood tier; superhost
status drives 3x booking volume at equivalent pricing; and minimum stay
length — not location — is the single strongest price predictor,
indicating that the short-stay and medium-term rental markets coexist
in this dataset and should be modeled separately.

---

## 2. Objectives & Scope

### 2.1 Assignment Objectives
This assignment required transforming raw Inside Airbnb public data into
engineering artifacts, analytical insights, and business recommendations
across five domains: data engineering, EDA, statistical analysis, machine
learning, and applied AI/NLP.

### 2.2 City Selection Rationale
**Barcelona** was selected as the single analysis city for the following
reasons:
- Rich, well-populated dataset (15,293 listings) with sufficient statistical
  power for hypothesis testing and ML modeling
- Strong external context (2028 regulatory phase-out) enabling business
  narrative beyond pure data description
- Mid-size complexity: large enough for meaningful analysis, small enough
  for depth-first treatment within the assignment timeline
- Less overused than NYC/London in data portfolios

### 2.3 Prioritization Rationale
Per the assignment's explicit design philosophy ("quality outweighs
quantity"), this submission prioritizes:
- Exceptional depth in Sections 2-7 over superficial coverage of all sections
- Rigorous assumption-checking in statistical analysis over mechanical
  test-running
- Honest documentation of failures, corrections, and limitations over
  presenting only clean results
- Real, verified business context (regulatory research) over generic
  "business implications"

### 2.4 Sections Completed
[See README for full completion table]

---

## 3. Dataset Overview

### 3.1 Source
Inside Airbnb (insideairbnb.com), Barcelona snapshot scraped June 2026.
An independent, non-commercial project that scrapes publicly available
Airbnb listing data.

### 3.2 Files Used
| File | Rows | Columns | Notes |
|---|---|---|---|
| listings.csv.gz (detailed) | 15,293 | 90 | Primary listings data |
| calendar.csv.gz | 5,623,190 | 5 | Availability only, no price |
| reviews.csv.gz (detailed) | 1,033,523 | 6 | Full review text |
| neighbourhoods.csv | 75 | 2 | Boundary names |
| neighbourhoods.geojson | 75 features | — | Polygon boundaries |

### 3.3 Entity Relationships
- **Listings** (grain: 1 row/listing) — core entity
- **Calendar** (grain: 1 row/listing/date) — 365 days forward per listing
- **Reviews** (grain: 1 row/review) — historical guest reviews
- **Hosts** (derived from listings) — 4,595 unique hosts
- **Neighbourhoods** — 69 with listings, 75 boundary polygons

### 3.4 Key Limitations
1. **12 columns 100% null** in listings (host_since, host_response_time,
   instant_bookable, etc.) — scrape-level gap, not a local error
2. **Calendar lacks pricing** — only availability + min/max nights;
   revenue estimation relies on pre-computed `estimated_revenue_l365d`
3. **Review text is multilingual** — 41.1% non-English; VADER sentiment
   analysis limited to English subset
4. **Single snapshot** — no longitudinal data; temporal trends inferred
   from review dates and calendar forward-bookings only
5. **Self-reported fields** — license, property type, amenities are
   host-declared and unverified

### 3.5 Assumptions
[All assumptions documented in reports/assumptions_and_decisions_log.md]

---

## 4. Methodology

### 4.1 Overall Approach
A depth-first, single-city approach was chosen deliberately. The analytical
pipeline followed this sequence:
1. Profile raw data before writing any cleaning logic
2. Document every finding and decision before proceeding
3. Verify hypotheses about data quality programmatically, not visually
4. Apply demand segmentation early (active/dormant/never-active) so all
   downstream analysis uses appropriate population filters
5. Check statistical assumptions before selecting tests
6. Report effect sizes alongside p-values throughout

### 4.2 Key Method Selections
| Decision | Choice | Rationale |
|---|---|---|
| Storage format | DuckDB + Parquet | Column-oriented, fast for analytics, no server required |
| Statistical tests | Mann-Whitney U / Kruskal-Wallis | Price/rating data severely non-normal; non-parametric required |
| ML target | log(price) | Reduces right-skew; improves linear model performance |
| Sentiment tool | VADER | Designed for short informal text; no GPU required at this scale |
| LLM provider | Groq (Llama-3.3-70b) | Free tier, fast inference, sufficient for structured summarization |

---

## 5. Engineering Approach

### 5.1 Pipeline Architecture
data/raw/ (gz files)
↓ src/profile_data.py
Raw profiling report
↓ src/clean_listings.py
↓ src/clean_calendar.py
↓ src/clean_reviews.py
data/processed/ (parquet files)
↓ src/build_warehouse.py
data/warehouse.duckdb (star schema)
↓ notebooks/
EDA → Stats → ML → NLP
↓
reports/ (figures, logs, briefings, PDF)

### 5.2 Star Schema Design
**Dimensions:**
- `dim_host` — 4,595 rows (grain: host)
- `dim_listing` — 15,293 rows (grain: listing, descriptive attributes)
- `dim_neighbourhood` — 69 rows
- `dim_date` — 374 rows (generated from calendar range)

**Facts:**
- `fact_listing_performance` — 15,293 rows (grain: listing, numeric KPIs)
- `fact_calendar` — 5,581,945 rows (grain: listing × date)
- `fact_reviews` — 1,026,820 rows (grain: review)

**View:**
- `v_listing_demand` — demand segmentation (active/dormant/never_active)

### 5.3 Data Quality Results
| Dataset | Raw Rows | Issues Found | Action |
|---|---|---|---|
| Listings | 15,293 | 12 cols 100% null, 12.7% price null | Dropped dead cols, flagged |
| Calendar | 5,623,190 | No price column, 113 orphan IDs | Documented, LEFT JOIN |
| Reviews | 1,033,523 | 114 null text, 77 orphan IDs | Dropped nulls, LEFT JOIN |

### 5.4 Engineering Decision Log
Full decision log: `reports/assumptions_and_decisions_log.md`
Key decisions: Parquet over CSV, LEFT JOIN orphan strategy, demand
segmentation as warehouse view, log-transform of price target for ML.

### 5.5 Production Readiness Discussion
For production deployment this pipeline would require:
- **Orchestration:** Prefect or Airflow DAG replacing manual script execution
- **Incremental processing:** CDC on listings table (track scrape_id changes)
- **Monitoring:** Row-count and null-rate alerts on each pipeline stage
- **Containerization:** Docker compose for environment reproducibility
- **Cloud:** S3 (raw) → Glue (transform) → Athena/Redshift (warehouse)
- **Multi-city scaling:** Config-driven city parameter; parallel processing
  via Prefect map() or Spark for 50+ cities

---

## 6. EDA Findings

### 6.1 Price Distribution
- Active listing median: €213.68 (vs full dataset €177.67 — dormant
  listings skew cheap, not expensive)
- Strong right-skew (skewness 5.42 for entire homes)
- Entire homes: median €240.10 | Private rooms: median €92.00
- 36.8% of listings have zero trailing-12-month occupancy

**Business interpretation:** The "typical" Barcelona Airbnb costs €213.68/
night for an active listing — 20% higher than naive statistics suggest,
because inactive/dormant listings disproportionately skew cheap.
Benchmarking against the raw mean (€240.62) or median (€177.67) without
demand-filtering would either over- or understate competitive pricing.

### 6.2 Geographic Analysis
Three clear pricing tiers by neighbourhood group (all differences
statistically confirmed in Section 7 / H4):
- **Premium:** Eixample (€257.50) — standalone tier
- **Upper-mid:** Sant Martí (€230), Gràcia (€217)
- **Mid:** Les Corts (€193), Sants-Montjuïc (€201)
- **Budget:** Nou Barris (€80), Sant Andreu (€92), Ciutat Vella (€125)

Notable: Barceloneta (famous beachfront) sits in the budget tier at
€88.86 median — a bimodal market of budget tourist flats and premium
beachfront properties, with the median reflecting the larger budget
segment rather than the neighbourhood's reputation.

### 6.3 Temporal & Seasonal Trends
- Historical demand (reviews) peaks in **May**
- Forward booking pressure (calendar availability) tightest in **July**
- Divergence explained by booking lead-time, not contradiction
- Winter trough: Dec-Jan (~35% below peak)
- Recency artifact: 2026-06/07 review data truncated to avoid
  misrepresenting the review-lag as a real demand drop

### 6.4 Host & Supply-Side Analysis
- Top 500 hosts (10.9%) control 61.1% of all listings
- Single largest operator: 588 listings (3.8% of market)
- 65% of hosts are single-listing casual operators — but control only
  19.5% of inventory
- Superhost listings: 25.9% of market, 3x more reviews, same price

### 6.5 Review & Demand-Side Analysis
- Price ↔ rating correlation: r=0.073 (essentially zero)
- High-review-count + low-rating: only 3 listings total (near-absent pattern)
- 2 of those 3 share the same host — not an independent market segment

---

## 7. Statistical Findings

All tests used non-parametric methods (Mann-Whitney U / Kruskal-Wallis)
due to confirmed violations of normality (skewness 2.53–8.97) and
homoscedasticity (Levene's p<0.000001) in all tested variables.

| Hypothesis | Test | Result | Effect Size | Conclusion |
|---|---|---|---|---|
| H1: Entire home > private room price | Mann-Whitney U | p<0.001 | r=0.695 (large) | ✅ Confirmed |
| H2: Superhost rating > non-superhost | Mann-Whitney U | p<0.001 | r=0.438 (moderate-large) | ✅ Confirmed |
| H3: 10+ reviews ≠ fewer reviews (price) | Mann-Whitney U | p<0.001 | r=0.544 (large) | ✅ Confirmed |
| H4: Neighbourhood prices differ | Kruskal-Wallis + Dunn's | p<0.001 | η²=0.095 | ✅ Confirmed |
| H5: Weekend ≠ weekday availability | Chi-square | p<0.001 | V=0.0035 (negligible) | ⚠️ Stat. sig, not practical |

**Key insight:** H5 demonstrates that statistical significance without
effect size is misleading — with 5.58M observations, a 0.39pp difference
tests as "significant" but is operationally meaningless. This is why
effect size reporting is non-negotiable.

**H4 detail:** 33 of 45 Bonferroni-corrected pairwise comparisons are
significant. Non-significant pairs reveal the three-tier structure.
Neighbourhood group explains 9.5% of price variance (η²=0.095) — real
but not the dominant driver (room type from H1 explains more).

---

## 8. Data Science Experiments

### 8.1 Feature Engineering
27 features engineered from raw data including:
- Physical attributes: accommodates, bathrooms, bedrooms, beds,
  beds_per_person
- Listing type: room_type (ordinally encoded), minimum_nights
- Performance: number_of_reviews, review_scores (4 dimensions),
  estimated_occupancy_l365d, availability_365
- Host: host_listings_count, hosts_time_as_host_years, is_superhost
- Geography: 9 neighbourhood group dummies (Eixample as reference dropped)
- Engineered flags: has_rating, beds_per_person

Target: log1p(price) — log transform reduces right-skew, improves
linear model calibration, and allows inverse-transform for interpretable
MAE/RMSE reporting.

### 8.2 Model Comparison (5-fold CV)
| Model | R² | MAE (EUR) | RMSE (EUR) | Train R² | Overfit |
|---|---|---|---|---|---|
| Ridge Regression | 0.805 | €57.13 | €115.50 | 0.808 | 0.003 |
| Random Forest | 0.845 | €52.48 | €110.64 | 0.913 | 0.068 |
| **XGBoost** | **0.857** | **€50.36** | **€106.86** | 0.940 | 0.083 |

XGBoost selected as final model. MAE of €50.36 on median price of
€213.68 = ~23.6% median error — respectable for unstructured real-world
data without text/amenity features.

### 8.3 SHAP Feature Importance
Top 5 by mean |SHAP value|:
1. `minimum_nights` (0.378) — dominant predictor; acts as short-stay vs
   medium-term rental proxy, not a genuine "minimum stay preference" signal
2. `room_type_enc` (0.171) — consistent with H1's 161% price gap finding
3. `accommodates` (0.138) — physical capacity outweighs location
4. `bathrooms` (0.050)
5. `number_of_reviews` (0.045)

`ng_Eixample` (0.025) is the only neighbourhood dummy with meaningful
weight — corroborates H4's finding that Eixample is the sole standalone
premium tier.

### 8.4 Residual Analysis
- Entire home / Private room: near-zero median error (well-calibrated
  for 95% of data)
- Shared rooms: -10.5% median error (systematic overprediction —
  thin segment with insufficient training signal)
- Nou Barris: -15.5% mean error (model anchors too high for cheapest
  district — training signal dominated by pricier central listings)
- Horta-Guinardó: std=50.47% (highest variance — small-n heterogeneous
  district)

---

## 9. AI/ML Experiments

### 9.1 Sentiment Analysis (VADER)
- 10,000 reviews sampled, language-detected via langdetect
- 58.9% English (5,885 reviews) used for VADER analysis
- Non-English correctly excluded (VADER is English-only; multilingual
  reviews artifactually score as neutral)

**Correlations (VADER compound vs numerical scores):**
| Score dimension | r |
|---|---|
| Overall rating | 0.245 |
| Value | 0.236 |
| Cleanliness | 0.209 |
| Location | 0.117 |
| Comment length | 0.071 |

Location shows the weakest correlation — guests describe location
objectively ("close to metro") rather than emotionally, so sentiment
tools miss this dimension. Sentiment and numerical scores are
complementary signals, not substitutes.

### 9.2 LLM Market Intelligence (Groq / Llama-3.3-70b)
A structured market intelligence briefing generator was built, taking
verified analytical findings as grounded context and generating:
- Executive briefing (C-suite / investor audience)
- Host briefing (individual property operator audience)

Key design choices to minimize hallucination:
- `temperature=0.3` (low, favors factual output)
- Explicit instruction: "only reference findings explicitly stated above"
- All statistics pre-computed from data, passed as context — LLM acts
  as writer, not analyst
- Output saved to `reports/market_intelligence_briefings.txt`

Full briefing text available in reports directory.

---

## 10. Visualizations

| Figure | File | Section |
|---|---|---|
| Price distribution + by room type | price_distribution.png | 6.1 |
| Neighbourhood price choropleth | price_choropleth.png | 6.2 |
| Seasonality dual-axis (reviews + availability) | seasonality_dual.png | 6.3 |
| Neighbourhood price tiers bar chart | neighbourhood_price_tiers.png | 6.4 |
| SHAP feature importance (bar) | shap_importance.png | 8.3 |
| SHAP beeswarm | shap_beeswarm.png | 8.3 |
| Residual analysis (4-panel) | residual_analysis.png | 8.4 |
| Sentiment analysis (3-panel) | sentiment_analysis.png | 9.1 |

All figures saved to `reports/figures/` at 150 DPI.

---

## 11. Business Recommendations

Based on the full analysis, the following recommendations are made for
different market participants:

### For Hosts (Current Operators)
1. **Target Superhost status as a primary growth lever** — superhosts
   receive 3x more bookings at equivalent pricing. The return on service
   quality investment is booking volume, not higher prices.
2. **Price against active-market medians, not city averages** — the
   active-listing median (€213.68) is 20% higher than the naive median
   (€177.67). Hosts benchmarking against the wrong baseline are
   systematically underpricing.
3. **Prepare for 2028 now** — all HUT licenses expire November 2028.
   Hosts without a verified HUTB license (61.5% of listings by our
   analysis) face either compliance costs or forced exit from the market.

### For Investors
1. **Eixample is the only neighbourhood tier that is statistically
   distinct from all others** — for premium positioning, it is the
   defensible choice. Upper-mid tier neighbourhoods (Sant Martí, Gràcia)
   are statistically indistinguishable from each other, offering
   competitive supply but no exclusive premium.
2. **Avoid interpreting the 2028 phase-out as an opportunity** — the
   supply reduction will not benefit remaining operators, as the legal
   mechanism eliminates the entire HUT license category, not just
   individual competitors.
3. **Revenue ceiling is capacity-driven, not location-driven** — the
   ML model shows `accommodates` (physical capacity) outweighs
   neighbourhood dummies in price prediction. Larger-capacity properties
   offer more pricing headroom than location premiums.

### For Platform/Policy Analysts
1. **Market concentration warrants monitoring** — 0.2% of hosts control
   17.6% of listings. The market structure is closer to institutional
   property management than peer-to-peer home-sharing.
2. **License compliance is a major gap** — only 890 verified HUTB
   numbers found across 15,293 listings (5.8%). Even accounting for
   multi-unit licenses and exemption categories, a large share of
   listings operate in a legally grey zone.
3. **Sentiment and numerical scores provide complementary signals** —
   a combined review-quality metric (NLP sentiment + structured scores)
   would more accurately surface listings with genuine guest goodwill
   vs. structurally inflated ratings.

---

## 12. Limitations & Caveats

1. **Single snapshot** — all findings reflect a single scrape (June 2026).
   No longitudinal tracking; trends inferred indirectly.
2. **Calendar lacks pricing** — revenue analysis relies on
   `estimated_revenue_l365d` (Inside Airbnb pre-computed), not
   independently derived.
3. **12 columns permanently null** — host_since, host_response_time,
   instant_bookable and 9 others are missing for all 15,293 listings
   in this scrape; host tenure and response analysis are unavailable.
4. **VADER English-only** — 41.1% of reviews excluded from sentiment
   analysis. Multilingual coverage requires GPU-scale transformer models.
5. **ML model biases** — systematic underprediction in Nou Barris
   (-15.5%) and shared rooms (-14.1%); model not suitable for pricing
   advice in these segments without correction.
6. **License field is self-reported** — HUTB numbers are host-declared;
   validity/expiry status not independently verified.
7. **Inside Airbnb methodology** — data is scraped, not from Airbnb's
   API. Prices may reflect listed rather than booked rates.

---

## 13. Future Improvements

1. **Amenity text features** — parse the `amenities` JSON field into
   binary flags (pool, parking, balcony etc.) and include in ML model;
   estimated R² improvement of 0.02-0.05
2. **Multilingual sentiment** — XLM-RoBERTa covers 100+ languages;
   would extend NLP analysis to full review corpus
3. **Separate medium-term rental model** — `minimum_nights` dominates
   as a proxy for rental type; training separate models for short-stay
   vs. medium-term would improve both
4. **Longitudinal scraping** — scrape the same city across multiple
   months to enable real trend analysis rather than snapshot inference
5. **Listing description NLP** — topic modeling and keyword extraction
   on listing descriptions to identify quality/positioning signals
6. **True RAG system** — build a retrieval-augmented generation system
   over the full reviews corpus for natural-language market querying
7. **Production MLOps** — implement model retraining pipeline triggered
   by new scrape ingestion, with drift monitoring and automated alerts

---

## 14. Reflection

### Prioritization decisions
This submission deliberately chose depth over breadth. The decision to
analyze a single city (Barcelona) rather than multiple cities was made
early and held consistently — the assignment's own design philosophy
states "quality outweighs quantity," and the regulatory context of
Barcelona offered external richness that a multi-city surface-level
approach would have sacrificed.

The most time-intensive phases were data engineering (Phase 3) and
statistical analysis (Phase 5) — both high-weight rubric categories.
The ML section (Phase 6) was scoped deliberately to 3 models with
rigorous validation rather than a larger experiment grid, prioritizing
honest residual analysis over impressive-looking leaderboard numbers.

### What I would do differently
- Build the demand segmentation view earlier (Phase 2 rather than
  discovering it mid-EDA) to avoid retroactive filtering across notebooks
- Include amenity features from the start of ML feature engineering
- Attempt multilingual sentiment analysis via a HuggingFace zero-shot
  classifier rather than defaulting to English-only VADER

### Key lessons
- Real data is messier than assignment descriptions suggest — 12 columns
  being 100% null across an entire dataset is not in any textbook example
- Effect size is not optional — H5 demonstrated this conclusively with
  5.58M observations producing p<10⁻¹⁶ for a 0.39pp difference
- Document decisions as you make them, not from memory — the
  assumptions log written in real-time was far more accurate and detailed
  than anything reconstructable retrospectively

---

## Appendix A: AI Usage Disclosure

### Tools Used
| Tool | Version/Model | Purpose |
|---|---|---|
| Claude (Anthropic) | claude-sonnet-4-6 | Project guidance, code review, debugging assistance, report structure |
| Groq API | llama-3.3-70b-versatile | LLM market intelligence briefing generation (Section 9.2) |

### AI-Assisted Sections
- **Code:** Pipeline scripts and notebook cells were developed interactively
  with Claude providing initial implementations which were then debugged,
  corrected, and validated against actual data outputs
- **Report structure:** Report outline and section framing developed with
  Claude; all numerical findings, interpretations, and business conclusions
  are the candidate's own analysis of real data
- **LLM briefings (Section 9.2):** Generated by Llama-3.3-70b via Groq;
  all input statistics are pre-computed from the actual dataset; LLM acted
  as writer not analyst

### Key Prompts
Available on request — the primary interaction was a multi-turn guided
analysis session where each phase was completed, verified with real data
output, and corrected before proceeding. Full conversation available.

### Output Validation
- Every code block was run against actual data and output verified before
  proceeding
- Statistical results cross-checked for direction and magnitude plausibility
- LLM briefing output verified against source findings for factual accuracy
- Multiple AI-suggested conclusions were rejected or modified when data
  contradicted them (documented in assumptions log: Finding 12 geojson
  encoding, Finding 11 Barceloneta hypothesis)

### Critical Assessment
Claude's initial hypothesis about the geojson encoding issue (Finding 12)
was wrong — the garbling was a terminal display artifact, not a real data
mismatch. This was caught by programmatic equality testing rather than
accepting the visual output. Similarly, the initial Barceloneta "small
subdivided apartments" hypothesis (Finding 11) was incorrect and corrected
by the data. These corrections are documented in the assumptions log as
examples of not accepting AI-generated hypotheses without verification.

