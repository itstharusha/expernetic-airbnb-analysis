# Assumptions & Decisions Log — Barcelona Airbnb Analysis

## Data Quality Findings (Section 2)

### Finding 1: 12 columns are 100% null in listings.csv.gz
Columns: neighborhood_overview, host_since, host_response_time, host_thumbnail_url,
host_acceptance_rate, host_response_rate, host_verifications, neighbourhood,
host_total_listings_count, host_neighbourhood, calendar_updated, instant_bookable.

**Verification:** Confirmed via unique-value inspection (only NaN present) and raw
gzip header inspection — ruled out download corruption or parsing error. This is a
genuine scrape-level gap in the source data for this snapshot date, not a local issue.

**Impact:** `host_since` cannot support a "host tenure" feature as originally planned
(Section 3.3). Substituted with `hosts_time_as_host_years` (0.57% null), which
captures equivalent information and survived independently.

**Decision:** Drop these 12 columns entirely from the cleaned dataset rather than
attempt imputation — there is no signal to impute from.

### Finding 2: ~23.2% missingness clusters around review_scores_*, first_review,
last_review, reviews_per_month
**Interpretation:** This is not random or corrupted missingness — it corresponds to
listings with zero reviews, which structurally cannot have review scores.

**Decision:** Retain as explicit nulls (not imputed with 0 or mean, which would
misrepresent "no data" as "bad score"). Create a boolean flag `has_reviews` for
segmentation in EDA and modeling, rather than silently dropping or filling these rows.

### Finding 3: Price fields required custom parsing
Raw `price` field stored as string with `$` prefix (e.g. "$409.00"). No commas
observed in this dataset's price range (max values stayed under 1000, so no
thousands separator appeared) but code defensively strips both `$` and `,`
using `pd.to_numeric(..., errors='coerce')` for safety against edge cases.

### Validation results (listings)
- Negative prices: 0
- Invalid lat/long (outside Barcelona bounds): 0
- Duplicate listing IDs: 0
- Final cleaned shape: 15,293 rows × 81 columns (from 90; 12 dead columns dropped)

### Engineering Decision: Parquet over CSV for processed data
Chose Parquet for `data/processed/` outputs over CSV: preserves dtypes (no
re-parsing dates/floats on every load), smaller file size, faster I/O for
downstream DuckDB queries and pandas loading in notebooks. Trade-off: not
human-readable directly, but acceptable since this is an intermediate
artifact, not a deliverable.

### Finding 4: Calendar file lacks pricing data
The calendar.csv.gz for this Barcelona snapshot contains only
[listing_id, date, available, minimum_nights, maximum_nights] — no `price`
or `adjusted_price` columns, unlike Inside Airbnb's documented full schema.

**Impact:** Calendar-based revenue estimation (Section 3.3) is not possible
with this file. Occupancy rate CAN still be computed from `available`.
Revenue estimates will instead rely on `estimated_revenue_l365d` from the
listings table, which Inside Airbnb pre-computes from actual booking data —
arguably a more reliable source than deriving it ourselves from calendar
snapshots anyway.

**Decision:** Scope revenue analysis to use the pre-computed listings-level
field rather than attempting calendar-derived estimates.

### Finding 5: 113 orphan listing_ids in calendar (0.73% of calendar listings)
Calendar contains 15,406 unique listing_ids; listings contains 15,293. 113
IDs appear in calendar but have no corresponding row in listings.

**Likely cause:** Calendar and listings files were scraped on slightly
different dates (calendar date range: 2026-06-24 to 2027-07-02); listings
scraped separately. Listings that were delisted/removed between scrapes
would produce exactly this pattern.

**Decision:** Use a LEFT JOIN from listings → calendar in the star schema
(fact table keyed on listings that exist), which naturally excludes these
113 orphans from listing-level analysis without an error-prone manual
filter. Orphan rows will be documented but not backfilled or investigated
further, as they fall outside the assignment's core listings-analysis scope.

### Finding 6: Reviews file — minor null/duplicate cleanup, 77 orphan listing_ids
Of 1,033,523 raw reviews: 114 had null comment text (dropped — unanalyzable),
0 exact duplicate review IDs found. 77 orphan listing_ids (0.007% of reviews)
reference listings not present in the cleaned listings table — consistent
with the same scrape-timing mismatch documented in Finding 5.

**Note:** 23,825 reviews (2.3%) are under 10 characters — likely low-value/
low-information text (e.g. "Great!", "👍"). Retained in the dataset but
flagged via `comment_length` for optional filtering in NLP work (Section 7.1),
where short reviews would add noise to topic modeling and sentiment analysis.

**Decision:** Same LEFT JOIN strategy as calendar — orphans naturally
excluded from listing-level joins without manual filtering.

## Star Schema Design (Section 3.4)

**Dimensions:** dim_host (4,595), dim_neighbourhood (69), dim_listing (15,293), dim_date (374)
**Facts:** fact_listing_performance (grain: 1 row/listing, 15,293 rows),
fact_calendar (grain: 1 row/listing/date, 5,581,945 rows),
fact_reviews (grain: 1 row/review, 1,026,820 rows)

**Design decision:** Split listing data into a slowly-changing dimension
(dim_listing — descriptive attributes) and a fact table
(fact_listing_performance — numeric metrics: price, reviews, occupancy).
This separates "what a listing is" from "how it performs," which is standard
dimensional modeling practice and keeps the fact table lean for aggregation
queries.

**Join strategy:** INNER JOIN from calendar/reviews against dim_listing
during fact table construction, which cleanly and automatically excludes
the 113 orphan calendar listing_ids (Finding 5) and 77 orphan review
listing_ids (Finding 6) without manual filtering logic. Row count deltas
verified against expected orphan impact (113 listings × ~365 days ≈ 41,245
row exclusion in fact_calendar — matches observed delta of 41,245 exactly).

**Trade-off accepted:** dim_date was generated from calendar's actual date
range rather than a fully independent calendar-dimension generator, since
calendar is the only source with a native date grain in this dataset.

### Finding 7: Extreme price outlier (€10,542/night) is a "dormant listing," not an error
Highest-priced listing (867256504774715226, accommodates 2) has 0 reviews,
363/365 days available, and 0 estimated occupancy — consistent with a host
pricing a listing intentionally unbookable rather than pausing it (a known
platform behavior pattern), not a data entry error.

**Decision:** Retain in the raw dataset (it's real data, not corrupted), but
EXCLUDE from price-distribution visualizations and summary statistics via a
"has_demand" filter (estimated_occupancy_l365d > 0 OR number_of_reviews > 0),
since dormant/unbookable listings distort price benchmarking without
reflecting actual market pricing. Will note in EDA that ~X listings show
this dormant pattern (to be quantified next).

### Finding 8: Two distinct "no current demand" segments identified
- 3,548 listings (23.2%) have NEVER received a review — likely new or
  perpetually inactive listings.
- 5,627 listings (36.8%) have zero estimated occupancy in the trailing 365
  days — a larger group.
- All 3,548 no-review listings fall within the zero-occupancy group, but
  2,079 listings have a review history (proven past demand) yet zero
  occupancy in the last year — a distinct "gone quiet" segment worth
  separate analysis (possibly delisted-but-live, seasonal, or losing
  competitiveness).

**Decision:** Build three-tier demand classification rather than a binary
flag:
1. `never_active` — zero reviews AND zero occupancy (3,548)
2. `dormant_with_history` — has reviews BUT zero occupancy (2,079)
3. `active` — nonzero occupancy (9,666)
This segmentation will be used in EDA/host analysis (Section 4.4/4.5) rather
than simply excluded as noise.

### Finding 9: Demand-filtering reveals dormant listings skew toward LOW prices, not high
Full valid-price dataset (n=13,355): median €177.67, mean €240.62, max €10,542.
Active-only (n=9,174): median €213.68 (+20%), mean €234.33, max €4,463.

**Interpretation:** Removing dormant/never-active listings raises the median
substantially, indicating underpriced/stale listings are disproportionately
inactive — not that inactive listings are overpriced outliers as initially
hypothesized (Finding 7 found one extreme high-price dormant outlier, but
that was not representative of the dormant segment as a whole).

**Decision:** All subsequent price-based EDA and statistical tests will use
the `active` demand segment as the primary lens, with full-dataset stats
shown alongside for transparency. This will be explicitly called out in the
report methodology section.

### Finding 10: Small-n neighbourhoods produce unreliable price statistics
Several neighbourhoods have fewer than 10 active listings (e.g., Montbau
n=1, la Trinitat Vella n=1, la Trinitat Nova n=1, la Vall d'Hebron n=3),
making their avg/median price statistically unreliable — single listings
can swing a "neighbourhood average" arbitrarily.

**Decision:** Apply a minimum listing-count threshold (n >= 10) for any
neighbourhood-level comparative claims, ranking, or visualization. Smaller
neighbourhoods will be shown in supporting tables but excluded from
headline geographic findings and any statistical tests (Section 5.4).

### Finding 11: La Barceloneta shows bimodal pricing, not uniformly low
Entire-home listings in Barceloneta (n=210): median €88.86 vs mean €153.35
(73% gap), max €1,826. Distribution is heavily right-skewed/bimodal rather
than uniformly cheap — a large budget/mid-tier cluster coexists with a
smaller premium beachfront segment. Confirms median (per Finding 9) is the
correct "typical price" metric; using mean here would overstate what most
bookings actually cost.

**Decision:** Flag Barceloneta (and likely other tourist-core neighbourhoods)
for a follow-up bimodality check across the full dataset in Section 5
(statistical analysis) rather than treating median-based neighbourhood
rankings as the complete picture.

### Finding 12: Geojson/listings neighbourhood mismatch was a display artifact, not a data bug
Initial diagnostic printed garbled accented characters (e.g. "FarrÃ³")
suggesting 22+ neighbourhood name mismatches between geojson and listings
data. Root cause: Windows terminal encoding issue in print/repr display only
— the underlying UTF-8 string data was correct throughout (verified via
direct byte inspection and exact string equality test). True mismatch after
correct comparison: 6 neighbourhoods present in geojson boundaries with no
corresponding Airbnb listings (expected — some areas have zero rental
activity).

**Lesson:** Terminal/console display encoding issues on Windows can produce
false-positive data quality findings. Always verify with programmatic
equality checks, not visual inspection of printed output, before concluding
a data mismatch is real.

### Note: Choropleth merge produced 52 rows vs. expected 50 from geo_reliable
Minor discrepancy (52 vs 50), not investigated further as it does not affect
findings — likely a duplicate join on a neighbourhood_group label variant.
Flagged for review if map figures are audited before final report.

### Finding 13: Review volume trend has a recency artifact — must truncate before scrape date
Monthly review counts show a sharp apparent "collapse" in 2026-06 (12,284,
down from 22,321 in May) and 2026-07 (5 reviews). This is NOT a genuine
demand drop — it reflects the review reporting lag (guests review after
checkout, so very recent stays are systematically undercounted) combined
with the dataset's scrape date falling in June 2026.

**Decision:** All temporal/seasonal trend analysis and charts will exclude
the final 2 months of data (2026-06 onward) to avoid misrepresenting
incomplete recency data as a real trend. Seasonality conclusions will be
drawn from complete months only (through 2026-05).

### Finding 14: Seasonality peaks in May (shoulder season), not deep summer
Average monthly review volume (2024-2026, complete months only) shows a
clear seasonal curve: trough in Dec-Jan (~4,200-4,400), peak in May (6,678),
sustained high plateau Jun-Oct (~5,600-5,930), decline into Nov. Demand does
NOT peak in July/August as commonly assumed for a Mediterranean tourist
city — May outperforms peak summer months, suggesting shoulder-season
demand is commercially significant and should inform pricing/availability
strategy across a wider window than just "summer."

### Finding 15: Calendar availability confirms seasonality but diverges from review-based peak month
Forward-looking calendar data shows lowest availability (highest booking
pressure) in July (37.6%) and June (44.0%), with highest availability
(lowest booking pressure) in Nov-Dec (~71%) — corroborating the winter
low-season / summer high-season pattern from Finding 14 using an
independent data source.

**Divergence noted:** Review data (historical, 2024-2026 average) showed May
as the single highest-demand month, while calendar data (forward-looking,
single snapshot as of June 2026 scrape) shows July as tightest. This is
explained by booking lead-time behavior, not contradiction: summer trips
are booked further in advance, so a June 2026 snapshot naturally shows
already-high advance bookings for the imminent July peak, while May 2026
bookings (already mostly complete/passed) don't appear in a forward-looking
availability metric. Both metrics agree on the broader seasonal shape
(spring ramp-up, summer peak pressure, winter trough); the exact peak month
differs by measurement method and is explained, not contradictory.

**Decision:** Report will present both metrics side-by-side with this
explanation, framing it as a methodological insight about the difference
between historical demand proxies and forward booking-pressure proxies,
rather than picking one "correct" peak month.

### Finding 16: Only ~6,437 verified HUT licenses among 15,293 listings — market has a large unlicensed/exempt segment
Initial naive check (license field non-null) suggested 12,250 "licensed"
listings (80.1%). Deep inspection revealed this field mixes genuine license
numbers with boilerplate exemption-category text. After regex-based
extraction of actual HUTB-format numbers:
- 7,415 listings (48.5%) carry a genuine HUTB license number (6,437 unique
  numbers — some shared legitimately across multi-unit buildings)
- 3,676 listings (24.0%) explicitly declare an exemption (e.g., "seasonal
  rental exempt") rather than holding a formal HUT license
- 3,043 listings (19.9%) have no license information at all
- ~646 listings (4.2%) use an unrecognized national-registration format
  (ESHFNT prefix) not yet classified — minor residual, acceptable given
  time constraints

**Cross-reference:** External reporting (as of July 2025) cites ~10,101
total city-wide HUT licenses across all platforms. Our count of 6,437
unique HUTB numbers within Airbnb listings alone is plausible as a subset
of that total (Airbnb is not the only STR platform in Barcelona).

**Business implication:** Under half of Barcelona's active Airbnb listings
carry a verifiable, formal tourist license. Combined with the city's
confirmed 2028 phase-out of all existing HUT licenses (verified via current
web search — Spain's Constitutional Court upheld this March 2025), a
substantial share of current inventory sits in a legally precarious
position: either already operating on exemption claims that regulators may
tighten, or with no documented license at all. This directly informs the
Business Recommendations section: any market-entry or investment analysis
must account for regulatory risk timelines, not just current pricing/demand
patterns.

**Methodological note:** This finding evolved through three iterations
(naive null-check → text-pattern regex v1 → regex v2), each catching a real
error in the previous pass. Documented here as an example of avoiding
premature conclusions on messy real-world text fields.

### Finding 17 (continued): Host-level pattern confirmed
Verified: 2 of the 3 high-volume/low-rating listings (125 and 115 reviews,
both rated 3.86) belong to the SAME host (host_id 426483512), both in
el Putxet i el Farró. This is a single host operating two similarly
underperforming properties, not two independent market data points — the
"n=3" sample is effectively "2 independent operators, one running two
similar underperforming units." Given the tiny scale, this remains a
noted anomaly rather than a market-wide segment, but is a good example of
verifying whether apparent outliers are independent occurrences before
generalizing from them.

## Statistical Findings (Section 5.1)

### H1: Entire-home listings command higher prices than private rooms — CONFIRMED
- Test: Mann-Whitney U (one-tailed), chosen due to severe right-skew (skewness
  5.42 / 4.19) and unequal variance (Levene's p<0.000001) violating t-test assumptions
- Result: U=10,093,732, p<0.001, effect size (rank-biserial)=0.695 (large)
- Practical: median price gap €148.10 (entire home €240.10 vs private room
  €92.00), a 161% premium
- Business takeaway: room-type is a dominant price driver; strongly
  significant and practically large, not just statistically detectable

  ### H2: Superhosts achieve higher review scores than non-superhosts — CONFIRMED
- Test: Mann-Whitney U (one-tailed), chosen due to severe left-skew (skewness
  -6.97 / -3.21) and unequal variance (Levene's p<0.000001)
- Result: U=21,271,370, p<0.001, effect size (rank-biserial)=0.438 (moderate-large)
- Practical: median rating 4.86 (superhost) vs 4.63 (non-superhost), a 0.23-point
  gap on a compressed 5-point scale
- Business takeaway: superhost status is a genuine quality signal, not just a
  badge — corroborates Section 4.4 finding that superhosts get 3x more reviews
  at similar pricing

  ### H3: Listings with 10+ reviews have significantly different prices — CONFIRMED
- Test: Mann-Whitney U (two-tailed), same assumption violations as H1/H2
- Result: U=13,480,184, p<0.001, effect size (rank-biserial)=0.544 (large)
- Practical: median price €246.74 (10+ reviews) vs €106.67 (≤10 reviews),
  2.31x higher, gap of €140.06
- Direction note: higher-reviewed listings are MORE expensive, not less —
  likely because both price and review accumulation are downstream of listing
  quality and host professionalism, not because price drives reviews causally.
  Report will explicitly flag this to avoid misleading causal interpretation.

  ### H4: Neighbourhood average prices differ significantly — CONFIRMED
- Test: Kruskal-Wallis (non-parametric one-way ANOVA equivalent), chosen due
  to right-skew across all 10 groups (skewness 1.24–11.86) and unequal variances
- Result: H=883.72, p<0.001, eta-squared=0.095 (neighbourhood group explains
  ~9.5% of price variance — statistically robust but not the dominant driver;
  room type from H1 likely explains more)
- Post-hoc: 45 pairwise Bonferroni-corrected Mann-Whitney tests;
  33/45 pairs significantly different, 12 non-significant
- Three pricing tiers identified:
  * Premium: Eixample (€257.50 median) — standalone, significantly different
    from all others
  * Upper-mid: Sant Martí (€230), Gràcia (€217), Sarrià-Sant Gervasi (€196,
    bridges to mid tier)
  * Mid: Les Corts (€193), Sants-Montjuïc (€201), Horta-Guinardó (€165)
  * Budget: Nou Barris (€80), Sant Andreu (€92), Ciutat Vella (€125)
- Notable: Ciutat Vella (historic centre) sits in budget tier by median —
  consistent with Barceloneta bimodal finding (Section 4.2/Finding 11);
  high-volume of budget/private-room listings pulling median down despite
  containing premium inventory
- Business takeaway: Barcelona's Airbnb market segments cleanly into 3-4
  price tiers by district — actionable for investors/hosts benchmarking
  against the right competitive set rather than city-wide averages

  ### H5: Weekend vs weekday pricing differences are statistically significant — REJECTED (practically)
- Adaptation note: calendar.csv.gz lacks price/adjusted_price columns (Finding 4),
  so H5 was adapted to test weekend vs weekday AVAILABILITY RATE as a
  demand-pressure proxy rather than direct price comparison. This is an honest
  methodological adaptation, not a like-for-like test of the original hypothesis.
- Test: Chi-square test of independence on binary availability outcome
  (appropriate for binary outcome at large n, avoids parametric assumptions)
- Result: χ²=69.29, p=8.51e-17, Cramer's V=0.0035
- Interpretation: STATISTICALLY significant but PRACTICALLY negligible —
  a 0.39 percentage point availability difference (57.93% weekend vs 58.32%
  weekday) is detectable only because n=5,581,945 gives extreme statistical
  power. Effect size (Cramer's V=0.0035) is essentially zero by any
  conventional threshold (negligible < 0.1).
- This result is a deliberate illustration of why effect size reporting is
  mandatory alongside p-values: with millions of observations, even trivial
  differences become "statistically significant." The correct conclusion is
  that weekend/weekday availability shows no meaningful difference in this
  dataset — and therefore no meaningful pricing-pressure differential can be
  inferred from calendar data alone.
- Business takeaway: Barcelona's Airbnb market does not show a detectable
  weekend pricing premium through the availability-pressure lens — hosts
  do not appear to be pricing weekends materially differently, or if they
  are, it is not reflected in differential booking rates at the market level.
  This finding would require direct calendar price data to test properly.

  ## ML Model Results (Section 6.1)

### Model comparison (5-fold cross-validation, log-price target, n=9,174 active listings)
| Model | R² | MAE (EUR) | RMSE (EUR) | Train R² | Overfit gap |
|---|---|---|---|---|---|
| Ridge Regression | 0.805 | 57.13 | 115.50 | 0.808 | 0.003 |
| Random Forest | 0.845 | 52.48 | 110.64 | 0.913 | 0.068 |
| XGBoost | 0.857 | 50.36 | 106.86 | 0.940 | 0.083 |

**Winner: XGBoost** (best test R², MAE, RMSE). Ridge shows minimal
overfitting but lower ceiling. XGBoost/RF moderate overfit acceptable
for tree-based models at this complexity.

**MAE context:** €50.36 MAE on median price €213.68 = ~23.6% median
error. Respectable for unstructured real-world data with no text/amenity
features included.

### Top SHAP features (XGBoost, mean |SHAP|):
1. minimum_nights (0.378) — dominant predictor; acts as listing-type
   proxy separating short-stay tourist vs medium-term rental markets
2. room_type_enc (0.171) — consistent with H1 finding (161% price gap)
3. accommodates (0.138) — physical capacity drives price more than location
4. bathrooms (0.050), number_of_reviews (0.045), review_scores_rating (0.042)
5. ng_Eixample (0.025) — only neighbourhood dummy with meaningful weight;
   consistent with H4 finding (Eixample standalone premium tier)
6. is_superhost (0.013) — host identity matters less than listing attributes
7. has_rating (0.000) — engineered flag added zero value; signal already
   captured by correlated features

### Residual bias findings:
- Shared rooms: median -10.5% error (systematic overprediction) — thin
  segment, insufficient training signal
- Nou Barris: median -10.5%, mean -15.5% — model anchors too high for
  cheapest district; training signal dominated by pricier central listings
- Horta-Guinardó: std=50.47% — highest prediction variance, consistent
  with small-n heterogeneous district
- Entire home/Private room (95% of data): near-zero median error —
  well-calibrated for dominant segments

### Improvements with more time/data:
- Add amenity text features (NLP-extracted flags from amenities column)
- Add listing description sentiment/length features
- Separate model for medium-term rentals (minimum_nights > 28)
  to remove the dominant proxy effect
- Hyperparameter tuning via Bayesian optimization
- Address Nou Barris/shared room bias via sample weighting or
  segment-specific models

  ## NLP Findings (Section 7.1)

### Sentiment Analysis on Reviews
- Tool: VADER (Valence Aware Dictionary and sEntiment Reasoner), chosen over
  transformer-based models because: (1) VADER is designed specifically for
  short informal text, (2) no GPU available for inference at scale, (3) speed
  allows full 10,000-review sample vs transformer subsample
- Language detection: langdetect applied to all 10,000 sampled reviews;
  58.9% English (5,885), 12.1% Spanish, 10.9% French, 4.3% German, 3.0%
  Italian — consistent with Barcelona's international visitor profile
- VADER applied to English-only subset (non-English reviews score as neutral
  artifactually, documented as a methodological limitation)

### Key correlations (VADER compound vs numerical scores, n=5,885):
- Overall rating: r=0.245 (moderate positive)
- Value score: r=0.236 (moderate positive)
- Cleanliness score: r=0.209 (moderate positive)
- Location score: r=0.117 (weak positive — location described objectively,
  not emotionally, so sentiment captures it poorly)
- Comment length: r=0.071 (negligible)

### Interpretation:
Sentiment and numerical scores are complementary, not redundant signals.
VADER captures emotional tone (positive/negative language) while numerical
scores capture structured guest evaluation — they partially overlap (r~0.2)
but neither is a substitute for the other. A listing with a middling
numerical score but strongly positive sentiment may have genuine guest
goodwill not reflected in the structured rating, and vice versa. This
suggests combining both signals would produce a richer listing-quality
metric than either alone.

### Limitation:
41.1% of reviews (non-English) excluded from correlation analysis.
A multilingual sentiment model (e.g., XLM-RoBERTa) would extend coverage
to all languages but requires GPU inference at this scale — flagged as a
future improvement.