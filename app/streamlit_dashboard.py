import os

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# --- PAGE CONFIG ---
st.set_page_config(page_title="Barcelona Airbnb Intelligence", page_icon="🏠", layout="wide")

# --- CUSTOM CSS ---
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #0056b3;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0056b3;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        text-transform: uppercase;
    }
    .ai-recommendation {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #28a745;
        margin-top: 20px;
        font-family: monospace;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- CONSTANTS & SETUP ---
DB_PATH = "data/warehouse.duckdb"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.joblib")
META_PATH = os.path.join(MODELS_DIR, "model_meta.joblib")
EXPLAINER_PATH = os.path.join(MODELS_DIR, "shap_explainer.joblib")
BRIEFING_PATH = "reports/market_intelligence_briefings.txt"

load_dotenv(".env")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None


# --- CACHED DATA & MODELS ---
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    return model, meta, explainer


@st.cache_data(ttl=3600)
def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    query = """
        SELECT f.listing_id, f.price, l.room_type, l.accommodates, 
               l.neighbourhood_name, n.neighbourhood_group, 
               h.host_is_superhost, h.host_id, f.review_scores_rating, f.number_of_reviews,
               v.demand_segment
        FROM fact_listing_performance f
        JOIN dim_listing l ON f.listing_id = l.listing_id
        JOIN dim_neighbourhood n ON l.neighbourhood_name = n.neighbourhood_name
        JOIN dim_host h ON l.host_id = h.host_id
        JOIN v_listing_demand v ON f.listing_id = v.listing_id
        WHERE f.price IS NOT NULL AND l.price_is_valid = true
    """
    df = con.execute(query).df()
    con.close()
    return df


@st.cache_data(ttl=3600)
def get_briefings():
    if os.path.exists(BRIEFING_PATH):
        with open(BRIEFING_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "Briefings not found."


# Load essentials
df_raw = load_data()
model, meta, explainer = load_models()

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("### 🏠 Expernetic Analytics")
st.sidebar.header("Filters")

neighbourhoods = sorted(df_raw["neighbourhood_group"].dropna().unique())
selected_ng = st.sidebar.multiselect("Neighbourhood Group", neighbourhoods, default=neighbourhoods)

room_types = sorted(df_raw["room_type"].dropna().unique())
selected_rt = st.sidebar.multiselect("Room Type", room_types, default=room_types)

min_price, max_price = float(df_raw["price"].min()), float(df_raw["price"].max())
selected_price = st.sidebar.slider("Price Range (EUR)", 0.0, 2000.0, (0.0, 1000.0))

demand_segments = sorted(df_raw["demand_segment"].dropna().unique())
selected_ds = st.sidebar.multiselect("Demand Segment", demand_segments, default=["active"])

# Apply filters
df_filtered = df_raw[
    (df_raw["neighbourhood_group"].isin(selected_ng))
    & (df_raw["room_type"].isin(selected_rt))
    & (df_raw["price"] >= selected_price[0])
    & (df_raw["price"] <= selected_price[1])
    & (df_raw["demand_segment"].isin(selected_ds))
]

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Market Overview",
        "🗺️ Geographic Analysis",
        "👥 Host Intelligence",
        "🤖 Price Advisor",
        "📰 AI Briefing",
    ]
)

# ==========================================
# TAB 1: Market Overview
# ==========================================
with tab1:
    st.header("Market Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            (
                f'<div class="metric-card">'
                f'<div class="metric-value">{len(df_filtered):,}</div>'
                f'<div class="metric-label">Total Listings</div></div>'
            ),
            unsafe_allow_html=True,
        )
    with col2:
        median_price = df_filtered["price"].median()
        st.markdown(
            (
                f'<div class="metric-card">'
                f'<div class="metric-value">€{median_price:.2f}</div>'
                f'<div class="metric-label">Median Price</div></div>'
            ),
            unsafe_allow_html=True,
        )
    with col3:
        avg_rating = df_filtered["review_scores_rating"].mean()
        st.markdown(
            (
                f'<div class="metric-card">'
                f'<div class="metric-value">{avg_rating:.2f}</div>'
                f'<div class="metric-label">Avg Rating</div></div>'
            ),
            unsafe_allow_html=True,
        )
    with col4:
        active_pct = (
            (len(df_filtered[df_filtered["demand_segment"] == "active"]) / len(df_filtered) * 100)
            if len(df_filtered) > 0
            else 0
        )
        st.markdown(
            (
                f'<div class="metric-card">'
                f'<div class="metric-value">{active_pct:.1f}%</div>'
                f'<div class="metric-label">% Active</div></div>'
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_filtered["price"], bins=40, kde=True, color="steelblue", ax=ax)
        ax.set_xlim(selected_price[0], selected_price[1])
        st.pyplot(fig)

    with col_chart2:
        st.subheader("Room Type Breakdown")
        rt_counts = df_filtered["room_type"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.pie(
            rt_counts.values,
            labels=rt_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("pastel"),
        )
        ax.axis("equal")
        st.pyplot(fig)

# ==========================================
# TAB 2: Geographic Analysis
# ==========================================
with tab2:
    st.header("Geographic Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Median Price by Neighbourhood Group")
        ng_price = (
            df_filtered.groupby("neighbourhood_group")["price"]
            .median()
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=ng_price.values, y=ng_price.index, palette="viridis", ax=ax)
        ax.set_xlabel("Median Price (EUR)")
        st.pyplot(fig)

    with col2:
        st.subheader("Market Stats by Neighbourhood Group")
        ng_stats = (
            df_filtered.groupby("neighbourhood_group")
            .agg(
                Count=("listing_id", "count"),
                Median_Price=("price", "median"),
                Avg_Rating=("review_scores_rating", "mean"),
            )
            .sort_values("Count", ascending=False)
        )
        st.dataframe(
            ng_stats.style.format({"Median_Price": "€{:.2f}", "Avg_Rating": "{:.2f}"}),
            use_container_width=True,
        )

    st.subheader("Top and Bottom Neighbourhoods (by Price)")
    n_stats = df_filtered.groupby("neighbourhood_name").agg(
        Count=("listing_id", "count"), Median_Price=("price", "median")
    )
    n_stats = n_stats[n_stats["Count"] >= 10]  # Filter out low counts

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Top 10 Most Expensive**")
        st.dataframe(
            n_stats.sort_values("Median_Price", ascending=False)
            .head(10)
            .style.format({"Median_Price": "€{:.2f}"})
        )
    with col_t2:
        st.markdown("**Top 10 Least Expensive**")
        st.dataframe(
            n_stats.sort_values("Median_Price", ascending=True)
            .head(10)
            .style.format({"Median_Price": "€{:.2f}"})
        )

# ==========================================
# TAB 3: Host Intelligence
# ==========================================
with tab3:
    st.header("Host Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Superhost Performance")
        sh_stats = df_filtered.groupby("host_is_superhost").agg(
            Count=("listing_id", "count"),
            Median_Price=("price", "median"),
            Avg_Rating=("review_scores_rating", "mean"),
            Median_Reviews=("number_of_reviews", "median"),
        )
        sh_stats.index = sh_stats.index.map({"t": "Superhost", "f": "Regular Host"})
        st.dataframe(
            sh_stats.style.format(
                {"Median_Price": "€{:.2f}", "Avg_Rating": "{:.2f}", "Median_Reviews": "{:.0f}"}
            )
        )

    with col2:
        st.subheader("Host Portfolio Distribution")
        host_counts = df_raw["host_id"].value_counts()
        single = (host_counts == 1).sum()
        multi = ((host_counts > 1) & (host_counts < 10)).sum()
        commercial = (host_counts >= 10).sum()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(
            [single, multi, commercial],
            labels=["Single Listing", "Multi (2-9)", "Commercial (10+)"],
            autopct="%1.1f%%",
            colors=sns.color_palette("Set2"),
        )
        ax.axis("equal")
        st.pyplot(fig)

    st.subheader("Market Concentration")
    total_listings = len(df_raw)
    top1 = host_counts.head(1).sum() / total_listings * 100
    top10 = host_counts.head(10).sum() / total_listings * 100
    top100 = host_counts.head(100).sum() / total_listings * 100
    top500 = host_counts.head(500).sum() / total_listings * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        x=["Top 1 Host", "Top 10 Hosts", "Top 100 Hosts", "Top 500 Hosts"],
        y=[top1, top10, top100, top500],
        palette="magma",
        ax=ax,
    )
    ax.set_ylabel("% of Total Listings")
    for i, v in enumerate([top1, top10, top100, top500]):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    st.pyplot(fig)

# ==========================================
# TAB 4: Price Advisor
# ==========================================
with tab4:
    st.header("🤖 AI-Powered Price Advisor")

    if not model:
        st.error("Model artefacts not found! Please run the notebook or `save_model.py` first.")
    else:
        listing_input = st.text_input("Enter Listing ID (e.g. 18674)")

        if st.button("Analyze Pricing"):
            if not listing_input.isdigit():
                st.warning("Please enter a valid numeric Listing ID.")
            else:
                lid = int(listing_input)
                with st.spinner("Analyzing listing and querying LLM..."):
                    try:
                        # 1. Fetch data
                        con = duckdb.connect(DB_PATH, read_only=True)
                        query = f"""
                            SELECT
                                f.listing_id,
                                f.price,
                                l.room_type,
                                l.accommodates,
                                l.bathrooms,
                                l.bedrooms,
                                l.beds,
                                l.neighbourhood_name,
                                n.neighbourhood_group,
                                h.host_is_superhost,
                                h.host_listings_count,
                                h.hosts_time_as_host_years,
                                f.number_of_reviews,
                                f.review_scores_rating,
                                f.review_scores_cleanliness,
                                f.review_scores_location,
                                f.review_scores_value,
                                f.availability_365,
                                f.minimum_nights,
                                f.estimated_occupancy_l365d,
                                v.demand_segment
                            FROM fact_listing_performance f
                            JOIN dim_listing l ON f.listing_id = l.listing_id
                            JOIN dim_neighbourhood n ON l.neighbourhood_name = n.neighbourhood_name
                            JOIN dim_host h ON l.host_id = h.host_id
                            JOIN v_listing_demand v ON f.listing_id = v.listing_id
                            WHERE f.listing_id = {lid}
                        """
                        raw_df = con.execute(query).df()

                        if raw_df.empty:
                            st.error(f"Listing ID {lid} not found in active performance tables.")
                            con.close()
                        else:
                            df = raw_df.copy()
                            raw_dict = df.iloc[0].to_dict()

                            # 2. Feature Engineering
                            df["is_superhost"] = (df["host_is_superhost"] == "t").astype(int)
                            df["has_rating"] = df["review_scores_rating"].notna().astype(int)

                            for col in [
                                "review_scores_rating",
                                "review_scores_cleanliness",
                                "review_scores_location",
                                "review_scores_value",
                            ]:
                                df[col] = df[col].fillna(0)

                            for col, med in meta["medians"].items():
                                if col in df.columns:
                                    df[col] = df[col].fillna(med)

                            df["beds_per_person"] = df["beds"] / df["accommodates"].clip(lower=1)
                            df["room_type_enc"] = df["room_type"].map(meta["room_type_enc"])

                            for col in meta["ng_dummy_cols"]:
                                df[col] = 0
                            ng_col = f"ng_{raw_dict['neighbourhood_group']}"
                            if ng_col in df.columns:
                                df[ng_col] = 1

                            X = df[meta["feature_cols"]].astype(float)

                            # 3. Predict
                            pred_log = model.predict(X)[0]
                            pred_price = np.expm1(pred_log)
                            current_price = raw_dict["price"]

                            # Market context
                            ng = raw_dict["neighbourhood_group"]
                            rt = raw_dict["room_type"]
                            safe_ng = ng.replace('"', "''")
                            med_query = (
                                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY price) "
                                "FROM stg_listings "
                                f"WHERE neighbourhood_group_cleansed = '{safe_ng}' "
                                f"AND room_type = '{rt}' "
                                "AND price > 0"
                            )
                            ng_rt_median = con.execute(med_query).fetchone()[0]

                            rank_query = (
                                "SELECT count(*) * 100.0 / "
                                "(SELECT count(*) FROM stg_listings "
                                f"WHERE neighbourhood_group_cleansed = '{safe_ng}' "
                                f"AND room_type = '{rt}' "
                                "AND price > 0) "
                                "FROM stg_listings "
                                f"WHERE neighbourhood_group_cleansed = '{safe_ng}' "
                                f"AND room_type = '{rt}' "
                                f"AND price <= {pred_price} "
                                "AND price > 0"
                            )
                            percentile = con.execute(rank_query).fetchone()[0]

                            con.close()

                            # 4. Display Metrics
                            st.subheader(f"Pricing Analysis for Listing {lid} ({rt} in {ng})")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Current Price", f"€{current_price:.2f}")
                            c2.metric(
                                "Predicted Fair Price",
                                f"€{pred_price:.2f}",
                                f"€{(pred_price - current_price):.2f}",
                            )
                            c3.metric("Neighbourhood Median", f"€{ng_rt_median:.2f}")
                            c4.metric("Market Position", f"{percentile:.0f}th percentile")

                            # 5. SHAP Waterfall
                            st.subheader("Key Price Drivers (SHAP)")
                            shap_values = explainer(X)

                            fig, ax = plt.subplots(figsize=(10, 6))
                            shap.plots.waterfall(shap_values[0], show=False)
                            st.pyplot(fig)
                            plt.clf()

                            # 6. Groq LLM Recommendation
                            if groq_client:
                                st.subheader("AI Recommendation")

                                feature_names = X.columns
                                shap_abs = np.abs(shap_values.values[0])
                                top_idx = np.argsort(shap_abs)[::-1][:5]

                                top_drivers = []
                                for i in top_idx:
                                    feat = feature_names[i]
                                    val = X.iloc[0, i]
                                    impact = shap_values.values[0, i]
                                    direction = "increases" if impact > 0 else "decreases"
                                    top_drivers.append(
                                        f"{feat} (value: {val:.1f}) {direction} "
                                        f"price by {abs(impact):.2f} log-euros"
                                    )

                                prompt = (
                                    f"""
                                You are an expert Airbnb pricing consultant. Provide specific,
                                actionable pricing advice in 3-4 short paragraphs.

                                Listing details:
                                - Type: {rt} in {ng}
                                - Accommodates: {raw_dict['accommodates']}
                                - Current Price: EUR {current_price:.2f}
                                - Model Predicted Fair Price: EUR {pred_price:.2f}
                                - Median Price for similar listings in this area:
                                  EUR {ng_rt_median:.2f}
                                - Market Position: The predicted price is in the
                                  {percentile:.0f}th percentile.

                                Top 5 factors influencing the predicted price:
                                """
                                    + "\\n".join([f"- {d}" for d in top_drivers])
                                    + "\\n\\nAnalyze these factors and give practical advice "
                                    "to the host on how to adjust their price or offering."
                                )

                                try:
                                    response = groq_client.chat.completions.create(
                                        messages=[
                                            {
                                                "role": "system",
                                                "content": (
                                                    "You are a professional, data-driven Airbnb "
                                                    "pricing consultant. Be direct, insightful, "
                                                    "and actionable."
                                                ),
                                            },
                                            {"role": "user", "content": prompt},
                                        ],
                                        model="llama-3.3-70b-versatile",
                                        temperature=0.7,
                                        max_tokens=500,
                                    )
                                    rec = response.choices[0].message.content
                                    st.markdown(
                                        f'<div class="ai-recommendation">{rec}</div>',
                                        unsafe_allow_html=True,
                                    )
                                except Exception as e:
                                    st.error(f"Failed to generate LLM recommendation: {e}")
                            else:
                                st.warning(
                                    "Groq API key not configured. "
                                    "Cannot generate AI recommendation."
                                )
                    except Exception as e:
                        st.error(f"Error analyzing listing: {e}")

# ==========================================
# TAB 5: AI Briefing
# ==========================================
with tab5:
    st.header("📰 Executive Market Briefings")

    briefings = get_briefings()
    st.text_area("Market Intelligence Reports", briefings, height=500)

    if st.button("Regenerate Briefings (Simulated)"):
        st.info(
            "In a full production environment, this would trigger a pipeline to "
            "query Groq with the latest data warehouse state."
        )
