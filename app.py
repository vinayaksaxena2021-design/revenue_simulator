# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Revenue Simulator",
    page_icon="💰",
    layout="wide",
)

st.title("💰 Product Revenue Simulator")
st.write(
    "Simulate revenue over time using your product catalog, stock levels, and "
    "some assumptions about demand."
)

# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_products(uploaded_file):
    """
    Load the product catalog either from an uploaded file
    or from the bundled products-100000.csv.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("products-100000.csv")

    # Basic sanity: ensure required columns exist
    required_cols = [
        "Index", "Name", "Brand", "Category", "Price", "Currency",
        "Stock", "Availability"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only what we need
    df = df[
        [
            "Index",
            "Name",
            "Brand",
            "Category",
            "Price",
            "Currency",
            "Stock",
            "Availability",
        ]
    ]

    # Drop any weird negative stocks or prices
    df = df[(df["Stock"] > 0) & (df["Price"] > 0)]

    return df


# ---------------------------------------------------------
# 3. SIDEBAR CONTROLS
# ---------------------------------------------------------

st.sidebar.header("📁 Data source")

uploaded = st.sidebar.file_uploader(
    "Upload products CSV (optional)",
    type=["csv"],
    help="If you don't upload anything, the bundled products-100000.csv will be used.",
)

try:
    products_df = load_products(uploaded)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(products_df):,} products")

# Filters
st.sidebar.header("🔎 Filters")

all_categories = sorted(products_df["Category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Categories",
    options=all_categories,
    default=all_categories,
)

all_avails = sorted(products_df["Availability"].dropna().unique().tolist())
selected_availability = st.sidebar.multiselect(
    "Availability",
    options=all_avails,
    default=all_avails,
)

filtered_df = products_df[
    products_df["Category"].isin(selected_categories)
    & products_df["Availability"].isin(selected_availability)
].copy()

if filtered_df.empty:
    st.warning("No products match your filters. Adjust filters to continue.")
    st.stop()

st.write(
    f"**Active products after filters:** {len(filtered_df):,} "
    f"({filtered_df['Category'].nunique()} categories)"
)

# ---------------------------------------------------------
# 4. AGGREGATE TO CATEGORY + AVAILABILITY LEVEL
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def aggregate_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate product-level data into Category + Availability groups.
    This keeps the simulation super fast even with 100k+ products.
    """
    grouped = (
        df.groupby(["Category", "Availability"], as_index=False)
        .agg(
            total_stock=("Stock", "sum"),
            avg_price=("Price", "mean"),
            product_count=("Index", "count"),
        )
    )
    return grouped


grouped_df = aggregate_products(filtered_df)

st.subheader("🔢 Aggregated Product Groups")
st.caption(
    "Revenue simulation runs at this group level (Category × Availability) "
    "so that it scales to very large catalogs."
)
st.dataframe(grouped_df, use_container_width=True)

# ---------------------------------------------------------
# 5. SIMULATION SETTINGS
# ---------------------------------------------------------

st.sidebar.header("📊 Simulation settings")

months = st.sidebar.slider(
    "Simulation horizon (months)",
    min_value=1,
    max_value=24,
    value=12,
)

base_sellthrough = st.sidebar.slider(
    "Base monthly sell-through (% of stock)",
    min_value=1,
    max_value=50,
    value=10,
    help="Roughly what % of remaining stock you expect to sell per month, "
         "before availability adjustments.",
)

volatility = st.sidebar.slider(
    "Demand volatility (% of mean)",
    min_value=0,
    max_value=100,
    value=30,
    help="Higher = more randomness in monthly demand.",
)

runs = st.sidebar.slider(
    "Number of Monte Carlo runs",
    min_value=100,
    max_value=2000,
    step=100,
    value=500,
)

seed = st.sidebar.number_input(
    "Random seed (optional)",
    min_value=0,
    max_value=1_000_000,
    value=42,
    help="Use a fixed seed for reproducible simulations.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Simulation is run at the aggregated level, so even large files (like your "
    "~16MB / 100k-row CSV) stay performant."
)

# ---------------------------------------------------------
# 6. SIMULATION LOGIC
# ---------------------------------------------------------

AVAILABILITY_FACTORS = {
    "in_stock": 1.0,
    "pre_order": 0.7,
    "discontinued": 0.2,
}


def get_availability_factor(avail: str) -> float:
    return AVAILABILITY_FACTORS.get(str(avail).lower(), 1.0)


@st.cache_data(show_spinner=True)
def simulate_revenue(
    groups: pd.DataFrame,
    months: int,
    base_sellthrough: float,
    volatility: float,
    runs: int,
    seed: int,
):
    """
    Monte Carlo simulation at Category × Availability level.

    - groups: DataFrame with total_stock, avg_price, Availability.
    - base_sellthrough: % of remaining stock sold per month (before availability factor).
    - volatility: std dev as % of mean demand.
    - runs: number of simulation runs.
    """
    rng = np.random.default_rng(seed)

    total_stock = groups["total_stock"].to_numpy(dtype=float)
    avg_price = groups["avg_price"].to_numpy(dtype=float)
    avail_factor = groups["Availability"].map(get_availability_factor).to_numpy(
        dtype=float
    )

    base_rate = base_sellthrough / 100.0
    demand_mean_per_month = total_stock * base_rate * avail_factor

    n_groups = len(groups)
    revenue_runs = np.zeros(runs, dtype=float)

    for r in range(runs):
        remaining_stock = total_stock.copy()
        total_revenue = 0.0

        for _ in range(months):
            mean = demand_mean_per_month
            std = mean * (volatility / 100.0)

            # Draw normal, clamp to non-negative
            demand = rng.normal(mean, std)
            demand = np.maximum(demand, 0.0)

            # Can't sell more than we have left
            sales = np.minimum(demand, remaining_stock)

            total_revenue += np.sum(sales * avg_price)
            remaining_stock -= sales

        revenue_runs[r] = total_revenue

    return revenue_runs


# Run simulation
with st.spinner("Running revenue simulation..."):
    revenue_runs = simulate_revenue(
        grouped_df,
        months=months,
        base_sellthrough=base_sellthrough,
        volatility=volatility,
        runs=runs,
        seed=seed,
    )

# ---------------------------------------------------------
# 7. RESULTS SUMMARY
# ---------------------------------------------------------

st.subheader("📈 Simulation Results")

if len(revenue_runs) == 0:
    st.error("Simulation produced no results.")
    st.stop()

mean_rev = float(np.mean(revenue_runs))
median_rev = float(np.median(revenue_runs))
p10 = float(np.percentile(revenue_runs, 10))
p90 = float(np.percentile(revenue_runs, 90))
best = float(np.max(revenue_runs))
worst = float(np.min(revenue_runs))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected revenue", f"${mean_rev:,.0f}")
col2.metric("Median revenue", f"${median_rev:,.0f}")
col3.metric("10–90% range", f"${p10:,.0f} – ${p90:,.0f}")
col4.metric("Best / Worst run", f"${best:,.0f}", f"min ${worst:,.0f}")

# Histogram
st.markdown("#### Revenue distribution across runs")

fig, ax = plt.subplots()
try:
    ax.hist(revenue_runs, bins=30)
except ValueError:
    ax.hist(revenue_runs, bins=1)
ax.set_xlabel("Total revenue over horizon")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# ---------------------------------------------------------
# 8. EXPECTED REVENUE BY GROUP (DETERMINISTIC VIEW)
# ---------------------------------------------------------

st.subheader("📊 Expected Revenue by Category × Availability")

# Deterministic "expected" revenue, assuming mean demand each month
grouped_df = grouped_df.copy()
grouped_df["availability_factor"] = grouped_df["Availability"].map(
    get_availability_factor
)
grouped_df["base_rate"] = base_sellthrough / 100.0
grouped_df["expected_monthly_units"] = (
    grouped_df["total_stock"]
    * grouped_df["base_rate"]
    * grouped_df["availability_factor"]
)
grouped_df["expected_units_over_horizon"] = (
    grouped_df["expected_monthly_units"] * months
).clip(upper=grouped_df["total_stock"])
grouped_df["expected_revenue"] = (
    grouped_df["expected_units_over_horizon"] * grouped_df["avg_price"]
)

grouped_df = grouped_df.sort_values("expected_revenue", ascending=False)

st.dataframe(grouped_df, use_container_width=True)

st.caption(
    "This table is a deterministic view using the mean demand each month. "
    "The Monte Carlo above shows how randomness can push you above or below these values."

)

