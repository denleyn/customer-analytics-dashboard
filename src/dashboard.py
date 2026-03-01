"""
Streamlit dashboard for the Customer Analytics Dashboard project.

Features:
- KPI cards: total customers, churn rate, average monthly spend.
- Line chart of monthly signups.
- Bar chart of churned vs active customers.
- Table of top 10 customers by spend.
- Churn prediction section using a logistic regression model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (
    compute_churn_rate,
    compute_monthly_trends,
    get_top_spenders,
    load_customers,
)
from model import ChurnModel, build_feature_matrix, train_churn_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "customers.db"


CUSTOM_CSS = """
<style>
.stApp {
    background-color: #0e1117;
    color: #e5e5e5;
}

[data-testid="stSidebar"] {
    background-color: #0e1117 !important;
}

[data-testid="stSidebar"] * {
    color: #e5e5e5 !important;
}

/* Date input styling inside the dark sidebar */
[data-testid="stSidebar"] input {
    color: #000000 !important;
    background-color: #ffffff !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] div[data-baseweb="input"] {
    background-color: #ffffff !important;
}

[data-testid="stSidebar"] input[class^="st-emotion-cache"][type="text"]::placeholder {
    color: #333333 !important;
    opacity: 1 !important;
}

.block-container {
    padding-top: 1.5rem;
}

.metric-row {
    display: flex;
    gap: 1rem;
}

.metric-card {
    flex: 1;
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
    box-shadow: 0 0 12px rgba(0, 0, 0, 0.45);
}

.metric-title {
    font-size: 0.9rem;
    color: #c0c0c0;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #ffffff;
}

.metric-sub {
    font-size: 0.8rem;
    color: #d0d0d0;
    margin-top: 0.15rem;
}

.metric-green {
    background: linear-gradient(135deg, #145a32, #27ae60);
}

.metric-red {
    background: linear-gradient(135deg, #7b241c, #e74c3c);
}

.metric-blue {
    background: linear-gradient(135deg, #154360, #2980b9);
}

.metric-purple {
    background: linear-gradient(135deg, #4a235a, #8e44ad);
}
</style>
"""


@st.cache_data(show_spinner=False)
def _load_customers_cached() -> pd.DataFrame:
    """Cached wrapper around `analysis.load_customers`."""
    return load_customers()


@st.cache_resource(show_spinner=False)
def _train_model_cached(df: pd.DataFrame) -> tuple[ChurnModel, dict]:
    """Train and cache the churn model so we don't retrain on every interaction."""
    model, metrics = train_churn_model(df)
    return model, metrics


def _render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("Customer Analytics Dashboard")
    st.markdown(
        "Interactive dashboard on synthetic customer data with churn analytics, "
        "revenue insights, and a simple churn prediction model."
    )


def _render_overview(df: pd.DataFrame, accuracy: float | None) -> None:
    """Render KPI cards and core charts."""
    st.subheader("Overview")

    churn_rate = compute_churn_rate(df)
    avg_spend = df["monthly_spend"].mean() if not df.empty else 0.0
    monthly_trends = compute_monthly_trends(df)
    total_customers = len(df)
    active_customers = int((df["churned"] == 0).sum()) if "churned" in df.columns else 0
    total_revenue = df["monthly_spend"].sum() if not df.empty else 0.0

    # KPI cards row
    kpi_cols = st.columns(4)

    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="metric-card metric-green">
                <div class="metric-title">Active customers</div>
                <div class="metric-value">{active_customers:,}</div>
                <div class="metric-sub">Total customers: {total_customers:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="metric-card metric-red">
                <div class="metric-title">Churn rate</div>
                <div class="metric-value">{churn_rate:.1%}</div>
                <div class="metric-sub">Estimated churned customers: {int(churn_rate * total_customers):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_cols[2]:
        st.markdown(
            f"""
            <div class="metric-card metric-blue">
                <div class="metric-title">Average monthly spend</div>
                <div class="metric-value">${avg_spend:,.2f}</div>
                <div class="metric-sub">Monthly revenue: ${total_revenue:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_cols[3]:
        acc_display = f"{accuracy:.1%}" if accuracy is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-card metric-purple">
                <div class="metric-title">Model accuracy</div>
                <div class="metric-value">{acc_display}</div>
                <div class="metric-sub">Logistic regression on filtered data</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Line chart: monthly signups
    if not monthly_trends.empty:
        st.markdown("#### Monthly signups")
        fig_signups = px.line(
            monthly_trends,
            x="month",
            y="signups",
            markers=True,
            template="plotly_dark",
        )
        fig_signups.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Month",
            yaxis_title="Signups",
            hovermode="x unified",
        )
        fig_signups.update_xaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
        fig_signups.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
        st.plotly_chart(fig_signups, use_container_width=True)

    # Bar chart: churned vs active
    st.markdown("#### Churned vs active customers")
    status_counts = (
        df["churned"]
        .map({0: "Active", 1: "Churned"})
        .value_counts()
        .reindex(["Active", "Churned"])
        .fillna(0)
        .astype(int)
    )
    status_df = status_counts.reset_index()
    status_df.columns = ["status", "count"]
    fig_status = px.bar(
        status_df,
        x="status",
        y="count",
        template="plotly_dark",
        color="status",
        color_discrete_map={"Active": "#27ae60", "Churned": "#e74c3c"},
    )
    fig_status.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Status",
        yaxis_title="Customers",
        showlegend=False,
    )
    fig_status.update_xaxes(showgrid=False)
    fig_status.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
    st.plotly_chart(fig_status, use_container_width=True)


def _assign_segment(df: pd.DataFrame) -> pd.Series:
    """Assign segment: High Value (top 25%), Mid Value (middle 50%), Low Value (bottom 25%)."""
    if df.empty or "monthly_spend" not in df.columns:
        return pd.Series(dtype=object)
    q25 = df["monthly_spend"].quantile(0.25)
    q75 = df["monthly_spend"].quantile(0.75)

    def segment(spend: float) -> str:
        if spend <= q25:
            return "Low Value"
        if spend <= q75:
            return "Mid Value"
        return "High Value"

    return df["monthly_spend"].map(segment)


def _render_customer_segments(df: pd.DataFrame) -> None:
    """Render Customer Segments: churn rate bar chart and revenue pie chart."""
    st.subheader("Customer Segments")

    if df.empty or "monthly_spend" not in df.columns:
        st.info("No data available for segments.")
        return

    df_seg = df.copy()
    df_seg["segment"] = _assign_segment(df_seg)
    segment_order = ["Low Value", "Mid Value", "High Value"]

    # Churn rate by segment
    churn_by_segment = (
        df_seg.groupby("segment", observed=True)["churned"]
        .mean()
        .reindex(segment_order)
        .fillna(0)
    )
    churn_df = churn_by_segment.reset_index()
    churn_df.columns = ["segment", "churn_rate"]

    fig_churn = px.bar(
        churn_df,
        x="segment",
        y="churn_rate",
        template="plotly_dark",
        color="churn_rate",
        color_continuous_scale="Reds",
    )
    fig_churn.update_layout(
        title="Churn rate by segment",
        xaxis_title="Segment",
        yaxis_title="Churn rate",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        xaxis={"categoryorder": "array", "categoryarray": segment_order},
    )
    fig_churn.update_xaxes(showgrid=False)
    fig_churn.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)")
    fig_churn.update_coloraxes(showscale=False)

    # Revenue by segment
    revenue_by_segment = (
        df_seg.groupby("segment", observed=True)["monthly_spend"]
        .sum()
        .reindex(segment_order)
        .fillna(0)
    )
    revenue_df = revenue_by_segment.reset_index()
    revenue_df.columns = ["segment", "revenue"]

    fig_pie = px.pie(
        revenue_df,
        values="revenue",
        names="segment",
        template="plotly_dark",
        color_discrete_sequence=["#2980b9", "#27ae60", "#8e44ad"],
    )
    fig_pie.update_layout(
        title="Revenue split by segment",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_churn, use_container_width=True)
    with col2:
        st.plotly_chart(fig_pie, use_container_width=True)


def _render_top_spenders(df: pd.DataFrame) -> None:
    """Render table of top 10 customers by spend."""
    st.subheader("Top 10 customers by spend")

    top_spenders = get_top_spenders(df, n=10)
    st.dataframe(
        top_spenders.assign(
            signup_date=top_spenders["signup_date"].dt.date,
            last_active=top_spenders["last_active"].dt.date,
        ),
        use_container_width=True,
    )


def _render_revenue_at_risk(df: pd.DataFrame) -> None:
    """Render revenue at risk from churned customers."""
    st.subheader("Revenue at Risk")

    total_revenue = df["monthly_spend"].sum() if not df.empty else 0.0
    churned_revenue = (
        df.loc[df["churned"] == 1, "monthly_spend"].sum() if "churned" in df.columns else 0.0
    )
    percent = (churned_revenue / total_revenue) if total_revenue > 0 else 0.0

    st.markdown(
        f"""
        <div class="metric-card metric-red">
            <div class="metric-title">Monthly revenue lost to churn</div>
            <div class="metric-value">${churned_revenue:,.0f}</div>
            <div class="metric-sub">{percent:.1%} of total monthly revenue</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_churn_prediction(df: pd.DataFrame, model: ChurnModel) -> None:
    """Render churn prediction UI for a single selected customer."""
    st.subheader("Churn prediction")

    customer_ids = df["customer_id"].tolist()
    if not customer_ids:
        st.info("No customers available for prediction.")
        return

    selected_id = st.selectbox("Select a customer", customer_ids)

    customer_row = df.loc[df["customer_id"] == selected_id].iloc[0:1]

    X, _y = build_feature_matrix(customer_row)
    proba = model.predict_proba(X)[0]
    prediction = "Churned" if proba >= 0.5 else "Active"

    st.markdown(
        f"**Predicted status:** {prediction}  \n"
        f"**Churn probability:** {proba:.1%}"
    )

    with st.expander("Customer details"):
        details = customer_row.copy()
        details["signup_date"] = details["signup_date"].dt.date
        details["last_active"] = details["last_active"].dt.date
        st.json(details.to_dict(orient="records")[0])


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(
        page_title="Customer Analytics Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _render_header()

    try:
        df = _load_customers_cached()
    except FileNotFoundError:
        st.error(
            f"Could not find the customer database at `{DB_PATH}`. "
            "Run `python src/data_generator.py` from the project root to generate it."
        )
        st.stop()
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to load customer data: {exc}")
        st.stop()

    if df.empty:
        st.warning("The customers table is empty. Generate data before using the dashboard.")
        st.stop()

    # Sidebar: title and signup date range filter
    st.sidebar.title("Customer Analytics Dashboard")
    min_date = df["signup_date"].min().date()
    max_date = df["signup_date"].max().date()
    date_range = st.sidebar.date_input("Signup date range", (min_date, max_date))

    if isinstance(date_range, tuple) or isinstance(date_range, list):
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date = end_date = date_range

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    mask = (df["signup_date"].dt.date >= start_date) & (df["signup_date"].dt.date <= end_date)
    df_filtered = df.loc[mask].copy()

    if df_filtered.empty:
        st.warning("No customers found in the selected signup date range.")
        st.stop()

    # Train model on filtered data and extract accuracy
    with st.spinner("Training churn model..."):
        churn_model, metrics = _train_model_cached(df_filtered)
    accuracy = metrics.get("accuracy")

    # Layout: main sections
    _render_overview(df_filtered, accuracy=accuracy)
    _render_revenue_at_risk(df_filtered)
    _render_customer_segments(df_filtered)
    _render_top_spenders(df_filtered)
    _render_churn_prediction(df_filtered, churn_model)


if __name__ == "__main__":
    main()

