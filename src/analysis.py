"""
Data analysis utilities using pandas.

Responsibilities:
- Load customer data from the SQLite database.
- Compute churn rate, monthly trends, and top spenders.
- Provide reusable functions that can be used by scripts and the Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "customers.db"


def load_customers() -> pd.DataFrame:
    """
    Load customer data from the SQLite database into a pandas DataFrame.

    The function parses `signup_date` and `last_active` as datetime columns.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {DB_PATH}. "
            "Run `data_generator.py` first to create it."
        )

    import sqlite3

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM customers", conn)

    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["last_active"] = pd.to_datetime(df["last_active"])

    return df


def compute_churn_rate(df: pd.DataFrame) -> float:
    """
    Compute overall churn rate as a fraction in [0, 1].

    Assumes `churned` is coded as 0 / 1.
    """
    if "churned" not in df.columns or len(df) == 0:
        return 0.0

    return float(df["churned"].mean())


def compute_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly signup, churn, and revenue trends.

    Returns a DataFrame indexed by month (period month) with columns:
    - month (Timestamp at start of month)
    - signups
    - churned_customers
    - active_customers
    - revenue
    """
    if df.empty:
        return pd.DataFrame(
            columns=["month", "signups", "churned_customers", "active_customers", "revenue"]
        )

    # Use signup month as the primary axis for trends
    df_trend = df.copy()
    df_trend["signup_month"] = df_trend["signup_date"].dt.to_period("M").dt.to_timestamp()

    grouped = df_trend.groupby("signup_month")

    trend = pd.DataFrame(
        {
            "month": grouped["customer_id"].count().index,
            "signups": grouped["customer_id"].count().values,
            "churned_customers": grouped["churned"].sum().values,
            "revenue": grouped["monthly_spend"].sum().values,
        }
    )

    # Approximate active customers per month as signups minus churned customers up to that month
    trend["cumulative_signups"] = trend["signups"].cumsum()
    trend["cumulative_churned"] = trend["churned_customers"].cumsum()
    trend["active_customers"] = trend["cumulative_signups"] - trend["cumulative_churned"]

    return trend[
        ["month", "signups", "churned_customers", "active_customers", "revenue"]
    ].sort_values("month")


def get_top_spenders(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return the top `n` customers by monthly spend.

    Columns returned:
    - customer_id
    - signup_date
    - last_active
    - monthly_spend
    - churned
    """
    if df.empty:
        return df

    cols = [
        "customer_id",
        "signup_date",
        "last_active",
        "monthly_spend",
        "churned",
    ]
    existing_cols = [c for c in cols if c in df.columns]

    return df.sort_values("monthly_spend", ascending=False)[existing_cols].head(n)


def main() -> None:
    """
    Simple CLI entry point for running basic analyses.

    Prints:
    - overall churn rate
    - a preview of monthly trends
    - top 10 spenders
    """
    df = load_customers()

    churn_rate = compute_churn_rate(df)
    monthly_trends = compute_monthly_trends(df)
    top_spenders = get_top_spenders(df, n=10)

    print("=== Overall churn rate ===")
    print(f"{churn_rate:.2%}")
    print()

    print("=== Monthly trends (head) ===")
    print(monthly_trends.head())
    print()

    print("=== Top 10 spenders ===")
    print(top_spenders.to_string(index=False))


if __name__ == "__main__":
    main()

