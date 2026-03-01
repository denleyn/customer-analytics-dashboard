"""
Data generator module.

Responsibility:
- Create a local SQLite database file under the `data/` directory.
- Generate fake customer records (e.g. 500+ rows) with fields:
  - customer_id
  - signup_date
  - last_active
  - monthly_spend
  - churned
"""
from __future__ import annotations

import os
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from faker import Faker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "customers.db"


def ensure_data_directory() -> None:
    """Create the `data` directory if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the customer database."""
    ensure_data_directory()
    return sqlite3.connect(DB_PATH)


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create the `customers` table, dropping any existing version.

    This keeps the script idempotent: each run starts from a clean table.
    """
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS customers")

    cursor.execute(
        """
        CREATE TABLE customers (
            customer_id   TEXT PRIMARY KEY,
            signup_date   TEXT NOT NULL,
            last_active   TEXT NOT NULL,
            monthly_spend REAL NOT NULL,
            churned       INTEGER NOT NULL CHECK (churned IN (0, 1))
        )
        """
    )

    conn.commit()


def generate_customer_records(n: int = 500) -> list[tuple[str, str, str, float, int]]:
    """
    Generate a list of fake customer records.

    Returns a list of tuples:
    (customer_id, signup_date, last_active, monthly_spend, churned)
    """
    fake = Faker()

    today = date.today()
    earliest_signup = today - timedelta(days=3 * 365)

    records: list[tuple[str, str, str, float, int]] = []

    for i in range(1, n + 1):
        customer_id = f"CUST-{i:04d}"

        # Random signup date in the last 3 years
        signup_offset_days = random.randint(0, (today - earliest_signup).days)
        signup_dt = earliest_signup + timedelta(days=signup_offset_days)

        # Last active date between signup and today
        last_active_offset_days = random.randint(0, (today - signup_dt).days)
        last_active_dt = signup_dt + timedelta(days=last_active_offset_days)

        # Monthly spend roughly between 5 and 500, skewed upward a bit
        base_spend = random.uniform(5, 300)
        bonus_spend = random.uniform(0, 200) if random.random() < 0.25 else 0.0
        monthly_spend = round(base_spend + bonus_spend, 2)

        # Simple churn logic: if not active for a long time, more likely churned
        days_since_active = (today - last_active_dt).days
        if days_since_active > 365:
            churn_prob = 0.9
        elif days_since_active > 180:
            churn_prob = 0.7
        elif days_since_active > 90:
            churn_prob = 0.4
        else:
            churn_prob = 0.15

        churned = int(random.random() < churn_prob)

        records.append(
            (
                customer_id,
                signup_dt.isoformat(),
                last_active_dt.isoformat(),
                monthly_spend,
                churned,
            )
        )

    return records


def populate_database(conn: sqlite3.Connection, records: list[tuple[str, str, str, float, int]]) -> None:
    """Insert generated customer records into the `customers` table."""
    cursor = conn.cursor()

    cursor.executemany(
        """
        INSERT INTO customers (
            customer_id,
            signup_date,
            last_active,
            monthly_spend,
            churned
        ) VALUES (?, ?, ?, ?, ?)
        """,
        records,
    )

    conn.commit()


def main() -> None:
    """Entry point for generating the SQLite database with fake data."""
    # Optional: make randomness reproducible if an environment variable is set
    seed_value = os.getenv("CUSTOMER_DASH_SEED")
    if seed_value is not None:
        try:
            seed_int = int(seed_value)
        except ValueError:
            seed_int = None
        if seed_int is not None:
            random.seed(seed_int)

    with get_connection() as conn:
        init_schema(conn)
        records = generate_customer_records(n=500)
        populate_database(conn, records)

    print(f"Generated {len(records)} customers in SQLite database at: {DB_PATH}")


if __name__ == "__main__":
    main()