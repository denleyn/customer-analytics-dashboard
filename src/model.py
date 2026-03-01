"""
Churn prediction model using scikit-learn logistic regression.

Responsibilities:
- Load and preprocess customer data from the SQLite database.
- Train a logistic regression classifier to predict churn.
- Expose helper functions for training and prediction that can be reused by the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "customers.db"


@dataclass
class ChurnModel:
    """Container for a trained churn model and its scaler."""

    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn probabilities for the given feature DataFrame."""
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn labels (0/1) for the given feature DataFrame."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)


def load_customers() -> pd.DataFrame:
    """
    Load customer data from the SQLite database into a pandas DataFrame.

    This mirrors the logic in `analysis.load_customers` but is kept local
    to avoid circular imports.
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


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y from the raw customer DataFrame.

    Features engineered:
    - tenure_days: days since signup
    - days_since_last_active: days since last_active
    - monthly_spend: as-is
    """
    today = pd.Timestamp.today().normalize()

    df_feat = df.copy()
    df_feat["tenure_days"] = (today - df_feat["signup_date"]).dt.days
    df_feat["days_since_last_active"] = (today - df_feat["last_active"]).dt.days

    feature_cols = ["tenure_days", "days_since_last_active", "monthly_spend"]

    X = df_feat[feature_cols]
    y = df_feat["churned"].astype(int)

    return X, y


def train_churn_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[ChurnModel, dict]:
    """
    Train a logistic regression churn model.

    Returns:
    - ChurnModel instance containing the fitted model, scaler, and feature list.
    - metrics dict with classification report text and basic scores.
    """
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    report = classification_report(y_test, y_pred, output_dict=False)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "classification_report": report,
        "accuracy": report_dict.get("accuracy"),
        "precision_churned": report_dict.get("1", {}).get("precision"),
        "recall_churned": report_dict.get("1", {}).get("recall"),
        "f1_churned": report_dict.get("1", {}).get("f1-score"),
    }

    churn_model = ChurnModel(
        model=clf,
        scaler=scaler,
        feature_names=list(X.columns),
    )

    return churn_model, metrics


def main() -> None:
    """
    CLI entry point for training and evaluating the churn model.

    Loads customers, trains a logistic regression model, and prints metrics.
    """
    df = load_customers()

    if df.empty:
        print("No customers found in the database. Generate data first.")
        return

    churn_model, metrics = train_churn_model(df)

    print("=== Churn model evaluation ===")
    print(metrics["classification_report"])
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(
        "Churned class - "
        f"precision: {metrics['precision_churned']:.3f}, "
        f"recall: {metrics['recall_churned']:.3f}, "
        f"f1: {metrics['f1_churned']:.3f}"
    )


if __name__ == "__main__":
    main()

