# Customer Analytics Dashboard

Python project for exploring customer behavior with a local SQLite database, data analysis using pandas, and a Streamlit dashboard for interactive visualizations and simple churn prediction.

## Project Structure

- `requirements.txt` – Python dependencies.
- `README.md` – Project overview and setup instructions.
- `data/` – SQLite database and any exported CSVs.
- `src/` – Application source code:
  - `__init__.py` – Marks `src` as a package.
  - `data_generator.py` – Creates a SQLite database with synthetic customer data (500+ rows).
  - `analysis.py` – Uses pandas to compute metrics and aggregates.
  - `model.py` – Trains a simple scikit-learn logistic regression churn model.
  - `dashboard.py` – Streamlit app showing churn rate, trends, and top spenders.

## Basic Usage (high level)

1. Create and populate the SQLite database (`data_generator.py`).
2. Run analysis scripts (`analysis.py`) or the Streamlit dashboard (`dashboard.py`).
3. Use the dashboard to explore churn, trends, and top spenders.

Implementation details for each script are added in later steps.

