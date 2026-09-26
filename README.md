# Telecom_churn_prediction# Customer Churn Prediction

End-to-end churn prediction system for a telecom company — Postgres-backed SQL EDA,
leakage-safe ML pipeline, SHAP interpretability, cost-based decision thresholding,
and a live Streamlit app querying the database directly.

> Work in progress — full write-up added on Day 7.

## Structure
- `sql/` — database creation, schema, and EDA queries
- `db/` — data loading script
- `notebooks/` — SQL EDA, feature engineering, modelling
- `app/` — Streamlit application
- `models/` — trained model artifact
- `outputs/plots/` — saved figures

## Setup (reproduce from scratch)

1. Install Postgres and start the server.
2. Connect to the default `postgres` database (it always exists) and run
   `sql/create_db.sql` to create `churn_db`.
3. Switch your connection to `churn_db` and run `sql/schema.sql` to create
   the 4 tables (`customers`, `services`, `billing`, `churn_status`).
4. Copy `.env.example` to `.env` and fill in your real Postgres host, port,
   database name, user, and password.
5. Install Python dependencies: `pip install -r requirements.txt`
6. Download the dataset (Kaggle: `blastchar/telco-customer-churn`) into
   `data/`, then run `python db/load_data.py` to clean and load it into
   Postgres.