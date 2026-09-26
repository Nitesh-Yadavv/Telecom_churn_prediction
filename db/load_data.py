"""
load_data.py
Reads the raw Telco churn CSV, cleans it, and loads it into the
normalized Postgres schema (customers, services, billing, churn_status).
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ---- 1. Load DB credentials from .env ----
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ---- 2. Read raw CSV ----
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")
df = pd.read_csv(CSV_PATH)

print(f"Loaded raw CSV: {df.shape[0]} rows, {df.shape[1]} columns")

# ---- 3. Clean known issues ----
# TotalCharges has blank strings for a few rows (new customers, tenure=0)
# Convert to numeric, coercing blanks to NaN, then fill with 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Standardize Yes/No -> boolean for binary columns
yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in yes_no_cols:
    df[col] = df[col].map({"Yes": True, "No": False})

df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool)

# ---- 4. Split into the 4 target tables ----
customers = df[[
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure"
]].rename(columns={
    "customerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "gender": "gender"
})

services = df[[
    "customerID", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]].rename(columns={
    "customerID": "customer_id",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies"
})

billing = df[[
    "customerID", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]].rename(columns={
    "customerID": "customer_id",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges"
})

churn_status = df[["customerID", "Churn"]].rename(columns={
    "customerID": "customer_id",
    "Churn": "churn"
})

# ---- 5. Load into Postgres ----
# Order matters: customers first (others reference it via foreign key)
customers.to_sql("customers", engine, if_exists="append", index=False)
print(f"Loaded {len(customers)} rows into customers")

services.to_sql("services", engine, if_exists="append", index=False)
print(f"Loaded {len(services)} rows into services")

billing.to_sql("billing", engine, if_exists="append", index=False)
print(f"Loaded {len(billing)} rows into billing")

churn_status.to_sql("churn_status", engine, if_exists="append", index=False)
print(f"Loaded {len(churn_status)} rows into churn_status")

print("Done — data loaded into Postgres.")