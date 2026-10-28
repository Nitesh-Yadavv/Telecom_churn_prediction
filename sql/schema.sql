-- schema.sql: FRESH install (an empty churn_db, or an empty Neon database).
-- For a database that already has the four original tables, run
-- migrations/002_split_scores_log.sql instead.

CREATE TABLE customers (
    customer_id      TEXT PRIMARY KEY,
    gender           VARCHAR(10),
    senior_citizen   BOOLEAN,
    partner          BOOLEAN,
    dependents       BOOLEAN,
    tenure           INTEGER,
    -- which side of the frozen train/test split (data/split_assignment.csv) this customer is on
    split            VARCHAR(5) CHECK (split IN ('train', 'test'))
);

CREATE TABLE services (
    customer_id       TEXT PRIMARY KEY REFERENCES customers(customer_id),
    phone_service     BOOLEAN,
    multiple_lines    VARCHAR(20),
    internet_service  VARCHAR(20),
    online_security   VARCHAR(20),
    online_backup     VARCHAR(20),
    device_protection VARCHAR(20),
    tech_support      VARCHAR(20),
    streaming_tv      VARCHAR(20),
    streaming_movies  VARCHAR(20)
);

CREATE TABLE billing (
    customer_id       TEXT PRIMARY KEY REFERENCES customers(customer_id),
    contract          VARCHAR(20),
    paperless_billing BOOLEAN,
    payment_method    VARCHAR(30),
    monthly_charges   NUMERIC(10,2),
    total_charges     NUMERIC(10,2)
);

CREATE TABLE churn_status (
    customer_id       TEXT PRIMARY KEY REFERENCES customers(customer_id),
    churn             BOOLEAN
);

-- One calibrated churn probability per customer, loaded from reports/scores.csv.
-- 'oof' = out-of-fold score for a training customer (slightly optimistic),
-- 'test' = the final model's prediction for a held-out customer (the honest one).
CREATE TABLE scores (
    customer_id    TEXT PRIMARY KEY REFERENCES customers(customer_id),
    probability    DOUBLE PRECISION NOT NULL CHECK (probability BETWEEN 0 AND 1),
    score_kind     VARCHAR(4) NOT NULL CHECK (score_kind IN ('oof', 'test')),
    band           VARCHAR(10) NOT NULL,
    flag           BOOLEAN NOT NULL,
    model_version  VARCHAR(60) NOT NULL
);

-- Single-customer predictions made through the API. No names or IDs: only what the form contained.
-- The API keeps this table small (a sample, and only the newest rows).
CREATE TABLE predictions_log (
    id             BIGSERIAL PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version  VARCHAR(60) NOT NULL,
    probability    DOUBLE PRECISION NOT NULL,
    band           VARCHAR(10) NOT NULL,
    flag           BOOLEAN NOT NULL,
    latency_ms     INTEGER,
    inputs         JSONB NOT NULL
);

CREATE INDEX idx_customers_split ON customers(split);
CREATE INDEX idx_scores_band ON scores(band);
CREATE INDEX idx_predictions_log_created ON predictions_log(created_at);