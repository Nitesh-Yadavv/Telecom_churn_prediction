CREATE TABLE customers (
    customer_id      VARCHAR(20) PRIMARY KEY,
    gender            VARCHAR(10),
    senior_citizen    BOOLEAN,
    partner           BOOLEAN,
    dependents        BOOLEAN,
    tenure            INTEGER
);

CREATE TABLE services (
    customer_id       VARCHAR(20) PRIMARY KEY REFERENCES customers(customer_id),
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
    customer_id       VARCHAR(20) PRIMARY KEY REFERENCES customers(customer_id),
    contract          VARCHAR(20),
    paperless_billing BOOLEAN,
    payment_method    VARCHAR(30),
    monthly_charges   NUMERIC(10,2),
    total_charges     NUMERIC(10,2)
);

CREATE TABLE churn_status (
    customer_id       VARCHAR(20) PRIMARY KEY REFERENCES customers(customer_id),
    churn             BOOLEAN
);