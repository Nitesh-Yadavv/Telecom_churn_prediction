-- roles_neon.sql: two limited database users for the deployed app.
-- Run in the Neon SQL Editor as the project's owner role, AFTER schema.sql, once.
-- Replace the two passwords first. Generate them locally and never commit the real ones:
--     python -c "import secrets; print(secrets.token_urlsafe(24))"
--
-- Roles created with SQL start with basic privileges only. (Roles created in the Neon Console are
-- administrators, which is exactly what the dashboard and the API must NOT use.)

REVOKE CREATE ON SCHEMA public FROM PUBLIC;

CREATE ROLE dashboard_ro LOGIN PASSWORD 'REPLACE_WITH_PASSWORD_1';
CREATE ROLE api_writer   LOGIN PASSWORD 'REPLACE_WITH_PASSWORD_2';

-- both may connect and look inside the public schema; neither may create or change anything by default
GRANT CONNECT ON DATABASE neondb TO dashboard_ro, api_writer;
GRANT USAGE ON SCHEMA public TO dashboard_ro, api_writer;

-- the dashboard: read-only, on every table (including tables added later)
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dashboard_ro;

-- the API: only the prediction log, and only what logging needs (insert new rows, trim old ones)
GRANT SELECT, INSERT, DELETE ON predictions_log TO api_writer;
GRANT USAGE ON SEQUENCE predictions_log_id_seq TO api_writer;