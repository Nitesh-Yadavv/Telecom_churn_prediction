"""
Proves the two limited database users can do only what they should, using the real connection
strings. Every test runs inside a transaction that is rolled back, so the database never changes.

Needs two lines in .env (never commit them):
    DATABASE_URL_RO=postgresql://dashboard_ro:...@HOST/neondb?sslmode=require
    DATABASE_URL_WRITER=postgresql://api_writer:...@HOST/neondb?sslmode=require

Run from the repo root:  python scripts/check_db_roles.py
"""
import os
import sys

import psycopg2
from dotenv import load_dotenv
from psycopg2 import errors

INSERT_LOG = ("INSERT INTO predictions_log (model_version, probability, band, flag, latency_ms, inputs) "
              "VALUES ('role-check', 0.5, 'High', true, 1, '{}')")

# (user, what we try, SQL, should it be allowed?)
CHECKS = [
    ("RO", "read scores", "SELECT COUNT(*) FROM scores", True),
    ("RO", "read customers", "SELECT COUNT(*) FROM customers", True),
    ("RO", "read the prediction log", "SELECT COUNT(*) FROM predictions_log", True),
    ("RO", "insert into the log", INSERT_LOG, False),
    ("RO", "update customers", "UPDATE customers SET tenure = tenure", False),
    ("RO", "delete scores", "DELETE FROM scores", False),
    ("RO", "create a table", "CREATE TABLE should_not_exist (x int)", False),
    ("RO", "drop a table", "DROP TABLE scores", False),
    ("WRITER", "insert into the log", INSERT_LOG, True),
    ("WRITER", "read the log", "SELECT COUNT(*) FROM predictions_log", True),
    ("WRITER", "trim old log rows", "DELETE FROM predictions_log WHERE id < 0", True),
    ("WRITER", "read customers", "SELECT COUNT(*) FROM customers", False),
    ("WRITER", "read scores", "SELECT COUNT(*) FROM scores", False),
    ("WRITER", "insert into scores", "INSERT INTO scores (customer_id) VALUES ('x')", False),
    ("WRITER", "update the log", "UPDATE predictions_log SET flag = flag", False),
    ("WRITER", "create a table", "CREATE TABLE should_not_exist (x int)", False),
]


def attempt(url: str, sql: str) -> str:
    """'allowed', 'denied' (a permission error) or 'error: ...' (something else went wrong)."""
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        return "allowed"
    except errors.InsufficientPrivilege:
        return "denied"
    except psycopg2.Error as exc:
        return f"error: {type(exc).__name__}"
    finally:
        conn.rollback()
        conn.close()


def main() -> int:
    load_dotenv()
    urls = {"RO": os.environ.get("DATABASE_URL_RO"), "WRITER": os.environ.get("DATABASE_URL_WRITER")}
    missing = [k for k, v in urls.items() if not v]
    if missing:
        print("Missing in .env: " + ", ".join("DATABASE_URL_" + m for m in missing))
        return 2
    failures = 0
    for user, what, sql, should_allow in CHECKS:
        result = attempt(urls[user], sql)
        ok = result == ("allowed" if should_allow else "denied")
        failures += not ok
        print(f"{'OK ' if ok else 'BAD'}  {user:6s} {what:26s} expected {'allowed' if should_allow else 'denied':8s} got {result}")
    print("\nAll checks passed." if not failures else f"\n{failures} check(s) FAILED.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())