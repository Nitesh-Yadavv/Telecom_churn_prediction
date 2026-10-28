"""
Integration tests for the loader against a REAL PostgreSQL. They run only when TEST_DATABASE_URL
points at a throwaway database whose name contains "test": the tests DROP everything in it.
Example:  set TEST_DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/churn_test
"""
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from churn_core.schema import clean_raw
from db.load_data import load_frames, prepare

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL is not set (needs a throwaway PostgreSQL)")
SQL = Path(__file__).resolve().parents[1] / "sql"


@pytest.fixture()
def engine():
    assert "test" in (make_url(URL).database or ""), "refusing to wipe a database whose name has no 'test'"
    eng = create_engine(URL)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    yield eng
    eng.dispose()


def run_sql_file(conn, path):
    """Send a whole .sql file as written (the migration contains % signs, which SQLAlchemy would treat
    as parameter markers if the file went through exec_driver_sql)."""
    cursor = conn.connection.cursor()
    cursor.execute(path.read_text())
    cursor.close()


def counts(conn):
    return {t: conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar_one()
            for t in ["customers", "services", "billing", "churn_status", "scores"]}


def test_fresh_schema_load_and_a_second_run_changes_nothing(engine, loader_parts):
    frames = prepare(*loader_parts)
    with engine.begin() as conn:
        run_sql_file(conn, SQL / "schema.sql")
    with engine.begin() as conn:
        load_frames(conn, frames)
        first = counts(conn)
    with engine.begin() as conn:
        load_frames(conn, frames)                      # again: upsert, no duplicates, no errors
        assert counts(conn) == first == dict.fromkeys(first, 10)
        row = conn.execute(text("SELECT split, probability, band FROM scores JOIN customers USING (customer_id) "
                                "ORDER BY customer_id LIMIT 1")).one()
        assert row.split in ("train", "test") and 0 <= row.probability <= 1 and row.band


def test_a_database_built_by_pandas_without_keys_is_repaired_by_the_migration(engine, loader_parts):
    """The situation on the author's laptop: plain tables with no primary or foreign keys."""
    frames = prepare(*loader_parts)
    clean = clean_raw(loader_parts[0])
    for name, cols in [("customers", ["customer_id", "gender", "senior_citizen", "partner", "dependents", "tenure"]),
                       ("services", frames["services"].columns.tolist()),
                       ("billing", frames["billing"].columns.tolist()),
                       ("churn_status", ["customer_id", "churn"])]:
        clean[cols].to_sql(name, engine, index=False)
    with engine.begin() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM pg_constraint WHERE contype = 'p' AND connamespace = 'public'::regnamespace")).scalar_one() == 0
        for _ in range(2):                             # twice: the migration must be safe to repeat
            run_sql_file(conn, SQL / "migrations" / "002_split_scores_log.sql")
        assert conn.execute(text("SELECT COUNT(*) FROM pg_constraint WHERE contype = 'p' AND connamespace = 'public'::regnamespace")).scalar_one() >= 6
        load_frames(conn, frames)
        assert counts(conn) == dict.fromkeys(counts(conn), 10)