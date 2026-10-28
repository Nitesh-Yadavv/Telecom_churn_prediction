"""

Loads the customer data and the model scores into PostgreSQL (local or Neon).

  * Cleans the raw CSV with the SAME function the app uses (churn_core.schema.clean_raw).
  * Fills customers.split from the frozen data/split_assignment.csv.
  * Fills the scores table from reports/scores.csv, with the band computed by churn_core.bands.
  * Safe to run again: rows are upserted (insert, or update if the customer already exists), and
    existing rows are never deleted or reordered.
  * All-or-nothing: everything is checked inside one transaction and rolled back if any check fails.
  * The target must be named, so a run can never silently go to the wrong database.

Run from the repo root (with `pip install -e .` done and the tables created from sql/schema.sql):

    python db/load_data.py --target local --dry-run     # check the files only, touch no database
    python db/load_data.py --target local
    python db/load_data.py --target neon
"""
import argparse
import os
import sys
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import URL

from churn_core import bands, config
from churn_core.schema import clean_raw

TABLE_COLUMNS = {
    "customers": ["customer_id", "gender", "senior_citizen", "partner", "dependents", "tenure", "split"],
    "services": ["customer_id", "phone_service", "multiple_lines", "internet_service", "online_security",
                 "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"],
    "billing": ["customer_id", "contract", "paperless_billing", "payment_method", "monthly_charges",
                "total_charges"],
    "churn_status": ["customer_id", "churn"],
    "scores": ["customer_id", "probability", "score_kind", "band", "flag", "model_version"],
}
LOAD_ORDER = ["customers", "services", "billing", "churn_status", "scores"]   # customers first (foreign keys)
CHUNK = 500


# ---------------- pure part: read the files and prepare one frame per table ----------------
def prepare(raw: pd.DataFrame, split: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """Validate the three input files against each other and build one DataFrame per table."""
    if raw["customerID"].duplicated().any():
        raise ValueError("the raw CSV contains duplicate customerID values")
    clean = clean_raw(raw)

    split = split[["customer_id", "split"]]
    if split["customer_id"].duplicated().any():
        raise ValueError("split_assignment.csv contains a customer more than once")
    clean = clean.merge(split, on="customer_id", how="left")
    if clean["split"].isna().any():
        raise ValueError(f"{int(clean['split'].isna().sum())} customers are missing from split_assignment.csv")

    s = scores.copy()
    if s["customer_id"].duplicated().any():
        raise ValueError("scores.csv contains a customer more than once")
    unknown = set(s["customer_id"]) - set(clean["customer_id"])
    if unknown:
        raise ValueError(f"scores.csv has {len(unknown)} customers that are not in the raw CSV")

    joined = s.merge(clean[["customer_id", "split", "monthly_charges"]], on="customer_id",
                     suffixes=("", "_db"))
    expected_kind = np.where(joined["split_db"] == "train", "oof", "test")
    if (joined["score_kind"].to_numpy() != expected_kind).any():
        raise ValueError("score_kind in scores.csv does not match the train/test split")
    if not np.allclose(joined["monthly_charges"], joined["monthly_charges_db"], atol=0.005):
        raise ValueError("monthly charges in scores.csv do not match the raw CSV (different customers?)")

    p = s["p_cal"].to_numpy(dtype=float)
    if (s["flag"].astype(bool).to_numpy() != (p >= config.default_cutoff())).any():
        raise ValueError("flag in scores.csv disagrees with the cutoff in model_config.json")

    scores_out = pd.DataFrame({
        "customer_id": s["customer_id"], "probability": p, "score_kind": s["score_kind"],
        "band": bands.assign_band(p), "flag": s["flag"].astype(bool), "model_version": s["model_version"],
    })
    frames = {name: clean[cols] for name, cols in TABLE_COLUMNS.items() if name != "scores"}
    frames["scores"] = scores_out
    return frames


def to_records(df: pd.DataFrame) -> list:
    """Rows as plain Python values (psycopg2 cannot send numpy types); missing values become None."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


# ---------------- database part ----------------
def upsert(conn, name: str, df: pd.DataFrame, key: str = "customer_id") -> None:
    table = Table(name, MetaData(), autoload_with=conn)
    unknown = [c for c in df.columns if c not in table.c]
    if unknown:
        raise RuntimeError(f"table {name} has no column(s) {unknown}; run the migration or sql/schema.sql")
    records = to_records(df)
    for i in range(0, len(records), CHUNK):
        stmt = pg_insert(table).values(records[i:i + CHUNK])
        stmt = stmt.on_conflict_do_update(index_elements=[key],
                                          set_={c: stmt.excluded[c] for c in df.columns if c != key})
        conn.execute(stmt)


def run_checks(conn, frames: dict) -> list:
    """(name, expected, actual) for numbers that must match what was just loaded."""
    def q(sql):
        return conn.execute(text(sql)).scalar_one()

    c, s = frames["customers"], frames["scores"]
    flagged_test = int(s.loc[s["customer_id"].isin(c.loc[c["split"] == "test", "customer_id"]), "flag"].sum())
    return [
        ("customers", len(c), q("SELECT COUNT(*) FROM customers")),
        ("services", len(frames["services"]), q("SELECT COUNT(*) FROM services")),
        ("billing", len(frames["billing"]), q("SELECT COUNT(*) FROM billing")),
        ("churn_status", len(frames["churn_status"]), q("SELECT COUNT(*) FROM churn_status")),
        ("scores", len(s), q("SELECT COUNT(*) FROM scores")),
        ("train customers", int((c["split"] == "train").sum()), q("SELECT COUNT(*) FROM customers WHERE split = 'train'")),
        ("test customers", int((c["split"] == "test").sum()), q("SELECT COUNT(*) FROM customers WHERE split = 'test'")),
        ("churned customers", int(frames["churn_status"]["churn"].sum()), q("SELECT COUNT(*) FROM churn_status WHERE churn")),
        ("flagged test customers", flagged_test,
         q("SELECT COUNT(*) FROM scores s JOIN customers c USING (customer_id) WHERE c.split = 'test' AND s.flag")),
        ("total monthly charges", round(float(frames["billing"]["monthly_charges"].sum()), 2),
         float(q("SELECT ROUND(SUM(monthly_charges)::numeric, 2) FROM billing"))),
    ]


def load_frames(conn, frames: dict) -> list:
    """Upsert every table, then verify. Raises (so the caller's transaction rolls back) on any mismatch."""
    for name in LOAD_ORDER:
        upsert(conn, name, frames[name])
    results = run_checks(conn, frames)
    bad = [r for r in results if abs(float(r[1]) - float(r[2])) > 0.01]
    if bad:
        raise RuntimeError("verification failed (the load was rolled back): " +
                           "; ".join(f"{n}: expected {e}, found {a}" for n, e, a in bad))
    return results


def make_engine(target: str):
    """(engine, a label that is safe to print). The password is never printed."""
    if target == "neon":
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise SystemExit("DATABASE_URL is not set (put the Neon connection string in .env)")
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url[len("postgresql://"):]
        host = urlsplit(url).hostname
    else:
        names = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
        missing = [n for n in names if not os.environ.get(n)]
        if missing:
            raise SystemExit(f"missing in .env: {', '.join(missing)}")
        host = os.environ["DB_HOST"]
        url = URL.create("postgresql+psycopg2", username=os.environ["DB_USER"],
                         password=os.environ["DB_PASSWORD"], host=host,
                         port=int(os.environ["DB_PORT"]), database=os.environ["DB_NAME"])
    return create_engine(url, pool_pre_ping=True), f"{target} (host: {host})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Load customers and scores into PostgreSQL.")
    ap.add_argument("--target", choices=["local", "neon"], required=True,
                    help="local = the DB_* settings in .env; neon = DATABASE_URL in .env")
    ap.add_argument("--dry-run", action="store_true", help="validate the files, touch no database")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation question for neon")
    args = ap.parse_args(argv)
    load_dotenv()

    frames = prepare(pd.read_csv(config.data_dir() / "Telco-Customer-Churn.csv"),
                     pd.read_csv(config.data_dir() / "split_assignment.csv"),
                     pd.read_csv(config.reports_dir() / "scores.csv"))
    print("Files are consistent:", {k: len(v) for k, v in frames.items()})
    if args.dry_run:
        return 0

    engine, label = make_engine(args.target)
    print("Target:", label)
    if args.target == "neon" and not args.yes and \
            input("Load into this remote database? Type yes to continue: ").strip().lower() != "yes":
        print("Cancelled.")
        return 1
    try:
        with engine.begin() as conn:
            results = load_frames(conn, frames)
    except Exception as exc:            # noqa: BLE001  (print one clear line, nothing was committed)
        print(f"FAILED, nothing was committed: {exc}")
        return 1
    for name, expected, actual in results:
        print(f"  {name:24s} expected {expected:>12} found {actual:>12}  OK")
    print("Committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())