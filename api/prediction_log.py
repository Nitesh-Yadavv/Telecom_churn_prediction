"""

Best-effort, anonymous log of single predictions, for monitoring the live demo.

Design rules (each one protects the free database or the visitor):
  * It can NEVER break a prediction: every failure is swallowed.
  * After a failure it pauses for a minute, so a database that is down or waking up cannot add its
    timeout to every request.
  * It samples and throttles (at most N writes a minute), so a flood of requests cannot flood the table.
  * It keeps only the newest rows (the oldest are trimmed every few writes), so the table cannot grow
    toward the free storage limit.
  * It stores what the form contained (no names, no IDs), the probability, the band and the timing.
  * Off by default: it does nothing unless PREDICTION_LOG_URL is set.
"""
import json
import logging
import os
import random
import threading
import time
from collections import deque

from sqlalchemy import create_engine, text

logger = logging.getLogger("prediction_log")

INSERT = text(
    "INSERT INTO predictions_log (model_version, probability, band, flag, latency_ms, inputs) "
    "VALUES (:model_version, :probability, :band, :flag, :latency_ms, CAST(:inputs AS JSONB))")
TRIM = text("DELETE FROM predictions_log WHERE id <= "
            "(SELECT id FROM predictions_log ORDER BY id DESC OFFSET :keep LIMIT 1)")


def _number(name: str, default, cast):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class PredictionLog:
    def __init__(self, write=None, sample_rate=1.0, max_per_minute=30, keep_rows=5000, trim_every=50,
                 pause_seconds=60, clock=time.monotonic, rng=random.random, dispose=None):
        self._write = write                  # callable(row: dict, trim_keep: int | None); None = disabled
        self.sample_rate, self.max_per_minute = sample_rate, max_per_minute
        self.keep_rows, self.trim_every, self.pause_seconds = keep_rows, trim_every, pause_seconds
        self._clock, self._rng, self._dispose = clock, rng, dispose
        self._lock = threading.Lock()
        self._times: deque = deque()
        self._count = 0
        self._paused_until = 0.0

    @property
    def enabled(self) -> bool:
        return self._write is not None

    @classmethod
    def from_env(cls) -> "PredictionLog":
        url = os.environ.get("PREDICTION_LOG_URL")
        settings = {"sample_rate": _number("LOG_SAMPLE_RATE", 1.0, float),
                    "max_per_minute": _number("LOG_MAX_PER_MINUTE", 30, int),
                    "keep_rows": _number("LOG_KEEP_ROWS", 5000, int)}
        if not url:
            return cls(**settings)
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg2://" + url[len("postgresql://"):]
        engine = create_engine(url, pool_size=1, max_overflow=0, pool_pre_ping=True, pool_recycle=240,
                               connect_args={"connect_timeout": 2})

        def write(row: dict, trim_keep) -> None:
            with engine.begin() as conn:
                conn.execute(text("SET LOCAL statement_timeout = 1500"))     # works through Neon's pooler
                conn.execute(INSERT, row)
                if trim_keep is not None:
                    conn.execute(TRIM, {"keep": trim_keep})

        return cls(write=write, dispose=engine.dispose, **settings)

    def record(self, inputs: dict, probability: float, band: str, flag: bool, model_version: str,
               latency_ms: int) -> bool:
        """Try to log one prediction. Returns True only if a row was written. Never raises."""
        if not self.enabled:
            return False
        now = self._clock()
        with self._lock:
            if now < self._paused_until or self._rng() >= self.sample_rate:
                return False
            while self._times and now - self._times[0] > 60:
                self._times.popleft()
            if len(self._times) >= self.max_per_minute:
                return False
            self._times.append(now)
            self._count += 1
            trim_due = self._count % self.trim_every == 0
        try:
            self._write({"model_version": model_version, "probability": float(probability), "band": band,
                         "flag": bool(flag), "latency_ms": int(latency_ms), "inputs": json.dumps(inputs)},
                        self.keep_rows if trim_due else None)
            return True
        except Exception as exc:        # noqa: BLE001  (logging must never break a prediction)
            logger.warning("prediction log write failed (%s); pausing for %ss", type(exc).__name__,
                           self.pause_seconds)
            with self._lock:
                self._paused_until = now + self.pause_seconds
            return False

    def close(self) -> None:
        if self._dispose:
            self._dispose()