"""
app/api_client.py

The dashboard's only way to reach the model: HTTP calls to the prediction API.
The dashboard holds no model code. The API key is read from the environment on the server side
and is never sent to the browser.
"""
import logging
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("api_client")

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
TIMEOUT = (10, 90)          # connect, read: generous, because a cold start can take several seconds
ATTEMPTS = 2


class ApiError(Exception):
    """Raised with a message that is safe and clear enough to show to the user."""


def _call(method: str, path: str, **kwargs) -> dict:
    if not API_KEY:
        raise ApiError("The dashboard has no API key configured (server setup problem).")
    last_error = None
    for attempt in range(ATTEMPTS):
        try:
            response = requests.request(method, f"{API_URL}{path}", headers={"X-API-Key": API_KEY},
                                        timeout=TIMEOUT, **kwargs)
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            # Written to the container log (never shown to the visitor, never includes the key)
            logger.warning("API call failed: %s %s%s -> %r", method, API_URL, path, exc)
            if attempt + 1 < ATTEMPTS:
                time.sleep(2)
    else:
        raise ApiError("Could not reach the prediction service. It may be waking up, "
                       "so please try again in a few seconds.") from last_error

    if response.status_code == 401:
        raise ApiError("The prediction service rejected the dashboard's API key (server setup problem).")
    if response.status_code == 422:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        raise ApiError(f"The service could not accept this input: {detail}")
    if not response.ok:
        raise ApiError(f"The prediction service returned an error (HTTP {response.status_code}).")
    return response.json()


def model_info() -> dict:
    return _call("GET", "/model-info")


def predict(customer: dict, assumptions: dict) -> dict:
    return _call("POST", "/predict", json={"customer": customer, "assumptions": assumptions})


def explain(customer: dict) -> dict:
    return _call("POST", "/explain", json={"customer": customer})