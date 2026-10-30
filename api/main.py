"""

The prediction service. It holds NO model logic of its own: every number comes from churn_core.
Run locally from the repo root:  uvicorn api.main:app --reload

Protection: every endpoint except /health needs the header  X-API-Key: <API_KEY>.  The key is read
from the environment (a .env file locally, Secret Manager in the cloud). The server refuses to
start without one. Uploaded files are processed in memory and never stored.
"""
import io
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from api.prediction_log import PredictionLog
from churn_core import bands, checks, config, economics, model
from churn_core.explain import explain
from churn_core.predict import predict, predict_customers
from churn_core.preprocess import load_bundle
from churn_core.schema import Customer, clean_raw, customer_frame

load_dotenv()

# Makes the docs page show an Authorize button where you paste the key.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

MAX_UPLOAD_BYTES = 2_000_000      # 2 MB
MAX_ROWS = 5_000

TEMPLATE_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
TEMPLATE_ROW = ["EXAMPLE-0001", "Female", 0, "No", "No", 5, "Yes", "No", "Fiber optic", "No", "No", "No",
                "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 89.9, 420.5]

EXAMPLE_CUSTOMER = {
    "gender": "Female", "senior_citizen": False, "partner": False, "dependents": False, "tenure": 5,
    "phone_service": True, "multiple_lines": "No", "internet_service": "Fiber optic",
    "online_security": "No", "online_backup": "No", "device_protection": "No", "tech_support": "No",
    "streaming_tv": "Yes", "streaming_movies": "Yes", "contract": "Month-to-month",
    "paperless_billing": True, "payment_method": "Electronic check",
    "monthly_charges": 89.9, "total_charges": 420.5,
}


# ---------------- request and response shapes ----------------
class AssumptionsIn(BaseModel):
    """Business assumptions. Leave a field out to use the saved default. They are assumptions."""
    model_config = ConfigDict(extra="forbid")
    r: float | None = Field(None, gt=0, le=1, description="offer success rate as a fraction (0.5 = 50%)")
    d: float | None = Field(None, ge=0, lt=1, description="discount as a fraction of the bill (0.2 = 20%)")
    months: int | None = Field(None, gt=0, le=60, description="months a retained customer keeps paying")


class AssumptionsOut(BaseModel):
    r: float
    d: float
    months: int


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"customer": EXAMPLE_CUSTOMER}})
    customer: Customer
    assumptions: AssumptionsIn = AssumptionsIn()


class PredictResponse(BaseModel):
    probability: float
    band: str
    band_description: str
    flag: bool
    expected_gain: float
    cutoff: float
    assumptions: AssumptionsOut
    warnings: list[str]
    model_version: str


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"customer": EXAMPLE_CUSTOMER}})
    customer: Customer


class Lever(BaseModel):
    text: str
    status: str


class WaterfallItem(BaseModel):
    feature: str
    label: str
    value: str
    effect: float


class Driver(WaterfallItem):
    direction: str
    sentence: str
    lever: Lever | None = None


class ExplainResponse(BaseModel):
    probability: float
    band: str
    base_logit: float
    raw_logit: float
    drivers: list[Driver]
    waterfall: list[WaterfallItem]
    note: str
    model_version: str


class BatchRow(BaseModel):
    row: int                       # spreadsheet row number (header = row 1)
    customer_id: str | None
    probability: float
    band: str
    flag: bool
    expected_gain: float
    reasons: list[str] | None = None


class BatchResponse(BaseModel):
    model_version: str
    cutoff: float
    assumptions: AssumptionsOut
    n_scored: int
    summary: dict
    issues: list[dict]
    results: list[BatchRow]


def resolve_assumptions(a: AssumptionsIn) -> config.Assumptions:
    default = config.default_assumptions()
    return config.Assumptions(r=default.r if a.r is None else a.r,
                              d=default.d if a.d is None else a.d,
                              months=default.months if a.months is None else a.months)


def score_batch(df: pd.DataFrame, a: config.Assumptions, reasons: bool) -> dict:
    """Check, clean and score an uploaded RAW-format frame. Rows with errors are left out."""
    report = checks.check_upload(df)
    body = {
        "model_version": config.MODEL_VERSION, "cutoff": economics.cutoff_from(a.r, a.d),
        "assumptions": {"r": a.r, "d": a.d, "months": a.months},
        "n_scored": report.n_valid, "summary": report.summary(),
        "issues": [asdict(i) for i in report.issues], "results": [],
    }
    if report.n_valid == 0:
        return body

    mask = report.valid_mask
    valid = df[mask].drop(columns=["Churn"], errors="ignore")
    clean = clean_raw(valid)
    out = predict(clean, r=a.r, d=a.d, months=a.months)
    why = explain(clean, top_k=3) if reasons else None

    ids = (valid["customerID"].astype(object).where(valid["customerID"].notna(), None).tolist()
           if "customerID" in valid.columns else [None] * len(valid))
    rows = np.flatnonzero(mask) + 2
    for k in range(len(out)):
        body["results"].append({
            "row": int(rows[k]), "customer_id": ids[k],
            "probability": float(out.probability.iloc[k]), "band": str(out.band.iloc[k]),
            "flag": bool(out.flag.iloc[k]), "expected_gain": float(out.expected_gain.iloc[k]),
            "reasons": [d["sentence"] for d in why[k]["drivers"]] if why else None,
        })
    return body


# ---------------- the app ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.environ.get("API_KEY")
    if not key:
        raise RuntimeError("API_KEY is not set. Put it in .env (local) or Secret Manager (cloud).")
    app.state.api_key = key
    app.state.prediction_log = PredictionLog.from_env()      # off unless PREDICTION_LOG_URL is set
    load_bundle()                         # load the encoder and both models once, at startup,
    model.load_models()                   # not on the first request
    yield
    app.state.prediction_log.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Churn prediction API", version=config.MODEL_VERSION, lifespan=lifespan)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = (request.headers.get("x-request-id") or uuid.uuid4().hex)[:64]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    def require_key(request: Request, key: str | None = Depends(api_key_header)) -> None:
        expected = request.app.state.api_key
        if key is None or not secrets.compare_digest(key.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": config.MODEL_VERSION}

    @app.get("/model-info", dependencies=[Depends(require_key)])
    def model_info(request: Request):
        a = config.default_assumptions()
        cfg = config.model_config()
        edges = bands.default_edges()
        return {
            "model_version": config.MODEL_VERSION,
            "cutoff": config.default_cutoff(),
            "assumptions": {"r": a.r, "d": a.d, "months": a.months},
            "bands": {"labels": list(bands.LABELS), "edges": list(edges),
                      "observed_churn_test": bands.published_table("test")},
            "test_metrics_at_cutoff": cfg.get("test_metrics_at_cutoff"),
            "library_versions": cfg.get("library_versions"),
            "upload_limits": {"max_bytes": MAX_UPLOAD_BYTES, "max_rows": MAX_ROWS},
            "prediction_log": {"enabled": request.app.state.prediction_log.enabled},
        }

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_key)])
    def predict_one(req: PredictRequest, request: Request):
        started = time.perf_counter()
        a = resolve_assumptions(req.assumptions)
        out = predict_customers([req.customer], r=a.r, d=a.d, months=a.months).iloc[0]
        response = PredictResponse(
            probability=float(out.probability), band=str(out.band),
            band_description=bands.DESCRIPTIONS[str(out.band)], flag=bool(out.flag),
            expected_gain=float(out.expected_gain), cutoff=economics.cutoff_from(a.r, a.d),
            assumptions=AssumptionsOut(r=a.r, d=a.d, months=a.months),
            warnings=checks.customer_warnings(req.customer.model_dump()),
            model_version=str(out.model_version))
        request.app.state.prediction_log.record(
            inputs={"customer": req.customer.model_dump(), "assumptions": response.assumptions.model_dump()},
            probability=response.probability, band=response.band, flag=response.flag,
            model_version=response.model_version, latency_ms=round((time.perf_counter() - started) * 1000))
        return response

    @app.post("/explain", response_model=ExplainResponse, dependencies=[Depends(require_key)])
    def explain_one(req: ExplainRequest):
        df = customer_frame([req.customer])
        out = predict(df).iloc[0]
        e = explain(df, top_k=3)[0]
        return ExplainResponse(probability=float(out.probability), band=str(out.band),
                               model_version=str(out.model_version), **e)

    @app.get("/upload-template", dependencies=[Depends(require_key)])
    def upload_template():
        text = ",".join(TEMPLATE_COLUMNS) + "\n" + ",".join(str(v) for v in TEMPLATE_ROW) + "\n"
        return Response(content=text, media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="churn_upload_template.csv"'})

    @app.post("/predict_batch", response_model=BatchResponse, dependencies=[Depends(require_key)])
    async def predict_batch(
        file: Annotated[UploadFile, File()],
        reasons: Annotated[bool, Query(description="add the top-3 plain-language reasons for each row")] = False,
        r: Annotated[float | None, Query(gt=0, le=1, description="offer success rate as a fraction (0.5 = 50%)")] = None,
        d: Annotated[float | None, Query(ge=0, lt=1, description="discount as a fraction (0.2 = 20%)")] = None,
        months: Annotated[int | None, Query(gt=0, le=60, description="months a retained customer keeps paying")] = None,
    ):
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // 1_000_000} MB")
        try:
            df = pd.read_csv(io.BytesIO(data), dtype=str)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise HTTPException(400, f"Could not read the file as a CSV ({type(exc).__name__})") from exc
        if len(df) > MAX_ROWS:
            raise HTTPException(413, f"Too many rows ({len(df)}); the limit is {MAX_ROWS}")

        a = resolve_assumptions(AssumptionsIn(r=r, d=d, months=months))
        body = await run_in_threadpool(score_batch, df, a, reasons)
        if body["n_scored"] == 0:                       # nothing could be scored: report why
            return JSONResponse(status_code=422, content=body)
        return body

    return app


app = create_app()