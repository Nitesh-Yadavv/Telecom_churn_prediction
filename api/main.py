"""

The prediction service. It holds NO model logic of its own: every number comes from churn_core.
Run locally from the repo root:  uvicorn api.main:app --reload

Protection: every endpoint except /health needs the header  X-API-Key: <API_KEY>.  The key is read
from the environment (a .env file locally, Secret Manager in the cloud). The server refuses to
start without one.
"""
import os
import secrets
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from churn_core import bands, checks, config, economics, model
from churn_core.predict import predict_customers
from churn_core.preprocess import load_bundle
from churn_core.schema import Customer

load_dotenv()

# Makes the docs page show an Authorize button where you paste the key.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------- request and response shapes ----------------
class AssumptionsIn(BaseModel):
    """Business assumptions. Leave a field out to use the saved default. They are assumptions."""
    model_config = ConfigDict(extra="forbid")
    r: float | None = Field(None, gt=0, le=1, description="offer success rate")
    d: float | None = Field(None, ge=0, lt=1, description="discount, as a share of the bill")
    months: int | None = Field(None, gt=0, le=60, description="months a retained customer keeps paying")


class AssumptionsOut(BaseModel):
    r: float
    d: float
    months: int


EXAMPLE_CUSTOMER = {
    "gender": "Female", "senior_citizen": False, "partner": False, "dependents": False, "tenure": 5,
    "phone_service": True, "multiple_lines": "No", "internet_service": "Fiber optic",
    "online_security": "No", "online_backup": "No", "device_protection": "No", "tech_support": "No",
    "streaming_tv": "Yes", "streaming_movies": "Yes", "contract": "Month-to-month",
    "paperless_billing": True, "payment_method": "Electronic check",
    "monthly_charges": 89.9, "total_charges": 420.5,
}


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


def resolve_assumptions(a: AssumptionsIn) -> config.Assumptions:
    default = config.default_assumptions()
    return config.Assumptions(r=default.r if a.r is None else a.r,
                              d=default.d if a.d is None else a.d,
                              months=default.months if a.months is None else a.months)


# ---------------- the app ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.environ.get("API_KEY")
    if not key:
        raise RuntimeError("API_KEY is not set. Put it in .env (local) or Secret Manager (cloud).")
    app.state.api_key = key
    load_bundle()                         # load the encoder and both models once, at startup,
    model.load_models()                   # not on the first request
    yield


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
    def model_info():
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
        }

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_key)])
    def predict_one(req: PredictRequest):
        a = resolve_assumptions(req.assumptions)
        out = predict_customers([req.customer], r=a.r, d=a.d, months=a.months).iloc[0]
        return PredictResponse(
            probability=float(out.probability), band=str(out.band),
            band_description=bands.DESCRIPTIONS[str(out.band)], flag=bool(out.flag),
            expected_gain=float(out.expected_gain), cutoff=economics.cutoff_from(a.r, a.d),
            assumptions=AssumptionsOut(r=a.r, d=a.d, months=a.months),
            warnings=checks.customer_warnings(req.customer.model_dump()),
            model_version=str(out.model_version))

    return app


app = create_app()