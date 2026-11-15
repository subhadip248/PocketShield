"""HTTP API for PocketShield URL and SMS analysis."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, StrictStr

from pocketshield.engine import DetectionEngine


class URLAnalysisRequest(BaseModel):
    url: StrictStr


class SMSAnalysisRequest(BaseModel):
    message: StrictStr


app = FastAPI(title="PocketShield API")
engine = DetectionEngine()


def _analyze_safely(method: str, value: str) -> dict[str, Any]:
    try:
        return getattr(engine, method)(value)
    except Exception as exc:
        # Keep internal exception details and paths out of the API response.
        raise HTTPException(
            status_code=500,
            detail="Analysis could not be completed.",
        ) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/analyze/url")
def analyze_url(request: URLAnalysisRequest) -> dict[str, Any]:
    return _analyze_safely("analyze_url", request.url)


@app.post("/analyze/sms")
def analyze_sms(request: SMSAnalysisRequest) -> dict[str, Any]:
    return _analyze_safely("analyze_sms", request.message)
