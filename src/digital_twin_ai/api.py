from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from .pipeline import run_pipeline


class PersonaGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str | None = None
    project_id: str
    random_state: int = 42
    n_synthetic_customers: int = Field(default=1000, ge=100)
    n_personas: int = Field(default=7, ge=2, le=20)
    excel_path: str = "./data/Digital Customer Twin.xlsx"
    output_dir: str = "./output"
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-3-flash"


class PersonaGenerateResponse(BaseModel):
    resource: str
    job_id: str | None = None
    project_id: str
    personas: list[dict]
    artifacts: dict[str, str]
    metadata: dict


app = FastAPI(title="digital-twin-ai", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/internal/personas/generate", response_model=PersonaGenerateResponse, status_code=status.HTTP_200_OK)
def generate_personas(request: PersonaGenerateRequest) -> PersonaGenerateResponse:
    try:
        metadata = run_pipeline(request.model_dump(exclude={"job_id", "project_id"}))
        personas_path = Path(metadata["outputs"]["personas"])
        personas = json.loads(personas_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI persona generation failed: {error}",
        ) from error

    return PersonaGenerateResponse(
        resource="personas",
        job_id=request.job_id,
        project_id=request.project_id,
        personas=personas,
        artifacts=metadata.get("outputs", {}),
        metadata=metadata,
    )
