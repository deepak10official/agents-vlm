from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.bbpou_participation import BBPouParticipation, agent, validate_document


app = FastAPI(
    title="Agents VLM API",
    description="API endpoints for document validation agents.",
    version="1.0.0",
)


class ValidateBBPouRequest(BaseModel):
    document_path: str = Field(
        description="Absolute or relative path to the BBPOU participation PDF document."
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agents/bbpou-participation/validate", response_model=BBPouParticipation)
def validate_bbpou_participation(payload: ValidateBBPouRequest) -> BBPouParticipation:
    path = Path(payload.document_path)
    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Document not found at path: {payload.document_path}",
        )
    if path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported for this endpoint.",
        )

    try:
        return validate_document(agent, path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {exc}",
        ) from exc
