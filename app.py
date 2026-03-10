from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.bbpou_participation import BBPouParticipation, agent, validate_document
from agents.gst_certificate import GstCertificateDetails, agent as gst_agent, validate_document as validate_gst_document
from agents.letter_from_sponser_bank import (
    LetterFromSponsorBankDetails,
    agent as sponsor_bank_agent,
    validate_document as validate_sponsor_bank_document,
)
from agents.ndc_letter import (
    NdcLetterDetails,
    agent as ndc_letter_agent,
    validate_document as validate_ndc_letter_document,
)
from agents.commencement_letter_to_rbi import (
    CommencementLetterToRbiDetails,
    agent as commencement_letter_agent,
    validate_document as validate_commencement_letter_document,
)
from agents.ifsc_and_settlement_account_confirmation import (
    IfscAndSettlementAccountConfirmationDetails,
    agent as ifsc_settlement_agent,
    validate_document as validate_ifsc_settlement_document,
)
from agents.escrow_account_details import (
    EscrowAccountDetails,
    agent as escrow_account_agent,
    validate_document as validate_escrow_account_document,
)
from agents.ai_clearance_from_entity import (
    AiClearanceFromEntityDetails,
    agent as ai_clearance_agent,
    validate_document as validate_ai_clearance_document,
)


app = FastAPI(
    title="Agents VLM API",
    description="API endpoints for document validation agents.",
    version="1.0.0",
)


class ValidateBBPouRequest(BaseModel):
    document_path: str = Field(
        description="Absolute or relative path to the BBPOU participation PDF document."
    )


class ValidateGstCertificateRequest(BaseModel):
    document_path: str = Field(
        description="Absolute or relative path to the GST certificate PDF document."
    )


class ValidateLetterFromSponsorBankRequest(BaseModel):
    document_path: str = Field(
        description="Absolute or relative path to the letter from sponsor bank PDF document."
    )


class ValidateNdcLetterRequest(BaseModel):
    document_path: str = Field(
        description="Absolute or relative path to the NDC letter / sponsor letter PDF document."
    )


class ValidateCommencementLetterToRbiRequest(BaseModel):
    document_path: str = Field(
        description="Absolute or relative path to the commencement letter to RBI PDF document."
    )


class ValidateIfscAndSettlementAccountConfirmationRequest(BaseModel):
    document_path: str = Field(
        description="Absolute or relative path to the IFSC and settlement account confirmation PDF document."
    )


class ValidateEscrowAccountDetailsRequest(BaseModel):
    document_path: str = Field(
        description="Absolute or relative path to the escrow account details PDF document."
    )


class ValidateAiClearanceFromEntityRequest(BaseModel):
    document_path: str = Field(
        description="Absolute or relative path to the AI clearance from entity PDF document."
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


@app.post("/agents/gst-certificate/validate", response_model=GstCertificateDetails)
def validate_gst_certificate(payload: ValidateGstCertificateRequest) -> GstCertificateDetails:
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
        return validate_gst_document(gst_agent, path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {exc}",
        ) from exc


@app.post("/agents/letter-from-sponsor-bank/validate", response_model=LetterFromSponsorBankDetails)
def validate_letter_from_sponsor_bank(payload: ValidateLetterFromSponsorBankRequest) -> LetterFromSponsorBankDetails:
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
        return validate_sponsor_bank_document(sponsor_bank_agent, path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {exc}",
        ) from exc


@app.post("/agents/ndc-letter/validate", response_model=NdcLetterDetails)
def validate_ndc_letter(payload: ValidateNdcLetterRequest) -> NdcLetterDetails:
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
        return validate_ndc_letter_document(ndc_letter_agent, path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {exc}",
        ) from exc


@app.post("/agents/commencement-letter-to-rbi/validate", response_model=CommencementLetterToRbiDetails)
def validate_commencement_letter_to_rbi(payload: ValidateCommencementLetterToRbiRequest) -> CommencementLetterToRbiDetails:
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
        return validate_commencement_letter_document(commencement_letter_agent, path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {exc}",
        ) from exc


@app.post("/agents/ifsc-and-settlement-account-confirmation/validate", response_model=IfscAndSettlementAccountConfirmationDetails)
def validate_ifsc_and_settlement_account_confirmation(payload: ValidateIfscAndSettlementAccountConfirmationRequest) -> IfscAndSettlementAccountConfirmationDetails:
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
        return validate_ifsc_settlement_document(ifsc_settlement_agent, path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {exc}",
        ) from exc


@app.post("/agents/escrow-account-details/validate", response_model=EscrowAccountDetails)
def validate_escrow_account_details(payload: ValidateEscrowAccountDetailsRequest) -> EscrowAccountDetails:
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
        return validate_escrow_account_document(escrow_account_agent, path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {exc}",
        ) from exc


@app.post("/agents/ai-clearance-from-entity/validate", response_model=AiClearanceFromEntityDetails)
def validate_ai_clearance_from_entity(payload: ValidateAiClearanceFromEntityRequest) -> AiClearanceFromEntityDetails:
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
        return validate_ai_clearance_document(ai_clearance_agent, path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {exc}",
        ) from exc
