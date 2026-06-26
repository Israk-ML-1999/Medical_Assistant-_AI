"""
app/patient_Summary/router.py

FastAPI router for the Patient Summary endpoint.
"""

from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.patient_Summary.case_request import PatientSummaryResponse
from app.patient_Summary.llm_service import generate_patient_summary

router = APIRouter(
    prefix="/patient-summary",
    tags=["Patient Summary"],
)


@router.post(
    "/generate",
    response_model=PatientSummaryResponse,
    summary="Generate a dynamic patient summary from documents, text, and/or conversation",
)
async def generate_patient_summary_endpoint(
    document_files: Optional[List[UploadFile]] = File(
        default=None, description="Optional multiple files (PDF/DOC/Image) - reports, prescriptions, etc."
    ),
    document_text: Optional[str] = Form(
        default=None, description="Optional raw document text (may include OCR output)"
    ),
    conversation: Optional[str] = Form(
        default=None, description="Optional doctor-patient conversation transcript"
    ),
    input_language: Optional[str] = Form(
        default=None, description="Optional input language (e.g. bn, en, hu, es, fr, de)"
    ),
    output_language: Optional[str] = Form(
        default=None, description="Optional output language (e.g. bn, en, hu, es, fr, de)"
    ),
):
    """
    Generates a dynamic patient summary (ICD coding, conditions, medications, allergies,
    surgeries/procedures, family/social history, and any other AI-identified clinically
    relevant sections) based on uploaded documents, raw text, and/or conversation.
    """
    if not document_files and not document_text and not conversation:
        raise HTTPException(
            status_code=400,
            detail="At least one of document_files, document_text, or conversation must be provided.",
        )

    try:
        result = await generate_patient_summary(
            document_files=document_files,
            document_text=document_text,
            conversation=conversation,
            input_language=input_language,
            output_language=output_language,
        )
        return PatientSummaryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Patient summary generation failed: {str(e)}")
