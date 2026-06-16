"""
app/SOAP_Note_generation/router.py

FastAPI router for the SOAP Note Generation endpoint.
"""

from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.SOAP_Note_generation.SOAP_request import SOAPNoteResponse
from app.SOAP_Note_generation.llm_service import generate_soap_note

router = APIRouter(
    prefix="/soap-note",
    tags=["SOAP Note Generation"],
)


@router.post(
    "/generate",
    response_model=SOAPNoteResponse,
    summary="Generate a SOAP note from documents, text, and/or conversation",
)
async def generate_soap_note_endpoint(
    document_files: Optional[List[UploadFile]] = File(
        default=None, description="Optional multiple files (PDF/DOC/Image) - reports, prescriptions, etc."
    ),
    document_text: Optional[str] = Form(
        default=None, description="Optional raw document text (may include OCR output)"
    ),
    conversation: Optional[str] = Form(
        default=None, description="Optional doctor-patient conversation transcript"
    ),
    user_instruction: Optional[str] = Form(
        default=None, description="Optional custom instruction from the doctor for better output"
    ),
):
    """
    Generates a structured SOAP note (Subjective, Objective, Assessment, Plan)
    based on uploaded medical documents, raw document text, conversation transcript,
    and optional custom instructions.
    """
    # At least one input source must be provided
    if not document_files and not document_text and not conversation:
        raise HTTPException(
            status_code=400,
            detail="At least one of document_files, document_text, or conversation must be provided.",
        )

    try:
        result = await generate_soap_note(
            document_files=document_files,
            document_text=document_text,
            conversation=conversation,
            user_instruction=user_instruction,
        )
        return SOAPNoteResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SOAP note generation failed: {str(e)}")
