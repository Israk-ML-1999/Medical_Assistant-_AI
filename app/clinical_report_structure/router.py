"""
app/Test_result/router.py

FastAPI router for the Test Result analysis endpoint.
"""

from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.Test_result.enhnce_request import TestResultResponse
from app.Test_result.llm_service import generate_test_results

router = APIRouter(
    prefix="/test-result",
    tags=["Test Result"],
)


@router.post(
    "/analyze",
    response_model=TestResultResponse,
    summary="Analyze documents/text/conversation and return structured test results",
)
async def analyze_test_results_endpoint(
    document_files: Optional[List[UploadFile]] = File(
        default=None, description="Optional multiple files (PDF/DOC/Image) - lab/imaging/ECG reports, etc."
    ),
    document_text: Optional[str] = Form(
        default=None, description="Optional raw document text (may include OCR output)"
    ),
    conversation: Optional[str] = Form(
        default=None, description="Optional doctor-patient conversation transcript"
    ),
    structured_format: Optional[str] = Form(
        default=None, description="Optional structured format"
    ),
):
    """
    Analyzes uploaded documents, raw text, and/or conversation to identify all
    tests performed, their results, and recommended follow-up tests, grouped into
    dynamic sections (Laboratory Results, Imaging, ECG & Cardiology,
    Microbiology & Culture, Recommended Follow-up Tests, etc.)
    """
    if not document_files and not document_text and not conversation:
        raise HTTPException(
            status_code=400,
            detail="At least one of document_files, document_text, or conversation must be provided.",
        )

    try:
        result = await generate_test_results(
            document_files=document_files,
            document_text=document_text,
            conversation=conversation,
        )
        return TestResultResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test result analysis failed: {str(e)}")
