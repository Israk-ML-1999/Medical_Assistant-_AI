"""
app/OCR_file_to_text/router.py

FastAPI router for the OCR (Document AI) file-to-text endpoint.
"""

from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.OCR_file_to_text.text_request import OCRResponse
from app.OCR_file_to_text.file_text_service import extract_text_from_files

router = APIRouter(
    prefix="/ocr",
    tags=["OCR File to Text"],
)


@router.post(
    "/extract-text",
    response_model=OCRResponse,
    summary="Extract structured text from uploaded PDF/Image/DOC files using Document AI",
)
async def extract_text_endpoint(
    files: List[UploadFile] = File(..., description="One or more files (PDF/Image/DOC) to OCR"),
):
    """
    Extracts text from one or more uploaded files using Google Document AI,
    preserving the original document structure as much as possible.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded.")

    try:
        result = await extract_text_from_files(files)
        return OCRResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(e)}")
