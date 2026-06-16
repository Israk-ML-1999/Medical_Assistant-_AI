"""
app/OCR_file_to_text/text_request.py

Pydantic schemas for the OCR (Document AI) endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SuccessfulFileResult(BaseModel):
    """Result for a successfully processed file."""
    file_name: str = Field(..., description="Name of the processed file")
    text: str = Field(..., description="Extracted text content with structure preserved")


class FailedFileResult(BaseModel):
    """Result for a file that failed to process."""
    file_name: str = Field(..., description="Name of the file that failed")
    error: str = Field(..., description="Error message describing the failure")


class OCRResponse(BaseModel):
    """Top-level OCR response containing successful and failed file results."""
    status: str = Field(default="success", description="Overall status of the OCR operation")
    results: List[SuccessfulFileResult] = Field(
        default_factory=list, description="List of successfully processed files with extracted text"
    )
    failed: Optional[List[FailedFileResult]] = Field(
        default=None, description="List of files that failed processing, if any"
    )
