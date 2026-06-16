"""
app/context_caching/router.py

FastAPI router for the Context Caching endpoint.
"""

from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.context_caching.cache_request import ContextCacheResponse
from app.context_caching.cache_service import extract_and_cache

router = APIRouter(
    prefix="/context-caching",
    tags=["Context Caching"],
)


@router.post(
    "/extract-document",
    response_model=ContextCacheResponse,
    summary="Extract text from documents and create a Vertex AI context cache",
)
async def extract_document_endpoint(
    files: List[UploadFile] = File(..., description="One or more files (PDF/Image) to extract and cache"),
):
    """
    Extracts text from one or more uploaded documents using Gemini, creates a
    Vertex AI Context Cache containing that text, and returns the extracted text
    along with the cache resource name for reuse in subsequent requests
    (e.g. the MedAi Chatbot endpoint).
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded.")

    try:
        result = await extract_and_cache(files)
        return ContextCacheResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context caching failed: {str(e)}")
