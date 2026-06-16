"""
app/OCR_file_to_text/file_text_service.py

Core logic for extracting structured text from PDF/Image/DOC files using
Google Document AI (via the Document-AI service account credentials).
Preserves document structure (paragraphs, line breaks) as much as possible.
"""

from functools import lru_cache
from typing import List

from fastapi import UploadFile
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.oauth2 import service_account

from config import settings, get_document_ai_credentials_path
from app.OCR_file_to_text.text_request import SuccessfulFileResult, FailedFileResult
from app.file_utils import files_to_bytes


@lru_cache()
def _get_documentai_client() -> documentai.DocumentProcessorServiceClient:
    """Initializes and caches the Document AI client using Document-AI credentials."""
    creds_path = get_document_ai_credentials_path()
    credentials = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    # Document AI regional endpoint
    api_endpoint = f"{settings.DOCAI_LOCATION}-documentai.googleapis.com"
    client_options = ClientOptions(api_endpoint=api_endpoint)

    return documentai.DocumentProcessorServiceClient(
        credentials=credentials, client_options=client_options
    )


def _get_processor_name() -> str:
    """Builds the full Document AI processor resource name."""
    return (
        f"projects/{settings.GCP_PROJECT_ID}/locations/"
        f"{settings.DOCAI_LOCATION}/processors/{settings.DOCAI_PROCESSOR_ID}"
    )


def _process_single_file(filename: str, content: bytes, mime_type: str) -> str:
    """
    Sends a single file to Document AI and returns the extracted text with
    structure preserved (Document AI's `text` field already preserves reading
    order and line breaks reasonably well).
    """
    client = _get_documentai_client()
    processor_name = _get_processor_name()

    raw_document = documentai.RawDocument(content=content, mime_type=mime_type)

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document,
    )

    result = client.process_document(request=request)
    document = result.document

    return document.text or ""


async def extract_text_from_files(files: List[UploadFile]) -> dict:
    """
    Processes multiple uploaded files through Document AI and returns extracted text
    for each, preserving document structure. Returns a dict suitable for OCRResponse.
    """
    file_data = await files_to_bytes(files)

    successful: List[SuccessfulFileResult] = []
    failed: List[FailedFileResult] = []

    for filename, content, mime_type in file_data:
        try:
            extracted_text = _process_single_file(filename, content, mime_type)
            successful.append(SuccessfulFileResult(file_name=filename, text=extracted_text))
        except Exception as e:
            failed.append(FailedFileResult(file_name=filename, error=str(e)))

    return {
        "status": "success" if successful else "error",
        "results": successful,
        "failed": failed if failed else None,
    }
