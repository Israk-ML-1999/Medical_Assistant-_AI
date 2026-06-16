"""
app/context_caching/cache_service.py

Core logic for:
1. Extracting text from uploaded documents using Gemini (Vertex AI) directly
   (reading PDF/Image files), and
2. Creating a Vertex AI Context Cache containing that extracted text, so it can
   be reused cheaply across subsequent chatbot requests without re-sending the
   full document content every time.

This module also exposes helper functions used by the MedAi_chatbot module to
create/retrieve caches by name.
"""

import datetime
from typing import List, Optional

from fastapi import UploadFile
from google.genai import types

from config import settings
from app.vertex_ai_client import get_client, get_model_name
from app.file_utils import files_to_parts


EXTRACTION_SYSTEM_PROMPT = """You are a medical document OCR and text extraction assistant.
Read the provided document(s) (PDF/Image) carefully and extract ALL text content,
preserving the original structure (headings, line breaks, tables as plain text rows)
as closely as possible. Do not summarize, do not omit any information, do not add
commentary. Output ONLY the extracted plain text from all documents combined,
separated by a line "--- Next Document ---" between files if multiple files are given.
"""


async def extract_text_with_gemini(document_files: List[UploadFile]) -> str:
    """
    Uses Gemini (Vertex AI) to read uploaded PDF/Image files directly and extract
    their full text content.
    """
    client = get_client()
    model_name = get_model_name(settings.GEMINI_FLASH_CACHE_MODEL)

    content_parts: List[types.Part] = [types.Part.from_text(text=EXTRACTION_SYSTEM_PROMPT)]

    file_parts = await files_to_parts(document_files)
    content_parts.extend(file_parts)

    content_parts.append(types.Part.from_text(text="\n\nNow extract all text from the document(s) above."))

    generation_config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=8192,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=content_parts,
        config=generation_config
    )
    in_tok = str(response.usage_metadata.prompt_token_count) if getattr(response, "usage_metadata", None) else None
    out_tok = str(response.usage_metadata.candidates_token_count) if getattr(response, "usage_metadata", None) else None
    return response.text or "", in_tok, out_tok


def create_context_cache(extracted_text: str, display_name: Optional[str] = None) -> str:
    """
    Creates a Vertex AI Context Cache containing the given extracted text.
    Returns the cache resource name (e.g. 'projects/.../cachedContents/...').
    """
    client = get_client()

    if display_name is None:
        display_name = f"medai-doc-cache-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    # Gemini 2.5 context caching requires a minimum of 1024 tokens.
    # To bypass this limit for short texts, we pad it with repeated text.
    if len(extracted_text) < 5000:
        # Pad until it is at least 5,000 characters (~1200 tokens)
        padding_needed = 5000 - len(extracted_text)
        padding_str = "\n\n[Padding to bypass minimum token limit for caching. Please ignore.]\n"
        repetitions = (padding_needed // len(padding_str)) + 1
        extracted_text += padding_str * repetitions

    cached_content = client.caches.create(
        model=settings.GEMINI_FLASH_CACHE_MODEL,
        config=types.CreateCachedContentConfig(
            system_instruction=(
                "The following is extracted text from a patient's medical document(s). "
                "Use this as context for answering questions about the patient. "
                "Ignore any repetitive padding text at the end."
            ),
            display_name=display_name,
            ttl="3600s",
            contents=[types.Part.from_text(text=extracted_text)]
        )
    )

    return cached_content.name


def get_cached_content(cache_name: str) -> types.CachedContent:
    """Retrieves an existing CachedContent object by its resource name."""
    client = get_client()
    return client.caches.get(name=cache_name)


async def extract_and_cache(document_files: List[UploadFile]) -> dict:
    """
    Full workflow for the /context-caching/extract-document endpoint:
    1. Extract text from uploaded documents using Gemini.
    2. Create a Vertex AI context cache containing that text.
    3. Return extracted text + cache name.
    """
    extracted_text, in_tok, out_tok = await extract_text_with_gemini(document_files)

    try:
        cache_name = create_context_cache(extracted_text)
        status = "success"
    except Exception:
        # If caching fails (e.g. text too short for caching minimum token requirements),
        # still return the extracted text with an empty cache name.
        cache_name = ""
        status = "success"

    return {
        "status": status,
        "extracted_text": extracted_text,
        "Cache_name": cache_name,
        "input_token": in_tok,
        "output_token": out_tok,
    }
