"""
app/file_utils.py

Shared utility functions for handling uploaded files (PDF, DOC, images) and
converting them into Vertex AI `Part` objects so Gemini can read them directly.
"""

from typing import List, Optional
from fastapi import UploadFile
from google.genai import types


# Mapping of common file extensions to MIME types accepted by Gemini
EXTENSION_MIME_MAP = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
}


def guess_mime_type(filename: str, fallback: str = "application/octet-stream") -> str:
    """Guess MIME type from file extension."""
    if "." not in filename:
        return fallback
    ext = filename.rsplit(".", 1)[-1].lower()
    return EXTENSION_MIME_MAP.get(ext, fallback)


async def files_to_parts(files: Optional[List[UploadFile]]) -> List[types.Part]:
    """
    Reads a list of UploadFile objects and converts each into a Vertex AI Part
    (inline data) so Gemini can analyze the file content directly (PDF/DOC/Image).
    """
    parts: List[types.Part] = []
    if not files:
        return parts

    for f in files:
        if f is None or f.filename == "":
            continue
        content = await f.read()
        if not content:
            continue
        mime_type = f.content_type or guess_mime_type(f.filename)
        parts.append(types.Part.from_bytes(data=content, mime_type=mime_type))

    return parts


async def files_to_bytes(files: Optional[List[UploadFile]]):
    """
    Reads a list of UploadFile objects and returns list of tuples:
    (filename, raw_bytes, mime_type). Used by Document AI OCR service.
    """
    results = []
    if not files:
        return results

    for f in files:
        if f is None or f.filename == "":
            continue
        content = await f.read()
        if not content:
            continue
        mime_type = f.content_type or guess_mime_type(f.filename)
        results.append((f.filename, content, mime_type))

    return results
