"""
app/context_caching/cache_request.py

Pydantic schemas for the Context Caching endpoint.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ContextCacheResponse(BaseModel):
    """Response after extracting text from documents and creating a context cache."""
    status: str = Field(..., description="Status of the operation: 'success' or 'error'")
    extracted_text: str = Field(..., description="Combined extracted text from all uploaded documents")
    Cache_name: str = Field(..., description="Vertex AI context cache resource name")
    input_token: Optional[str] = Field(default=None, description="Number of tokens in the prompt")
    output_token: Optional[str] = Field(default=None, description="Number of tokens in the generated response")
