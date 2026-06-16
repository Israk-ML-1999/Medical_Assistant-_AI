"""
app/MedAi_chatbot/chat_request.py

Pydantic schemas for the MedAi Chatbot endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ConversationPair(BaseModel):
    """A single question-answer pair in the conversation history."""
    user_query: str = Field(..., description="The doctor's previous question")
    chat_respons: str = Field(..., description="The chatbot's previous response")


class ChatRequestSchema(BaseModel):
    """Request schema for the MedAi Chatbot endpoint."""
    question: str = Field(..., description="The doctor's current question")
    live_transcript: Optional[str] = Field(
        default=None, description="Live doctor-patient conversation transcript (not cached)"
    )
    conversation_history: Optional[List[ConversationPair]] = Field(
        default_factory=list, description="Previous chat history pairs (not cached)"
    )
    cache_name: Optional[str] = Field(
        default=None,
        description="Existing context cache name/id. If null/empty, document_texts must be provided.",
    )
    document_texts: Optional[List[str]] = Field(
        default_factory=list,
        description="Document text(s) to cache (only used when cache_name is not provided)",
    )


class ChatResponseSchema(BaseModel):
    """Response schema for the MedAi Chatbot endpoint."""
    status: str = Field(..., description="'success' or 'error'")
    chat_respons: str = Field(..., description="The chatbot's answer to the doctor's question")
    new_Cache_id: Optional[str] = Field(
        default=None, description="New cache id if a new cache was created, else null"
    )
    input_token: Optional[str] = Field(default=None, description="Number of tokens in the prompt")
    output_token: Optional[str] = Field(default=None, description="Number of tokens in the generated response")
