from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class ConversationPair(BaseModel):
    user_query: str
    chat_respons: str

class ClinicalReportRequest(BaseModel):
    live_transcript: str = Field(..., description="Live transcript between doctor and patient")
    conversation_history: Optional[List[ConversationPair]] = Field(default=[], description="Conversation history as query/response pairs")
    document_texts: Optional[List[str]] = Field(default=[], description="Extracted texts from documents/reports")
    example_structure: List[Dict[str, Any]] = Field(..., description="The example JSON structure to follow for the report")

class ClinicalReportResponse(BaseModel):
    report: Any = Field(..., description="The generated report in JSON format following the example_structure")
    input_tokens: Optional[str] = Field(default=None, description="Number of tokens in the prompt")
    output_tokens: Optional[str] = Field(default=None, description="Number of tokens in the generated response")
