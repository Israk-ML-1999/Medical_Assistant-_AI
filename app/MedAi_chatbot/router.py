"""
app/MedAi_chatbot/router.py

FastAPI router for the MedAi Chatbot endpoint.
"""

from fastapi import APIRouter, HTTPException

from app.MedAi_chatbot.chat_request import ChatRequestSchema, ChatResponseSchema
from app.MedAi_chatbot.llm_service import generate_chat_response

router = APIRouter(
    prefix="/medai-chatbot",
    tags=["MedAi Chatbot"],
)


@router.post(
    "/chat",
    response_model=ChatResponseSchema,
    summary="Chat with MedAI using cached document context, live transcript, and chat history",
)
async def chat_endpoint(payload: ChatRequestSchema):
    """
    Handles a chatbot turn:
    - If `cache_name` is provided, reuses the existing Vertex AI context cache.
    - If `cache_name` is not provided but `document_texts` is, generates the
      answer and creates a new context cache in parallel, returning new_Cache_id.
    - If neither is provided, answers from live_transcript + chat_history only.
    """
    try:
        result = await generate_chat_response(
            question=payload.question,
            live_transcript=payload.live_transcript,
            conversation_history=payload.conversation_history,
            cache_name=payload.cache_name,
            document_texts=payload.document_texts,
        )
        return ChatResponseSchema(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot response generation failed: {str(e)}")
