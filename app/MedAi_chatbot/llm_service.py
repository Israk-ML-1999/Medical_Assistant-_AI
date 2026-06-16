"""
app/MedAi_chatbot/llm_service.py
"""

import asyncio
from typing import List, Optional

from google.genai import types

from config import settings
from app.vertex_ai_client import get_client, get_model_name
from app.context_caching.cache_service import create_context_cache, get_cached_content
from app.MedAi_chatbot.chat_request import ConversationPair


SYSTEM_PROMPT = """You are MedAI, a clinical assistant chatbot helping a licensed doctor
during a patient consultation. You may be given:
- Cached or provided patient document text (lab reports, prescriptions, history, etc.),
  which may contain OCR artifacts - interpret medical meaning carefully.
- A live doctor-patient conversation transcript (ongoing consultation).
- Previous chat history with this doctor.
- The doctor's current question.

If the doctor asks a casual greeting (e.g., 'hi', 'hello') or a general medical question, answer it normally and politely. 
If the doctor asks about the patient but no patient documents or transcript are provided in the context, clearly state in Bengali or English that you do not have any patient information at this moment.
Otherwise, answer the doctor's question CONCISELY and ACCURATELY using only the information available in the provided context.
Respond in the same language the doctor used in their question English or any language. Do not include any JSON or markdown - just a direct, natural-language answer suitable for display in a chat UI.
"""


def _build_history_text(conversation_history: Optional[List[ConversationPair]]) -> str:
    if not conversation_history:
        return ""
    lines = []
    for pair in conversation_history:
        lines.append(f"Doctor: {pair.user_query}")
        lines.append(f"MedAI: {pair.chat_respons}")
    return "\n".join(lines)


def _build_user_content_parts(
    question: str,
    live_transcript: Optional[str],
    conversation_history: Optional[List[ConversationPair]],
    document_text: Optional[str] = None,
) -> List[str]:  # <-- এখানে List[types.Part] এর বদলে List[str] ব্যবহার করা হচ্ছে
    """Builds the non-cached portion of the prompt as plain strings."""
    parts: List[str] = []

    if document_text:
        parts.append(f"--- PATIENT DOCUMENT TEXT ---\n{document_text}\n")

    history_text = _build_history_text(conversation_history)
    if history_text:
        parts.append(f"--- PREVIOUS CHAT HISTORY ---\n{history_text}\n")

    if live_transcript:
        parts.append(f"--- LIVE CONSULTATION TRANSCRIPT ---\n{live_transcript}\n")

    parts.append(f"--- DOCTOR'S QUESTION ---\n{question}\n\nProvide your answer now.")

    return parts


def _generate_answer_with_cache(cache_name: str, user_parts: List[str]) -> str:
    """Generates an answer using an existing Vertex AI context cache."""
    client = get_client()

    generation_config = types.GenerateContentConfig(
        temperature=0.2, 
        max_output_tokens=1024,
        cached_content=cache_name
    )
    response = client.models.generate_content(
        model=settings.GEMINI_FLASH_CACHE_MODEL,
        contents=user_parts,
        config=generation_config
    )
    in_tok = str(response.usage_metadata.prompt_token_count) if getattr(response, "usage_metadata", None) else None
    out_tok = str(response.usage_metadata.candidates_token_count) if getattr(response, "usage_metadata", None) else None
    return response.text or "", in_tok, out_tok


def _generate_answer_no_cache(user_parts: List[str]) -> str:
    """Generates an answer without using any context cache."""
    client = get_client()
    model_name = get_model_name()

    generation_config = types.GenerateContentConfig(
        temperature=0.2, 
        max_output_tokens=1024,
        system_instruction=SYSTEM_PROMPT  
    )
    
    response = client.models.generate_content(
        model=model_name,
        contents=user_parts,
        config=generation_config
    )
    in_tok = str(response.usage_metadata.prompt_token_count) if getattr(response, "usage_metadata", None) else None
    out_tok = str(response.usage_metadata.candidates_token_count) if getattr(response, "usage_metadata", None) else None
    return response.text or "", in_tok, out_tok


def _create_cache_from_documents(document_texts: List[str]) -> str:
    """Creates a new Vertex AI context cache from combined document texts."""
    combined_text = "\n\n--- Next Document ---\n\n".join(document_texts)
    return create_context_cache(combined_text)


async def generate_chat_response(
    question: str,
    live_transcript: Optional[str],
    conversation_history: Optional[List[ConversationPair]],
    cache_name: Optional[str],
    document_texts: Optional[List[str]],
) -> dict:
    """
    Main entry point. Implements the three scenarios described in the module docstring.
    Returns: {"status": "success", "chat_respons": str, "new_Cache_id": Optional[str]}
    """

    has_cache_name = bool(cache_name) and cache_name.strip().lower() not in ("", "null", "none", "string")
    
    # Filter out default swagger values from document_texts
    valid_docs = []
    if document_texts:
        valid_docs = [t for t in document_texts if t and t.strip().lower() not in ("", "null", "none", "string")]
        
    has_document_texts = len(valid_docs) > 0

    # ------------------------------------------------------------------
    # Case 1: cache_name provided -> reuse cache, no new cache created
    # ------------------------------------------------------------------
    if has_cache_name:
        user_parts = _build_user_content_parts(
            question=question,
            live_transcript=live_transcript,
            conversation_history=conversation_history,
            document_text=None,
        )

        loop = asyncio.get_event_loop()
        try:
            answer, in_tok, out_tok = await loop.run_in_executor(
                None, _generate_answer_with_cache, cache_name.strip(), user_parts
            )
        except Exception as e:
            answer = f"Sorry, there was a problem creating the answer using cache.: {str(e)}"
            in_tok, out_tok = None, None

        return {
            "status": "success",
            "chat_respons": answer.strip(),
            "new_Cache_id": None,
            "input_token": in_tok,
            "output_token": out_tok,
        }

    # ------------------------------------------------------------------
    # Case 2: no cache_name but document_texts provided ->
    #   PARALLEL: generate answer (using doc text directly) + create new cache
    # ------------------------------------------------------------------
    if has_document_texts:
        combined_doc_text = "\n\n--- Next Document ---\n\n".join(valid_docs)

        user_parts = _build_user_content_parts(
            question=question,
            live_transcript=live_transcript,
            conversation_history=conversation_history,
            document_text=combined_doc_text,
        )

        loop = asyncio.get_event_loop()

        answer_task = loop.run_in_executor(None, _generate_answer_no_cache, user_parts)
        cache_task = loop.run_in_executor(None, _create_cache_from_documents, valid_docs)

        answer_result, new_cache_name = await asyncio.gather(answer_task, cache_task, return_exceptions=True)

        in_tok, out_tok = None, None
        if isinstance(answer_result, Exception):
            answer = f"Sorry, there was a problem creating the answer.: {str(answer_result)}"
        else:
            answer, in_tok, out_tok = answer_result

        if isinstance(new_cache_name, Exception):
            new_cache_name = None

        return {
            "status": "success",
            "chat_respons": str(answer).strip(),
            "new_Cache_id": new_cache_name,
            "input_token": in_tok,
            "output_token": out_tok,
        }

    # ------------------------------------------------------------------
    # Case 3: no cache_name, no document_texts -> answer from transcript/history only
    # ------------------------------------------------------------------
    user_parts = _build_user_content_parts(
        question=question,
        live_transcript=live_transcript,
        conversation_history=conversation_history,
        document_text=None,
    )

    loop = asyncio.get_event_loop()
    try:
        answer, in_tok, out_tok = await loop.run_in_executor(None, _generate_answer_no_cache, user_parts)
    except Exception as e:
        answer = f"Sorry, there was a problem creating the answer.: {str(e)}"
        in_tok, out_tok = None, None

    return {
        "status": "success",
        "chat_respons": answer.strip(),
        "new_Cache_id": None,
        "input_token": in_tok,
        "output_token": out_tok,
    }