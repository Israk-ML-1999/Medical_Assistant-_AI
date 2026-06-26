import asyncio
import json
from typing import Any, Dict, List, Optional
from google.genai import types

from config import settings
from app.vertex_ai_client import get_client, get_model_name

SYSTEM_PROMPT = """You are a STRICT Clinical Scribe AI. You are NOT the treating physician.
Your ONLY task is to extract and document facts exactly as they appear in the provided inputs (live transcript, conversation history, document texts). 

CRITICAL ANTI-HALLUCINATION RULES:
1. ZERO INVENTION (NO PRESCRIBING): DO NOT invent, suggest, or prescribe medications, dosages, treatments, follow-up dates, or specific ICD codes. If a medicine or plan is not EXPLICITLY spoken by the doctor in the transcript, mentioned in the conversation history, or written in the documents, it DOES NOT exist. Logical clinical deductions are STRICTLY FORBIDDEN.
2. STRICT FACT GROUNDING: Every single claim in your final report must have direct evidence in the provided input text.
3. HANDLING MISSING DATA (NULL VALUES): If the "example_structure" contains keys like "Discharge Medications", "Follow-up Plan", or "red_flag", but the inputs contain NO information about them, you MUST write "None documented during this encounter" or "Not applicable". DO NOT copy dummy data from the example structure.
4. STRICT SKELETON: Use the exact JSON keys provided in the "example_structure". Do not add or remove structural keys.

CHAIN OF THOUGHT (_clinical_thought_process):
Before generating the final report, you MUST think step-by-step in the "_clinical_thought_process" key. Explicitly ask yourself:
- "Did the doctor explicitly mention any medications in the transcript/history? (Yes/No)"
- "If No, I must write 'None documented' in the Discharge Medications section."
- "Did the doctor explicitly give a follow-up date? (Yes/No)"

FINAL OUTPUT:
After your thought process, output the structured report under the key "final_report". Provide your output as strictly valid, parseable JSON.
"""

def _build_language_instruction(input_language: Optional[str], output_language: Optional[str]) -> str:
    lang_map = {
        "bn": "Bangla",
        "en": "English",
        "hu": "Hungarian",
        "es": "Spanish",
        "fr": "French",
        "de": "German"
    }
    
    in_lang_name = lang_map.get(input_language.lower()) if input_language else None
    out_lang_name = lang_map.get(output_language.lower()) if output_language else None
    
    instructions = ["\n\n--- LANGUAGE RULES ---"]
    
    if in_lang_name:
        instructions.append(f"1. The input information is primarily in: {in_lang_name}. Analyze it in this language context.")
    else:
        instructions.append("1. Automatically identify the language of the input medical information.")
        
    if out_lang_name:
        instructions.append(f"2. You MUST generate all text content/values inside the final JSON report in this language: {out_lang_name}.")
    else:
        instructions.append("2. You MUST generate all text content/values inside the final JSON report in the same language as the patient document/input information.")
        
    instructions.append("3. Keep the JSON structure keys exactly as provided in the 'example_structure'. Do not translate the JSON keys themselves, only the populated values/details inside those keys.")
    
    return "\n".join(instructions)


def _build_prompt(
    live_transcript: str,
    conversation_history: Optional[List[Any]],
    document_texts: Optional[List[str]],
    example_structure: List[Dict[str, Any]],
    input_language: Optional[str] = None,
    output_language: Optional[str] = None,
) -> str:
    prompt_parts = []
    
    if document_texts and len(document_texts) > 0:
        docs_str = "\n\n--- Next Document ---\n\n".join(document_texts)
        prompt_parts.append(f"--- DOCUMENT TEXTS ---\n{docs_str}")
        
    if conversation_history and len(conversation_history) > 0:
        history_lines = []
        for pair in conversation_history:
            query = pair.user_query if hasattr(pair, 'user_query') else pair.get("user_query", "")
            resp = pair.chat_respons if hasattr(pair, 'chat_respons') else pair.get("chat_respons", "")
            history_lines.append(f"User: {query}")
            history_lines.append(f"AI Assistant: {resp}")
        
        chat_str = "\n\n".join(history_lines)
        prompt_parts.append(f"--- PREVIOUS CONVERSATION HISTORY ---\n{chat_str}")
        
    prompt_parts.append(f"--- LIVE TRANSCRIPT ---\n{live_transcript}")
    
    structure_str = json.dumps(example_structure, ensure_ascii=False, indent=2)
    prompt_parts.append(f"--- SKELETON TO STRICTLY FOLLOW (IGNORE DUMMY DATA) ---\n{structure_str}")
    
    # Add language rules
    prompt_parts.append(_build_language_instruction(input_language, output_language))
    
    prompt_parts.append("\nGenerate the JSON output now, starting with your '_clinical_thought_process'.")
    return "\n\n".join(prompt_parts)

def _generate_report(prompt: str) -> dict:
    client = get_client()
    model_name = get_model_name()
    
    generation_config = types.GenerateContentConfig(
        temperature=0.2, 
        max_output_tokens=8192,
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json"
    )
    
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=generation_config
    )
    
    in_tok = str(response.usage_metadata.prompt_token_count) if getattr(response, "usage_metadata", None) else None
    out_tok = str(response.usage_metadata.candidates_token_count) if getattr(response, "usage_metadata", None) else None
    
    try:
        raw_json = json.loads(response.text)
        report_data = raw_json
    except Exception:
        report_data = {"error": "Failed to parse LLM output as JSON", "raw_output": response.text}
        
    return {
        "report": report_data,
        "input_tokens": in_tok,
        "output_tokens": out_tok
    }

async def generate_clinical_report_service(
    live_transcript: str,
    conversation_history: Optional[List[Any]],
    document_texts: Optional[List[str]],
    example_structure: List[Dict[str, Any]],
    input_language: Optional[str] = None,
    output_language: Optional[str] = None,
) -> dict:
    prompt = _build_prompt(live_transcript, conversation_history, document_texts, example_structure, input_language, output_language)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _generate_report, prompt)
    return result