"""
app/patient_Summary/llm_service.py

Core logic for generating a dynamic Patient Summary using Gemini via Vertex AI
(Agent Platform). The AI acts as a professional clinical analyst, identifies relevant
sections (ICD Coding, Conditions, Medications, Allergies, Family History, Social History,
Surgeries/Procedures, custom clinical parameters, etc.) DYNAMICALLY based on the
provided documents/text/conversation, and returns each as {title, items:[{name, description}]}.
"""

import json
import re
from typing import List, Optional

from fastapi import UploadFile
from google.genai import types

from app.vertex_ai_client import get_client, get_model_name
from app.file_utils import files_to_parts


# ----------------------------------------------------------------------
# Professional system prompt for dynamic patient summary generation
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior clinical data analyst and certified medical coder assistant,
working for a licensed physician. Your job is to analyze patient medical documents
(reports, prescriptions, discharge summaries, conversation transcripts) and produce a
structured, DYNAMIC "Patient Summary" broken into clinically meaningful sections.

INPUT MAY INCLUDE:
1. Uploaded files (PDF/Image/DOC) - read and analyze directly.
2. Raw document text - this text MAY contain OCR artifacts (broken words, noise,
   misplaced line breaks). Carefully interpret the intended medical meaning.
3. Doctor-patient conversation transcript.

YOUR TASK:
Act as a professional doctor reviewing the chart. Identify and extract ALL clinically
relevant categories present in the data. Common categories include (but are NOT limited to):
- "ICD Coding" (use real ICD-10 codes as the "name" field with their description as "description")
- "Conditions" / "Diagnoses"
- "Medications"
- "Allergies"
- "Surgeries and Procedures"
- "Family History"
- "Social History"
- Any other clinically relevant custom parameter found in the documents
  (e.g., "Immunizations", "Vital Trends", "Lab Abnormalities", etc.)

SECTION TITLES ARE DYNAMIC: only include sections that are actually supported by the
provided data. Do NOT include empty sections. Do NOT force all categories if not present.

OUTPUT FORMAT (STRICT):
Return a single valid JSON object with this exact shape:
{
  "patient_summary": [
    {
      "title": "<section title>",
      "items": [
        {"name": "<short name/code>", "description": "<detailed description>"}
      ]
    }
  ]
}

RULES:
- For "ICD Coding" sections, "name" = ICD-10 code, "description" = condition name/explanation.
- For other sections, "name" = item name/title, "description" = clinically detailed explanation.
- Resolve OCR noise intelligently; prioritize medical plausibility.
- Do not fabricate data not supported by the provided sources.
- Output JSON only - no markdown, no commentary, no code fences.
"""


def _extract_json(text: str) -> dict:
    """Extracts a JSON object from the model's raw text output, handling code fences."""
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    if not text.startswith("{"):
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(1)

    return json.loads(text)


async def generate_patient_summary(
    document_files: Optional[List[UploadFile]] = None,
    document_text: Optional[str] = None,
    conversation: Optional[str] = None,
) -> dict:
    """
    Generates a dynamic patient summary using Gemini (Vertex AI).
    Returns a dict: {"patient_summary": [ {title, items:[{name, description}]} ]}
    """

    client = get_client()
    model_name = get_model_name()

    content_parts: List[types.Part] = [types.Part.from_text(text=SYSTEM_PROMPT)]

    file_parts = await files_to_parts(document_files)
    if file_parts:
        content_parts.append(types.Part.from_text(text="\n\n--- UPLOADED MEDICAL DOCUMENT(S) ---"))
        content_parts.extend(file_parts)

    if document_text and document_text.strip():
        content_parts.append(
            types.Part.from_text(text=f"\n\n--- DOCUMENT TEXT (may include OCR artifacts) ---\n{document_text.strip()}")
        )

    if conversation and conversation.strip():
        content_parts.append(
            types.Part.from_text(text=f"\n\n--- DOCTOR-PATIENT CONVERSATION TRANSCRIPT ---\n{conversation.strip()}")
        )

    content_parts.append(
        types.Part.from_text(text=
            "\n\nNow analyze the above and return the patient_summary JSON object as instructed. JSON only."
        )
    )

    generation_config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=8192,
        response_mime_type="application/json",
    )

    response = client.models.generate_content(
        model=model_name,
        contents=content_parts,
        config=generation_config,
    )

    raw_text = response.text

    try:
        parsed = _extract_json(raw_text)
    except (json.JSONDecodeError, AttributeError, TypeError):
        parsed = {"patient_summary": []}

    if "patient_summary" not in parsed or not isinstance(parsed["patient_summary"], list):
        parsed["patient_summary"] = []

    if getattr(response, "usage_metadata", None):
        parsed["input_token"] = str(response.usage_metadata.prompt_token_count)
        parsed["output_token"] = str(response.usage_metadata.candidates_token_count)

    return parsed
