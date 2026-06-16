"""
app/SOAP_Note_generation/llm_service.py

Core logic for generating a SOAP note (Subjective, Objective, Assessment, Plan)
using Gemini via Vertex AI (Agent Platform). Handles multi-file input (PDF/DOC/Image),
raw document text (often OCR output), doctor-patient conversation transcripts,
and optional custom instructions from the doctor.
"""

import json
import re
from typing import List, Optional

from fastapi import UploadFile
from google.genai import types

from app.vertex_ai_client import get_client, get_model_name
from app.file_utils import files_to_parts


# ----------------------------------------------------------------------
# Base professional system prompt for SOAP note generation
# ----------------------------------------------------------------------
BASE_SYSTEM_PROMPT = """You are an expert clinical documentation assistant specialized in generating
SOAP (Subjective, Objective, Assessment, Plan) notes for licensed medical doctors.

You will be given one or more of the following inputs:
1. Uploaded medical documents (lab reports, prescriptions, discharge summaries, imaging reports, etc.)
   as raw files (PDF, images, or DOC) - read and analyze them directly.
2. Raw document text - this text MAY have been extracted via OCR, so it can contain
   spelling errors, broken line breaks, misplaced characters, or formatting noise.
   Carefully interpret the intended medical meaning despite OCR artifacts.
3. A doctor-patient conversation transcript.
4. Optional custom instructions from the doctor to refine the output.

YOUR TASK:
Generate a clinically accurate, well-structured SOAP note based on ALL available information.

GUIDELINES:
- Subjective: Patient's reported symptoms, history of present illness, complaints,
  relevant past medical/family/social history as mentioned by the patient or in conversation.
- Objective: Vital signs, physical exam findings, lab/imaging results, and any
  measurable/observable clinical data found in the documents.
- Assessment: Doctor's clinical impression, differential diagnosis, and diagnosis
  based on the Subjective and Objective findings. Use professional medical terminology.
- Plan: Treatment plan, medications (with dosage if available), procedures, referrals,
  patient education, and follow-up recommendations.

IMPORTANT RULES:
- If information for a section is not available, write a concise clinically appropriate
  statement (e.g., "Not documented in provided records.") instead of leaving it empty.
- Do not fabricate clinical findings that are not supported by the provided data.
- Resolve OCR noise intelligently - prioritize medical plausibility.
- If the doctor provides custom instructions, follow them with high priority while
  maintaining clinical accuracy.
- Output MUST be valid JSON only, with exactly these four keys: "Subjective", "Objective",
  "Assessment", "Plan". No markdown, no extra commentary, no code fences.
"""


def _extract_json(text: str) -> dict:
    """
    Extracts a JSON object from the model's raw text output.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    text = text.strip()

    # Remove markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # If still not starting with '{', try to find the first JSON object
    if not text.startswith("{"):
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(1)

    return json.loads(text)


async def generate_soap_note(
    document_files: Optional[List[UploadFile]] = None,
    document_text: Optional[str] = None,
    conversation: Optional[str] = None,
    user_instruction: Optional[str] = None,
) -> dict:
    """
    Generates a SOAP note using Gemini (Vertex AI) based on uploaded files,
    raw document text, conversation transcript, and optional custom instructions.

    Returns a dict with keys: Subjective, Objective, Assessment, Plan.
    """

    client = get_client()
    model_name = get_model_name()

    # Build the multi-part content list for the model
    content_parts: List[types.Part] = []

    # 1. Add the system prompt (with optional custom instruction appended)
    system_text = BASE_SYSTEM_PROMPT
    if user_instruction:
        system_text += f"\n\nDOCTOR'S CUSTOM INSTRUCTIONS (high priority):\n{user_instruction}\n"

    content_parts.append(types.Part.from_text(text=system_text))

    # 2. Add uploaded files (PDF/DOC/Image) directly so Gemini can read them
    file_parts = await files_to_parts(document_files)
    if file_parts:
        content_parts.append(types.Part.from_text(text="\n\n--- UPLOADED MEDICAL DOCUMENT(S) ---"))
        content_parts.extend(file_parts)

    # 3. Add raw OCR/document text if provided
    if document_text and document_text.strip():
        content_parts.append(
            types.Part.from_text(text=f"\n\n--- DOCUMENT TEXT (may include OCR artifacts) ---\n{document_text.strip()}")
        )

    # 4. Add conversation transcript if provided
    if conversation and conversation.strip():
        content_parts.append(
            types.Part.from_text(text=f"\n\n--- DOCTOR-PATIENT CONVERSATION TRANSCRIPT ---\n{conversation.strip()}")
        )

    # 5. Final instruction reminder
    content_parts.append(
        types.Part.from_text(text=
            "\n\nNow generate the SOAP note as a single valid JSON object with keys: "
            "Subjective, Objective, Assessment, Plan. Output JSON only."
        )
    )

    generation_config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=4096,
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
        # Fallback: return raw text wrapped so the API doesn't crash
        parsed = {
            "Subjective": "Unable to parse model output.",
            "Objective": "Unable to parse model output.",
            "Assessment": "Unable to parse model output.",
            "Plan": raw_text if isinstance(raw_text, str) else str(raw_text),
        }

    # Ensure all required keys are present
    for key in ["Subjective", "Objective", "Assessment", "Plan"]:
        parsed.setdefault(key, "Not documented in provided records.")

    if getattr(response, "usage_metadata", None):
        parsed["input_token"] = str(response.usage_metadata.prompt_token_count)
        parsed["output_token"] = str(response.usage_metadata.candidates_token_count)

    return parsed
