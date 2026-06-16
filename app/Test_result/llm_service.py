"""
app/Test_result/llm_service.py

Core logic for the Test Result analysis endpoint using Gemini via Vertex AI
(Agent Platform). Analyzes uploaded documents/text/conversation to identify all
tests performed, results, and recommended follow-up tests, grouping them into
dynamic sections (Laboratory Results, Imaging, ECG & Cardiology, Microbiology &
Culture, Recommended Follow-up Tests, etc.)
"""

import json
import re
from typing import List, Optional

from fastapi import UploadFile
from google.genai import types

from app.vertex_ai_client import get_client, get_model_name
from app.file_utils import files_to_parts


SYSTEM_PROMPT = """You are an expert clinical pathologist and diagnostic analyst assisting
a licensed physician. Your task is to carefully review uploaded medical documents
(lab reports, imaging reports, ECG/echo reports, microbiology/culture reports,
prescriptions, etc.), raw document text, and/or doctor-patient conversation transcripts,
and produce a structured summary of ALL test results found, plus any recommended
follow-up tests.

INPUT MAY INCLUDE:
1. Uploaded files (PDF/Image/DOC) - read and analyze directly.
2. Raw document text - this text MAY contain OCR artifacts (broken words, noise,
   misplaced line breaks, misread numbers/units). Carefully interpret the intended
   medical meaning, especially for lab values and units.
3. Doctor-patient conversation transcript - may mention tests done, pending tests,
   or tests the doctor recommends next.

YOUR TASK:
Identify and group findings into clinically meaningful DYNAMIC sections. Common sections
include (but are not limited to):
- "Laboratory Results" - ALL lab values found (CBC, blood sugar, HbA1c, renal/liver
  function, lipid profile, etc.) with values, units, and interpretation (normal/abnormal).
- "Imaging" - X-ray, CT, MRI, USG findings.
- "ECG & Cardiology" - ECG, echocardiogram, stress test findings.
- "Microbiology & Culture" - culture/sensitivity results, infection findings.
- "Recommended Follow-up Tests" - tests the doctor should order next, and when (timing
  if mentioned or clinically appropriate).
- Any other clinically relevant test category found in the documents.

SECTION RULES:
- Only include sections that are actually supported by the provided data.
- Each section's "description" must be a single, informative, well-written paragraph
  summarizing ALL relevant findings for that section (combine multiple data points).
- Resolve OCR noise intelligently; prioritize medical plausibility for values and units.
- Do not fabricate results not supported by the data.

OUTPUT FORMAT (STRICT):
Return a single valid JSON object with this exact shape:
{
  "test_results": [
    {"title": "<section title>", "description": "<detailed informative paragraph>"}
  ]
}
Output JSON only - no markdown, no commentary, no code fences.
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


async def generate_test_results(
    document_files: Optional[List[UploadFile]] = None,
    document_text: Optional[str] = None,
    conversation: Optional[str] = None,
) -> dict:
    """
    Analyzes documents/text/conversation and returns a dict:
    {"test_results": [ {title, description} ]}
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
            "\n\nNow analyze the above and return the test_results JSON object as instructed. JSON only."
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
        parsed = {"test_results": []}

    if "test_results" not in parsed or not isinstance(parsed["test_results"], list):
        parsed["test_results"] = []

    if getattr(response, "usage_metadata", None):
        parsed["input_token"] = str(response.usage_metadata.prompt_token_count)
        parsed["output_token"] = str(response.usage_metadata.candidates_token_count)

    return parsed
