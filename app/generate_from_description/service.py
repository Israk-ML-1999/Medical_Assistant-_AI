import json
from typing import Dict, Any, List, Optional, Literal
from google import genai
from google.genai import types
from google.oauth2 import service_account
from pydantic import BaseModel, Field, field_validator, ValidationError
from config import settings
from app.vertex_ai_client import get_client, get_model_name


# ---------- Assistant Configuration Schema ----------
class AssistantConfig(BaseModel):
    name: str = Field(..., example="Psychiatry Discharge Assistant")
    specialty: str = Field(..., example="Psychiatry")
    clinical_role: str = Field(..., example="Attending Physician")
    output_language: str = Field("English", example="English")
    document_type: str = Field("SOAP Note", example="Discharge Summary")
    care_setting: Literal[
        "ED", "Inpatient Ward", "ICU/HDU", "Ambulatory Outpatient",
        "Procedural", "Telehealth", "Community"
    ] = Field("Ambulatory Outpatient", example="Inpatient Ward")
    instructions: str = Field("", example="Write concise discharge summaries. Include ICD‑10 codes in the Coding section.")
    output_format: Literal["Full Paragraphs", "Bullet Points", "Mixed Format"] = Field("Mixed Format", example="Bullet Points")
    patient_population: Literal["All Ages", "Geriatric (>60)", "Paediatric (<18)", "Adult (18-60)"] = Field("All Ages", example="All Ages")
    sections: List[str] = Field(
        ["Subjective", "Objective", "Assessment", "Plan", "Coding"],
        example=["Reason for admission", "Hospital course", "Discharge medications", "Follow-up plan", "Coding"]
    )
    description: str = Field("", example="Assistant for generating psychiatry discharge summaries with ICD‑10 coding.")
    tags: List[str] = Field([], example=["Psychiatry", "Discharge", "ICD-10"])
    text_example: Optional[str] = Field(None, example="S - SUBJECTIVE: ...\nO - OBJECTIVE: ...")
    visibility: str = Field("private", example="private")
    input_token: Optional[str] = Field(default=None, description="Number of tokens in the prompt")
    output_token: Optional[str] = Field(default=None, description="Number of tokens in the generated response")

    @field_validator("specialty")
    def validate_specialty(cls, v):
        allowed = [
            "Cardiology", "Neurology", "Psychiatry", "General Medicine",
            "Dentistry", "Pharmacy", "Internal Medicine", "Pediatrics",
            "Surgery", "Orthopedics", "Radiology", "Pathology"
        ]
        for a in allowed:
            if v.lower() == a.lower():
                return a
        raise ValueError(f"Specialty must be one of {allowed}")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Psychiatry Discharge Assistant",
                "specialty": "Psychiatry",
                "clinical_role": "Attending Physician",
                "output_language": "English",
                "document_type": "Discharge Summary",
                "care_setting": "Inpatient Ward",
                "instructions": "Write concise discharge summaries. Include ICD‑10 codes in the Coding section.",
                "output_format": "Bullet Points",
                "patient_population": "All Ages",
                "sections": [
                    "Reason for admission",
                    "Hospital course",
                    "Discharge medications",
                    "Follow-up plan",
                    "Coding"
                ],
                "description": "Assistant for generating psychiatry discharge summaries with ICD‑10 coding.",
                "tags": ["Psychiatry", "Discharge", "ICD-10"],
                "text_example": None,
                "visibility": "private"
            }
        }

# ---------- Validation Service ----------
class ValidationService:
    @staticmethod
    def validate_assistant_config(json_data: dict):
        try:
            return AssistantConfig(**json_data)
        except ValidationError as e:
            raise ValueError(f"Invalid assistant config: {e.errors()}")

validator = ValidationService()

# ---------- Prompt Builder ----------
class PromptBuilder:
    @staticmethod
    def build_assistant_config_prompt(description: str, schema_json: dict) -> str:
        return f"""
You are an AI that converts user requests into a medical assistant configuration.
Output a JSON object that matches this schema exactly:
{schema_json}

Follow these guidelines:
- `specialty` must be one of: Cardiology, Neurology, Psychiatry, General Medicine, Dentistry, Pharmacy, Internal Medicine, Pediatrics, Surgery, Orthopedics, Radiology, Pathology.
- `output_format` must be "Full Paragraphs", "Bullet Points", or "Mixed Format".
- `patient_population` must be "All Ages", "Geriatric (>60)", "Paediatric (<18)", or "Adult (18-60)".
- `care_setting` must be "ED", "Inpatient Ward", "ICU/HDU", "Ambulatory Outpatient", "Procedural", "Telehealth", "Community".
- `document_type` is free‑text.
- `output_language` is free‑text.
- `instructions` should capture the tone, rules, and behaviour.
- `description` is a 2‑3 sentence explanation (for humans).
- `tags` are relevant keywords.
- `text_example` is optional.

User request: {description}
Return ONLY valid JSON.
"""

    @staticmethod
    def build_generate_note_prompt(transcript: str, assistant: Dict[str, Any], note_type: str) -> str:
        system_parts = [
            f"You are a {assistant.get('clinical_role', 'clinician')} in {assistant.get('specialty', 'general medicine')}.",
            f"Language: {assistant.get('output_language', 'English')}.",
            f"Document type: {assistant.get('document_type', 'SOAP Note')}.",
            f"Care setting: {assistant.get('care_setting', 'Ambulatory Outpatient')}.",
            f"Patient population: {assistant.get('patient_population', 'All Ages')}.",
            f"Output format: {assistant.get('output_format', 'Mixed Format')}.",
        ]
        instructions = assistant.get('instructions', '')
        if instructions:
            system_parts.append(f"Additional instructions: {instructions}")
        text_example = assistant.get('text_example', '')
        if text_example:
            system_parts.append(f"\nHere is an example of the desired writing style:\n{text_example}\n")

        sections = assistant.get('sections', ['Subjective', 'Objective', 'Assessment', 'Plan'])
        if any("coding" in s.lower() or "icd" in s.lower() for s in sections):
            system_parts.append(
                "If a 'Coding' or 'ICD-10' section is present, extract or suggest relevant ICD-10 codes from the transcript/document and list them clearly in that section. Use the format: 'ICD-10 Code - Description'."
            )
        system_parts.append(f"Use the following sections: {', '.join(sections)}.")
        system_parts.append("Base your writing strictly on the provided transcript.")
        system_parts.append("Highlight critical safety issues with a 'RED FLAG:' line.")
        system = " ".join(system_parts)

        return f"""{system}

Write a {note_type} note using the sections above.
Transcript:
{transcript}

Output format: JSON object with keys exactly matching the section names.
You may add an extra key "red_flag" if needed.
Return ONLY valid JSON.
"""

# ---------- LLM Service ----------
class LLMService:
    def __init__(self):
        self.client = get_client()

    async def generate_structured(self, prompt: str, schema: Dict[str, Any], model: str = "flash"):
        if not self.client:
            raise ValueError("Vertex AI client not configured")
        model_name = get_model_name()
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.2
            )
        )
        in_tok = str(response.usage_metadata.prompt_token_count) if getattr(response, "usage_metadata", None) else None
        out_tok = str(response.usage_metadata.candidates_token_count) if getattr(response, "usage_metadata", None) else None
        return response.text, in_tok, out_tok

llm_service = LLMService()

# ---------- Business Logic ----------
async def generate_assistant_config(description: str):
    if not llm_service.client:
        raise ValueError("Vertex AI client not configured")
    schema = AssistantConfig.model_json_schema()
    prompt = PromptBuilder.build_assistant_config_prompt(description, schema)
    json_output, in_tok, out_tok = await llm_service.generate_structured(prompt, schema)
    config_dict = json.loads(json_output)
    config_dict["input_token"] = in_tok
    config_dict["output_token"] = out_tok
    validated = validator.validate_assistant_config(config_dict)
    return validated.dict()