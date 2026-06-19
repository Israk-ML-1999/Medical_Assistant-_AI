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

# ---------- Enhanced Dummy Content ----------
def get_dummy_content(document_type: str) -> str:
    """
    Return dummy content (placeholder patient data) relevant to the document type category.
    This content is used only for preview – real sessions use actual patient data.
    """
    doc_lower = document_type.lower()

    # 1. Surgical / Operative / Procedure
    if any(k in doc_lower for k in ["operative", "pre-operative", "post-operative", "procedure", "anesthesia", "surgery", "operation"]):
        return """**Patient & Procedure Data:**
Patient: 72M, scheduled for elective right hemicolectomy.
Procedure: Laparoscopic right hemicolectomy for colon cancer.
Findings: Tumor in ascending colon, no metastasis.
Complications: None.
Surgeon: Dr. Smith.
Vitals: BP 130/80, HR 72, Temp 36.8°C, SpO2 98%.
"""

    # 2. Radiology / Imaging / Endoscopy / ECG / EEG
    if any(k in doc_lower for k in ["radiology", "x-ray", "ct", "mri", "ultrasound", "imaging", "endoscopy", "ecg", "eeg"]):
        return """**Imaging Study:**
Patient: 55F, CT chest with contrast.
Indication: Evaluate lung nodules.
Findings: 3mm nodule in right upper lobe, stable; no mediastinal lymphadenopathy.
Impression: Benign likely; follow-up in 6 months.
"""

    # 3. Laboratory / Pathology / Biopsy
    if any(k in doc_lower for k in ["laboratory", "lab", "pathology", "biopsy", "hematology", "chemistry", "microbiology"]):
        return """**Lab Results:**
Patient: 45M, CBC and metabolic panel.
Hb: 14.2 g/dL, WBC: 6.5 x10^3, Platelets: 220 x10^3.
Sodium: 140, Potassium: 4.2, Creatinine: 1.0.
Glucose: 98 mg/dL.
Impression: Normal labs.
"""

    # 4. Discharge / Admission / Transfer
    if any(k in doc_lower for k in ["discharge", "admission", "transfer"]):
        return """**Patient Summary:**
Patient: 68F, admitted with community-acquired pneumonia.
Hospital course: Treated with IV antibiotics, improved.
Discharge medications: Amoxicillin 500mg TID x 7 days.
Follow-up plan: Primary care in 1 week.
"""

    # 5. Psychiatry / Mental Health / Psychological
    if any(k in doc_lower for k in ["psychiatry", "psychiatric", "mental health", "depression", "anxiety", "psychological"]):
        return """**Psychiatric Assessment:**
Patient: 54M, presents with alcohol dependence and anxiety.
History: Heavy alcohol use, previous detox admissions.
Mental State Exam: Agitated, confabulatory, poor insight.
Plan: Detox protocol, thiamine, psychotherapy.
**ICD‑10 codes applicable:**
- F10.27 – Alcohol dependence with alcohol-induced persisting amnestic disorder
- F05 – Delirium due to known physiological condition
- L97.919 – Non-pressure chronic ulcer of unspecified lower leg with unspecified severity
"""

    # 6. Medication / Prescription / Chemotherapy / Treatment Plan
    if any(k in doc_lower for k in ["medication", "prescription", "reconciliation", "review", "treatment plan", "chemotherapy", "pharmacotherapy", "drug"]):
        return """**Medication Review:**
Patient: 62F, history of hypertension and type 2 diabetes.
Current medications:
- Lisinopril 10mg daily
- Metformin 500mg BID
- Atorvastatin 20mg daily
Recent changes: Added amlodipine 5mg daily for BP control.
Allergies: Penicillin (rash).
Recommendation: Monitor blood pressure and adjust as needed.
"""

    # 7. Nursing / Allied Health / Physiotherapy / Dietitian / Social Work / Wound Care
    if any(k in doc_lower for k in ["nursing", "physiotherapy", "dietitian", "nutrition", "social work", "wound care", "allied health"]):
        return """**Nursing / Allied Health Assessment:**
Patient: 78M, post‑hip replacement surgery.
Mobility: Requires assistance with walking; using a walker.
Wound: Surgical incision clean, dry, intact, no signs of infection.
Nutrition: Appetite improving; tolerating solid foods.
Plan: Continue physical therapy, monitor wound, encourage ambulation.
"""

    # 8. Legal / Administrative / Consent / Certificate / Insurance
    if any(k in doc_lower for k in ["legal", "administrative", "consent", "certificate", "insurance", "medical expert opinion"]):
        return """**Legal / Administrative Document:**
Patient: 45M, requesting sick leave certificate.
Diagnosis: Acute low back pain.
Duration: 2 weeks.
Work restrictions: Sedentary duties only.
Recommendation: Provide certificate for 2 weeks, reassess after.
"""

    # 9. Progress / Follow‑up / Outpatient / Clinic Letter / Handover
    if any(k in doc_lower for k in ["progress note", "follow-up", "outpatient", "clinic letter", "on-call", "handover", "triage"]):
        return """**Progress Note:**
Patient: 56F, admitted with COPD exacerbation.
Today: Symptoms improving; oxygen saturation 94% on room air.
Plan: Continue steroids and antibiotics; consider discharge tomorrow.
"""

    # 10. Summary / Report / Tumor Board / Case Report / Incident Report
    if any(k in doc_lower for k in ["patient summary", "tumor board", "mdt", "case report", "incident report", "autopsy report"]):
        return """**Clinical Summary:**
Patient: 60M, diagnosed with stage III colorectal cancer.
Multidisciplinary team reviewed: Surgery planned, followed by adjuvant chemotherapy.
Risk factors: Family history, smoking.
Plan: PET‑CT for staging; discuss with oncologist.
"""

    # 11. Consultation / Referral Letter
    if any(k in doc_lower for k in ["consultation", "referral"]):
        return """**Referral / Consultation:**
Referring clinician: Dr. Jones, Cardiology.
Referred to: Dr. Smith, Cardiothoracic Surgery.
Reason: Patient with severe aortic stenosis, symptomatic.
History: Dyspnoea, angina, syncope.
Echo: Severe aortic stenosis (valve area 0.8 cm²).
Request: Surgical evaluation for valve replacement.
"""

    # 12. SOAP / General Clinical Notes (fallback for all others)
    return """**Patient Interview:**
Patient: "I have chest pain and shortness of breath."
Doctor: "When did it start?"
Patient: "Two hours ago."
Vitals: BP 140/90, HR 88, Temp 37.2°C, SpO2 95%.
Exam: Lungs clear, heart sounds normal.
Assessment: Possible angina, rule out MI.
Plan: ECG, troponin, aspirin, cardiology consult.
"""

# ---------- Business Logic ----------
async def preview_assistant(assistant: AssistantConfig):
    if not llm_service.client:
        raise ValueError("Vertex AI client not configured")
    dummy_content = get_dummy_content(assistant.document_type)
    prompt = PromptBuilder.build_generate_note_prompt(dummy_content, assistant.dict(), "Preview")
    sections = assistant.sections
    properties = {s: {"type": "string"} for s in sections}
    properties["red_flag"] = {"type": "string", "description": "Optional critical safety alert"}
    schema = {"type": "object", "properties": properties, "required": sections}
    json_output, in_tok, out_tok = await llm_service.generate_structured(prompt, schema, model="flash")
    
    result = json.loads(json_output)
    result["input_token"] = in_tok
    result["output_token"] = out_tok
    return result