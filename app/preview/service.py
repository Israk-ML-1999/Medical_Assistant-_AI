import json
from typing import Dict, Any, List, Optional, Literal
from google.genai import types
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
        """
        Builds the prompt for generating a clinical note.
        - If transcript is empty (preview mode), the AI generates a detailed sample note
          that follows the standard format for the document type.
        - If a real transcript is provided (session mode), the AI uses that content.
        """
        # Build system prompt from assistant configuration
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
                "If a 'Coding' or 'ICD-10' section is present, extract or suggest relevant ICD-10 codes and list them clearly in that section. Use the format: 'ICD-10 Code - Description'."
            )
        system_parts.append(f"Use the following sections: {', '.join(sections)}.")
        system_parts.append("Highlight critical safety issues with a 'RED FLAG:' line.")
        system = " ".join(system_parts)

        # Get template guidance for the document type
        template_guidance = get_template_guidance(assistant.get('document_type', ''))

        # User prompt: sample generation or real transcript
        if not transcript or transcript.strip() == "":
            user_prompt = f"""
You are now creating a **realistic sample clinical note** for demonstration purposes.
Based on the assistant configuration above, generate a detailed {assistant.get('document_type', 'clinical note')} for a hypothetical patient.

{template_guidance}

The note should be clinically appropriate, informative, and follow the sections exactly.
Do not use placeholders like '[Patient name]' – invent plausible details.
Include specific history, examination findings, diagnosis, management plan, and relevant ICD‑10 codes in the Coding section if present.
Follow the standard format for this document type as described above.

Write a {note_type} note using the sections above.
Return ONLY valid JSON with keys matching the sections.
"""
        else:
            user_prompt = f"""
Write a {note_type} note using the sections above.
Transcript:
{transcript}
"""

        return f"""{system}

{user_prompt}

Output format: JSON object with keys exactly matching the section names.
You may add an extra key "red_flag" if needed.
Return ONLY valid JSON.
"""

# ---------- Template Guidance ----------
def get_template_guidance(document_type: str) -> str:
    """Return specific formatting and structural guidance for the document type."""
    doc_lower = document_type.lower()

    # Admission & Discharge
    if any(k in doc_lower for k in ["discharge summary", "discharge", "zárójelentés"]):
        return """
The standard Discharge Summary format includes:
- Patient demographics (name, DOB, MRN, mother's name, admission/discharge dates)
- Diagnoses with ICD-10 codes (Principal and Secondary)
- Clinical course & interventions (reason for admission, procedures, summary of stay)
- Discharge status & plan (condition at discharge, medication table, follow-up, statutory labor status)
Use bullet points for medications (Brand | Dose | Frequency | Duration). Include a table for medications if possible.
"""
    elif "admission note" in doc_lower:
        return """
The standard Admission Note includes: Chief Complaint, History of Present Illness (HPI), Past Medical History, Medications & Allergies, Physical Exam (vitals, systems), Initial Impression, Differential Diagnosis, and Initial Plan.
Use clear headings and bullet points for differentials.
"""
    elif "transfer summary" in doc_lower:
        return """
The Transfer Summary includes: Sending/Receiving Unit, Date of Transfer, Patient Status Overview (diagnoses, hospital course), Current Clinical Status (vitals, infusions, outstanding results), and Reason for Transfer.
"""
    elif "referral letter" in doc_lower:
        return """
The Referral Letter includes: To/From, Date, RE line, Purpose of Referral, Clinical Summary (presentation, investigations, current medications), and Request.
Use a formal, professional tone.
"""
    elif "death summary" in doc_lower:
        return """
The Death Summary includes: Patient demographics, Final Diagnoses (immediate and underlying cause), Summary of Events (clinical course, terminal event, pronouncement), and Notification/Post-Mortem details.
Use precise, chronological language.
"""

    # Clinical Notes
    elif "progress note" in doc_lower:
        return """
The Progress Note includes: Date/Time, 24-Hour Events & Overview, Objective Data (vitals, physical exam, I/O, diagnostic updates), Assessment & Plan (evolution of active problems, care transitions).
Use bullet points for key updates.
"""
    elif "soap note" in doc_lower:
        return """
The SOAP Note includes: Subjective (patient's symptoms), Objective (vitals, exam, data), Assessment (clinical synthesis), Plan (diagnostics, therapeutics, patient education).
Use clear subheadings and bullet points.
"""
    elif "history & physical" in doc_lower or "h&p" in doc_lower:
        return """
The H&P includes: Chief Complaint, HPI (OLDCARTS), PMH, Family History, Social History, Review of Systems, Physical Exam (head-to-toe), Assessment/Plan (ranked problem list).
Use comprehensive, detailed descriptions.
"""
    elif "consultation note" in doc_lower:
        return """
The Consultation Note includes: Requesting Physician, Consulting Specialty, Reason for Consult, Findings & Review, Recommendations.
Use numbered recommendations.
"""
    elif "follow-up" in doc_lower or "outpatient note" in doc_lower:
        return """
The Outpatient Follow-up Note includes: Interval Since Last Visit, Current Status, Objective Assessment (vitals, targeted exam), Assessment, Plan (maintain/adjust regimen, next visit).
"""
    elif "clinic letter" in doc_lower or "outpatient letter" in doc_lower:
        return """
The Clinic Letter includes: Date, To/From, RE, Dear Dr., Summary of Consultation, Treatment Plan, and Sincerely.
Use a formal letter format.
"""
    elif "nursing note" in doc_lower:
        return """
The Nursing Note includes: Shift/Date/Time, Patient Care Overview (neurological, CV/resp, GI/GU, wound, safety, interventions delivered).
Use clear, concise, objective language.
"""
    elif "on-call" in doc_lower or "handover note" in doc_lower:
        return """
The On-Call/Handover Note includes: Date/Shift, Patient, Acute Event, Assessment Findings (vitals, status), Interventions Performed, Expected Action.
Use clear, concise, and actionable language.
"""
    elif "sbar" in doc_lower:
        return """
The SBAR Handover Note includes: Situation (current status), Background (history, course), Assessment (clinical interpretation), Recommendation (specific actions).
Use clear, structured headings.
"""
    elif "triage note" in doc_lower:
        return """
The Triage Note includes: Arrival Time, Presentation Mode, Chief Complaint, Triage Acuity Score, Initial Vitals, Critical Presentation Summary, Disposition.
Use objective, urgent language.
"""

    # Surgical Documents
    elif "pre-operative" in doc_lower:
        return """
The Pre-Operative Assessment includes: Proposed Surgery, ASA Classification, Surgical Risk Screening (airway, CV/pulmonary, coagulation), Pre-Operative Diagnostics, Final Disposition (cleared/conditional).
Use structured checklist style.
"""
    elif "operative report" in doc_lower:
        return """
The Operative Report includes: Date/Time, Primary Surgeon/Assistants/Anesthesiologist, Operative Titles (pre/post-op diagnoses, procedure performed), Specifics (counts, EBL, specimens, drains), Operative Description in detail (chronological walkthrough).
Use formal, precise surgical language.
"""
    elif "post-operative" in doc_lower:
        return """
The Post-Operative Note includes: Procedure Completed, Surgeon, PACU Progress & Status (vitals, neuro, surgical site, drains, fluid status), Post-Op Plans & Orders (pain management, DVT prophylaxis, diet).
"""
    elif "anesthesia record" in doc_lower:
        return """
The Anesthesia Record includes: Type, Pre-Anesthetic Check, Induction Timeline & Agents, Maintenance & Monitoring Parameters, Total Fluids, Reversal Agents.
Use precise drug names and dosages.
"""
    elif "procedure note" in doc_lower:
        return """
The Procedure Note includes: Date/Time, Performing Clinician, Procedure Performed, Indication, Procedure Walkthrough (consent, prep, execution, findings, complications).
Use clear, stepwise description.
"""

    # Summaries & Reports
    elif "patient summary" in doc_lower:
        return """
The Patient Summary includes: Demographics, Active Problem List, Core Standing Medication Profile, Critical Allergies & Alerts, Recent Key Values.
Use a dashboard-style format.
"""
    elif "tumor board" in doc_lower or "mdt" in doc_lower:
        return """
The Tumor Board/MDT Summary includes: Date, Panel, Patient Presentation (staging, performance status, clinical summary), Diagnostic Review, Consensus Recommendations.
Use multidisciplinary, collaborative language.
"""
    elif "case report" in doc_lower:
        return """
The Case Report includes: Title, Abstract, Case Presentation (patient profiles, diagnostic workup, managed pathway), Discussion, Conclusion.
Use academic, narrative style.
"""
    elif "incident report" in doc_lower:
        return """
The Incident Report includes: Date/Time/Location, Person Reporting, Type of Incident, Persons Involved, Factual Chronological Description, Consequences & Immediate Actions, Notifications.
Use objective, non‑speculative language.
"""
    elif "autopsy report" in doc_lower:
        return """
The Autopsy Report includes: Case Ref, Pathologist, External Examination, Internal System Examination (CV, respiratory, CNS, GI), Histopathology & Toxicology, Final Diagnoses.
Use formal, systematic, and objective language.
"""

    # Diagnostics & Results
    elif "radiology report" in doc_lower:
        return """
The Radiology Report includes: Exam Type, Date, Ordering Provider, Reporting Radiologist, Clinical Indications, Comparison Study, Technical Protocol, Findings (system-based), Impression.
Use clear, descriptive medical imaging language.
"""
    elif "pathology report" in doc_lower:
        return """
The Pathology Report includes: Specimen Source, Clinical Diagnosis, Accession Number, Macroscopic Description, Microscopic Description, Diagnostic Impression.
Use precise histological terminology.
"""
    elif "laboratory result" in doc_lower:
        return """
The Laboratory Result Interpretation includes: Requested Panel, Collection Date, Critical/Out‑of‑Range Parameters, Clinical Synthesis, Recommended Actions.
Use structured tables for lab values and clear clinical reasoning.
"""
    elif "ecg" in doc_lower or "eeg" in doc_lower:
        return """
The ECG/EEG Report includes: Type, Patient Status, Recording Specifics (metrics, waveforms), Findings/Phenomena, Clinical Impression.
Use precise electrophysiological terminology.
"""
    elif "endoscopy report" in doc_lower:
        return """
The Endoscopy Report includes: Procedure Type, Sedation, Instrument, Anatomical Regions Evaluated & Findings, Interventions Done, Final Diagnostic Impression.
Use stepwise, systematic descriptions.
"""
    elif "biopsy report" in doc_lower:
        return """
The Biopsy Report includes: Anatomical Target Site, Guidance Technique, Number of Cores, Microscopic Architecture Synopsis, Immunohistochemistry Profile, Final Impression.
Use clear pathological details.
"""

    # Medications & Orders
    elif "medication reconciliation" in doc_lower:
        return """
The Medication Reconciliation includes: Verification Sources, Table of Drug/Strength/Pre‑Admission Dose/Hospital Status/Disposition, Reconciliation Date, Reviewer.
Use a table format for medications.
"""
    elif "prescription" in doc_lower:
        return """
The Prescription includes: Clinic/Hospital Source, Patient Name/DOB/Date, Rx details (Drug, Strength, Quantity, Sig, Refills), Prescriber Signature.
Use standard prescription format.
"""
    elif "medication review" in doc_lower:
        return """
The Medication Review includes: Review Type, Pharmaceutical Interactions Evaluated, Adherence Assessment, Recommended Optimization Steps.
Use clear, actionable language.
"""
    elif "treatment plan" in doc_lower:
        return """
The Treatment Plan includes: Patient Diagnosis Focus, Primary Care Coordinator, Short‑Term & Long‑Term Goals, Intervention Timeline.
Use structured, goal‑oriented language.
"""
    elif "chemotherapy protocol" in doc_lower:
        return """
The Chemotherapy Protocol includes: Regimen Name, Indication, Cycle Parameters, Patient Metrics, Pre‑medications, Active Antineoplastic Agent Dosing Matrix.
Use precise oncological dosing language and tables.
"""

    # Legal & Administrative
    elif "informed consent" in doc_lower:
        return """
The Informed Consent includes: Patient Name/DOB/MRN, Proposed Procedure, Physician Acknowledgement, Patient Declaration, Signatures.
Use formal, legal language.
"""
    elif "sick leave certificate" in doc_lower:
        return """
The Sick Leave Certificate includes: Patient Registration Details, Certification Matrix (temporarily unfit), Validity Window, Re‑evaluation Date, Issuing Clinic/Practice Stamp, Physician Signature.
Use official, administrative language.
"""
    elif "disability certificate" in doc_lower:
        return """
The Disability Certificate includes: Patient Profile, Clinical Impairment Identification, Functional Anatomical Capacity Evaluation, Expert Clinical Assessment Determination, Signature.
Use formal, comprehensive evaluation language.
"""
    elif "insurance" in doc_lower or "medical report" in doc_lower:
        return """
The Insurance Medical Report includes: Insurance Claim Reference, Patient Details, Chronological Clinical Presentation (dates, diagnostics, procedures, prognosis), Certification.
Use formal, evidence‑based language.
"""
    elif "medical expert opinion" in doc_lower:
        return """
The Medical Expert Opinion includes: Commissioning Authority, Evaluating Expert, Case Scope, Documentation Reviewed, Expert Analysis, Conclusions/Rulings.
Use formal, authoritative language.
"""

    # Nursing & Allied Health
    elif "nursing assessment" in doc_lower:
        return """
The Nursing Assessment includes: Admitting Ward/Date/Time, Core Physiological Risks (fall, pressure injury, cognitive), System Component Mapping (nutrition, elimination, pain), Discharge Barriers.
Use comprehensive, holistic language.
"""
    elif "physiotherapy" in doc_lower:
        return """
The Physiotherapy Note includes: Session Number, Patient Mobility Baseline, Therapeutic Objective, Functional Status Metrics (ROM, strength, gait), Therapy Delivered, Patient Tolerance, Plan.
Use clear, rehabilitation‑focused language.
"""
    elif "dietitian" in doc_lower or "nutrition" in doc_lower:
        return """
The Dietitian/Nutrition Note includes: Reason for Consult, Anthropometric Data, Clinical Nutritional Assessment, Dietary Management, Oral Nutritional Supplements, Enteral/Parenteral specifications.
Use clear nutritional terminology.
"""
    elif "social work" in doc_lower:
        return """
The Social Work Note includes: Reason for Referral, Psychosocial Findings Summary, Coordination Action & Case Placement Steps.
Use compassionate, solution‑oriented language.
"""
    elif "wound care" in doc_lower:
        return """
The Wound Care Note includes: Anatomical Wound Site, Wound Classification, Objective Characteristics (dimensions, undermining, tissue composition, exudate, peri‑wound), Therapeutic Treatment Procedures Applied.
Use precise wound management terminology.
"""

    # Mental Health
    elif "psychiatric evaluation" in doc_lower:
        return """
The Psychiatric Evaluation includes: Reason for Intake, History of Psychiatric Illness, Substance Use Profile, Mental Status Examination (appearance, speech, mood, thought, perception, risk, insight), Diagnostic Impression (DSM‑5/ICD‑10), Management Scheduling Plan.
Use comprehensive, empathetic, and professional language.
"""
    elif "psychological report" in doc_lower:
        return """
The Psychological Report includes: Referral Question, Assessment Tools, Test Results Synthesis (cognitive, personality), Behavioral Observations, Formulation, Therapeutic Recommendations.
Use clear psychological terminology and structured test results.
"""
    elif "mental health progress note" in doc_lower:
        return """
The Mental Health Progress Note includes: Session Type, Patient Treatment Focus, Therapeutic Work Summary (subjective, MSE, interventions), Client Homework/Plan.
Use collaborative, therapeutic language.
"""

    # Fallback: General SOAP format
    else:
        return """
Generate a comprehensive clinical note following the sections provided. Include detailed history, examination, assessment, and management plan. Use bullet points for clarity if appropriate. Ensure the note is realistic and clinically sound.
"""

# ---------- LLM Service ----------
class LLMService:
    def __init__(self):
        self.client = get_client()

    async def generate_structured(self, prompt: str, schema: Dict[str, Any], model: str = "flash") -> tuple:
        """
        Generate structured JSON output from Gemini.
        Returns (text, input_token, output_token).
        """
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
        # Extract token counts
        in_tok = str(response.usage_metadata.prompt_token_count) if hasattr(response, "usage_metadata") and response.usage_metadata else None
        out_tok = str(response.usage_metadata.candidates_token_count) if hasattr(response, "usage_metadata") and response.usage_metadata else None
        return response.text, in_tok, out_tok

llm_service = LLMService()

# ---------- Business Logic ----------
async def preview_assistant(assistant: AssistantConfig):
    """
    Generate a preview clinical note and return it with token counts.
    """
    if not llm_service.client:
        raise ValueError("Vertex AI client not configured")

    # Pass empty transcript to trigger sample note generation
    prompt = PromptBuilder.build_generate_note_prompt("", assistant.dict(), "Preview")
    sections = assistant.sections
    properties = {s: {"type": "string"} for s in sections}
    properties["red_flag"] = {"type": "string", "description": "Optional critical safety alert"}
    schema = {"type": "object", "properties": properties, "required": sections}
    
    json_output, in_tok, out_tok = await llm_service.generate_structured(prompt, schema, model="flash")
    note = json.loads(json_output)
    
    # Attach token counts to the response (you can return them separately if needed)
    return {
        "note": note,
        "input_token": in_tok,
        "output_token": out_tok
    }