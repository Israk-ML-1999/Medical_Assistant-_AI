from pydantic import BaseModel, Field

class GenerateFromDescriptionRequest(BaseModel):
    description: str = Field(
        ...,
        example="Create a psychiatry discharge summary assistant with sections: Reason for admission, Hospital course, Discharge medications, Follow-up plan, Coding. Include ICD‑10 codes in the Coding section. Use bullet points. Care setting: Inpatient Ward. Clinical role: Attending Physician."
    )