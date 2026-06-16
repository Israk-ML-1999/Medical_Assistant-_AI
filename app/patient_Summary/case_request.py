"""
app/patient_Summary/case_request.py

Pydantic schemas for the Patient Summary endpoint.
The summary sections are DYNAMIC - AI decides the titles based on document analysis,
so we model it as a flexible array of {title, items: [{name, description}]}.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PatientSummaryItem(BaseModel):
    """A single item within a patient summary section (e.g., one ICD code, one medication)."""
    name: str = Field(..., description="Name / code / short label of the item")
    description: str = Field(..., description="Detailed description / explanation of the item")


class PatientSummarySection(BaseModel):
    """A dynamic section of the patient summary (e.g., 'ICD Coding', 'Medications')."""
    title: str = Field(..., description="Section title, dynamically identified by AI")
    items: List[PatientSummaryItem] = Field(default_factory=list, description="List of items under this section")


class PatientSummaryResponse(BaseModel):
    """Top-level response containing the array of dynamic summary sections."""
    patient_summary: List[PatientSummarySection] = Field(
        ..., description="Array of dynamically identified patient summary sections"
    )
    input_token: Optional[str] = Field(default=None, description="Number of tokens in the prompt")
    output_token: Optional[str] = Field(default=None, description="Number of tokens in the generated response")
