"""
app/clinical_report_structure/router.py

FastAPI router for generating clinical report based on example structure.
"""

from fastapi import APIRouter, HTTPException
from app.clinical_report_structure.Report_request import ClinicalReportRequest, ClinicalReportResponse
from app.clinical_report_structure.report_llm_service import generate_clinical_report_service

router = APIRouter(
    prefix="/clinical-report-structure",
    tags=["Clinical Report Structure"],
)

@router.post(
    "/generate",
    response_model=ClinicalReportResponse,
    summary="Generate patient report following a specific example structure",
)
async def generate_clinical_report_endpoint(request: ClinicalReportRequest):
    """
    Analyzes live transcript, chat history, and document texts, 
    and generates a report strictly following the provided example_structure format.
    """
    try:
        result = await generate_clinical_report_service(
            live_transcript=request.live_transcript,
            chat_history=request.chat_history,
            document_texts=request.document_texts,
            example_structure=request.example_structure,
        )
        return ClinicalReportResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clinical report generation failed: {str(e)}")
