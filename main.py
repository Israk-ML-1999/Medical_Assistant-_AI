"""
main.py

Entry point for the Doctor Template Generation FastAPI application.
Run with: uvicorn main:app --reload
Swagger UI available at: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI

from config import settings

from app.SOAP_Note_generation.router import router as soap_router
from app.patient_Summary.router import router as patient_summary_router
from app.Test_result.router import router as test_result_router
from app.context_caching.router import router as context_caching_router
from app.MedAi_chatbot.router import router as medai_chatbot_router
from app.OCR_file_to_text.router import router as ocr_router
from app.preview.router import router as preview_router
from app.generate_from_description.router import router as generate_from_description_router
from app.clinical_report_structure.router import router as clinical_report_structure_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI-powered medical documentation system for doctors: SOAP note generation, "
        "patient summary, test result analysis, OCR (Document AI), context caching, "
        "and a MedAI chatbot - all powered by Gemini via Vertex AI (Agent Platform)."
    ),
    version="1.0.0",
    debug=settings.DEBUG,
)

# Fixed middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# Register all routers
app.include_router(soap_router)
app.include_router(patient_summary_router)
app.include_router(test_result_router)
app.include_router(context_caching_router)
app.include_router(medai_chatbot_router)
app.include_router(ocr_router)
app.include_router(preview_router)
app.include_router(generate_from_description_router)
app.include_router(clinical_report_structure_router)


@app.get("/", tags=["Health"])
async def root():
    """Health check / welcome endpoint."""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "docs": "/docs",
    }
