from fastapi import APIRouter, HTTPException
from app.generate_from_description.request import GenerateFromDescriptionRequest
from app.generate_from_description.service import generate_assistant_config

router = APIRouter(tags=["generate-from-description"])

@router.post("/assistants/generate-from-description")
async def generate_assistant(req: GenerateFromDescriptionRequest):
    try:
        return await generate_assistant_config(req.description)
    except Exception as e:
        raise HTTPException(400, f"AI generation failed: {str(e)}")