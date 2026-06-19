from fastapi import APIRouter, HTTPException
from .request import AssistantConfig
from .service import preview_assistant

router = APIRouter(tags=["preview"])

@router.post("/preview", status_code=200)
async def preview(assistant: AssistantConfig):
    try:
        return await preview_assistant(assistant)
    except Exception as e:
        raise HTTPException(500, f"Preview failed: {str(e)}")