from fastapi import APIRouter, File, UploadFile
from app.services import process_video

router = APIRouter()

@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    result = await process_video(file)
    return result
