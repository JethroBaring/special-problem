from fastapi import APIRouter, WebSocket
from app.ws.ws_video import process_realtime_video
from app.ws.ws_audio import process_realtime_audio

router = APIRouter()

@router.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await process_realtime_audio(websocket)

@router.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await process_realtime_video(websocket)
