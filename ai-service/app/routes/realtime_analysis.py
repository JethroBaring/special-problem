from fastapi import APIRouter, WebSocket
from app.ws.ws_realtime import process_realtime_audio

router = APIRouter()

@router.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await process_realtime_audio(websocket)
