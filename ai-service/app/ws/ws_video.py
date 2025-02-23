import asyncio
import base64
import cv2
import numpy as np
from deepface import DeepFace
from fastapi import WebSocket
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def convert_numpy_values(obj):
    """Convert numpy values to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_values(item) for item in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    return obj

async def process_realtime_video(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected for real-time video analysis.")

    try:
        while True:
            # Receive base64 image from frontend
            data = await websocket.receive_text()

            # Decode base64 image
            image_data = data.split(",")[1] if "," in data else data
            frame_bytes = base64.b64decode(image_data)

            # Convert bytes to OpenCV image
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("⚠️ Invalid frame received. Skipping...")
                continue

            # Convert BGR to RGB for DeepFace
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                # Analyze emotions with DeepFace
                analysis = await asyncio.to_thread(
                    lambda: DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
                )
                
                print(f"✅ DeepFace Output: {analysis}")  # Debugging line
                
                # ✅ Ensure `analysis` is always treated as a list
                if isinstance(analysis, list) and len(analysis) > 0:
                    analysis = analysis[0]  # Get the first result

                # Convert numpy values to Python native types
                analysis = convert_numpy_values(analysis)

                # Extract emotion safely
                dominant_emotion = analysis.get("dominant_emotion", "N/A")
                emotions = analysis.get("emotion", {})

                # Send results back
                await websocket.send_json({
                    "dominant_emotion": dominant_emotion,
                    "emotion_scores": emotions
                })

            except Exception as e:
                print(f"❌ DeepFace error: {e}")

    except Exception as e:
        print(f"❌ WebSocket error: {e}")

    finally:
        print("⚠️ WebSocket closed.")