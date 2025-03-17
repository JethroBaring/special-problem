from deepface import DeepFace
import cv2
import numpy as np
import base64
import traceback
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

async def analyze_facial_emotion(frame_bytes):
    try:
        print(f"📥 Starting facial analysis with data type: {type(frame_bytes)}")

        # Decode base64 string if needed
        if isinstance(frame_bytes, str):
            if "base64," in frame_bytes:
                frame_bytes = frame_bytes.split("base64,")[1]

            frame_data = base64.b64decode(frame_bytes)
            print(f"📏 Decoded base64 string to {len(frame_data)} bytes")
        else:
            frame_data = frame_bytes

        # Convert to NumPy array
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print(f"❌ Failed to decode frame. Input length: {len(frame_data)} bytes")
            return {"dominant_emotion": "Decode Error", "emotion_score": 0}

        print(f"📸 Decoded image of shape: {frame.shape}")

        # Convert BGR to RGB (DeepFace expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"🎨 Converted to RGB image of shape: {frame_rgb.shape}")

        # Directly pass the NumPy array to DeepFace
        print("🚀 Running DeepFace analysis...")
        analysis = DeepFace.analyze(
            img_path=frame_rgb,  # Pass NumPy array directly
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )

        if isinstance(analysis, list) and analysis:
            result = analysis[0]
            dominant_emotion = result.get("dominant_emotion", "N/A")
            emotion_scores = result.get("emotion", {})
            highest_score = emotion_scores.get(dominant_emotion, 0) if dominant_emotion != "N/A" else 0

            print(f"✅ Analysis complete. Dominant emotion: {dominant_emotion}, Score: {highest_score}")

            return {
                "dominant_emotion": dominant_emotion,
                "emotion_scores": emotion_scores,
                "emotion_score": highest_score
            }
        else:
            print(f"⚠️ Unexpected analysis result format: {analysis}")
            return {"dominant_emotion": "Format Error", "emotion_score": 0}

    except Exception as e:
        print(f"❌ DeepFace analysis error: {e}")
        traceback.print_exc()
        return {"dominant_emotion": "Analysis Error", "emotion_score": 0}
