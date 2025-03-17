# posture.py
import cv2
import numpy as np
import mediapipe as mp
import base64

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

async def analyze_posture(frame_bytes):
    try:
        # Check if we received base64 string or raw bytes
        if isinstance(frame_bytes, str):
            # Decode base64 string
            frame_data = base64.b64decode(frame_bytes)
        else:
            # Already bytes, use directly
            frame_data = frame_bytes
            
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print(f"📏 Frame size: {len(frame_bytes)} bytes")
            print("⚠️ Invalid frame data posture.")
            return {"posture_feedback": "No valid frame"}

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run posture analysis
        results = pose.process(frame_rgb)
        if not results.pose_landmarks:
            return {"posture_feedback": "No person detected"}

        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        head_tilt = abs(left_shoulder.y - right_shoulder.y)
        feedback = "Good posture" if head_tilt < 0.05 else "Head tilted"

        return {"posture_feedback": feedback, "confidence": 1.0 - head_tilt}

    except Exception as e:
        print(f"⚠️ Posture analysis error: {e}")
        return {"posture_feedback": "Error", "confidence": 0}