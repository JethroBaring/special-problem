import time

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands()
pose = mp_pose.Pose()

# Helper function: Analyze emotional trends
def analyze_emotions(emotion_counts):
    """
    Analyze trends in emotions based on the count of emotions observed.
    """
    if not emotion_counts:
        return "Neutral"
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    return dominant_emotion

# Initialize variables for emotion trends
emotion_counts = {}
start_time = time.time()

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Analyze face for emotions
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        current_emotion = analysis[0]['dominant_emotion']
        emotion_counts[current_emotion] = emotion_counts.get(current_emotion, 0) + 1
    except Exception as e:
        current_emotion = "Unknown"

    # Analyze hand gestures
    hand_label = "Unknown"
    results_hands = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results_hands.multi_hand_landmarks:
        hand_label = "Expressive"

    # Analyze posture for confidence
    results_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    posture_label = "Unknown"
    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        posture_label = "Upright" if landmarks[mp_pose.PoseLandmark.NOSE.value].y < landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y else "Slouching"

    # Display emotional trends
    elapsed_time = time.time() - start_time
    if elapsed_time > 10:  # Analyze trends every 10 seconds
        dominant_emotion = analyze_emotions(emotion_counts)
        emotion_counts = {}  # Reset for the next interval
        start_time = time.time()
    else:
        dominant_emotion = "Analyzing..."

    # Display results
    cv2.putText(frame, f"Current Emotion: {current_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Dominant Emotion: {dominant_emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Posture: {posture_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Hand Gesture: {hand_label}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Comprehensive Behavioral Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
