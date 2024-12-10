import cv2
import mediapipe as mp
import openai
from deepface import DeepFace

# OpenAI API setup
openai.api_key = "your_openai_api_key"

# Mediapipe setup for gesture tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Vision, mission, and core values
company_vision = "To innovate and inspire through technology."
company_mission = "Deliver high-quality, impactful solutions."
company_values = "Integrity, collaboration, and customer focus."

# Questions and metadata
questions = [
    {"text": "How do you handle conflict in a team?", "category": "Core Values"},
    {"text": "Where do you see yourself in 5 years?", "category": "Vision"},
    {"text": "What motivates you to excel at work?", "category": "Mission"}
]

# Video capture
cap = cv2.VideoCapture(0)

for idx, question in enumerate(questions):
    print(f"Question {idx + 1}: {question['text']}")
    print("Answer verbally and use gestures...")

    candidate_response = input("Summarize your response for the record: ")  # Simulate a transcript
    dominant_emotion = "Neutral"
    gesture_score = "Unknown"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze facial emotion
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
        except Exception:
            pass

        # Analyze gestures
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            gesture_score = "Expressive"
        else:
            gesture_score = "Calm"

        # Display video with current emotion and gesture data
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture_score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Interview Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        break  # Move to next question after some processing

    # GPT Analysis
    input_prompt = f"""
    The company has the following principles:
    Vision: {company_vision}
    Mission: {company_mission}
    Core Values: {company_values}

    Question: {question['text']}
    Candidate's Answer: {candidate_response}
    Observations:
    - Emotion: {dominant_emotion}
    - Gesture: {gesture_score}

    Assess how well the candidate aligns with the company's culture and provide a qualitative summary.
    """
    gpt_response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=input_prompt,
        max_tokens=200
    )

    print("\nGPT Analysis:")
    print(gpt_response['choices'][0]['text'].strip())

    # Add a small pause between questions
    input("Press Enter to proceed to the next question...")

cap.release()
cv2.destroyAllWindows()
