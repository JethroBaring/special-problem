import cv2
import mediapipe as mp
import pyttsx3
import threading
import tkinter as tk
from PIL import Image, ImageTk
from deepface import DeepFace
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
import speech_recognition as sr
import time
import whisper


# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize text-to-speech
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Initialize sentiment analyzer
nltk.download('vader_lexicon')
nltk.download('wordnet')
sia = SentimentIntensityAnalyzer()

# Expand keywords for company values
def expand_keywords(words):
    expanded = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    return list(expanded)

# Company values keywords
company_values = {
    "innovation": expand_keywords(["creative", "new ideas", "invent", "improve"]),
    "teamwork": expand_keywords(["collaborate", "together", "support", "help"]),
    "integrity": expand_keywords(["honest", "trustworthy", "ethical"]),
    "growth": expand_keywords(["learn", "improve", "challenge", "adapt"]),
}

# Speak function with threading
def speak(text, callback=None):
    def run():
        engine.say(text)
        engine.runAndWait()
        if callback:
            root.after(100, callback)
    threading.Thread(target=run, daemon=True).start()

# Process video frame
def process_frame():
    global cap, video_label, photo
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image=img)
        video_label.config(image=photo)
        video_label.image = photo
    root.after(10, process_frame)

# Emotion analysis (run every 2 seconds)
def analyze_emotion():
    global cap
    ret, frame = cap.read()
    if ret:
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
        except Exception:
            dominant_emotion = "Unknown"
        emotion_label.config(text=f"Emotion: {dominant_emotion}")
    root.after(2000, analyze_emotion)

# Evaluate response
def evaluate_response(response):
    sentiment_score = sia.polarity_scores(response)['compound']
    keyword_match = {key: sum(response.lower().count(word) for word in words) for key, words in company_values.items()}
    total_score = sum(keyword_match.values()) + sentiment_score * 5  # Scale sentiment score
    
    if total_score >= 8:
        feedback = "Strong culture fit."
    elif total_score >= 4:
        feedback = "Moderate fit but can improve."
    else:
        feedback = "Weak culture fit."
    
    feedback_label.config(text=f"Feedback: {feedback}")

# Capture spoken response in a separate thread
previous_response = None  # Start as None

def capture_response():
    global previous_response
    def run():
        print("🎤 Listening...")  # Debugging
        with sr.Microphone(device_index=2) as source:
            listening_label.config(text="Listening...")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)  # Adjust for background noise
                audio = recognizer.listen(source, phrase_time_limit=10)  # Limit listening time
                print("Captured audio data")  # Debugging
                
                # Save audio for Whisper
                # audio_path = "./speech.wav"
                # with open(audio_path, "wb") as f:
                #     f.write(audio.get_wav_data())
                
                # Load Whisper Model
                # model = whisper.load_model("base")  # Options: "tiny", "base", "small", "medium", "large"
                # result = model.transcribe(audio_path)
                MyText = recognizer.recognize_google(audio)
                response = MyText
                print(f"✅ Recognized response: {response}")  # Debugging
                
                root.after(0, update_response, response)
            except sr.UnknownValueError:
                print("⚠ Could not understand audio.")  # Debugging
                root.after(0, update_response, "Could not understand audio.")
            except sr.RequestError:
                print("⚠ Speech recognition service error.")  # Debugging
                root.after(0, update_response, "Speech recognition service error.")
            except Exception as e:
                print(f"⚠ Unexpected error: {e}")  # Debugging
                root.after(0, update_response, "Error occurred while processing audio.")
    threading.Thread(target=run, daemon=True).start()

def update_response(response):
    global previous_response
    previous_response = response  # Ensure it's stored correctly
    response_label.config(text=f"Response: {response}")
    evaluate_response(response)

# Start interview
def start_interview():
    global start_button
    start_button.config(state=tk.DISABLED)
    listening_label.config(text="")
    speak("Welcome to the AI-powered interview. Please be yourself and answer honestly.", callback=enable_next_question)

# Enable next question
def enable_next_question():
    start_button.config(text="Next Question", command=next_question, state=tk.NORMAL)

# Next question
def next_question():
    global question_index, question_label, previous_response
    if question_index < len(questions):
        # Show the previous response before asking the new question
        if previous_response:
            response_label.config(text=f"Previous Response: {previous_response}")
        else:
            response_label.config(text="Previous Response: None")

        listening_label.config(text="")  # Ensure "Listening..." isn't shown while AI speaks
        speak(questions[question_index], callback=lambda: root.after(500, capture_response))
        question_index += 1
    else:
        speak("Thank you for participating in the interview.")
        start_button.config(state=tk.DISABLED)
        listening_label.config(text="")

# Sample questions
questions = [
    "How do you handle conflict in a team?",
    "Where do you see yourself in 5 years?",
    "What motivates you to excel at work?"
]
question_index = 0

# Setup GUI
root = tk.Tk()
root.title("AI Interview")
root.geometry("900x600")

# Left Side - Video Feed
video_frame = tk.Frame(root, width=640, height=480)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)
video_label = tk.Label(video_frame)
video_label.pack()

# Right Side - Question and Details
info_frame = tk.Frame(root, width=260, height=480)
info_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
question_label = tk.Label(info_frame, text="Press Start to Begin", font=("Arial", 14), wraplength=250)
question_label.pack(pady=20)
emotion_label = tk.Label(info_frame, text="Emotion: Unknown", font=("Arial", 12))
emotion_label.pack(pady=10)
listening_label = tk.Label(info_frame, text="", font=("Arial", 12), fg="red")
listening_label.pack(pady=5)
response_label = tk.Label(info_frame, text="Response: ", font=("Arial", 12), wraplength=250)
response_label.pack(pady=10)
feedback_label = tk.Label(info_frame, text="Feedback: ", font=("Arial", 12))
feedback_label.pack(pady=10)
start_button = tk.Button(info_frame, text="Start", command=start_interview, font=("Arial", 12))
start_button.pack(pady=10)

# Open video capture
cap = cv2.VideoCapture(0)
process_frame()
analyze_emotion()

root.mainloop()
cap.release()
cv2.destroyAllWindows()