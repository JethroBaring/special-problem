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
import pyaudio
import numpy as np
import librosa
import tkinter as tk
from scipy.signal import find_peaks
import sounddevice as sd  # Ensure this is included
import scipy
# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# Initialize text-to-speech
engine = pyttsx3.init()
recognizer = sr.Recognizer()
import google.generativeai as genai  # Import Gemini API

genai.configure(api_key="AIzaSyDwLPKzexRfCby7a9lQZYelhlWbx3Smylc")

# Store responses
responses = []
dominant_emotion = "Unknown"  # Global variable for emotion
posture_status = "Neutral" 
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
        
        # Analyze Posture
        posture_status = analyze_posture(frame)
        posture_label.config(text=f"Posture: {posture_status}")  # Update UI
    root.after(10, process_frame)

# Emotion analysis (run every 2 seconds)
def analyze_emotion():
    global cap, dominant_emotion
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
                r = {}
                r["question"] = questions[question_index-1]
                r["response"] = MyText
                r["posture"] = posture_status
                r["emotion"] = dominant_emotion
                print(r)
                responses.append(r)
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

# Evaluate interview using Gemini AI
def analyze_with_gemini():
    # Convert each dictionary in responses to a formatted string
    formatted_responses = [
        f"Question: {r['question']}\nResponse: {r['response']}\nPosture: {r['posture']}\nEmotion: {r['emotion']}"
        for r in responses
    ]

    # Join formatted responses with new lines
    full_text = "\n\n".join(formatted_responses)

    prompt = f"""
    You are an AI HR assistant evaluating a candidate’s interview performance.
    The company values include innovation, teamwork, integrity, and growth.
    
    Below are the candidate's responses, along with detected body posture and emotions:
    
    {full_text}

    Provide:
    - Strengths
    - Areas for improvement
    - A culture fit score (0-10)
    """

    model = genai.GenerativeModel("gemini-pro")  # Use Gemini AI model
    response = model.generate_content(prompt)

    feedback_label.config(text=f"Gemini AI Feedback:\n{response.text}")


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
        analyze_with_gemini()

def check_camera_type():
    # Try opening the default camera (0)
    cap = cv2.VideoCapture(0)
    built_in_camera = cap.isOpened()
    cap.release()
    
    # Try opening another camera (1)
    cap = cv2.VideoCapture(1)
    external_camera = cap.isOpened()
    cap.release()

    if built_in_camera and external_camera:
        return "Camera: Built-in & External Detected"
    elif built_in_camera:
        return "Camera: Built-in"
    elif external_camera:
        return "Camera: External Only"
    else:
        return "No Camera Detected"

# Detect Microphone Type
def list_microphones():
    audio = pyaudio.PyAudio()
    mic_info = "No Microphone Detected"
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            mic_info = f"Mic: {device_info['name']} ({'Built-in' if 'internal' in device_info['name'].lower() else 'External'})"
            break  # Assume the first valid input is the primary mic
    audio.terminate()
    return mic_info

# Update hardware labels
def update_hardware_info():
    camera_label.config(text=check_camera_type())
    mic_label.config(text=list_microphones())

# Analyze environment noise level
def analyze_environment_noise():
    sample_rate = 16000  # Sampling rate in Hz
    duration = 2  # Capture duration in seconds

    def record_audio():
        audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        noise_level = np.mean(np.abs(audio_data))
        noise_label.config(text=f"Background Noise: {'High' if noise_level > 0.02 else 'Low'}")

    threading.Thread(target=record_audio, daemon=True).start()

# Analyze speech tone
def analyze_audio_tone():
    sample_rate = 16000  # Sampling rate in Hz
    duration = 2  # Capture duration in seconds

    def record_audio():
        audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio_data = audio_data.flatten()

        # Compute Pitch
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

        # Compute Loudness
        avg_loudness = np.mean(librosa.feature.rms(y=audio_data))

        # Compute Speech Rate (based on peaks in amplitude)
        peaks, _ = scipy.signal.find_peaks(librosa.feature.rms(y=audio_data).flatten(), height=0.02, distance=sample_rate//4)
        speech_rate = len(peaks) / duration

        # Update UI
        pitch_label.config(text=f"Pitch: {round(avg_pitch, 2)} Hz")
        loudness_label.config(text=f"Loudness: {round(avg_loudness, 4)}")
        speech_rate_label.config(text=f"Speech Rate: {round(speech_rate, 2)} words/sec")

    threading.Thread(target=record_audio, daemon=True).start()

def analyze_posture(frame):
    global posture_status
    """Analyzes upper-body posture for interview scenarios."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get key upper-body points
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate shoulder alignment
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)

        # Calculate head tilt (nose position relative to shoulders)
        head_tilt = abs(nose.x - ((left_shoulder.x + right_shoulder.x) / 2))

        # Determine posture based on values
        if shoulder_diff < 0.03 and head_tilt < 0.02:
            posture_status = "Good Posture"
            return "Good Posture ✅"
        elif shoulder_diff > 0.05:
            posture_status = "Uneven Shoulders"
            return "Uneven Shoulders ⚠"
        elif head_tilt > 0.03:
            posture_status = "Head Tilt"
            return "Head Tilt ⚠"
        else:
            posture_status = "Unclear Posture"
            return "Unclear Posture"
    
    return "No Person Detected ❌"

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
# Hardware Info Labels
camera_label = tk.Label(info_frame, text="Detecting Camera...", font=("Arial", 12))
camera_label.pack(pady=5)
mic_label = tk.Label(info_frame, text="Detecting Microphone...", font=("Arial", 12))
mic_label.pack(pady=5)

# Noise & Speech Analysis Labels
noise_label = tk.Label(info_frame, text="Background Noise: Analyzing...", font=("Arial", 10))
noise_label.pack(pady=5)
pitch_label = tk.Label(info_frame, text="Pitch: Analyzing...", font=("Arial", 10))
pitch_label.pack(pady=5)
loudness_label = tk.Label(info_frame, text="Loudness: Analyzing...", font=("Arial", 10))
loudness_label.pack(pady=5)
speech_rate_label = tk.Label(info_frame, text="Speech Rate: Analyzing...", font=("Arial", 10))
speech_rate_label.pack(pady=5)

# Posture
posture_label = tk.Label(info_frame, text="Posture: Analyzing...", font=("Arial", 12))
posture_label.pack(pady=10)

# Function to continuously update noise & audio analysis
def update_audio_analysis():
    analyze_environment_noise()
    analyze_audio_tone()
    root.after(5000, update_audio_analysis)  # Run every 5 seconds

# Start Updating
update_audio_analysis()


# Open video capture
cap = cv2.VideoCapture(0)
process_frame()
analyze_emotion()
# Update hardware info at startup
update_hardware_info()
root.mainloop()
cap.release()
cv2.destroyAllWindows()