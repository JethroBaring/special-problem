import whisper
from deepface import DeepFace
from transformers import pipeline
from moviepy import VideoFileClip

# Load AI models
speech_model = whisper.load_model("base")  # Speech-to-text
sentiment_model = pipeline("sentiment-analysis")  # Sentiment
emotion_model = DeepFace  # Facial expression

async def process_video(file):
    video_path = f"/tmp/{file.filename}"
    
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Extract audio from video
    clip = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".wav")
    clip.audio.write_audiofile(audio_path)

    # Speech-to-text transcription
    transcription = speech_model.transcribe(audio_path)["text"]

    # Sentiment Analysis
    sentiment = sentiment_model(transcription)[0]

    # Facial Emotion Analysis
    emotion = emotion_model.analyze(video_path, actions=['emotion'])[0]["dominant_emotion"]

    return {
        "transcription": transcription,
        "sentiment": sentiment["label"],
        "confidence": sentiment["score"],
        "emotion": emotion,
    }
