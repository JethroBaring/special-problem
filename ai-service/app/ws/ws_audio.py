import whisper
from transformers import pipeline
import asyncio
import subprocess
import tempfile
import os

# Load AI models
speech_model = whisper.load_model("base")  # Whisper for speech-to-text
sentiment_model = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

async def process_realtime_audio(websocket):
    await websocket.accept()
    print("WebSocket connected for real-time analysis.")

    while True:
        try:
            print("Waiting for audio input...")
            audio_chunk = await websocket.receive_bytes()
            print("Received audio data")

            # Validate received audio
            if not audio_chunk or len(audio_chunk) < 100:
                print("Received empty or invalid audio data. Skipping processing.")
                continue

            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm:
                temp_webm.write(audio_chunk)
                temp_webm.flush()
                webm_path = temp_webm.name

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                wav_path = temp_wav.name

            # Convert WebM to WAV using FFmpeg
            convert_command = [
                "ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
            ]
            result = subprocess.run(convert_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                print(f"FFmpeg conversion failed: {result.stderr.decode()}")
                continue

            # Transcribe audio
            transcription = speech_model.transcribe(wav_path).get("text", "")

            # Analyze sentiment
            sentiment = sentiment_model(transcription)[0]

            response = {
                "transcription": "xd",
                "sentiment": sentiment["label"],
                "confidence": f"{sentiment['score'] * 100:.2f}%"
            }
            await websocket.send_json(response)
        
        except Exception as e:
            print("Error processing audio:", e)
            break
        
        finally:
            # Cleanup
            for path in (webm_path, wav_path):
                if os.path.exists(path):
                    os.remove(path)
