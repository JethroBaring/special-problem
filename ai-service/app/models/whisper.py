import whisper

# Load Whisper model once
speech_model = whisper.load_model("base")

async def analyze_speech(audio_path):
    try:
        # Transcribe audio
        result = speech_model.transcribe(audio_path)

        # Extract transcription
        transcription = result.get("text", "")
        return {"transcription": transcription}

    except Exception as e:
        print(f"⚠️ Speech analysis error: {e}")
        return None
