from faster_whisper import WhisperModel
from .anonymize import anonymize_full

def transcribe_audio(file_path, model_size="large"):
    print("🧠 Starte Transkription...")
    model = WhisperModel(model_size, device="cpu")
    segments, info = model.transcribe(file_path)
    text = " ".join([segment.text for segment in segments])
    anonymized_text = anonymize_full(text)
    return anonymized_text