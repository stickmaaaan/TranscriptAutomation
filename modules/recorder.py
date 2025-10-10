import sounddevice as sd
import wavio
import tempfile
import os

def record_audio(duration=10, samplerate=16000):
    print(f"Aufnahme gestartet ({duration} Sekunden)...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "aufnahme.wav")
    wavio.write(file_path, audio, samplerate, sampwidth=2)
    print(f"Gespeichert unter: {file_path}")
    return file_path