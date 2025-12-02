#modules/preprocessing.py

import soundfile as sf
import numpy as np
import librosa
import tempfile

def preprocess_audio(input_path: str):
    y, sr = librosa.load(input_path, sr=16000, mono=True)

    # Verhindert Clipping
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / max_val * 0.95  # maximal 95% Lautstärke

    # Abspeichern als PCM16 WAV
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp.name, y, 16000, subtype='PCM_16')
    print(f"[INFO] Audio preprocessing abgeschlossen: {temp.name}")
    return temp.name
