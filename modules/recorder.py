#modules/recorder.py

import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wavfile

def list_microphones():
    """
    Gibt alle Eingabegeräte mit echten sounddevice-Indizes zurück.
    Jedes Element: { 'id': idx, 'label': 'idx <name> (N in)', 'samplerate': int, 'channels': int }
    """
    devices = sd.query_devices()
    mics = []
    for idx, dev in enumerate(devices):
        # Nur echte Eingabegeräte (mind. 1 input channel)
        if dev.get("max_input_channels", 0) > 0:
            mics.append({
                "id": idx,
                "label": f"{idx}: {dev.get('name', 'unknown')} ({int(dev.get('max_input_channels',0))} in)",
                "name": dev.get("name", "unknown"),
                "channels": int(dev.get("max_input_channels", 1)),
                "samplerate": int(dev.get("default_samplerate", 16000))
            })
    return mics


def get_input_level(mic, duration=0.1):
    """
    Misst den Pegel für das gegebene Mikrofon-Objekt (siehe list_microphones).
    Rückgabewert: Ganzzahl 0..100
    """
    if mic is None:
        return 0
    try:
        sr = int(mic.get("samplerate", 16000))
        idx = mic["id"]
        ch = max(1, int(mic.get("channels", 1)))
        rec = sd.rec(int(duration * sr), samplerate=sr, channels=ch, device=idx, dtype="float32")
        sd.wait()
        # Falls stereo/multi: mix to mono for level computation
        if rec.ndim > 1:
            rec_mono = np.mean(rec, axis=1)
        else:
            rec_mono = rec
        volume_norm = np.linalg.norm(rec_mono) / rec_mono.size
        return min(int(volume_norm * 1000), 100)
    except Exception:
        return 0


def record_audio(duration, mic):
    """
    Nimmt Audio vom ausgewählten 'mic' (Objekt aus list_microphones) auf.
    - verwendet native Kanäle des Geräts beim Recording (für Kompatibilität),
    - downmixt danach sauber auf Mono,
    - speichert als PCM_16 WAV und gibt den Pfad zurück.
    """
    if mic is None:
        print("[FEHLER] Kein Mikrofon-Objekt übergeben.")
        return None

    device_index = mic["id"]
    samplerate = int(mic.get("samplerate", 16000))
    channels = int(mic.get("channels", 1))

    # Schutz: valide samplerate
    if samplerate <= 0:
        samplerate = 16000

    print("[INFO] Starte Audioaufnahme...")
    print(f"[INFO] Aufnahmegerät: {mic['label']}")
    print(f"[INFO] Sample Rate: {samplerate} Hz, Channels: {channels}")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        # Versuche mit nativer Kanalanzahl aufzunehmen (ALSA/USB-Geräte wollen oft native channels)
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, device=device_index, dtype="float32")
        sd.wait()

        # Downmix falls nötig (mono für Whisper/Pyannote)
        if recording.ndim > 1 and recording.shape[1] > 1:
            mono = np.mean(recording, axis=1)
        else:
            mono = recording.reshape(-1)

        # Konvertiere zu PCM16 (Skalierung & Clipping-Schutz)
        mono = np.clip(mono, -1.0, 1.0)
        pcm16 = (mono * 32767).astype(np.int16)

        wavfile.write(temp_file.name, samplerate, pcm16)
        print(f"[OK] Aufnahme gespeichert unter: {temp_file.name}")
        return temp_file.name

    except Exception as e:
        print(f"[FEHLER] Audioaufnahme fehlgeschlagen: {e}")
        return None
