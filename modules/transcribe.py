# modules/transcribe.py

import os
import torch
from .speaker_diarization import diarize_audio, fallback_diarization
from .anonymize import anonymize_text

def find_speaker_for_time(diar_segments, timestamp):
    """Finde das passende Speaker-Segment für einen gegebenen Zeitpunkt."""
    for seg in diar_segments:
        if seg["start"] <= timestamp <= seg["end"]:
            return seg["speaker"]
    return "Unbekannt"

def transcribe_audio(
    file_path,
    model_size="large",
    preprocessing_enabled=True,
    anonymizer_enabled=True,
    diarization_enabled=True,
    timestamps_enabled=True,
    force_dummy=False,
    hf_token=None,
    return_debug=True
):
    """
    Vollständig modularisierte Transkription + Diarization + Anonymizer
    - preprocessing_enabled: Audio preprocessing an/aus
    - anonymizer_enabled: Text anonymisieren an/aus
    - diarization_enabled: Diarization an/aus (wenn False: keine Sprecher-Segmentierung!)
    - timestamps_enabled: Zeitstempel anzeigen an/aus
    - force_dummy: Dummy-Fallback erzwingen
    - hf_token: Huggingface Token, optional
    - return_debug: Debug-Info zurückgeben
    Rückgabe: (formatted_text, debug_dict)
    """

    debug = {}
    created_temp = None

    # 1️⃣ Vorverarbeitung
    if preprocessing_enabled:
        try:
            from .preprocessing import preprocess_audio
            cleaned_path = preprocess_audio(file_path)
            created_temp = cleaned_path
            print(f"[INFO] Audio preprocessing abgeschlossen: {cleaned_path}")
        except Exception as e:
            print(f"[WARNUNG] Preprocessing fehlgeschlagen: {e}")
            cleaned_path = file_path
    else:
        cleaned_path = file_path
        print("[INFO] Preprocessing deaktiviert, Originalaudio wird verwendet.")

    # 2️⃣ Whisper-Transkription
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from faster_whisper import WhisperModel
    print("[INFO] Starte Transkription mit Whisper...")
    model = WhisperModel(model_size, device=device)
    segments, _ = model.transcribe(cleaned_path, beam_size=5)
    transcript_segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in segments
    ]
    print(f"[OK] Transkription abgeschlossen: {len(transcript_segments)} Segmente")
    debug["transcript_segments"] = transcript_segments.copy()

    # 3️⃣ Sprecher-Diarization (nur wenn aktiviert!)
    diar_segments = None
    if diarization_enabled:
        diar_segments = diarize_audio(
            cleaned_path,
            force_dummy=force_dummy,
            hf_token=hf_token
        )
        print("[INFO] Diarization Segmente erhalten:", len(diar_segments))
        debug["diar_segments"] = diar_segments.copy()
    else:
        print("[INFO] Diarization deaktiviert")
        debug["diar_segments"] = []

    # 4️⃣ Mapping + optionaler Anonymizer
    speaker_map = {}
    def map_speaker(label):
        # Wenn Dummy-Fallback aktiv ist: IMMER "Person", keine Nummerierung
        if force_dummy:
            return "Person"

        if label not in speaker_map:
            speaker_map[label] = f"Person {len(speaker_map)+1}"
        return speaker_map[label]

    final_transcript = []
    for seg in transcript_segments:
        text = seg["text"]
        
        # Anonymizer anwenden (wenn aktiviert)
        if anonymizer_enabled:
            try:
                text = anonymize_text(text)
            except Exception as e:
                print(f"[WARNUNG] Anonymizer fehlgeschlagen: {e}; Originaltext wird verwendet")
        
        # Sprecher-Mapping (nur wenn Diarization aktiviert)
        if diarization_enabled and diar_segments:
            speaker_label = find_speaker_for_time(diar_segments, seg["start"])
            mapped = map_speaker(speaker_label)
            final_transcript.append({
                "speaker_label": speaker_label,
                "mapped": mapped,
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "has_speaker": True
            })

            
        else:
            # KEINE Sprecher-Info, nur Text
            final_transcript.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "has_speaker": False
            })

    # 5️⃣ temporäre Datei löschen
    try:
        if created_temp and created_temp != file_path:
            os.remove(created_temp)
    except Exception:
        pass

    # 6️⃣ Ausgabe formatieren
    lines = []
    for s in final_transcript:
        # Zeitstempel-Prefix (nur wenn aktiviert)
        timestamp_prefix = f"[{s['start']:.2f}-{s['end']:.2f}] " if timestamps_enabled else ""
        
        if s.get("has_speaker", False):
            # MIT Sprecher-Segmentierung
            dummy_prefix = "[DUMMY-Fallback] " if force_dummy else ""
            lines.append(f"{dummy_prefix}{timestamp_prefix}{s['mapped']}: {s['text']}")
        else:
            # OHNE Sprecher-Segmentierung
            lines.append(f"{timestamp_prefix}{s['text']}")

    return ("\n".join(lines), debug if return_debug else None)