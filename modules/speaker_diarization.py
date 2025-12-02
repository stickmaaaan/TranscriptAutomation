# modules/speaker_diarization.py

import os
import sys
import traceback
import torch
import json
# Versuche pyannote-Imports
try:
    from pyannote.audio import Pipeline
except Exception:
    Pipeline = None


def load_hf_token(require_token=False):
    """
    Lädt den Huggingface Token aus config.json.
    - require_token=True: Fehler, wenn kein Token vorhanden.
    - require_token=False: Gibt None zurück, falls kein Token vorhanden.
    """
    config_path = os.path.join(os.path.dirname(__file__), "../config.json")
    try:
        with open(config_path, "r") as f:
            conf = json.load(f)
    except FileNotFoundError:
        conf = {}
    hf_token = conf.get("HF_TOKEN", None)
    if require_token and not hf_token:
        raise ValueError("Huggingface Token fehlt in config.json")
    return hf_token


# Fallback-Diarization (immer gültiges end)
def fallback_diarization(audio_file: str):
    return [{"start": 0.0, "end": float("inf"), "speaker": "Person-DUMMY"}]


def _print_env_info():
    """Druckt Versionsinfo zur schnellen Diagnose (hilfreich beim Debug)."""
    try:
        import pyannote.audio as pa
        pa_ver = getattr(pa, "__version__", "unknown")
    except Exception:
        pa_ver = "not-installed"
    try:
        import huggingface_hub as hf_hub
        hf_ver = getattr(hf_hub, "__version__", "unknown")
    except Exception:
        hf_ver = "not-installed"
    try:
        import torch as _torch
        torch_ver = _torch.__version__
        cuda = _torch.cuda.is_available()
    except Exception:
        torch_ver = "not-installed"
        cuda = False

    print("[ENV] pyannote.audio:", pa_ver)
    print("[ENV] huggingface_hub:", hf_ver)
    print("[ENV] torch:", torch_ver, " CUDA_AVAILABLE=", cuda)
    print("[ENV] python:", sys.version.splitlines()[0])


def _try_pipeline_from_pretrained(model_id: str, hf_token: str):
    """Versuch: Pipeline.from_pretrained (Standard)."""
    if Pipeline is None:
        raise RuntimeError("pyannote.audio.Pipeline ist nicht importierbar in dieser Umgebung.")
    return Pipeline.from_pretrained(model_id, use_auth_token=hf_token)


def _try_alternative_speakerdiarization(model_id: str, hf_token: str):
    """
    Versuch: neuere/andere API (pyannote.audio.pipelines.SpeakerDiarization),
    oder alternative Konstruktion — nur, falls vorhanden.
    """
    try:
        # Import lokal, damit ImportError gehandhabt werden kann
        from pyannote.audio.pipelines import SpeakerDiarization
        return SpeakerDiarization.from_pretrained(model_id, use_auth_token=hf_token)
    except Exception as e:
        raise


def diarize_audio(audio_file: str, force_dummy=False, hf_token=None):
    """
    Robust loader: versucht mehrere Wege, ein pyannote-Pipeline-Modell zu laden.
    Gibt Liste von segments zurück: [{"start": float, "end": float, "speaker": str}, ...]
    Bei Fehlern -> fallback_diarization.
    """

    fallback=fallback_diarization(audio_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        hf_token = load_hf_token()   # <-- Token sicher laden
    except Exception as e:
        print(f"[WARNUNG] HF_TOKEN konnte nicht geladen werden: {e}")
        return fallback

    if not hf_token:
        print("[WARNUNG] HF_TOKEN konnte nicht geladen werden, Dummy-Fallback wird genutzt.")
        return fallback

    # Wenn kein Token, sofort fallback
    if force_dummy:
        print("[INFO] Dummy-Fallback erzwungen, keine echte Diarization.")
        return fallback

    # Drucke Umgebungsinfo für Diagnose
    _print_env_info()

    model_ids_to_try = [
        "pyannote/speaker-diarization-precision-2",
        "pyannote/speaker-diarization-3.1",
        "pyannote/speaker-diarization",
    ]

    last_exc = None
    for model_id in model_ids_to_try:
        try:
            print(f"[INFO] Versuche Pipeline.from_pretrained('{model_id}')")
            pipeline = _try_pipeline_from_pretrained(model_id, hf_token)
            pipeline.to(device)
            print(f"[OK] Pipeline loaded from {model_id} via Pipeline.from_pretrained")
            break
        except Exception as e:
            last_exc = e
            print(f"[DEBUG] Pipeline.from_pretrained('{model_id}') failed: {e}")
            # detailliertere Fehlerausgabe nur für logs
            traceback.print_exc()
            # versuche alternative API
            try:
                print(f"[INFO] Versuche alternative loader für {model_id}")
                pipeline = _try_alternative_speakerdiarization(model_id, hf_token)
                pipeline.to(device)
                print(f"[OK] Pipeline loaded from {model_id} via SpeakerDiarization.from_pretrained")
                break
            except Exception as e2:
                last_exc = e2
                print(f"[DEBUG] Alternative loader failed for {model_id}: {e2}")
                traceback.print_exc()
                continue
    else:
        # alle Versuche gescheitert
        print(f"[FEHLER] Konnte kein pyannote-Pipeline-Modell laden. Letzter Fehler: {last_exc}")
        return fallback

    # Falls pipeline existiert, führe Diarization aus
    try:
        print("[INFO] Führe Diarization aus (this can take time)...")
        diarization = pipeline(audio_file)
        segments = []
        # diarization kann unterschiedliche Typen zurückgeben; robust iterieren
        try:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker)
                })
        except Exception:
            # Manche pipelines geben ein Annotation-Objekt, itersegments statt itertracks
            try:
                for segment, track in diarization.items():
                    # segment könnte (start, end) oder Timestamp-Objekt sein
                    st = float(getattr(segment, "start", segment[0]))
                    ed = float(getattr(segment, "end", segment[1]))
                    sp = str(track)
                    segments.append({"start": st, "end": ed, "speaker": sp})
            except Exception:
                print("[WARNUNG] Unerwartetes Diarization-Resultat; verwende Fallback.")
                return fallback

        if not segments:
            print("[WARNUNG] Keine Segmente erkannt, Fallback wird genutzt.")
            return fallback

        # alles gut
        return segments

    except Exception as e:
        print(f"[FEHLER] Fehler während Diarization: {e}")
        traceback.print_exc()
        return fallback

