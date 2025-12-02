import streamlit as st
from modules.recorder import list_microphones, get_input_level, record_audio
from modules.transcribe import transcribe_audio
from modules.speaker_diarization import load_hf_token
import time
import threading
import tempfile
import json
import os

st.set_page_config(page_title="Transkriptor", layout="wide")
st.title("🤖 Transkript Automatisierung")

# ---------------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------------
def validate_hf_token(token):
    """Überprüft ob der HF Token gültig ist"""
    if not token:
        return False, "Kein Token vorhanden"
    
    if len(token) < 20:
        return False, "Token ist unvollständig (zu kurz)"
    
    if not token.startswith("hf_"):
        return False, "Token hat falsches Format (sollte mit 'hf_' beginnen)"
    
    # Versuche echte API-Validierung
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami(token=token)
        return True, "Token ist gültig ✓"
    except ImportError:
        return None, "huggingface_hub nicht installiert - Format scheint OK"
    except Exception as e:
        return False, f"Token ist ungültig: {str(e)}"

def save_config(token):
    """Speichert den Token in config.json"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "w") as f:
            json.dump({"HF_TOKEN": token}, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Fehler beim Speichern: {e}")
        return False

def export_transcript(text, format_type):
    """Exportiert das Transkript als Download"""
    if format_type == "txt":
        return text.encode('utf-8'), "text/plain", "transkript.txt"
    elif format_type == "json":
        lines = text.split("\n")
        json_data = {"segments": []}
        for line in lines:
            if line.strip():
                json_data["segments"].append({"text": line})
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        return json_str.encode('utf-8'), "application/json", "transkript.json"

# ---------------------------------------------------------
# TOKEN-VERWALTUNG IN SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("🔑 Huggingface Token")

try:
    current_token = load_hf_token(require_token=False)
except Exception:
    current_token = None

if current_token:
    masked_token = current_token[:7] + "..." + current_token[-4:] if len(current_token) > 11 else "***"
    st.sidebar.text(f"Token: {masked_token}")
    
    is_valid, msg = validate_hf_token(current_token)
    if is_valid:
        st.sidebar.success(msg)
    elif is_valid is False:
        st.sidebar.error(msg)
    else:
        st.sidebar.warning(msg)
else:
    st.sidebar.warning("⚠️ Kein Token in config.json gefunden")

# Token validieren Button
if st.sidebar.button("🔍 Token überprüfen"):
    if current_token:
        with st.sidebar:
            with st.spinner("Validiere Token..."):
                is_valid, msg = validate_hf_token(current_token)
                if is_valid:
                    st.success(msg)
                elif is_valid is False:
                    st.error(msg)
                else:
                    st.info(msg)
    else:
        st.sidebar.error("Kein Token vorhanden")

# Token bearbeiten
with st.sidebar.expander("Token bearbeiten"):
    new_token = st.text_input("Neuer Token", type="password", key="new_token_input")
    if st.button("Token speichern"):
        if new_token:
            if save_config(new_token):
                st.success("Token gespeichert! Seite wird neu geladen...")
                time.sleep(1)
                st.rerun()
        else:
            st.error("Bitte einen Token eingeben")

# ---------------------------------------------------------
# STATES
# ---------------------------------------------------------
if "processing" not in st.session_state:
    st.session_state.processing = False

if "audio_file_path" not in st.session_state:
    st.session_state.audio_file_path = None

if "transcript_result" not in st.session_state:
    st.session_state.transcript_result = None

if "debug_info" not in st.session_state:
    st.session_state.debug_info = None

# ---------------------------------------------------------
# DISABLE UI IF PROCESSING
# ---------------------------------------------------------
def ui_disabled():
    return {"disabled": st.session_state.processing}

# ---------------------------------------------------------
# SIDEBAR - PIPELINE EINSTELLUNGEN
# ---------------------------------------------------------
st.sidebar.header("⚙️ Pipeline Einstellungen")

preprocessing_enabled = st.sidebar.checkbox(
    "Preprocessing aktivieren (Resample/Normalize)", value=False, **ui_disabled()
)

anonymizer_enabled = st.sidebar.checkbox(
    "Text-Anonymizer aktivieren", value=False, **ui_disabled()
)

diarization_enabled = st.sidebar.checkbox(
    "Sprechererkennung aktivieren", value=False, **ui_disabled()
)

timestamps_enabled = st.sidebar.checkbox(
    "Zeitstempel anzeigen", value=False, **ui_disabled()
)

force_dummy_fallback = st.sidebar.checkbox(
    "Sprechererkennung-Fallback erzwingen (Person-DUMMY)", value=False, **ui_disabled()
)

show_diar_debug = st.sidebar.checkbox(
    "Debug: Diarization Segmente anzeigen", value=False, **ui_disabled()
)

show_transcript_debug = st.sidebar.checkbox(
    "Debug: Transcript Segmente anzeigen", value=False, **ui_disabled()
)

# ---------------------------------------------------------
# MODE
# ---------------------------------------------------------
mode = st.radio("Modus wählen", ["Aufnahme", "Datei hochladen"], **ui_disabled())

# ---------------------------------------------------------
# AUDIO RECORDING
# ---------------------------------------------------------
if mode == "Aufnahme":
    mics = list_microphones()
    if not mics:
        st.error("Kein Mikrofon gefunden!")
        st.stop()

    mic_labels = [m["label"] for m in mics]
    selected_label = st.selectbox("Mikrofon wählen", mic_labels, **ui_disabled())
    selected_mic = next(m for m in mics if m["label"] == selected_label)

    duration = st.slider("Aufnahmedauer (Sekunden)", 5, 120, 10, **ui_disabled())

    show_level = st.checkbox("Mikrofonpegel anzeigen", **ui_disabled())
    level_bar = st.progress(0)
    stop_thread = False

    def monitor_level(microphone):
        while not stop_thread:
            try:
                level = get_input_level(microphone, duration=0.1)
                level_bar.progress(level)
            except Exception:
                level_bar.progress(0)
            time.sleep(0.1)

    if show_level and not st.session_state.processing:
        t = threading.Thread(target=monitor_level, args=(selected_mic,), daemon=True)
        t.start()

    if st.button("🎤 Aufnahme starten", **ui_disabled()):
        stop_thread = True
        path = record_audio(duration, selected_mic)
        if path:
            st.success("Aufnahme abgeschlossen!")
            st.session_state.audio_file_path = path
            st.session_state.transcript_result = None  # Reset old results
            st.session_state.debug_info = None
        else:
            st.error("Aufnahme fehlgeschlagen.")

# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------
elif mode == "Datei hochladen":
    uploaded_file = st.file_uploader(
        "Wähle eine Audiodatei (.wav, .mp3, .m4a)",
        type=["wav", "mp3", "m4a"],
        **ui_disabled()
    )
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(uploaded_file.read())
        temp_file.flush()
        st.session_state.audio_file_path = temp_file.name
        st.session_state.transcript_result = None  # Reset old results
        st.session_state.debug_info = None
        st.success("Datei hochgeladen!")

# ---------------------------------------------------------
# START BUTTON
# ---------------------------------------------------------
st.markdown("---")
start_btn = st.button(
    "🚀 Transkription starten", 
    disabled=(not st.session_state.audio_file_path) or st.session_state.processing
)

if start_btn:
    st.session_state.processing = True
    st.rerun()

# ---------------------------------------------------------
# PIPELINE EXECUTION
# ---------------------------------------------------------
if st.session_state.processing:
    if st.session_state.audio_file_path:
        with st.spinner("Verarbeite Audio..."):
            result_text, debug = transcribe_audio(
                st.session_state.audio_file_path,
                model_size="large",
                preprocessing_enabled=preprocessing_enabled,
                anonymizer_enabled=anonymizer_enabled,
                diarization_enabled=diarization_enabled,
                timestamps_enabled=timestamps_enabled,
                force_dummy=force_dummy_fallback,
                hf_token=current_token,
            )
            
            st.session_state.transcript_result = result_text
            st.session_state.debug_info = debug
            st.session_state.processing = False
            # Kein st.rerun() hier! Einfach weiterlaufen lassen

# ---------------------------------------------------------
# ERGEBNISSE ANZEIGEN
# ---------------------------------------------------------
if st.session_state.transcript_result:
    st.success("✅ Transkription abgeschlossen!")
    
    st.text_area(
        "Transkript", 
        st.session_state.transcript_result, 
        height=300,
        key="transcript_display"
    )
    
    # Export-Buttons und Verwerfen-Button in einer Reihe
    col1, col2, col3 = st.columns(3)
    with col1:
        txt_data, txt_mime, txt_name = export_transcript(st.session_state.transcript_result, "txt")
        st.download_button(
            label="📄 Als TXT exportieren",
            data=txt_data,
            file_name=txt_name,
            mime=txt_mime
        )
    with col2:
        json_data, json_mime, json_name = export_transcript(st.session_state.transcript_result, "json")
        st.download_button(
            label="📋 Als JSON exportieren",
            data=json_data,
            file_name=json_name,
            mime=json_mime
        )
    with col3:
        if st.button("🗑️ Verwerfen & Neustart"):
            # Alle relevanten States zurücksetzen
            st.session_state.transcript_result = None
            st.session_state.debug_info = None
            st.session_state.audio_file_path = None
            st.session_state.processing = False
            st.rerun()
    
    # Debug-Informationen
    if show_diar_debug and st.session_state.debug_info:
        st.write("### Debug: Diarization Segmente")
        diar_segments = st.session_state.debug_info.get("diar_segments", [])
        for d in diar_segments:
            st.write(f"{d.get('start'):.2f}s — {d.get('end'):.2f}s : {d.get('speaker')}")

    if show_transcript_debug and st.session_state.debug_info:
        st.write("### Debug: Transcript Segmente")
        trans_segments = st.session_state.debug_info.get("transcript_segments", [])
        for t in trans_segments:
            st.write(f"{t.get('start'):.2f}s — {t.get('end'):.2f}s : {t.get('text')}")