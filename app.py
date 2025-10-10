import streamlit as st
from modules.transcribe import transcribe_audio
from modules.recorder import record_audio
import os

st.set_page_config(page_title="Lokaler Transkriptor", layout="centered")

st.title("🎧 Lokaler Transkriptor (Prototyp)")

option = st.radio("Wähle Eingabemethode:", ("Audio aufnehmen", "Audiodatei hochladen"))

if option == "Audio aufnehmen":
    duration = st.slider("Aufnahmedauer (Sekunden):", 5, 60, 10)
    if st.button("🎙️ Aufnahme starten"):
        path = record_audio(duration)
        st.success(f"Datei gespeichert: {path}")
        if st.button("🧠 Transkription starten"):
            text = transcribe_audio(path)
            st.text_area("Transkript:", text, height=300)

elif option == "Audiodatei hochladen":
    file = st.file_uploader("Wähle eine Audiodatei (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])
    if file:
        with open("temp_audio.wav", "wb") as f:
            f.write(file.getbuffer())
        if st.button("🧠 Transkription starten"):
            text = transcribe_audio("temp_audio.wav")
            st.text_area("Transkript:", text, height=300)

st.checkbox("Automatisch anonymisieren", value=True)