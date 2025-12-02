# 🤖 Transkriptor — datenschutzfreundliche Transkript-Automatisierung

**Transkriptor** ist eine modulare Streamlit-Anwendung zur lokalen bzw. privaten
Transkription und kontrollierten Anonymisierung von Audioaufnahmen.  
Ziel ist ein praxisfähiges Werkzeug für z. B. Soziale Arbeit, Forschung oder Dokumentation,
bei dem sensible Daten lokal verbleiben und über eine optionale Sprechererkennung
(„diarization“) strukturiert werden können.

---

![screenshot](images/screenshot.png)


## 🔎 Hauptfunktionen

- Aufnahme über das Mikrofon (Auswahl des Eingabegeräts).
- Upload von Audiodateien (.wav, .mp3, .m4a).
- Transkription mit `faster-whisper` (lokal, optional GPU-beschleunigt).
- Optionale Vorverarbeitung: Resampling, Normalisierung, einfache Rauschminderung.
- Optionale Anonymisierung (spaCy NER → Ersetzen von PER/LOC/ORG etc.).
- Optionale Sprecher-Diarization (pyannote-Modelle, Hugging Face Token nötig für einige Modelle).
- Debug- und Segmentanzeige (ein-/ausschaltbar).
- Fallback-Modus: Wenn kein HF-Token vorhanden, läuft die App trotzdem — allerdings ohne echte Diarization (Dummy-Speaker).

---


## 🗂 Projektstruktur
```plaintext
Transkriptor/
├── transcriptor/
│ ├── app.py
│ └──  modules/
│ │    ├── recorder.py
│ │    ├── transcribe.py
│ │    ├── speaker_diarization.py
│ │    ├── preprocessing.py
│ │    └── anonymize.py
│ └── config.json
│ ├── requirements.txt
│ ├── README.md
└── .gitignore
```


## 🤗 Hugging Face Token — Schritt-für-Schritt (kurzanleitung)
Dieser Schritt ist nur wichtig, wenn die Sprechererkennung genutzt werden soll.

Wenn das nicht gwünscht ist, kann man den Schritt hier überspringen.



1. Melde dich bei https://huggingface.co an (oder registriere dich).
2. Gehe zu deinem Profil → Settings → Access Tokens (oder: https://huggingface.co/settings/tokens).
3. Erstelle einen neuen Token (New token).
4. Wähle einen aussagekräftigen Namen.
5. Scope: read (oder repo/read:models) — für die meisten Anwendungsfälle reicht read.
6. Kopiere den Token und füge ihn in config.json ein.

ℹ️ Zugriff auf pyannote-Modelle / gated Modelle:
Einige pyannote-Modelle sind gated — das heißt: Du musst auf der jeweiligen HF-Modelldetailseite die Bedingungen explizit akzeptieren (Button „I accept“).
Erst nachdem du die Bedingungen akzeptiert hast und einen korrekten HF-Token benutzt, lässt sich das Modell per `Pipeline.from_pretrained("pyannote/...", use_auth_token=HF_TOKEN)` laden.
Wenn du keinen Token hast oder die Bedingungen nicht akzeptiert sind, fällt das System in den Dummy-Fallback zurück (keine echte Sprechertrennung).

Folgende pyannote-Modelle müssen auf Huggingface akzeptiert werden:

[pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

[pyannote/speaker-diarization-precision-2](https://huggingface.co/pyannote/speaker-diarization-precision-2)

[pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)



## ⚙️ Installation (lokal)

### 1. Repo klonen:
```bash
git clone https://github.com/stickmaaaan/TranscriptAutomation.git
cd transcriptor/transcriptor
```

### 2. Virtuelle Umgebung erstellen und aktivieren:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows (PowerShell: .venv\Scripts\Activate.ps1)
```

### 3. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```
Hinweis: Manche Abhängigkeiten (z. B. torch) sollten passend zur Hardware (CPU / NVIDIA CUDA / ROCm) installiert werden — siehe Abschnitt GPU & Kompatibilität weiter unten.


### 4. config.json vorbereiten:
Siehe: Hugging Face Token
```json
{
    "HF_TOKEN": "hf_.......",
}
```

### 5. App starten:
```bash
streamlit run transcriptor/app.py
```


## 📦 Requirements
Für die Sprechererkennung (pyannote) ist ein Account auf Huggingface.co notwendig!

| Paket            | Zweck                                            |
| ---------------- | ------------------------------------------------ |
| `streamlit`      | Web-UI der Anwendung                             |
| `faster-whisper` | Schnelle Whisper-Transkription auf CPU/GPU       |
| `sounddevice`    | Mikrofonaufnahme                                 |
| `wavio`          | Speichern von WAV-Audio                          |
| `numpy`          | Signalverarbeitung, Audiopuffer                  |
| `scipy`          | Resampling / Preprocessing                       |
| `pydub`          | Formatkonvertierung, Schneiden, Normalisieren    |
| `librosa`        | Audioanalyse (z. B. Lautstärke, Samplerate)      |
| `noisereduce`    | Rauschunterdrückung für Preprocessing            |
| `spacy`          | NLP-Anonymisierung (Namen, Orte, Organisationen) |
| `torch`          | Tensor-Backend / GPU-Beschleunigung              |
| `torchaudio`     | Audio-Backend für Torch                          |
| `pyannote.audio` | **Speaker-Diarization** (Sprechertrennung)       |








## 🪲 Bekannte Probleme
Da die GPU-Nutzung über CUDA läuft, funktioniert die GPU-Nutzung nur mit NVIDIA Grafikkarten!


###   🧠 GPU & Kompatibilität (Warum in der Praxis meist NVIDIA/CUDA)

Kurzfassung: Viele ML-Frameworks (PyTorch, TensorRT, CTranslate2, manche model-optimizations) verwenden CUDA, das proprietäre Compute-Framework von NVIDIA.

NVIDIA GPUs + CUDA die zuverlässigste, mainstream-kompatible Option für GPU-beschleunigte Inferenz.

NVIDIA wird standardmäßig unterstützt mit passenden CUDA-Treibern.


AMD GPUs können mit ROCm in einigen Setups arbeiten, aber ROCm-Unterstützung ist hardware- und distribution-spezifisch (nur bestimmte AMD-Karten, oft Linux-only).


Intel GPUs (neuere Integrationslösungen): experimental, nicht allgemein unterstützt in vielen Libraries (Stand: Standardeinsatzfälle).

Empfohlene Schritte, wenn du GPU nutzen willst:

 - Installiere passende NVIDIA-Treiber + CUDA Toolkit (Version passend zu deiner PyTorch-Version).

```bash
# Beispiel (nicht exakt für jede Version)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Es wird dringend Empfohlen GPU-Nutzung zu verwenden!!!

Wenn GPU nicht benutzt wird, dann folgt ein fallback auf CPU-Nutzung — App läuft weiter, nur langsamer und es kann zu Abstürzen führen.




## 🧭 Usage / UI-Bedienung (kurzanleitung)

1. Starte die App: `streamlit run transcriptor/app.py.`

2. Wähle in der Sidebar:

  - Preprocessing an/aus

  - Anonymizer an/aus

  - Diarization an/aus

  - Force Dummy-Fallback (zum Debug/Test)

3. Nimm eine Aufnahme auf oder lade eine Datei hoch.

4. Klicke Transkription starten — während der Verarbeitung wird die UI gesperrt. Ergebnis wird angezeigt, danach ist die UI wieder aktiv.

5. Transkript als `.txt` oder `.json` exportieren. 

Debug-Ansichten zeigen Transkript- und Diarization-Segmente (optional).



## 🛠 Troubleshooting:

„Huggingface Token fehlt“
→ config.json prüfen. Wenn kein Token vorhanden ist, fällt die App in den Dummy-Fallback (keine echte Sprechererkennung).

„pyannote-Model konnte nicht geladen werden / Zugriff verweigert“
→ Prüfe, ob du die Modellseite geöffnet und die Bedingungen akzeptiert hast. Stelle sicher, dass der HF_TOKEN die richtige Scope/Privilegien hat.

„torch.cuda.is_available() == False“
→ Prüfe GPU-Treiber & CUDA-Installation. Alternativ läuft alles auf CPU, aber deutlich langsamer.

Audio-Recording Fehler (PortAudio / sounddevice)
→ Prüfe sd.query_devices(), wähle richtigen Index, achte auf korrekte Kanalanzahl (mono vs. stereo) und kompatible Sample Rate (16kHZ).

spaCy Modell fehlt (z. B. de_core_news_lg)
→ Installieren: python -m spacy download de_core_news_lg oder wähle ein kleineres Modell (sm / md) falls Speicher knapp ist.


