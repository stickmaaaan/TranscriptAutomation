# 🎙️ Transkriptor - Lokale Audio-Transkription mit Sprechererkennung und Text-Anonymizer!

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)

**Transkriptor** ist eine modulare Streamlit-Anwendung zur **lokalen und datenschutzkonformen** Transkription und kontrollierten Anonymisierung von Audioaufnahmen. Das Tool wurde für Soziale Arbeit, Forschung, Dokumentation und andere sensible Anwendungsfälle entwickelt, bei denen Daten lokal verbleiben müssen.

![Transkriptor Screenshot](images/screenshot.png)

---

## 📋 Inhaltsverzeichnis

- [Features](#-features)
- [Systemanforderungen](#-systemanforderungen)
- [Warum nur NVIDIA GPU?](#-warum-nur-nvidia-gpu)
- [Installation](#-installation)
  - [1. Repository klonen](#1-repository-klonen)
  - [2. Python Virtual Environment erstellen](#2-python-virtual-environment-erstellen)
  - [3. Abhängigkeiten installieren](#3-abhängigkeiten-installieren)
  - [4. Hugging Face Token einrichten](#4-hugging-face-token-einrichten-optional)
  - [5. spaCy Modell installieren](#5-spacy-modell-installieren)
  - [6. GPU-Setup (NVIDIA)](#6-gpu-setup-nvidia-empfohlen)
- [Konfiguration](#-konfiguration)
- [Verwendung](#-verwendung)
- [Projektstruktur](#-projektstruktur)
- [Abhängigkeiten](#-abhängigkeiten)
- [Troubleshooting](#-troubleshooting)
- [Zukünftige Erweiterungen](#-zukünftige-erweiterungen)
- [Lizenz](#-lizenz)

---

## ✨ Features

### 🎤 Audio-Input
- **Mikrofonaufnahme** mit Auswahl des Eingabegeräts
- **Live-Pegelanzeige** während der Aufnahme
- **Datei-Upload** für `.wav`, `.mp3`, `.m4a`

### 🔊 Audio-Processing
- **Vorverarbeitung (optional)**: Resampling auf 16kHz, Normalisierung, Rauschunterdrückung
- **Lokale Transkription** mit [faster-whisper](https://github.com/guillaumekln/faster-whisper) (GPU-beschleunigt)
- **Zeitstempel** für jedes Segment (ein-/ausschaltbar)

### 👥 Sprechererkennung
- **Speaker Diarization** mit [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- Anonymisierte Sprecher als "Person 1", "Person 2", etc.
- **Dummy-Fallback** falls kein Hugging Face Token vorhanden

### 🔒 Datenschutz & Anonymisierung
- **NER-basierte Anonymisierung** mit [spaCy](https://spacy.io/) (deutsches Modell)
- Ersetzt automatisch:
  - Personennamen (`[PER]`)
  - Orte (`[LOC]`)
  - Organisationen (`[ORG]`)
  - E-Mails (`[EMAIL]`)
  - Telefonnummern (`[TELEFON]`)
  - Daten (`[DATUM]`)
  - Zahlen (`[ZAHL]`)

### 📤 Export & Debugging
- **Export als `.txt`** oder **`.json`**
- **Debug-Modi**: Anzeige von Diarization- und Transkript-Segmenten
- **Verwerfen & Neustart** Button für schnelles Zurücksetzen

---

## 💻 Systemanforderungen

| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| **Betriebssystem** | Windows 10, macOS 10.15+, Linux (Ubuntu 20.04+) | Windows 11, macOS 12+, Ubuntu 22.04+ |
| **Python** | 3.8+ | 3.10+ |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | - | NVIDIA GPU mit 6+ GB VRAM |
| **CUDA** | - | CUDA 11.8 oder 12.x |
| **Speicher** | 5 GB (Modelle) | 10 GB+ |

⚠️ **Wichtig**: Ohne GPU läuft die Transkription auf CPU (deutlich langsamer, kann zu Abstürzen führen bei großen Dateien).

---

## 🚀 Warum nur NVIDIA GPU?

**Kurze Antwort**: Die meisten ML-Frameworks (PyTorch, TensorFlow, CTranslate2) nutzen **CUDA**, NVIDIAs proprietäre Compute-Plattform.

### Technischer Hintergrund

1. **CUDA-Abhängigkeit**: 
   - `faster-whisper` basiert auf **CTranslate2**, das für CUDA optimiert ist
   - `pyannote.audio` nutzt **PyTorch**, das primär CUDA unterstützt
   - NVIDIA-Treiber + CUDA Toolkit sind Standard in der ML-Community

2. **Alternativen und ihre Einschränkungen**:
   - **AMD ROCm**: Nur auf bestimmten AMD-Karten, oft Linux-only, instabil
   - **Intel GPUs**: Experimentell, kaum Library-Support
   - **Apple Silicon (M1/M2/M3)**: PyTorch hat MPS-Support, aber CTranslate2 nicht vollständig kompatibel

3. **Warum keine CPU-Optimierung?**:
   - CPU-Inferenz ist 10-50x langsamer als GPU
   - Große Whisper-Modelle (`large-v3`) benötigen GPU-VRAM für performante Verarbeitung
   - Speaker Diarization (pyannote) ist extrem rechenintensiv

**Empfehlung**: NVIDIA GPU mit mindestens 6 GB VRAM (z.B. RTX 3060, RTX 4060, GTX 1660 Ti)

---

## 📦 Installation

### 1. Repository klonen

```bash
git clone https://github.com/stickmaaaan/TranscriptAutomation.git
cd TranscriptAutomation
```

### 2. Python Virtual Environment erstellen

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows (CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Abhängigkeiten installieren

#### Standard-Installation (CPU-only)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### GPU-Installation (NVIDIA CUDA)

**Wichtig**: Zuerst NVIDIA-Treiber + CUDA Toolkit installieren!

##### CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

##### CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**CUDA Toolkit Download**: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

**PyTorch Installation Wizard**: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### 4. Hugging Face Token einrichten (Optional)

⚠️ **Nur notwendig für Speaker Diarization!**

#### a) Hugging Face Account erstellen
1. Registriere dich bei [Hugging Face](https://huggingface.co/join)
2. Gehe zu [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Klicke auf **"New token"**
4. Name: z.B. `transkriptor-token`
5. Type: **Read**
6. Kopiere den Token (beginnt mit `hf_...`)

#### b) Modelle akzeptieren (Gated Models)

Folgende Modelle müssen manuell akzeptiert werden:

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/speaker-diarization-precision-2](https://huggingface.co/pyannote/speaker-diarization-precision-2)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

**Klicke auf jeder Seite auf "Agree and access repository"**

#### c) Token in `config.json` eintragen

```json
{
  "HF_TOKEN": "hf_deinTokenHier..."
}
```

**Alternative**: Token direkt in der UI eingeben (Sidebar → "Token bearbeiten")

### 5. spaCy Modell installieren

Für deutsche Anonymisierung:

```bash
python -m spacy download de_core_news_lg
```

**Alternative Modelle** (falls Speicher knapp):
- `de_core_news_md` (mittlere Genauigkeit, 40 MB)
- `de_core_news_sm` (niedrige Genauigkeit, 15 MB)

**spaCy Modelle**: [https://spacy.io/models/de](https://spacy.io/models/de)

### 6. GPU-Setup (NVIDIA, empfohlen)

#### Treiber-Check (Windows/Linux)
```bash
nvidia-smi
```

**Erwartete Ausgabe**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2   |
+-----------------------------------------------------------------------------+
```

Falls `nvidia-smi` nicht funktioniert:
- **Windows**: [NVIDIA-Treiber Download](https://www.nvidia.com/Download/index.aspx)
- **Linux**: `sudo apt install nvidia-driver-535` (Ubuntu)

#### PyTorch GPU-Test
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

**Erwartete Ausgabe**:
```
PyTorch Version: 2.1.0+cu118
CUDA verfügbar: True
CUDA Version: 11.8
GPU Name: NVIDIA GeForce RTX 3060
```

---

## 🎯 Verwendung

### App starten

```bash
cd transcriptor
streamlit run app.py
```

**Erwartete Ausgabe**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Workflow

1. **Token überprüfen** (Sidebar)
   - Grüner Haken ✅ = Token gültig
   - Roter Fehler ❌ = Token ungültig/fehlt

2. **Pipeline-Einstellungen** (Sidebar, alle optional)
   - ☑️ Preprocessing aktivieren
   - ☑️ Text-Anonymizer aktivieren
   - ☑️ Sprecher-Diarization aktivieren
   - ☑️ Zeitstempel anzeigen
   - ☑️ Dummy-Fallback erzwingen (für Tests ohne Token)

3. **Audio-Input wählen**
   - **Aufnahme**: Mikrofon auswählen → Dauer einstellen → Aufnahme starten
   - **Upload**: Datei hochladen (`.wav`, `.mp3`, `.m4a`)

4. **Transkription starten**
   - Button "🚀 Transkription starten" klicken
   - Warten (je nach GPU/CPU: 1-10 Minuten pro Stunde Audio)

5. **Ergebnisse exportieren**
   - 📄 **Als TXT exportieren**: Reiner Text mit Zeitstempeln
   - 📋 **Als JSON exportieren**: Strukturierte Daten
   - 🗑️ **Verwerfen & Neustart**: Session zurücksetzen

### Beispiel-Output

**Mit Diarization + Zeitstempel**:
```
[0.00-2.50] Person 1: Guten Tag, wie kann ich Ihnen helfen?
[2.50-5.00] Person 2: Ich hätte gerne Informationen zu [ORG].
[5.00-8.20] Person 1: Natürlich, [PER] kann Ihnen da weiterhelfen.
```

**Nur Transkript (ohne Optionen)**:
```
Guten Tag, wie kann ich Ihnen helfen?
Ich hätte gerne Informationen zu dem Projekt.
Natürlich, unser Kollege kann Ihnen da weiterhelfen.
```

---

## 📁 Projektstruktur

```
TranscriptAutomation/
│
├── transcriptor/
│   ├── app.py                      # Haupt-Streamlit-App
│   ├── config.json                 # HF Token (optional, nicht in Git)
│   │
│   └── modules/
│       ├── recorder.py             # Mikrofonaufnahme + Audio-Handling
│       ├── transcribe.py           # Whisper-Transkription (Hauptlogik)
│       ├── speaker_diarization.py  # pyannote Speaker Diarization
│       ├── preprocessing.py        # Audio-Normalisierung/Resampling
│       └── anonymize.py            # spaCy NER-Anonymisierung
│
├── requirements.txt                # Python-Abhängigkeiten
├── README.md                       # Diese Datei
├── .gitignore                      # Git-Ausschlüsse
└── screenshot/
    └── TranscriptorWF.png          # UI-Screenshot
```

---

## 📦 Abhängigkeiten

### Core Libraries

| Paket | Version | Zweck |
|-------|---------|-------|
| `streamlit` | >=1.28.0 | Web-UI Framework |
| `faster-whisper` | >=0.10.0 | Whisper-Transkription (CTranslate2) |
| `torch` | >=2.0.0 | ML-Backend / GPU-Beschleunigung |
| `torchaudio` | >=2.0.0 | Audio-Tensor-Operationen |
| `pyannote.audio` | >=3.0.0 | Speaker Diarization |

### Audio-Processing

| Paket | Version | Zweck |
|-------|---------|-------|
| `sounddevice` | >=0.4.6 | Mikrofonaufnahme (PortAudio) |
| `scipy` | >=1.10.0 | Audio-Resampling |
| `librosa` | >=0.10.0 | Audio-Feature-Extraction |
| `soundfile` | >=0.12.0 | WAV/FLAC/OGG I/O |
| `noisereduce` | >=3.0.0 | Rauschunterdrückung |

### NLP & Anonymisierung

| Paket | Version | Zweck |
|-------|---------|-------|
| `spacy` | >=3.6.0 | NER (Named Entity Recognition) |
| `de_core_news_lg` | >=3.6.0 | Deutsches Sprachmodell |

### Utilities

| Paket | Version | Zweck |
|-------|---------|-------|
| `numpy` | >=1.24.0 | Array-Operationen |
| `huggingface_hub` | >=0.19.0 | HF Model Download/Auth |

### `requirements.txt` (Vollständig)

```txt
streamlit>=1.28.0
faster-whisper>=0.10.0
torch>=2.0.0
torchaudio>=2.0.0
pyannote.audio>=3.0.0
sounddevice>=0.4.6
scipy>=1.10.0
librosa>=0.10.0
soundfile>=0.12.0
noisereduce>=3.0.0
spacy>=3.6.0
numpy>=1.24.0
huggingface-hub>=0.19.0
```

**Installation (siehe oben)**:
```bash
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

### ❌ Fehler: "Huggingface Token fehlt"

**Lösung**:
1. Token in `config.json` oder UI eingeben (Sidebar → "Token bearbeiten")
2. Falls vorhanden: Modelle auf HF akzeptieren (siehe [Hugging Face Token](#4-hugging-face-token-einrichten-optional))

### ❌ Fehler: "pyannote-Model konnte nicht geladen werden"

**Ursachen**:
- Modelle nicht auf Hugging Face akzeptiert
- Token hat falsche Permissions (muss `read` sein)
- Netzwerkprobleme beim Download

**Lösung**:
1. Manuell auf HF einloggen und Modelle akzeptieren
2. Token neu erstellen mit `read`-Scope
3. Proxy/Firewall-Einstellungen prüfen

### ❌ Fehler: "torch.cuda.is_available() == False"

**Ursachen**:
- PyTorch für CPU installiert (nicht CUDA)
- NVIDIA-Treiber fehlt/veraltet
- CUDA Toolkit nicht installiert

**Lösung**:
```bash
# GPU-Check
nvidia-smi

# PyTorch neu installieren (CUDA 11.8)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Test
python -c "import torch; print(torch.cuda.is_available())"
```

### ❌ Fehler: "PortAudio / sounddevice Fehler"

**Linux**:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**macOS**:
```bash
brew install portaudio
```

**Windows**: Normalerweise kein Problem, falls doch:
1. Visual C++ Redistributables installieren
2. `pip install sounddevice --force-reinstall`

### ❌ Fehler: "spaCy Modell 'de_core_news_lg' nicht gefunden"

**Lösung**:
```bash
python -m spacy download de_core_news_lg
```

### ⚠️ Warnung: "Transkription sehr langsam (CPU)"

**Ursache**: Keine NVIDIA GPU verfügbar oder CUDA nicht richtig Installiert!

**Temporäre Lösung**:
- Kleineres Whisper-Modell nutzen (`base`, `small` statt `large`)
- Kürzere Audio-Dateien verarbeiten
- Geduld haben (10-30 Minuten für 1 Stunde Audio auf CPU)

**Dauerhafte Lösung**: NVIDIA GPU installieren

### 🐛 Debug-Modus aktivieren

In der Sidebar:
- ☑️ Debug: Diarization Segmente anzeigen
- ☑️ Debug: Transcript Segmente anzeigen

Zeigt detaillierte Zeitstempel und Speaker-Labels.

---

## 🚀 Zukünftige Erweiterungen

### 🔄 In Planung

#### 1. **Live-Transkription (Speech-to-Text in Echtzeit)**
- Streaming-Transkription während der Aufnahme
- Intuitive User Experience mit Live-Feedback
- WebSocket-basierte Architektur für niedrige Latenz

#### 2. **Webhosting mit externer GPU-Anbindung**
- Cloud-Deployment (AWS/GCP/Azure) mit GPU-Instanzen
- Ressourcenschonend für Client (nur Browser benötigt)
- Schnellere Verarbeitung durch dedizierte Server-GPUs
- Multi-User-Support mit Queue-System

#### 3. **Inhaltzusammenfassung auf anonymisierter Grundlage**
- LLM-basierte Zusammenfassung (GPT-4, Claude, Llama)
- Automatische Extraktion der wichtigsten Punkte
- Konjunktiv-Umschreibung für objektive Darstellung
- Export als Executive Summary

---

## 📄 Lizenz

Dieses Projekt steht unter der **MIT-Lizenz**. Siehe [LICENSE](LICENSE) für Details.

### Verwendete Libraries & Modelle

- **faster-whisper**: [MIT License](https://github.com/guillaumekln/faster-whisper/blob/master/LICENSE)
- **pyannote.audio**: [MIT License](https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE)
- **spaCy**: [MIT License](https://github.com/explosion/spaCy/blob/master/LICENSE)
- **Whisper (OpenAI)**: [MIT License](https://github.com/openai/whisper/blob/main/LICENSE)

---

## 📞 Support & Kontakt

**GitHub Issues**: [TranscriptAutomation/issues](https://github.com/stickmaaaan/TranscriptAutomation/issues)

**Häufige Probleme**: Siehe [Troubleshooting](#-troubleshooting)

---

## 🙏 Danksagungen

- **OpenAI Whisper**: State-of-the-Art Speech-to-Text
- **pyannote.audio**: Robuste Speaker Diarization
- **spaCy**: Schnelle NLP-Pipeline
- **Streamlit**: Einfaches UI-Framework

---

**Made with ❤️ for privacy-conscious transcription**
