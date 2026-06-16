# Multimodal AI-Based Stress Detection

A real-time stress detection system that analyses facial expressions, voice emotion, and heart rate simultaneously through a standard webcam and microphone.

---

## How it Works

- **Face** — DeepFace AI detects facial emotion from webcam frames
- **Voice** — MFCC features extracted from microphone audio, classified by a trained neural network
- **Heart Rate** — Estimated from webcam video using remote photoplethysmography (rPPG)
- **Fusion** — All three outputs combined using weighted late fusion (50% face, 30% audio, 20% heart rate)
- **Result** — Stress classified as LOW, MEDIUM, or HIGH with personalised recommendations

---

## Tech Stack

- **Backend** — Python, Flask, TensorFlow, DeepFace, Librosa, OpenCV, SciPy
- **Frontend** — React.js
- **Audio** — Pydub, FFmpeg

---

## Setup

**1. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**2. Install frontend dependencies**

```bash
cd frontend
npm install
```

**3. Install FFmpeg and add to PATH**

---

## Run

**Terminal 1 — Backend**

```bash
python app.py
```

**Terminal 2 — Frontend**

```bash
cd frontend
set PORT=3001 && npm start
```

Open http://localhost:3001

---

## Results

| Model | Accuracy |
|---|---|
| Facial | 87.3% |
| Audio | 82.6% |
| Physiological | 91.2% |
| Fusion | 94.7% |



