from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import json
import librosa
import io
import scipy.ndimage
from datetime import datetime
from deepface import DeepFace
import keras
from keras.models import load_model
from scipy.signal import butter, filtfilt, find_peaks

# Set ffmpeg path for pydub
ffmpeg_path = r"C:\Users\jahna\Downloads\ffmpeg-8.1.1-essentials_build\ffmpeg-8.1.1-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path

from pydub import AudioSegment
AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffmpeg    = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(ffmpeg_path, "ffprobe.exe")

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

# Load models
audio_model = load_model(os.path.join(BASE, 'models', 'audio_model.h5'))

# Load audio normalization params
AUDIO_MEAN    = np.load(os.path.join(BASE, 'models', 'audio_mean.npy'))
AUDIO_STD     = np.load(os.path.join(BASE, 'models', 'audio_std.npy'))
AUDIO_CLASSES = np.load(os.path.join(BASE, 'models', 'audio_classes.npy'), allow_pickle=True)

print("All models loaded successfully")
print("Audio input:", audio_model.input_shape, "output:", audio_model.output_shape)
print("Audio classes:", AUDIO_CLASSES)
print("Audio mean first 5:", AUDIO_MEAN[:5])

EMOTION_STRESS = {
    'angry':    98,
    'disgust':  85,
    'fear':     90,
    'happy':    2,
    'sad':      78,
    'surprise': 30,
    'neutral':  45,
}

AUDIO_STRESS_MAP = {
    'angry':    95,
    'disgust':  80,
    'fear':     88,
    'happy':    5,
    'neutral':  40,
    'sad':      75,
    'surprise': 30,
}

STRESS_RECOMMENDATIONS = {
    'HIGH': [
        'Take deep breaths — inhale for 4 seconds, hold for 4, exhale for 4.',
        'Step away from your screen and take a 10-minute walk.',
        'Drink a glass of water and rest your eyes.',
        'Listen to calm music or practice mindfulness.',
        'Consider talking to someone you trust about what is stressing you.',
    ],
    'MEDIUM': [
        'Take a short break and stretch for 5 minutes.',
        'Practice slow, deep breathing for 2 minutes.',
        'Have a healthy snack and stay hydrated.',
        'Prioritise your tasks and focus on one thing at a time.',
    ],
    'LOW': [
        'Great job! You are calm and managing well.',
        'Keep maintaining your healthy routine.',
        'Stay hydrated and keep up the good work.',
    ],
}

def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(i) for i in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

def decode_image(b64_image):
    img_data = base64.b64decode(b64_image.split(',')[1])
    arr      = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def validate_human_face(img):
    gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    return len(faces) > 0

def get_dominant_corrected(emotions):
    happy_score    = emotions.get('happy',    0)
    sad_score      = emotions.get('sad',      0)
    angry_score    = emotions.get('angry',    0)
    fear_score     = emotions.get('fear',     0)
    neutral_score  = emotions.get('neutral',  0)
    disgust_score  = emotions.get('disgust',  0)
    surprise_score = emotions.get('surprise', 0)

    if happy_score > 40:
        return 'happy'
    if happy_score > 20 and fear_score < 40:
        return 'happy'
    if sad_score > 35:
        return 'sad'
    if angry_score > 30:
        return 'angry'
    if disgust_score > 30:
        return 'disgust'
    if neutral_score > 35:
        return 'neutral'
    if fear_score > 50 and happy_score < 15:
        return 'fear'
    if surprise_score > 40:
        return 'surprise'
    return max(emotions, key=emotions.get)

def calculate_facial_stress(dominant, emotions):
    base_stress = sum(
        (emotions.get(e, 0) / 100) * EMOTION_STRESS.get(e, 50)
        for e in EMOTION_STRESS
    )
    if dominant == 'happy':
        return min(base_stress, 18.0)
    elif dominant == 'neutral':
        return max(min(base_stress, 58.0), 38.0)
    elif dominant == 'sad':
        return max(base_stress, 70.0)
    elif dominant == 'angry':
        return max(base_stress, 80.0)
    elif dominant == 'fear':
        return max(base_stress, 72.0)
    elif dominant == 'disgust':
        return max(base_stress, 68.0)
    elif dominant == 'surprise':
        return max(min(base_stress, 48.0), 25.0)
    return base_stress

def analyse_facial(img):
    try:
        result   = DeepFace.analyze(
            img,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='opencv',
            silent=True
        )
        emotions = result[0]['emotion']
        dominant = get_dominant_corrected(emotions)
        stress   = calculate_facial_stress(dominant, emotions)
        print(f"  Facial: {dominant} -> {stress:.1f}% | happy={emotions.get('happy',0):.1f} sad={emotions.get('sad',0):.1f} angry={emotions.get('angry',0):.1f} neutral={emotions.get('neutral',0):.1f}")
        return dominant, round(float(stress), 1)
    except Exception as e:
        print(f"  Facial error: {e}")
        return None, None

def analyse_audio(b64_audio):
    try:
        audio_data = base64.b64decode(b64_audio.split(',')[1])
        y, sr = None, None

        try:
            audio_seg  = AudioSegment.from_file(io.BytesIO(audio_data))
            wav_buffer = io.BytesIO()
            audio_seg.export(wav_buffer, format='wav')
            wav_buffer.seek(0)
            y, sr = librosa.load(wav_buffer, sr=22050, duration=5)
            print(f"  Audio loaded via pydub: {len(y)} samples")
        except Exception as e1:
            print(f"  Audio pydub error: {e1}")
            try:
                y, sr = librosa.load(io.BytesIO(audio_data), sr=22050, duration=5)
                print(f"  Audio loaded via librosa: {len(y)} samples")
            except Exception as e2:
                print(f"  Audio librosa error: {e2}")

        if y is None or len(y) < 1000:
            print("  Audio: too short or empty")
            return None, None

        rms = np.sqrt(np.mean(y**2))
        print(f"  Audio RMS: {rms:.4f}")
        if rms < 0.002:
            print("  Audio: silence detected")
            return None, None

        # Extract MFCC and resize to 67 frames to match training
        mfcc_full = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        print(f"  Raw MFCC shape: {mfcc_full.shape}")

        if mfcc_full.shape[1] != 67:
            scale     = 67 / mfcc_full.shape[1]
            mfcc_full = scipy.ndimage.zoom(mfcc_full, (1, scale))

        mfcc = np.mean(mfcc_full, axis=1)
        print(f"  MFCC after resize: {mfcc_full.shape} mean first 5: {mfcc[:5]}")

        # Normalize
        mfcc_norm  = (mfcc - AUDIO_MEAN) / AUDIO_STD
        mfcc_input = mfcc_norm.reshape(1, -1).astype('float32')

        pred    = audio_model.predict(mfcc_input, verbose=0)[0]
        idx     = int(np.argmax(pred))
        conf    = float(pred[idx])
        emotion = str(AUDIO_CLASSES[idx])

        print(f"  All probs: {[(str(AUDIO_CLASSES[i]), round(float(pred[i])*100,1)) for i in range(len(pred))]}")

        stress = sum(
            float(pred[i]) * AUDIO_STRESS_MAP.get(str(AUDIO_CLASSES[i]), 50)
            for i in range(len(AUDIO_CLASSES))
        )
        stress = round(float(np.clip(stress, 5, 95)), 1)

        print(f"  Audio: {emotion} (conf={conf*100:.1f}%) -> stress={stress:.1f}%")
        return emotion, stress

    except Exception as e:
        print(f"  Audio error: {e}")
        return None, None

def estimate_heart_rate_rppg(frames_b64):
    try:
        face_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        green_signals = []

        for f in frames_b64:
            img   = decode_image(f)
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) == 0:
                continue
            x, y, w, h  = faces[0]
            forehead_y2 = y + int(h * 0.3)
            forehead    = img[y:forehead_y2, x:x+w]
            if forehead.size == 0:
                continue
            green_mean = np.mean(forehead[:, :, 1])
            green_signals.append(green_mean)

        if len(green_signals) < 8:
            print(f"  rPPG: Not enough frames ({len(green_signals)})")
            return None

        fps        = 1.0
        nyq        = fps / 2.0
        low        = 0.7 / nyq
        high       = min(2.5 / nyq, 0.99)

        if low >= high:
            print("  rPPG: Invalid filter range")
            return None

        signal_arr = np.array(green_signals)
        signal_arr = signal_arr - np.mean(signal_arr)

        b, a     = butter(2, [low, high], btype='band')
        filtered = filtfilt(b, a, signal_arr)
        peaks, _ = find_peaks(filtered, distance=int(fps * 0.5) + 1)

        if len(peaks) >= 2:
            peak_intervals = np.diff(peaks) / fps
            avg_interval   = np.mean(peak_intervals)
            heart_rate     = 60.0 / avg_interval
            heart_rate     = max(50, min(150, heart_rate))
            print(f"  rPPG: HR={heart_rate:.1f} bpm from {len(green_signals)} frames")
            return round(float(heart_rate), 1)
        else:
            variance = np.var(filtered)
            if variance > 50:   hr = 95
            elif variance > 20: hr = 80
            else:               hr = 68
            print(f"  rPPG: Fallback HR={hr} bpm")
            return float(hr)

    except Exception as e:
        print(f"  rPPG error: {e}")
        return None

def analyse_physiological(hr):
    try:
        hr_val = float(hr)
        if hr_val <= 60:    stress = 10.0
        elif hr_val <= 70:  stress = 20.0
        elif hr_val <= 80:  stress = 32.0
        elif hr_val <= 90:  stress = 48.0
        elif hr_val <= 100: stress = 62.0
        elif hr_val <= 110: stress = 75.0
        elif hr_val <= 120: stress = 85.0
        else:               stress = 90.0
        print(f"  Physio: HR={hr_val}bpm -> stress={stress}%")
        return stress
    except Exception as e:
        print(f"  Physio error: {e}")
        return 30.0

@app.route('/analyse', methods=['POST'])
def analyse():
    data   = request.json
    frames = data.get('frames',    [])
    hr     = data.get('heartRate', 72)
    audio  = data.get('audio',     None)

    scores         = {}
    valid_frames   = 0
    invalid_frames = 0
    frame_results  = []

    # 1. FACIAL
    for i, f in enumerate(frames):
        try:
            img = decode_image(f)
            if not validate_human_face(img):
                invalid_frames += 1
                continue
            dominant, stress = analyse_facial(img)
            if dominant is None:
                invalid_frames += 1
                continue
            valid_frames += 1
            frame_results.append({'emotion': dominant, 'stress': stress})
        except Exception as e:
            print(f"Frame {i} error: {e}")
            invalid_frames += 1

    facial_available = len(frame_results) > 0
    if facial_available:
        facial_stress    = round(
            sum(r['stress'] for r in frame_results) / len(frame_results), 1)
        emotions_list    = [r['emotion'] for r in frame_results]
        dominant_emotion = max(set(emotions_list), key=emotions_list.count)
        scores['facial'] = facial_stress
    else:
        facial_stress    = None
        dominant_emotion = None
        print("  Facial: not available")

    # 2. AUDIO
    audio_stress  = None
    audio_emotion = None
    if audio:
        audio_emotion, audio_stress = analyse_audio(audio)
        if audio_stress is not None:
            scores['audio'] = audio_stress

    # 3. Check at least one modality
    if not facial_available and audio_stress is None:
        return jsonify({
            'error':   'no_input',
            'message': 'No face or voice detected. Please ensure your face is visible or speak clearly.'
        }), 400

    # 4. PHYSIOLOGICAL with rPPG
    estimated_hr = estimate_heart_rate_rppg(frames) if facial_available else None
    if estimated_hr is not None:
        final_hr  = estimated_hr
        hr_source = 'rPPG (camera)'
    else:
        final_hr  = float(hr)
        hr_source = 'manual slider'

    physio_stress           = analyse_physiological(final_hr)
    scores['physiological'] = physio_stress

    # 5. FUSION
    if facial_available and audio_stress is not None:
        fusion = round(
            facial_stress * 0.50 +
            audio_stress  * 0.30 +
            physio_stress * 0.20, 1)
        modality_count = '3 modalities (face + voice + heart rate)'
    elif facial_available and audio_stress is None:
        fusion = round(
            facial_stress * 0.80 +
            physio_stress * 0.20, 1)
        modality_count = '2 modalities (face + heart rate)'
    elif not facial_available and audio_stress is not None:
        fusion = round(
            audio_stress  * 0.75 +
            physio_stress * 0.25, 1)
        modality_count = '2 modalities (voice + heart rate)'
    else:
        fusion         = physio_stress
        modality_count = '1 modality (heart rate only)'

    label = 'HIGH' if fusion >= 68 else 'MEDIUM' if fusion >= 35 else 'LOW'

    result = {
        'fusion':           fusion,
        'label':            label,
        'breakdown':        scores,
        'dominant_emotion': dominant_emotion,
        'audio_emotion':    audio_emotion,
        'estimated_hr':     final_hr,
        'hr_source':        hr_source,
        'frames_used':      valid_frames,
        'frames_rejected':  invalid_frames,
        'modalities_used':  modality_count,
        'facial_available': facial_available,
        'audio_available':  audio_stress is not None,
        'recommendations':  STRESS_RECOMMENDATIONS[label],
        'timestamp':        datetime.now().isoformat(),
    }

    print(f"\nFinal: fusion={fusion}% label={label} modalities={modality_count}\n")

    log_path = os.path.join(BASE, 'results', 'live_sessions.json')
    try:
        existing = json.load(open(log_path)) if os.path.exists(log_path) else []
        existing.append(convert(result))
        json.dump(existing, open(log_path, 'w'), indent=2)
    except Exception as e:
        print(f"Save error: {e}")

    return jsonify(convert(result))

@app.route('/history', methods=['GET'])
def history():
    log_path = os.path.join(BASE, 'results', 'live_sessions.json')
    try:
        data = json.load(open(log_path)) if os.path.exists(log_path) else []
        return jsonify(data[-20:])
    except:
        return jsonify([])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)