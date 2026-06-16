def map_emotion_to_stress(emotion_label):
    emotion_label = emotion_label.lower()

    stress_classes = ["angry", "fear", "disgust", "sad", "stress"]
    non_stress_classes = ["happy", "neutral", "baseline", "amusement"]

    if emotion_label in stress_classes:
        return 1
    elif emotion_label in non_stress_classes:
        return 0
    elif emotion_label == "surprise":
        return 1
    else:
        return 0