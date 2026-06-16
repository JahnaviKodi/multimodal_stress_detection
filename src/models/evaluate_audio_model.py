import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

INPUT_FOLDER = r"data/processed/audio"
MODEL_PATH = r"models/audio_model.h5"

X = []
y = []

for root, dirs, files in os.walk(INPUT_FOLDER):
    for file_name in files:
        if file_name.endswith(".npy"):
            file_path = os.path.join(root, file_name)
            features = np.load(file_path)
            features_mean = np.mean(features, axis=1)
            class_name = os.path.basename(root)

            X.append(features_mean)
            y.append(class_name)

X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
    X, y_categorical, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = load_model(MODEL_PATH)

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

print("Classification Report:\n")
print(classification_report(y_test_raw, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test_raw, y_pred))