import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

INPUT_FOLDER = r"data/processed/audio"
MODEL_SAVE_PATH = r"models/audio_model.h5"

X = []
y = []

for root, dirs, files in os.walk(INPUT_FOLDER):
    for file_name in files:
        if not file_name.endswith(".npy"):
            continue

        file_path = os.path.join(root, file_name)
        features = np.load(file_path)

        class_name = os.path.basename(root)
        features_mean = np.mean(features, axis=1)

        X.append(features_mean)
        y.append(class_name)

X = np.array(X)
y = np.array(y)

print("Total audio samples:", len(X))
print("Audio feature shape:", X.shape)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nAudio Test Accuracy: {accuracy:.4f}")
print(f"Audio Test Loss: {loss:.4f}")

os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Audio model saved to: {MODEL_SAVE_PATH}")
print("Label classes:", list(label_encoder.classes_))