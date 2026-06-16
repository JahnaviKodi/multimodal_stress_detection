import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np

INPUT_CSV = r"data/processed/physiological/wesad_features.csv"
MODEL_PATH = r"models/physiological_model.h5"
SAVE_PATH = r"results/physiological/confusion_matrix.png"

df = pd.read_csv(INPUT_CSV)

drop_columns = ["subject"]

def map_stress(x):
    if x < 2:
        return "baseline"
    elif x < 3:
        return "stress"
    else:
        return "amusement"

df["target"] = df["label_mean"].apply(map_stress)

for col in ["label_mean", "label_std", "label_min", "label_max"]:
    if col in df.columns:
        drop_columns.append(col)

X = df.drop(columns=[c for c in drop_columns if c in df.columns] + ["target"], errors="ignore")
y = df["target"]

X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
    X_scaled, y_categorical, y_encoded, test_size=0.2, random_state=42
)

model = load_model(MODEL_PATH)
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test_raw, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Physiological Model Confusion Matrix")
plt.tight_layout()

os.makedirs("results/physiological", exist_ok=True)
plt.savefig(SAVE_PATH)
print(f"Saved to {SAVE_PATH}")