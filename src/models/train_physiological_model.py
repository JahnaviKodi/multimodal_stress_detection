import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

INPUT_CSV = r"data/processed/physiological/wesad_features.csv"
MODEL_SAVE_PATH = r"models/physiological_model.h5"

df = pd.read_csv(INPUT_CSV)

drop_columns = ["subject"]

if "label_mean" in df.columns:
    def map_stress(x):
        if x < 2:
            return "baseline"
        elif x < 3:
            return "stress"
        else:
            return "amusement"

    df["target"] = df["label_mean"].apply(map_stress)
    drop_columns.append("label_mean")

for col in ["label_std", "label_min", "label_max"]:
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

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=8
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nPhysiological Test Accuracy: {accuracy:.4f}")
print(f"Physiological Test Loss: {loss:.4f}")

os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Physiological model saved to: {MODEL_SAVE_PATH}")
print("Label classes:", list(label_encoder.classes_))