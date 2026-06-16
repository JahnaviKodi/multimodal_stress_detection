import os
import pickle
import numpy as np
import pandas as pd

INPUT_FOLDER = r"data/raw/emotion/archive (3)/WESAD"
OUTPUT_FOLDER = r"data/processed/physiological"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

all_rows = []

for subject_name in os.listdir(INPUT_FOLDER):
    subject_path = os.path.join(INPUT_FOLDER, subject_name)

    if not os.path.isdir(subject_path):
        continue

    pkl_file = os.path.join(subject_path, f"{subject_name}.pkl")

    if not os.path.exists(pkl_file):
        print(f"Skipping {subject_name}: {pkl_file} not found")
        continue

    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        label = data.get("label", None)
        signal = data.get("signal", {})

        chest = signal.get("chest", {})
        wrist = signal.get("wrist", {})

        row = {
            "subject": subject_name
        }

        if label is not None:
            row["label_mean"] = float(np.mean(label))
            row["label_std"] = float(np.std(label))
            row["label_min"] = float(np.min(label))
            row["label_max"] = float(np.max(label))

        for key, value in chest.items():
            try:
                arr = np.array(value).astype(float).flatten()
                row[f"chest_{key}_mean"] = float(np.mean(arr))
                row[f"chest_{key}_std"] = float(np.std(arr))
                row[f"chest_{key}_min"] = float(np.min(arr))
                row[f"chest_{key}_max"] = float(np.max(arr))
            except Exception:
                pass

        for key, value in wrist.items():
            try:
                arr = np.array(value).astype(float).flatten()
                row[f"wrist_{key}_mean"] = float(np.mean(arr))
                row[f"wrist_{key}_std"] = float(np.std(arr))
                row[f"wrist_{key}_min"] = float(np.min(arr))
                row[f"wrist_{key}_max"] = float(np.max(arr))
            except Exception:
                pass

        all_rows.append(row)
        print(f"Processed WESAD subject: {subject_name}")

    except Exception as e:
        print(f"Error processing {subject_name}: {e}")

df = pd.DataFrame(all_rows)

output_csv = os.path.join(OUTPUT_FOLDER, "wesad_features.csv")
df.to_csv(output_csv, index=False)

print("\nWESAD preprocessing completed.")
print(f"Saved features to: {output_csv}")
print(f"Total subjects processed: {len(df)}")