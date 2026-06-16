import os
import numpy as np
import librosa

INPUT_FOLDER = "data/raw/audio/archive (2)"
OUTPUT_FOLDER = "data/processed/audio"
VALID_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

total_processed = 0
total_skipped = 0

for root, dirs, files in os.walk(INPUT_FOLDER):
    for file_name in files:
        if not file_name.lower().endswith(VALID_EXTENSIONS):
            continue

        input_path = os.path.join(root, file_name)

        relative_path = os.path.relpath(root, INPUT_FOLDER)
        output_subfolder = os.path.join(OUTPUT_FOLDER, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        try:
            signal, sample_rate = librosa.load(input_path, sr=22050, duration=5)

            mfcc = librosa.feature.mfcc(
                y=signal,
                sr=sample_rate,
                n_mfcc=40
            )

            output_name = os.path.splitext(file_name)[0] + ".npy"
            output_path = os.path.join(output_subfolder, output_name)

            np.save(output_path, mfcc)
            print(f"Processed: {input_path}")
            total_processed += 1

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            total_skipped += 1

print("\nAudio preprocessing completed.")
print(f"Total processed: {total_processed}")
print(f"Total skipped: {total_skipped}")