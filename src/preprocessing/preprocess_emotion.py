import os
import cv2

INPUT_FOLDER = "data/raw/emotion"
OUTPUT_FOLDER = "data/processed/emotion"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

for file_name in os.listdir(INPUT_FOLDER):
    if not file_name.lower().endswith(valid_extensions):
        continue

    input_path = os.path.join(INPUT_FOLDER, file_name)
    output_path = os.path.join(OUTPUT_FOLDER, file_name)

    image = cv2.imread(input_path)

    if image is None:
        print(f"Skipping unreadable file: {file_name}")
        continue

    image = cv2.resize(image, (48, 48))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(output_path, gray)
    print(f"Processed: {file_name}")

print("Emotion preprocessing completed.")