import os
import cv2

INPUT_FOLDER = r"data/raw/facial/archive (1)"
OUTPUT_FOLDER = r"data/processed/facial"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

total_processed = 0
total_skipped = 0

for split_name in os.listdir(INPUT_FOLDER):
    split_input_path = os.path.join(INPUT_FOLDER, split_name)

    if not os.path.isdir(split_input_path):
        continue

    split_output_path = os.path.join(OUTPUT_FOLDER, split_name)
    os.makedirs(split_output_path, exist_ok=True)

    for class_name in os.listdir(split_input_path):
        class_input_path = os.path.join(split_input_path, class_name)

        if not os.path.isdir(class_input_path):
            continue

        class_output_path = os.path.join(split_output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        for file_name in os.listdir(class_input_path):
            if not file_name.lower().endswith(VALID_EXTENSIONS):
                continue

            input_path = os.path.join(class_input_path, file_name)
            output_path = os.path.join(class_output_path, file_name)

            image = cv2.imread(input_path)

            if image is None:
                print(f"Skipping unreadable file: {input_path}")
                total_skipped += 1
                continue

            image = cv2.resize(image, (48, 48))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(output_path, gray)
            print(f"Processed: {split_name}/{class_name}/{file_name}")
            total_processed += 1

print("\nFacial preprocessing completed.")
print(f"Total processed: {total_processed}")
print(f"Total skipped: {total_skipped}")