import os
import cv2
import pandas as pd

# =========================
# CONFIG
# =========================
IMAGES_DIR = r"C:\Users\Stefano\Documents\Uni\UniLu\JupyterServers\CVIA\data\spark-2024-detection-test\images"
CSV_PATH = r"..\SubmissionOutputs\Detection\yolo11n\detection.csv"

WINDOW_NAME = "YOLO Detection Viewer"
"""Simple script to evaluate detection CSV files by visualizing bounding boxes on images (Notebook version is easier to use)."""
# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

if df.empty:
    raise ValueError("Csv file is empty or not found.")

index = 0
num_images = len(df)

# =========================
# MAIN LOOP
# =========================
while True:
    row = df.iloc[index]

    filename = row["filename"]
    class_name = row["class"]
    bbox = eval(row["bbox"])  # (x_min, y_min, x_max, y_max)

    image_path = os.path.join(IMAGES_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Image not found: {filename}")
        continue

    x_min, y_min, x_max, y_max = bbox

    # draw bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # draw label
    label = f"{class_name}"
    cv2.putText(
        image,
        label,
        (x_min, max(y_min - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # info text
    info = f"{index+1}/{num_images} - {filename}"
    cv2.putText(
        image,
        info,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(WINDOW_NAME, image)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        index = min(index + 1, num_images - 1)
    elif key == ord('p'):
        index = max(index - 1, 0)

cv2.destroyAllWindows()
