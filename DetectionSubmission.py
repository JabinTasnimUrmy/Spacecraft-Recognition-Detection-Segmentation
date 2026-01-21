import os
import csv
from ultralytics import YOLO
from tqdm import tqdm
import cv2

# =========================
# CONFIG
# =========================
MODEL_PATH = r"models\yolo11n(20Percent).pt"
TEST_DIR = r"C:\Users\Stefano\Documents\Uni\UniLu\JupyterServers\CVIA\data\spark-2024-detection-test\images"
OUTPUT_CSV = r"Results\detection.csv"

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

image_files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

results_rows = []

# =========================
# INFERENCE LOOP
# =========================
for filename in tqdm(image_files, desc="Running detection", unit="img"):
    image_path = os.path.join(TEST_DIR, filename)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    results = model(image_path, verbose=False)[0]

    if len(results.boxes) > 0:
        best_idx = results.boxes.conf.argmax()

        box = results.boxes.xyxy[best_idx].cpu().numpy()
        cls_id = int(results.boxes.cls[best_idx])
        class_name = model.names[cls_id]

        x_min, y_min, x_max, y_max = map(int, box)
    else:
        # Fallback bbox (centro immagine)
        x_min = w // 4
        y_min = h // 4
        x_max = x_min + w // 2
        y_max = y_min + h // 2
        class_name = list(model.names.values())[0]

    # Clamp bbox
    x_min = max(0, min(x_min, w - 1))
    y_min = max(0, min(y_min, h - 1))
    x_max = max(0, min(x_max, w))
    y_max = max(0, min(y_max, h))

    bbox_str = f"({x_min}, {y_min}, {x_max}, {y_max})"

    results_rows.append([filename, class_name, bbox_str])

# =========================
# SAVE CSV (STANDARD)
# =========================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "class", "bbox"])
    writer.writerows(results_rows)

print(f"[OK] Saved {OUTPUT_CSV} with {len(results_rows)} detections")
