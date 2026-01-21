import os
import glob
import csv
from tqdm import tqdm

import torch
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import argparse


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CURRENT_DIR = Path(__file__).parent.resolve()
MODELS_FOLDER_DIR = CURRENT_DIR.parent.parent / "models" / "detection" 
TEST_IMAGES_DIR = CURRENT_DIR.parent.parent.parent / "data" / "spark-2024-detection-test" / "images"
TEST_IMAGES_NUMBER = 20000 # change this number according to the size of the test set

CONF_THRESHOLD = 0.25 # min confidence for prediction




def get_test_images(folder):
    """
    Collect and return all test image paths, sorted by filename.
    """
    return sorted(glob.glob(os.path.join(folder, "*.jpg")))


def fallback_bbox(width, height):
    """
    Return a reasonable default bounding box if YOLO detects nothing.
    The box is centered and covers roughly 1/4 of the image.
    """
    cx, cy = width // 2, height // 2
    w, h = width // 4, height // 4
    return cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the pretrained YOLO model
    model = YOLO(MODEL_PATH)
    model.to(device)
    model.eval()

    # Mapping from class index to class name
    names = model.names
    print(f"Classes: {names}")

    # Load all test image paths
    image_paths = get_test_images(TEST_IMAGES_DIR)
    print(f"Found {len(image_paths)} test images")

    # Sanity check: only process the first 100 images
    #image_paths = image_paths[:100]

    results = []

    # Run inference image by image
    for img_path in tqdm(image_paths, desc="Running YOLO inference"):
        filename = os.path.basename(img_path)

        # Load image only to get its width and height
        img = Image.open(img_path)
        W, H = img.size

        # Run YOLO prediction on the image
        pred = model.predict(
            source=img_path,
            conf=CONF_THRESHOLD,
            device=device,
            verbose=False
        )[0]

        # If at least one bounding box is detected
        if pred.boxes is not None and len(pred.boxes) > 0:
            boxes = pred.boxes

            # Select the box with the highest confidence
            best_idx = boxes.conf.argmax().item()

            # Extract bounding box coordinates
            x_min, y_min, x_max, y_max = map(
                int, boxes.xyxy[best_idx].tolist()
            )

            # Get class name
            class_name = names[int(boxes.cls[best_idx].item())]

        # If no detection is found, use a fallback box
        else:
            x_min, y_min, x_max, y_max = fallback_bbox(W, H)
            class_name = list(names.values())[0]

        # Store one row for the submission CSV
        results.append([
            filename,
            class_name,
            f"({x_min}, {y_min}, {x_max}, {y_max})"
        ])

    # Write results to detection.csv
    output_csv = os.path.join(OUTPUT_DIR, "detection.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class", "bbox"])
        writer.writerows(results)

    print(f"Saved detection.csv with {len(results)} entries")
    if len(results) != TEST_IMAGES_NUMBER:
        print("Warning: Number of results does not match expected number of test images!")
        exit(1) 


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_dir", type=str, help="path to output directory", required=True)
    argparser.add_argument("--model_path", type=str, help="path to detection model file", required=True)
    args = argparser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    MODEL_FOLDER_DIR = MODELS_FOLDER_DIR / args.model_path
    model = list(MODEL_FOLDER_DIR.rglob("*.pt"))
    if len(model) == 0:
        raise FileNotFoundError(f"No model file found in {MODEL_FOLDER_DIR}")
    elif len(model) > 1:
        raise ValueError(f"Multiple model files found in {MODEL_FOLDER_DIR}: {model}")
    MODEL_PATH = MODEL_FOLDER_DIR / args.model_path / model[0].name
    print("Test images dir:", TEST_IMAGES_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True) # create about dir
    main()