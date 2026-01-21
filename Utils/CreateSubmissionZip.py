import zipfile
import os
from pathlib import Path
from tqdm import tqdm
import argparse


argparser = argparse.ArgumentParser(description="Create submission zip file.")
argparser.add_argument('--segmentation_model', type=str, default="segformer_cropped", required=False, help="Segmentation model name.")
argparser.add_argument('--detection_model', type=str, default="yolo11s", help="Detection model name.")
argparser.add_argument('--output_zip', type=str, default="submission_test.zip", help="Output zip file name.")
argparser.add_argument("--submission_dir", type=str, default="SubmissionOutputs", help="Base directory for submission outputs.")
args = argparser.parse_args()
SEGMENTATION_MODEL=args.segmentation_model
DETECTION_MODEL = args.detection_model
SUBMISSION_DIR = args.submission_dir
OUTPUT_ZIP = args.output_zip
# Define base paths
base_dir = Path(__file__).parent.parent
detection_path = base_dir / SUBMISSION_DIR / "Detection" / DETECTION_MODEL
segmentation_path = base_dir / SUBMISSION_DIR / "Segmentation" / SEGMENTATION_MODEL
zip_path = base_dir / OUTPUT_ZIP

# Collect all files first
print("Collecting files...")
all_files = []
for folder_path in [detection_path, segmentation_path]:
    if folder_path.exists():
        print(f"  ✓ Processing: {folder_path}")
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                all_files.append(file_path)
    else:
        print(f"  ✗ Folder not  found: {folder_path}")

print(f"\nCreating zip with {len(all_files)} file...")
# Create zip file
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in tqdm(all_files, desc="Adding file"):
        arcname = file_path.name
        zipf.write(file_path, arcname)

print(f"\nSubmission zip created: {zip_path}")