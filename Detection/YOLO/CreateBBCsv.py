import os
import glob
import csv
from tqdm import tqdm
import ast
import torch
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
import argparse

import sys
utils_path = Path(__file__).resolve().parent.parent.parent / "Utils"
sys.path.append(str(utils_path))
from Utils.utils import PyTorchSPARKDataset
from utils import *
""""OBSOLETE (Using the bounding boxes from the original dataset is faster and more accurate -> use directly GenrateCroppedDataset.py)
        Script to create detection CSV files using yolo to generate the cropped dataset .  """

class_map = {
    'VenusExpress': 0, 'Cheops': 1, 'LisaPathfinder': 2, 'ObservationSat1': 3,
    'Proba2': 4, 'Proba3': 5, 'Proba3ocs': 6, 'Smart1': 7, 'Soho': 8, 'XMM Newton': 9
}
DATA_DIR = os.path.join(r"C:\Users\Stefano\Documents\Uni\UniLu\CVIA\data" )





def main():
    argparser = argparse.ArgumentParser(description="Generate detection CSV for cropped dataset.")
    argparser.add_argument('--split', type=str, default='train', help='Dataset split to process (train/val/test)')
    argparser.add_argument("--output_dir", type=str, default= os.path.join(Path(__file__).resolve().parent.parent,"BoundingBoxesTest" ), help="Output directory for detection CSV")
    args = argparser.parse_args()
    SPLIT = args.split
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True) # create about dir
    data = PyTorchSPARKDataset(class_map=class_map,root_dir=DATA_DIR,split = SPLIT)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device : " ,device)

    all_image_paths = []
    all_filenames = []
    all_masknames =[]
    all_classnames = []
    all_bboxes = []
    for idx in range(len(data)):
        row = data.labels.iloc[idx]
        sat_name = row['Class']
        img_name = row['Image name']
        mask_name  = row['Mask name']
        bbox = row['Bounding box']
        path = os.path.join(data.root_dir, "images", sat_name, data.split, img_name)
        all_image_paths.append(path)
        all_filenames.append(img_name)
        all_masknames.append(mask_name)
        all_classnames.append(sat_name)
        all_bboxes.append(bbox)

    results = []
    batch_size = 256

    for i in tqdm(range(0, len(all_image_paths), batch_size), desc="Processing"):
        batch_paths = all_image_paths[i : i + batch_size]
        batch_fnames = all_filenames[i : i + batch_size]
        batch_masknames = all_masknames[i : i + batch_size]
        batch_classnames = all_classnames[i : i + batch_size]
        batch_bboxes = all_bboxes[i : i + batch_size]

        for j in range(0,len(batch_fnames)):
            fname = batch_fnames[j]
            maskname = batch_masknames[j]

            class_name = batch_classnames[j]
        
            x_min,y_min,x_max,y_max = ast.literal_eval(batch_bboxes[j])
            results.append([fname, maskname,class_name, f"({x_min}, {y_min}, {x_max}, {y_max})"])

    # Save to CSV
    output_csv = os.path.join(OUTPUT_DIR, f"detection_{SPLIT}.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename","maskname", "class" ,"bbox"])
        writer.writerows(results)

    print(f"Saved detection.csv with {len(results)} entries")


if __name__ == "__main__":
    main()