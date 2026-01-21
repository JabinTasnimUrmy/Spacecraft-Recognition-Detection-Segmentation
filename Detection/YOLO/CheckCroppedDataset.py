import json

import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
from tqdm import tqdm
utils_path = Path(__file__).resolve().parent.parent.parent / "Utils"
sys.path.append(str(utils_path))
from utils import *

import argparse

def fast_check_file_existence(dataset):
    missing_files = []
    
    print(f"Starting fast existence check for {len(dataset)} samples...")
    
    # Iterate through the dataset labels directly -> faster than using DataLoader
    for idx, row in tqdm(dataset.labels.iterrows(), total=len(dataset.labels)):
        sat_name = row['Class']
        img_name = row['Image name']
        mask_name = row['Mask name']
        
       
        image_path = os.path.join(dataset.root_dir, 'images', sat_name, dataset.split, img_name) #get the full image path
        mask_path = os.path.join(dataset.root_dir, 'mask', sat_name, dataset.split, mask_name) #get the full mask path
        
        # Check image existence
        if not os.path.exists(image_path):
            missing_files.append({"index": idx, "type": "IMAGE", "path": image_path}) # if not exist append to the missing files list
            
        # Check mask existence
        if not os.path.exists(mask_path):
            missing_files.append({"index": idx, "type": "MASK", "path": mask_path})

    return missing_files



if __name__ == "__main__":
    # CUSTOM UTILS
    args = argparse.ArgumentParser()
    args.add_argument('--split', type=str, default='train', help='Dataset split to process (train/val/test)')
    args.add_argument("--data_dir", type=str, default=Path(__file__).resolve().parent / "spark_cropped_pp", help="Path to the dataset to be checked directory")
    parsed_args = args.parse_args()
    CROPPED_DATASET_DIR = Path(parsed_args.data_dir)
    SPLIT = parsed_args.split
    
    
    ORIGINAL_DATASET_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" 
    CURRENT_DIR=Path(__file__).resolve().parent


    class_map = {
        'VenusExpress': 0, 'Cheops': 1, 'LisaPathfinder': 2, 'ObservationSat1': 3,
        'Proba2': 4, 'Proba3': 5, 'Proba3ocs': 6, 'Smart1': 7, 'Soho': 8, 'XMM Newton': 9
    }
    cropped_dataset = PyTorchSPARKDataset(class_map=class_map,split=SPLIT,root_dir=CROPPED_DATASET_DIR)
    original_dataset = PyTorchSPARKDataset(class_map=class_map,split=SPLIT,root_dir=ORIGINAL_DATASET_DIR)
    # Final Summary Report
    missing_list_cropped = fast_check_file_existence(cropped_dataset) # check on the cropped dataset

    if not missing_list_cropped:
        print("\n✅ Success: All files exist on disk.")
    else:
        print(f"\n❌ Found {len(missing_list_cropped)} missing files.")
        with open(os.path.join(CURRENT_DIR,f'missing_files_{SPLIT}.json'), 'w') as f:
            json.dump(missing_list_cropped, f, indent=4)

        print("Saved to missing_files.json")
        



    print("#"*30)
    print("Check on the original dataset")
    missing_list = fast_check_file_existence(original_dataset) # check on the original dataset -> to ensure that missing files are not due to missing files in the original dataset

    if not missing_list:
        print("\n✅ Success: All files exist on disk.")
    else:
        print(f"\n❌ Found {len(missing_list)} missing files.")