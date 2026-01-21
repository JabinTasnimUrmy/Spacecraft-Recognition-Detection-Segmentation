import os
import shutil
from ast import literal_eval
from skimage import io
import yaml
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import SPARKDataset
import warnings
import uuid
from pathlib import Path


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent / "data"  #INPUT DIR
YOLO_OUT = Path(__file__).resolve().parent.parent.parent / "data"/"yolo_dataset" #Output dir for YOLO
USE_LINKS = False
SPLITS = ["train", "val"]

#SPARKDataset classes
class_map = {
    'VenusExpress': 0, 'Cheops': 1, 'LisaPathfinder': 2, 'ObservationSat1': 3,
    'Proba2': 4, 'Proba3': 5, 'Proba3ocs': 6, 'Smart1': 7, 'Soho': 8, 'XMM Newton': 9
}


# ---------------------------------------------------------
# GLOBALS FOR WORKERS
# ---------------------------------------------------------
DATASET = None
OUT_DIRS = None


def worker_init(dataset_obj, out_dirs):
    """Initialize global worker copies."""
    global DATASET, OUT_DIRS
    DATASET = dataset_obj
    OUT_DIRS = out_dirs


# ---------------------------------------------------------
# UTILITY: Write YOLO TXT
# ---------------------------------------------------------
def save_yolo_label(label_path, class_id, bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox

    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2
    cy = y_min + h / 2

    cx /= img_w
    cy /= img_h
    w /= img_w
    h /= img_h

    with open(label_path, "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------
# UTILITY: Save mask for YOLO segmentation
# ---------------------------------------------------------
def save_mask(mask, out_path):
    if mask.max() <= 1:
        mask = mask * 255

    io.imsave(out_path, mask.astype(np.uint8))


# ---------------------------------------------------------
# WORKER FUNCTION
# ---------------------------------------------------------
def process_item(args):
    i, labels_path, root_dir, split, class_map, out_dirs = args
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*low contrast image.*",
            category=UserWarning,
            module="skimage"
        )

        # Load the single label row (safe & lightweight)
        row = labels_path.iloc[i]

        sat_name = row["Class"]
        img_name = row["Image name"]
        mask_name = row["Mask name"]
        bbox = literal_eval(row["Bounding box"])

        img_path = f"{root_dir}/images/{sat_name}/{split}/{img_name}"
        mask_path = f"{root_dir}/mask/{sat_name}/{split}/{mask_name}"

        

        class_id = class_map[sat_name]
        img_base = os.path.splitext(img_name)[0]


        rand_prefix = uuid.uuid4().hex[:7]
        # Output paths
        out_img = os.path.join(out_dirs["images"], rand_prefix+"_"+sat_name+"_"+img_name)
        out_mask = os.path.join(out_dirs["masks"], rand_prefix+"_"+sat_name+"_"+img_base + ".png")
        out_lbl  = os.path.join(out_dirs["labels"], rand_prefix+"_"+sat_name+"_"+img_base + ".txt")
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        if(USE_LINKS):
            os.symlink(img_path,out_img)
            
            if mask.max() <= 1: #fallback to heavy method if conversion is needed
                mask = mask * 255
                
                io.imsave(out_mask, mask.astype(np.uint8))
            else:
                os.symlink(mask_path,out_mask)        
        else:
            
            io.imsave(out_img, img.astype(np.uint8))
            if mask.max() <= 1: #fallback to
                mask = mask * 255
            io.imsave(out_mask, mask.astype(np.uint8))
        
        
        

        h, w = img.shape[:2]
        save_yolo_label(out_lbl, class_id, bbox, w, h)

        return True



# ---------------------------------------------------------
# CONVERSION PROCESS (PARALLEL)
# ---------------------------------------------------------
def convert_split(split, workers=8):

    print(f"\nConverting split: {split}")

    # Load labels CSV once
    dataset = SPARKDataset(class_map=class_map, root_dir=ROOT_DIR, split=split)
    labels = dataset.labels

    img_out = os.path.join(YOLO_OUT, "images", split)
    lbl_out = os.path.join(YOLO_OUT, "labels", split)
    mask_out = os.path.join(YOLO_OUT, "masks", split)

    os.makedirs(img_out,  exist_ok=True)
    os.makedirs(lbl_out,  exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    out_dirs = {
        "images": img_out,
        "labels": lbl_out,
        "masks": mask_out
    }

    # Prepare tasks
    tasks = [
        (i, labels, ROOT_DIR, split, class_map, out_dirs)
        for i in range(len(labels))
    ]

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_item, t) for t in tasks]

        for _ in tqdm(
            as_completed(futures),
            total=len(tasks),
            desc=f"Processing {split}"
        ):
            pass

    print(f"Finished split: {split}")



# ---------------------------------------------------------
# DATA.YAML GENERATION
# ---------------------------------------------------------
def write_data_yaml():
    data_yaml = {
        "path": YOLO_OUT,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {v: k for k, v in class_map.items()},
        "task": "detect",
        "segmentation": True
    }

    with open(os.path.join(YOLO_OUT, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    print("Wrote data.yaml")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    print("Converting SPARK dataset → YOLO format")

    if os.path.exists(YOLO_OUT):
        print("Clearing previous output folder...")
        shutil.rmtree(YOLO_OUT)
    if(USE_LINKS):
        print("Using Links")
    os.makedirs(YOLO_OUT, exist_ok=True)

    for split in SPLITS:
        convert_split(split, workers=8)

    write_data_yaml()

    print("Conversion complete.")
