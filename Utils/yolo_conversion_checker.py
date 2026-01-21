import os
from utils import SPARKDataset
ROOT_DIR = r"c:\Users\Stefano\Documents\Uni\UniLu\JupyterServers\CVIA\data"        # Original spark dataset
YOLO_OUT = "../yolo_dataset"  # genrated dataset YOLO 



SPLITS = ["train", "val"]

if __name__ == "__main__":
    for split in SPLITS:
        spark_ds = SPARKDataset(class_map={}, root_dir=ROOT_DIR, split=split)
        n_spark = len(spark_ds)
        yolo_img_dir = os.path.join(YOLO_OUT, "images", split)
        yolo_masks_dir = os.path.join(YOLO_OUT, "masks", split)
        n_yolo_images = len([f for f in os.listdir(yolo_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]) # lowercasing for robustness
        n_yolo_masks = len([f for f in os.listdir(yolo_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]) # lowercasing for robustness
        print(f"\n=== {split.upper()} ===")
        print("SPARK dataset (CSV count):", n_spark)
        print("YOLO final images:\t", n_yolo_images)
        print("YOLO final masks: \t", n_yolo_masks)
        
        print("\n---Results---")
        if n_spark == n_yolo_images:
            print("Images: OK")
        else:
            print("NOT ENOUGH IMAGES")
        if n_spark == n_yolo_masks:
            print("Maks: OK")
        else:
            print("NOT ENOUGH MASKS")
