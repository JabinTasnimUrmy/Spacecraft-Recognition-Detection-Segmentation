import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ast import literal_eval
import argparse



TARGET_SIZE = (1024, 1024)

def save_rgb_letterboxed_crops(csv_path, data_root, output_root, split="val"):
    """Generate 1024x1024 RGB crops using letterboxing"""
    # expansion factor is not really necessary(it was used in preliminary experiments where the bounding boxes where predicted by a detection model)
    df = pd.read_csv(csv_path)
    os.makedirs(output_root, exist_ok=True)
    
    print(f"Generating 1024x1024 RGB crops  using letterboxing %)")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        cname = row['class']
        mask_fname = row["maskname"]
        bbox = literal_eval(row['bbox'])
        
        img_path = os.path.join(data_root, "images", cname, split, fname)
        mask_path = os.path.join(data_root, "mask", cname, split, mask_fname)

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(img_path)
            continue

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB") 
        W_img, H_img = img.size
        
        # 1. Expansion 5% (probably not necessay)
        x_min, y_min, x_max, y_max = bbox
        dw = (x_max - x_min)
        dh = (y_max - y_min) 
        
        x_min_new = max(0, int(x_min - dw))
        y_min_new = max(0, int(y_min - dh))
        x_max_new = min(W_img, int(x_max + dw))
        y_max_new = min(H_img, int(y_max + dh))
        
        # 2. Crop
        crop_img = img.crop((x_min_new, y_min_new, x_max_new, y_max_new))
        crop_mask = mask.crop((x_min_new, y_min_new, x_max_new, y_max_new))

        # 3. Letterboxing (
        crop_img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
        # NEAREST to  avoid creating new boarders
        crop_mask.thumbnail(TARGET_SIZE, Image.Resampling.NEAREST)

        # 4. Black canvas
        final_img = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
        final_mask = Image.new("RGB", TARGET_SIZE, (0, 0, 0))

        offset = (
            (TARGET_SIZE[0] - crop_img.size[0]) // 2,
            (TARGET_SIZE[1] - crop_img.size[1]) // 2
        )

        final_img.paste(crop_img, offset)
        final_mask.paste(crop_mask, offset)
        # 5. Saving using the original structure
        out_img_dir = os.path.join(output_root, "images", cname, split)
        out_mask_dir = os.path.join(output_root, "mask", cname, split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        final_img.save(os.path.join(out_img_dir, fname))
        final_mask.save(os.path.join(out_mask_dir, mask_fname))



def save_pixel_perfect_crops(csv_path, data_root, output_root, split="val"):
    df = pd.read_csv(csv_path)
    os.makedirs(output_root, exist_ok=True)
    
    print(f"Generating 1024x1024 crops using pure translation (No Resize)")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        cname = row['class']
        mask_fname = row["maskname"]
        
        # Using Ground Truth Bounding Box directly (no expansion needed)
        bbox = literal_eval(row['bbox'])
        x_min, y_min, x_max, y_max = bbox
        
        img_path = os.path.join(data_root, "images", cname, split, fname)
        mask_path = os.path.join(data_root, "mask", cname, split, mask_fname)

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue # skip missing files

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        # 1. Direct Crop: No expansion, no filters, just raw pixels
        crop_img = img.crop((x_min, y_min, x_max, y_max))
        crop_mask = mask.crop((x_min, y_min, x_max, y_max))
        
        cw, ch = crop_img.size

        # 2. Create Black Canvas: 1024x1024 identity background
        final_img = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
        final_mask = Image.new("RGB", TARGET_SIZE, (0, 0, 0))

        # 3. Deterministic Offset: Using integer division to center the crop
        offset_x = (TARGET_SIZE[0] - cw) // 2 
        offset_y = (TARGET_SIZE[1] - ch) // 2

        # 4. Paste crop onto the black canvas at the calculated offset -> crop always centered
        final_img.paste(crop_img, (offset_x, offset_y))
        final_mask.paste(crop_mask, (offset_x, offset_y))

        # 5. Save output maintaining original structure
        out_img_dir = os.path.join(output_root, "images", cname, split)
        out_mask_dir = os.path.join(output_root, "mask", cname, split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)
        png_fname = fname.rsplit('.', 1)[0] + '.png'
        png_mask_fname = mask_fname.rsplit('.', 1)[0] + '.png'
        final_img.save(os.path.join(out_img_dir, png_fname))
        final_mask.save(os.path.join(out_mask_dir, png_mask_fname))


DATA_DIR = r"C:\Users\Stefano\Documents\Uni\UniLu\CVIA\data"

def save_satellite_size_crops(csv_path, data_root, output_root, split="val"):
    df = pd.read_csv(csv_path)
    os.makedirs(output_root, exist_ok=True)
    
    print(f"Generating raw crops from bounding boxes...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        cname = row['class']
        mask_fname = row["maskname"]
        
        # Using Ground Truth Bounding Box directly
        bbox = literal_eval(row['bbox'])
        x_min, y_min, x_max, y_max = bbox
        
        img_path = os.path.join(data_root, "images", cname, split, fname)
        mask_path = os.path.join(data_root, "mask", cname, split, mask_fname)

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue 

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        # 1. Direct Crop: Extracts only the pixels within the bbox
        crop_img = img.crop((x_min, y_min, x_max, y_max))
        crop_mask = mask.crop((x_min, y_min, x_max, y_max))
        
        # 2. Setup output directories
        out_img_dir = os.path.join(output_root, "images", cname, split)
        out_mask_dir = os.path.join(output_root, "mask", cname, split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)
        
        # 3. Save the cropped versions directly
        
        png_mask_fname = mask_fname.rsplit('.', 1)[0] + '.png'
        
        crop_img.save(os.path.join(out_img_dir, fname))
        crop_mask.save(os.path.join(out_mask_dir, png_mask_fname))


def save_centered_512_crops(csv_path, data_root, output_root, split="val"):
    target_size = 512
    df = pd.read_csv(csv_path)
    os.makedirs(output_root, exist_ok=True)
    
    print(f"Generating centered {target_size}x{target_size} crops...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        cname = row['class']
        mask_fname = row["maskname"]
        
        bbox = literal_eval(row['bbox'])
        x_min, y_min, x_max, y_max = bbox
        
        img_path = os.path.join(data_root, "images", cname, split, fname)
        mask_path = os.path.join(data_root, "mask", cname, split, mask_fname)

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue 

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        # 1. Extract the raw crop
        crop_img = img.crop((x_min, y_min, x_max, y_max))
        crop_mask = mask.crop((x_min, y_min, x_max, y_max))
        
        # 2. Calculate Scaling
        w, h = crop_img.size
        max_side = max(w, h)
        
        if max_side > target_size:
            scale = target_size / max_side
            new_w, new_h = int(w * scale), int(h * scale)
            # Use Resize (Image.Resampling.BILINEAR for img, NEAREST for mask)
            crop_img = crop_img.resize((new_w, new_h), resample=Image.BILINEAR)
            crop_mask = crop_mask.resize((new_w, new_h), resample=Image.NEAREST)
        
        # 3. Calculate Centering Pads
        # After scaling (if any), get the new dimensions
        curr_w, curr_h = crop_img.size
        
        pad_left = (target_size - curr_w) // 2
        pad_top = (target_size - curr_h) // 2
        
        # 4. Create the 512x512 Canvas (Black/Zero background)
        final_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        final_mask = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        
        # Paste the crop into the center
        final_img.paste(crop_img, (pad_left, pad_top))
        final_mask.paste(crop_mask, (pad_left, pad_top))
        
        # 5. Setup output directories and save
        out_img_dir = os.path.join(output_root, "images", cname, split)
        out_mask_dir = os.path.join(output_root, "mask", cname, split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)
        
        png_mask_fname = mask_fname.rsplit('.', 1)[0] + '.png'
        
        final_img.save(os.path.join(out_img_dir, fname))
        final_mask.save(os.path.join(out_mask_dir, png_mask_fname))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train Segformer with custom CLI arguments.")
    parser.add_argument('--split', type=str, default='train', help='Dataset split to process (train/val/test)')
    parser.add_argument("--output_dir", type=str, default="./spark_cropped", help="Output directory for cropped dataset")
    args = parser.parse_args()  

    SPLIT = args.split
    OUTPUT_DIR = args.output_dir
    save_centered_512_crops(
        csv_path=f"../BoundingBoxes/detection_{SPLIT}.csv", 
        data_root=DATA_DIR, 
        output_root=OUTPUT_DIR, 
        split=SPLIT
    )