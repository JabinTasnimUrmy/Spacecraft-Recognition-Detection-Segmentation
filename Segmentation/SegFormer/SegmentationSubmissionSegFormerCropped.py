import os
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ast import literal_eval
import segmentation_models_pytorch as smp
from tqdm import tqdm
import argparse 
from pathlib import Path
import yaml
# --- CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.resolve()
TEST_IMG_DIR = os.path.join(CURRENT_DIR.parent.parent.parent,"data", "spark-2024-segmentation-test", "stream-1-test")
TEST_CSV_PATH = r"SubmissionOutputs\Detection\yolo11s\detection_task2.csv" # Il CSV con le BBox del test set
RAW_OUTPUT_DIR = r"..\..\predicted_masks\segformer_cropped" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (512, 512) # Input size for Segformer
ORIG_RES = (1024, 1024)   # Original image resolution
EXPANSION_FACTOR = 0.05


class SPARKTestCroppedInference(Dataset):
    def __init__(self, img_dir, csv_path, target_size=(512, 512),pixel_perfect=False,satellite_crop=False):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.target_size = target_size
        self.pixel_perfect = pixel_perfect
        self.satellite_crop = satellite_crop
        
        # : Resize (letterbox) + ToTensor
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        bbox = literal_eval(row['bbox'])
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        W_orig, H_orig = image.size
        x_min, y_min, x_max, y_max = bbox

        if(not self.pixel_perfect):
            #WORKFLOW WITH EXPANSION + LETTERBOXING + RESIZING TO 512x512 (NOT PIXEL PERFECT)
            # 1. Expansion BBox 5% (to compensate  possible model inaccuracies)
           
            dw, dh = (x_max - x_min) * EXPANSION_FACTOR, (y_max - y_min) * EXPANSION_FACTOR
            x1, y1 = max(0, int(x_min - dw)), max(0, int(y_min - dh))
            x2, y2 = min(W_orig, int(x_max + dw)), min(H_orig, int(y_max + dh))
            
            # 2. Crop e Letterboxing
            crop = image.crop((x1, y1, x2, y2))
            # thumbnail keep   aspect ratio
            crop.thumbnail(self.target_size, Image.Resampling.LANCZOS)
            
            final_img = Image.new("RGB", self.target_size, (0, 0, 0))
            offset = ((self.target_size[0] - crop.size[0]) // 2, (self.target_size[1] - crop.size[1]) // 2)
            final_img.paste(crop, offset)

            return self.transform(final_img), img_name, np.array([x1, y1, x2, y2])
        elif (self.satellite_crop):

            # 1. Extract the raw crop
            crop_img = image.crop((x_min, y_min, x_max, y_max))
        
            # 2. Calculate Scaling
            w, h = crop_img.size
            max_side = max(w, h)
            
            if max_side > 512:
                scale = 512 / max_side
                new_w, new_h = int(w * scale), int(h * scale)

                crop_img = crop_img.resize((new_w, new_h), resample=Image.BILINEAR)
            
            # 3. Calculate Centering Pads
            # After scaling (if any), get the new dimensions
            curr_w, curr_h = crop_img.size
            
            pad_left = (512 - curr_w) // 2
            pad_top = (512 - curr_h) // 2
            
            # 4. Create the 512x512 Canvas (Black/Zero background)
            final_img = Image.new("RGB", (512, 512), (0, 0, 0))
            final_mask = Image.new("RGB", (512, 512), (0, 0, 0))
            
            # Paste the crop into the center
            final_img.paste(crop_img, (pad_left, pad_top))

            return self.transform(final_img), img_name, np.array([x_min, y_min, x_max, y_max])
        else:
            crop = image.crop((x_min, y_min, x_max, y_max))
            cw, ch = crop.size
            canvas_size = ORIG_RES
            # 2. Create the 1024x1024 intermediate black canvas
            canvas_1024 = Image.new("RGB", canvas_size, (0, 0, 0))

            # 3. Calculate deterministic offset for centering on 1024
            offset_x = (canvas_size[0] - cw) // 2
            offset_y = (canvas_size[1] - ch) // 2

            # 4. Paste crop into 1024 canvas (1:1 pixel ratio here)
            canvas_1024.paste(crop, (offset_x, offset_y))

            # 5. Global resize to 512x512 for Segformer
            # This scales everything (satellite + canvas) by exactly 0.5x
            input_tensor = self.transform(canvas_1024) 

            # Return the tensor and the bbox for the unshift logic
            return input_tensor, img_name, np.array([x_min, y_min, x_max, y_max])

def unshift_mask_rgb(pred_mask, bbox_expanded, target_canvas_size=(512, 512), final_res=(1024, 1024)):
    """
   Unshifts the prediction mask from the 1024x1024 canvas back to the original coordinates.
    """
    x1, y1, x2, y2 = bbox_expanded
    crop_w, crop_h = x2 - x1, y2 - y1

    # Inversione scaling thumbnail
    scale = min(target_canvas_size[0] / crop_w, target_canvas_size[1] / crop_h)
    if scale > 1.0: scale = 1.0
    
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    offset_x = (target_canvas_size[0] - new_w) // 2
    offset_y = (target_canvas_size[1] - new_h) // 2

    # Estrazione area utile dalla predizione 512x512
    useful_mask = pred_mask[offset_y : offset_y + new_h, offset_x : offset_x + new_w]

    # Resize alla dimensione del crop originale
    mask_resized = cv2.resize(useful_mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    preds = np.argmax(mask_resized, axis=-1).astype(np.uint8)
    mask_rgb_1024 = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    mask_rgb_1024[preds == 1] = [255, 0, 0] # Body
    mask_rgb_1024[preds == 2] = [0, 0, 255] # Panels

    # Posizionamento nel frame originale 1024x1024
    full_mask_rgb = np.zeros((*final_res, 3), dtype=np.uint8)
    full_mask_rgb[y1:y2, x1:x2] = mask_rgb_1024
    
    return full_mask_rgb
def unshift_mask_rgb_pp(pred_mask, bbox, canvas_size=(512, 512), final_res=(1024, 1024)):
    """
    Reconstructs the full-resolution mask by inverting the 1024->512 pipeline.
    """
    # 1. Upscale the 512x512 prediction back to the 1024x1024 intermediate canvas
    # Use INTER_NEAREST to maintain discrete class colors [255, 0, 0] etc.
    if pred_mask.shape[0] != 1024 or pred_mask.shape[1] != 1024:
        mask_1024 = cv2.resize(pred_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    else:
        mask_1024 = pred_mask

    x1, y1, x2, y2 = bbox
    crop_w, crop_h = x2 - x1, y2 - y1

    # 2. Calculate the SAME offsets used in the Dataset (must be identical)
    offset_x = (final_res[0] - crop_w) // 2
    offset_y = (final_res[1] - crop_h) // 2

    # 3. Extract the 'Useful Area' using slicing
    # Since mask_1024 is back to 1:1 ratio, we slice exactly crop_w x crop_h
    useful_area = mask_1024[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]

    # 4. Initialize the final 1024x1024 full frame
    full_frame = np.zeros((*final_res, 3), dtype=np.uint8)

    # 5. Place the satellite back into its original global coordinates
    full_frame[y1:y2, x1:x2] = useful_area

    return full_frame




def unshift_mask_satellite_crop(pred_mask, bbox_gt, target_size=512,original_res=(1024, 1024),):
    
    x1, y1, x2, y2 = bbox_gt
    orig_w, orig_h = x2 - x1, y2 - y1
    max_side = max(orig_w, orig_h)

    # 2. Determine Scale and Scaled Dimensions
    scale = 1.0
    if max_side > target_size:
        scale = target_size / max_side
    
    curr_w, curr_h = int(orig_w * scale), int(orig_h * scale)

    # 3. Calculate identical offset used during generation
    offset_x = (target_size - curr_w) // 2
    offset_y = (target_size - curr_h) // 2

    # 4. Extract the "Useful Area" from the 512x512 canvas
    useful_area = pred_mask[offset_y : offset_y + curr_h, offset_x : offset_x + curr_w]

    # 5. Reverse Scaling (if object was > 512)
    if scale != 1.0:
        useful_area = cv2.resize(useful_area, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) #mask so we use NEAREST

    # 6. Reconstruct the original frame
    if useful_area.ndim == 3:
        full_res_frame = np.zeros((*original_res, useful_area.shape[2]), dtype=useful_area.dtype)
    else:
        full_res_frame = np.zeros(original_res, dtype=useful_area.dtype)

    # 7. Final Shift: Paste back to original coordinates
    full_res_frame[y1:y2, x1:x2] = useful_area
    preds = np.argmax(full_res_frame, axis=-1).astype(np.uint8)
    mask_rgb_1024 = np.zeros((original_res[0],original_res[0], 3), dtype=np.uint8)
    mask_rgb_1024[preds == 1] = [255, 0, 0] # Body
    mask_rgb_1024[preds == 2] = [0, 0, 255] # Panels
    
    return mask_rgb_1024
if __name__ == '__main__':
    #Arguments Definition
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--test_img_dir', type=str, default=TEST_IMG_DIR, help='test image directory')
    argparser.add_argument('--test_csv_path', type=str, default=TEST_CSV_PATH, help='test csv path with bboxes')
    argparser.add_argument("--pixel_perfect", action='store_true', help="Use pixel-perfect resizing and cropping")
    argparser.add_argument('--satellite_crop', action='store_true', help='Crop to satellite fixed size')
    argparser.add_argument('--model_path', type=str, required=True, help='Path to the segmentation model folder (not the .pth file)')
    argparser.add_argument("--output_dir", type=str, help="Directory to save predicted masks", required=True)
    args = argparser.parse_args()

    MODEL_DIR = Path(args.model_path)  
    RAW_OUTPUT_DIR = Path(args.output_dir)
    
    os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
    encoder_name="mit_b0"
    model = list(MODEL_DIR.rglob("*.pth"))
    if len(model) == 0:
        raise FileNotFoundError(f"No model file found in {MODEL_DIR}")
    elif len(model) > 1:
        raise ValueError(f"Multiple model files found in {MODEL_DIR}: {model}")
    args_path = MODEL_DIR / "args.yaml"
    if os.path.exists(args_path):   
        with open(args_path, 'r') as f:
            saved_args = yaml.safe_load(f)
            if "image_size" in saved_args:
                print(f"Overriding target size with saved model argument: {saved_args['image_size']}x{saved_args['image_size']}")
                img_size = saved_args['image_size']
                TARGET_SIZE = [img_size, img_size]
            if "encoder" in saved_args: #Optional field -> initially i didnt include it, so for backward compatibility i check its existence
                encoder_name = saved_args["encoder"]
    MODEL_PATH = MODEL_DIR / model[0].name
    #Parsing Arguments
    
    TEST_IMG_DIR = args.test_img_dir
    TEST_CSV_PATH = args.test_csv_path

    test_ds = SPARKTestCroppedInference(TEST_IMG_DIR, TEST_CSV_PATH, target_size=TARGET_SIZE,pixel_perfect=args.pixel_perfect,satellite_crop=args.satellite_crop)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Model Loading
    model = smp.Segformer(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=3).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Generating Reconstructed Full-Res Masks...")
    with torch.no_grad():
        for imgs, filenames, bboxes in tqdm(test_loader):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy() # [Batch, 512, 512]

            for i in range(len(filenames)):
                # 1. Create the mask
                prob_map_512 = np.transpose(probs[i], (1, 2, 0))
                # 2. Unshift
                if(args.pixel_perfect):
                    # Pixel-perfect reconstruction (resizing handled outside the unshift function)
                    upscale = cv2.resize(prob_map_512, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    preds = np.argmax(upscale, axis=-1).astype(np.uint8)
                    mask_rgb_1024 = np.zeros((ORIG_RES[0], ORIG_RES[1], 3), dtype=np.uint8)
                    mask_rgb_1024[preds == 1] = [255, 0, 0] # Body
                    mask_rgb_1024[preds == 2] = [0, 0, 255] # Panels
                    final_mask_1024 = unshift_mask_rgb_pp(mask_rgb_1024, bboxes[i].numpy(), ORIG_RES, ORIG_RES)
                elif(args.satellite_crop):
                    # Satellite-crop reconstruction
                    final_mask_1024 = unshift_mask_satellite_crop(prob_map_512, bboxes[i].numpy(), TARGET_SIZE[0], ORIG_RES) 
                else:
                    final_mask_1024 = unshift_mask_rgb(prob_map_512, bboxes[i].numpy(), TARGET_SIZE, ORIG_RES)#gives result in 1024x1024
       

                # 3. Save the final mask
                mask_filename = filenames[i].replace("_img.jpg", "_layer.png")
                save_path = os.path.join(RAW_OUTPUT_DIR, mask_filename)
                
                # RGB -> BGR per OpenCV
                cv2.imwrite(save_path, cv2.cvtColor(final_mask_1024, cv2.COLOR_RGB2BGR))
    print(f"\nSuccess! Now you can run:")
    print(f"python PrepareImagesForSubmission.py --input_dir {str(RAW_OUTPUT_DIR)[3:]} --output_dir ../SubmissionOutputs/Segmentation/SegformerCropped")