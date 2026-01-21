import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from segmentation_models_pytorch.encoders import get_encoder
import segmentation_models_pytorch as smp
from tqdm import tqdm
from pathlib import Path
import argparse
# --- CONFIGURATION ---
# Path to the test images folder
CURRENT_DIR = Path(__file__).parent.resolve()
TEST_IMG_DIR = os.path.join(CURRENT_DIR.parent.parent.parent,"data", "spark-2024-segmentation-test", "stream-1-test")
# Where to save the PNG masks
RAW_OUTPUT_DIR = r"..\..\predicted_masks\segformer-test" 
# Model weights



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class SPARKTestInference(Dataset):
    def __init__(self, img_dir, target_size=(512, 512)):
        self.img_dir = img_dir
        self.target_size = target_size
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), img_name

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_path', type=str, required=True, help='Path to the segmentation model folder (not the .pth file)')
    argparser.add_argument("--output_dir", type=str, help="Directory to save predicted masks", required=True)
    args = argparser.parse_args()

    MODEL_DIR = Path(args.model_path)  
    RAW_OUTPUT_DIR = Path(args.output_dir)

    model = list(MODEL_DIR.rglob("*.pth"))
    if len(model) == 0:
        raise FileNotFoundError(f"No model file found in {MODEL_DIR}")
    elif len(model) > 1:
        raise ValueError(f"Multiple model files found in {MODEL_DIR}: {model}")
    MODEL_PATH = MODEL_DIR / model[0].name
    print(f"Predictions will be saved to: {RAW_OUTPUT_DIR}")

    test_ds = SPARKTestInference(TEST_IMG_DIR)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # 1. LOAD MODEL
    model = smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=3).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. RUN INFERENCE
    print("Generating masks...")
    with torch.no_grad():
        for imgs, filenames in tqdm(test_loader):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            # Argmax to get classes (0, 1, 2)
            probs = torch.softmax(outputs, dim=1).cpu().numpy() # [Batch, 512, 512]

            for i in range(len(filenames)):
                # Convert class indices to RGB (mapping used in training)
                # Class 0: Black [0,0,0], Class 1: Red [255,0,0], Class 2: Blue [0,0,255]
                prob_map_512 = np.transpose(probs[i], (1, 2, 0))
                upscale = cv2.resize(prob_map_512, (1024, 1024), interpolation=cv2.INTER_LINEAR) # Upscale to 1024x1024 using bilinear to achieve smoother edges
                preds = np.argmax(upscale, axis=-1).astype(np.uint8)

                mask_rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
                mask_rgb[preds == 1] = [255, 0, 0]
                mask_rgb[preds == 2] = [0, 0, 255]
       
     
                
                # Naming convention: test_00000_img.jpg -> test_00000_layer.png
                mask_filename = filenames[i].replace("_img.jpg", "_layer.png")
                save_path = os.path.join(RAW_OUTPUT_DIR, mask_filename)
                
                # Save as PNG (RGB)
                # OpenCV uses BGR by default, convert  using cv2.imwrite
                cv2.imwrite(save_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

    print(f"\nSuccess! Now you can run:")
    print(f"python PrepareImagesForSubmission.py --input_dir {str(RAW_OUTPUT_DIR)[3:]} --output_dir ../SubmissionOutputs/Segmentation/Segformer")
