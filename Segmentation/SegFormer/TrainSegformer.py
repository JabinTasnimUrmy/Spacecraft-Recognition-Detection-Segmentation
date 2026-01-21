import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast

import argparse
import yaml

import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path

# CUSTOM UTILS
CURRENT_DIR = Path(__file__).resolve().parent
utils_path = CURRENT_DIR.parent.parent / "Utils"
sys.path.append(str(utils_path))
from utils import *

segmentation_utils_path = CURRENT_DIR.parent / "Utils"
sys.path.append(str(segmentation_utils_path))

# GLOBAL VARS
ROOT_DIR = CURRENT_DIR.parent.parent.parent / "data" 
BASE_OUTPUT_DIR = os.path.join("..", "..", "models", "segmentation", "segformer")

class_map = {
    'VenusExpress': 0, 'Cheops': 1, 'LisaPathfinder': 2, 'ObservationSat1': 3,
    'Proba2': 4, 'Proba3': 5, 'Proba3ocs': 6, 'Smart1': 7, 'Soho': 8, 'XMM Newton': 9
}

def train(fraction, num_epochs,batch_size, patience, use_custom_loss, output_dir,l_r,workers_train,workers_val,encoder_name,target_size=(512,512),use_new_custom_loss=False):
    # Ensure the specific output subdirectory exists
    os.makedirs(output_dir, exist_ok=True)
    if(use_custom_loss and use_new_custom_loss):
        raise ValueError("Cannot use both custom_loss and custom_loss_new flags simultaneously.")
    # MODEL INSTANTIATION
    model = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3, 
        classes=3 # 3 classes: background, satellite body, solar panel
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model to device: {device}")
    model.to(device)
    print("Using image size:", target_size)
    # DATASET LOADING
    train_dataset_full = ResizedSPARKDataset(
        class_map=class_map, 
        root_dir=ROOT_DIR, 
        split='train', 
        target_size=target_size
    )
    
    # Create training subset based on fraction
    train_indices = np.random.choice(len(train_dataset_full), int(fraction * len(train_dataset_full)), replace=False)
    train_subset = Subset(train_dataset_full, train_indices)

    val_dataset_full = ResizedSPARKDataset(
        class_map=class_map,
        root_dir=ROOT_DIR,
        target_size=target_size,
        split='val'
    )
    
    # Create validation subset based on fraction
    val_indices = np.random.choice(len(val_dataset_full), int(fraction * len(val_dataset_full)), replace=False)
    val_subset = Subset(val_dataset_full, val_indices)

    # DATALOADER DEFINITION
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers_train,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        num_workers=workers_val,
        pin_memory=True,
        persistent_workers=True
    )

    # LOSS, OPTIMIZER, SCALER
    weights = torch.tensor([0.2, 1.0, 0.8]).to(device) # Class weights for CE Loss (Background, Body, Panels)
    dice_loss_fn = smp.losses.DiceLoss(mode='multiclass')
    ce_loss_fn = nn.CrossEntropyLoss(weight=weights)
    focal_loss_fn = smp.losses.FocalLoss(mode='multiclass')
   
    optimizer = optim.AdamW(model.parameters(), lr=l_r,weight_decay=1e-2)
    scaler = GradScaler() # For Mixed Precision -> faster training on modern GPUs

    # TRACKING BEST MODEL
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"Training started: Custom Loss={use_custom_loss}, Output={output_dir}")

    for epoch in range(num_epochs):
        if epochs_without_improvement > patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for batch in pbar:
            imgs = batch['img'].to(device, non_blocking=True)
            masks_rgb = batch['mask'].to(device, non_blocking=True)

            red_ch = masks_rgb[:, 0, :, :]
            blue_ch = masks_rgb[:, 2, :, :]

            # 2. Initialize the class index mask with zeros (Background)
            # Shape: [Batch, Height, Width]
            masks = torch.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3]), 
                                dtype=torch.long, device=device)

            # 3. Define a threshold to ignore noise (anti-aliasing or compression artifacts)
            # Use 0.5 if masks are normalized [0-1], or 127 if they are [0-255]
            thresh = 0.5 if masks_rgb.max() <= 1.0 else 127

            # 4. Assign Class 1 (Body)
            # Criteria: Red is stronger than (or equal to) Blue AND Red is above threshold
            masks[(red_ch >= blue_ch) & (red_ch > thresh)] = 1

            # 5. Assign Class 2 (Panels)
            # Criteria: Blue is strictly stronger than Red AND Blue is above threshold
    
            masks[(blue_ch > red_ch) & (blue_ch > thresh)] = 2 #> ensures mutual exclusivity with Body
            # 2. Init mask
            optimizer.zero_grad(set_to_none=True)

            # Forward with Mixed Precision
            with autocast(device_type='cuda'):
                # Base loss is always CE
                if not use_new_custom_loss:
                    loss = ce_loss_fn(outputs := model(imgs), masks)
                    # Add Dice loss if custom_loss flag is set
                    if use_custom_loss:
                        loss += dice_loss_fn(outputs, masks)
                        loss += focal_loss_fn(outputs, masks)
                else:
                    outputs = model(imgs)
                    loss = dice_loss_fn(outputs, masks)
                    loss += focal_loss_fn(outputs, masks)

            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_epoch_loss = running_loss / len(train_loader.dataset)
        
        # VALIDATION PHASE
        model.eval() # Set model to evaluation mode -> no weights update
        val_running_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Validation {epoch+1}", unit="batch", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                imgs = batch['img'].to(device, non_blocking=True)
                masks_rgb = batch['mask'].to(device, non_blocking=True)
                masks = torch.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3]), 
                                    dtype=torch.long, device=device)
                

                red_ch = masks_rgb[:, 0, :, :]
                blue_ch = masks_rgb[:, 2, :, :]
                # 3. Define a threshold to ignore noise (anti-aliasing or compression artifacts)
                # Use 0.5 if masks are normalized [0-1], or 127 if they are [0-255]
                thresh = 0.5 if masks_rgb.max() <= 1.0 else 127

                # 4. Assign Class 1 (Body)
                # Criteria: Red is stronger than (or equal to) Blue AND Red is above threshold
                masks[(red_ch >= blue_ch) & (red_ch > thresh)] = 1

                # 5. Assign Class 2 (Panels)
                # Criteria: Blue is strictly stronger than Red AND Blue is above threshold
                masks[(blue_ch > red_ch) & (blue_ch > thresh)] = 2
                with autocast(device_type='cuda'):
                    if not use_new_custom_loss:
                        loss = ce_loss_fn(outputs := model(imgs), masks)
                        # Add Dice loss if custom_loss flag is set
                        if use_custom_loss:
                            loss += dice_loss_fn(outputs, masks)
                            loss += focal_loss_fn(outputs, masks)
                    else:
                        outputs = model(imgs)
                        loss = dice_loss_fn(outputs, masks)
                        loss += focal_loss_fn(outputs, masks)

                
                val_running_loss += loss.item() * imgs.size(0)
                val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        
        val_epoch_loss = val_running_loss / len(val_subset)
        
        print(f"\nEpoch {epoch+1}: Train Loss = {train_epoch_loss:.4f} | Val Loss = {val_epoch_loss:.4f}")
        
        epochs_without_improvement += 1 # Increment early stopping counter
        # Save best model based on validation loss
        if val_epoch_loss < best_val_loss:
            epochs_without_improvement = 0 #reset counter
            best_val_loss = val_epoch_loss 
            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)
            print(f"*** New Best Model Saved to {save_path} ***")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Segformer with custom CLI arguments.")
    
    # Training Arguments
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument("--image_size", type=int, default=512, help="Input image size (assumed square)")
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--fraction', type=float, default=0.1, help='Fraction of the dataset to use')
    parser.add_argument('--workers_train', type=int, default=8, help='Number of workers for train dataloader')
    parser.add_argument('--workers_val', type=int, default=8, help='Number of workers for val dataloader')
    parser.add_argument("--cropped_dataset",action='store_true', help='Use cropped dataset for training')
    parser.add_argument("--cropped_dataset_dir",type=str, default="spark_cropped", help='Path to cropped dataset directory')
    parser.add_argument("--cropped_satellite",action='store_true', help='Use cropped satellite dataset for training')
    # Logic Flags
    parser.add_argument('--custom_loss', action='store_true', help='Enable CE + Dice loss')
    parser.add_argument('--custom_loss_new', action='store_true', help='Enable alternative custom loss function')
    parser.add_argument('--suffix', type=str, default="", help='Subdirectory name for saving weights')
    parser.add_argument("--encoder",type=str, default="mit_b0", help='Encoder version for segformer')
    
    args = parser.parse_args()
    model_input_size = (args.image_size, args.image_size)
    use_cropped_dataset= args.cropped_dataset
    if args.cropped_satellite:
        use_cropped_dataset = True
    if(use_cropped_dataset):
        CROP_DIR = args.cropped_dataset_dir
        ROOT_DIR = os.path.join(ROOT_DIR,CROP_DIR)
    # Define final output path as a subdirectory
    if(args.custom_loss):
        BASE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR,"CustomLoss")
    elif(args.custom_loss_new):
        BASE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR,"CustomLossNew")
    else:
        BASE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR,"CE")
    final_dir = os.path.join(BASE_OUTPUT_DIR, args.suffix) if args.suffix else BASE_OUTPUT_DIR
    os.makedirs(final_dir, exist_ok=True)
    yaml_path = os.path.join(final_dir, "args.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Configuration saved to {yaml_path}")

    train(
        fraction=args.fraction,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        target_size=model_input_size,
        use_custom_loss=args.custom_loss,
        output_dir=final_dir,
        l_r=args.l_r,
        workers_train=args.workers_train,
        workers_val=args.workers_val,
        encoder_name=args.encoder
    )