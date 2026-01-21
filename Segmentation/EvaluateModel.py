import torch
from torchmetrics.classification import MulticlassJaccardIndex
import numpy as np
from tqdm import tqdm  # Ensure tqdm is imported
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader,Subset
from pathlib import Path
import argparse
import sys
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
utils_path = CURRENT_DIR.parent / "Utils"
sys.path.append(str(utils_path))
segmentation_utils_path = CURRENT_DIR / "Utils"
sys.path.append(str(segmentation_utils_path))

from utils import *


from segmentation_utils_spark import print_iou_results,rgb_mask_to_indices,upscale_prediction_results_bilinear,ModelData,EvaluationData,save_evaluation_results,unshift_mask_pp,unshift_mask,unshift_mask_sat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Segformer with custom CLI arguments.")
    parser.add_argument("--fraction", type=float, default=0.01, help="Fraction of validation dataset to use (default: 0.1)")
    parser.add_argument("--model_type", type=str, default="segformer", help="Type of model to evaluate (default: Segformer)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model directory (relative to the models dir)", required=True)
    parser.add_argument("--crop_type", type=str, default=None, help="Type of crop to use (default: None)", required=False)
    parser.add_argument("--model_res", type=int, default=512, help="Model resolution (default: 512)")
    args = parser.parse_args()
    FRACTION = args.fraction
    MODEL_TYPE = args.model_type
    MODEL_RES = (args.model_res, args.model_res)

    MODELS_PATH = CURRENT_DIR.parent /"models"/"segmentation"
    #determining model type subfolder
    if MODEL_TYPE.lower() == "segformer":
        MODELS_PATH = MODELS_PATH / "segformer"
    elif MODEL_TYPE.lower() == "unet":
        MODELS_PATH = MODELS_PATH / "unet"
    ROOT_DIR = CURRENT_DIR.parent.parent / "data" 


    # Set crop directory based on crop type argument (Only if requested)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    # MODEL LOADING
    MODEL_DIR = MODELS_PATH / args.model_path
    #Chosing the model type
    if(str.lower(MODEL_TYPE) == "segformer"): 
        #SEGFORMER MODEL
        #Determining encoder type from config file
        encoder_name="mit_b0"
        config_path = MODEL_DIR / "args.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if "encoder" in config: #Optional field -> initially i didnt include it, so for backward compatibility i check its existence
                encoder_name = config["encoder"]
        print(f"Evaluating Segformer {MODEL_DIR.name} (encoder={encoder_name}) using device {DEVICE}")
        model = smp.Segformer(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=3).to(DEVICE)

    elif(str.lower(MODEL_TYPE) == "unet"):
        #UNET MODEL
        print(f"Evaluating UNet {MODEL_DIR.name} using device {DEVICE}")
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3).to(DEVICE)

    else:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}. Supported types are 'segformer' and 'unet'.")
    # loading the best model weights
    checkpoint = torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    
    
    # VALIDATION DATASET AND DATALOADER
    val_ds = ResizedSPARKDataset(class_map=CLASS_MAP, root_dir=ROOT_DIR, split='val', target_size=MODEL_RES)
    val_ds_cropped = None
    CROP_DIR = None
    if args.crop_type is not None:
        if args.crop_type.lower() == "cropped":
            CROP_DIR = ROOT_DIR / "spark_cropped"
        elif args.crop_type.lower() == "cropped_pp":
            CROP_DIR = ROOT_DIR / "spark_cropped_pp_png"
        elif args.crop_type.lower() == "sat":
            CROP_DIR = ROOT_DIR / "spark_satellite_crop_fixed_size"
        else:
            raise ValueError(f"Unsupported crop type: {args.crop_type}. Supported types are 'cropped' and 'cropped_pp'.")
        val_ds_cropped = ResizedSPARKDataset(class_map=CLASS_MAP, root_dir=CROP_DIR, split='val', target_size=  MODEL_RES)
        
    if FRACTION < 1.0:
        num_samples = int(len(val_ds) * FRACTION)
        
        np.random.seed(42) # for reproducibilty
        # Randomly select indices for the subset
        indices = np.random.choice(len(val_ds), num_samples, replace=False)
        val_dataset = Subset(val_ds, indices)
        if val_ds_cropped is not None:
            val_dataset_cropped = Subset(val_ds_cropped, indices)
        print(f"Evaluating on a subset of {num_samples} images (Fraction: {FRACTION})...")
    else:
        val_dataset = val_ds
        print(f"Evaluating on the full validation set ({len(val_dataset)} images)...")

    # 3. Setup DataLoader
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    val_cropped_loader = None
    if val_ds_cropped is not None:
        val_cropped_loader = DataLoader(val_dataset_cropped, batch_size=32, shuffle=False, num_workers=0)
        cropped_iter = iter(val_cropped_loader)

    # Initialize Metrics
    val_iou_full_res = MulticlassJaccardIndex(num_classes=3, average=None).to(DEVICE)
    val_iou_model_res = MulticlassJaccardIndex(num_classes=3, average=None).to(DEVICE)
    # Evaluation Loop with tqdm
    with torch.no_grad():
        # Wrap val_loader with tqdm for a progress bar
        for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):
            batch_cropped = None
            if val_ds_cropped is not None:
                batch_cropped = next(cropped_iter)
                images = batch_cropped["img"].to(DEVICE)
            else:
                images = batch["img"].to(DEVICE) 
            masks_full_res = batch["original_size_mask"].to(DEVICE)
            masks_model_res = batch["mask"].to(DEVICE)
            #Process the masks
            processed_masks_model_res = rgb_mask_to_indices(masks_model_res)    
            processed_masks_full_res = rgb_mask_to_indices(masks_full_res)
            

            outputs = model(images) # Model Res (ex. 512x512)

                
            preds_model_res = torch.argmax(outputs,dim=1) 
            preds_full_res = upscale_prediction_results_bilinear(outputs,(1024,1024))
            if args.crop_type is not None:
                crop_type = str.lower(args.crop_type)   
                if(crop_type == "cropped_pp"):
                    for i in range(len(preds_full_res)):
                        preds_full_res[i] = unshift_mask_pp(preds_full_res[i], batch["bbox"][i],original_res=(1024,1024))
                elif(crop_type == "cropped"):
                    for i in range(len(preds_full_res)):
                        preds_full_res[i] = unshift_mask(preds_full_res[i], batch["bbox"][i], canvas_size=(1024,1024), original_res=(1024,1024))
                elif(crop_type=="sat"):
                    for i in range(len(preds_full_res)):
                     preds_full_res[i] = unshift_mask_sat(outputs[i], batch["bbox"][i], original_res=(1024,1024))
                preds_model_res = torch.nn.functional.interpolate(preds_full_res.unsqueeze(1).float(), size=MODEL_RES, mode='nearest').squeeze(1).long()
            

            val_iou_model_res.update(preds_model_res,processed_masks_model_res)
            val_iou_full_res.update(preds_full_res, processed_masks_full_res)

    # At the end of validation, compute final results
    final_per_class_full_res = val_iou_full_res.compute()
    final_per_class_model_res = val_iou_model_res.compute()
    #Preparing data to save them
    model_data = ModelData(MODEL_DIR.name,MODEL_DIR)
    model_res_evaluation_data = EvaluationData(fraction=FRACTION,resolution=MODEL_RES,results=final_per_class_model_res)
    full_res_evaluation_data = EvaluationData(fraction=FRACTION,resolution=(1024,1024),results=final_per_class_full_res)


    save_evaluation_results(model_data,[model_res_evaluation_data,full_res_evaluation_data],MODEL_DIR)
    print("\n" + "="*35)
    print("FINAL EVALUATION RESULTS" )
    print(f"{FRACTION*100}% of Validation Set used")
    print("="*35)
    print(f"Model Res({MODEL_RES[0]}x{MODEL_RES[0]})")
    print("="*35)
    print_iou_results(final_per_class_model_res)
    print("="*35)
    print("Full Res(1024x1024)")
    print("="*35)
    print_iou_results(final_per_class_full_res)
    