import torch
from torchmetrics.classification import MulticlassJaccardIndex
import yaml
import os
import numpy as np
import cv2

class EvaluationData():
    def __init__(self, fraction, resolution, results):
        self.fraction = fraction
        self.resolution = resolution        
        # Convert PyTorch Tensor to a list of floats so YAML can serialize it
        self.results_list = results.cpu().tolist() if torch.is_tensor(results) else results

        self.class_names = ['Background', 'Spacecraft Body', 'Solar Panels','Mean IoU']
        self.results_list.append(torch.mean(results).item())#adding mean IoU

    def to_dict(self):
        """Maps IoU scores to class names for a readable YAML output."""
        # Create a dictionary pairing names with scores
        results_with_names = {
            name: score for name, score in zip(self.class_names, self.results_list)
        }
        
        return {
            "fraction": self.fraction,
            "resolution": list(self.resolution),
            "results": results_with_names
        }

class ModelData():
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        # Path objects must be converted to strings for YAML
        self.model_path = str(model_path)

    def to_dict(self):
        """Returns a dictionary representation of the model metadata."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path
        }

def rgb_mask_to_indices(rgb_mask):
    """
    Converts RGB masks to class indices.
    Supports:
    - Single mask: (3, H, W) -> returns (H, W)
    - Batch: (B, 3, H, W) -> returns (B, H, W)
    """
    ndim = rgb_mask.ndim
    
    # 1. Explicitly extract channels based on input dimensions
    if ndim == 3:
        # Single Image: [Channels, Height, Width]
        red_ch = rgb_mask[0, :, :]
        blue_ch = rgb_mask[2, :, :]
    elif ndim == 4:
        # Batch: [Batch, Channels, Height, Width]
        red_ch = rgb_mask[:, 0, :, :]
        blue_ch = rgb_mask[:, 2, :, :]
    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions, got {ndim} (shape: {rgb_mask.shape})")
    # 2. Determine threshold
    threshold = 0.5 if rgb_mask.max() <= 1.0 else 127
    # 3. Initialize output
    gt_indices = torch.zeros(red_ch.shape, dtype=torch.long, device=rgb_mask.device)     # red_ch.shape to automatically match (H, W) or (B, H, W)
    # 4. Convert
    # Class 1: Body (Red)
    body_mask = (red_ch > threshold) & (red_ch >= blue_ch)
    gt_indices[body_mask] = 1
    
    # Class 2: Panels (Blue)
    panel_mask = (blue_ch > threshold) & (blue_ch > red_ch)
    gt_indices[panel_mask] = 2
    
    return gt_indices


def upscale_prediction_results_bilinear(ouptuts,target_size):
    probs = torch.softmax(ouptuts, dim=1)
    preds_upscaled = torch.nn.functional.interpolate(probs, size=target_size, mode='bilinear', align_corners=False)
    return torch.argmax(preds_upscaled, dim=1)

#####################
##UNSHIFT SECTION
#####################
def unshift_mask_pp(pred_mask, bbox,original_res=(1024, 1024)):
    """
    Reconstructs the full-resolution mask by inverting the 1024->512 pipeline.
    """

    if pred_mask.shape[0] != 1024 or pred_mask.shape[1] != 1024:
        raise ValueError("pred_mask must be of size 1024x1024 for unshift operation.")
    else:
        mask_1024 = pred_mask

    x1, y1, x2, y2 = bbox
    crop_w, crop_h = x2 - x1, y2 - y1

    # 2. Calculate the SAME offsets used in the Dataset (must be identical)
    offset_x = (original_res[0] - crop_w) // 2
    offset_y = (original_res[1] - crop_h) // 2

    # 3. Extract the 'Useful Area' using slicing
    # Since mask_1024 is back to 1:1 ratio, we slice exactly crop_w x crop_h
    useful_area = mask_1024[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w]

    # 4. Initialize the final 1024x1024 full frame
    full_frame = torch.zeros(original_res, dtype=torch.uint8, device=pred_mask.device)

    # 5. Place the satellite back into its original global coordinates
    full_frame[y1:y2, x1:x2] = useful_area

    return full_frame
def unshift_mask(pred_mask, bbox, canvas_size=(512, 512), original_res=(1024, 1024)):
    """
   Unshifts the prediction mask from the 1024x1024 canvas back to the original coordinates.
    """
    x1, y1, x2, y2 = bbox
    crop_w, crop_h = x2 - x1, y2 - y1

    # Inverting scaling thumbnail
    scale = min(canvas_size[0] / crop_w, canvas_size[1] / crop_h)
    if scale > 1.0: scale = 1.0
    
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    offset_x = (canvas_size[0] - new_w) // 2
    offset_y = (canvas_size[1] - new_h) // 2

    # Extract useful area from the 512x512 prediction
    
    useful_mask = pred_mask[offset_y : offset_y + new_h, offset_x : offset_x + new_w]
    

    # Resize to original crop size
    full_frame = torch.zeros(original_res, dtype=torch.uint8, device=pred_mask.device)
    full_frame[y1:y2, x1:x2] = useful_mask

    return full_frame

def unshift_mask_sat(pred_mask, bbox_gt, target_size=512,original_res=(1024, 1024),):
    
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
    useful_area = pred_mask[:, offset_y : offset_y + curr_h, offset_x : offset_x + curr_w]

    # 5. Reverse Scaling (if object was > 512)
    if scale != 1.0:
        useful_area=useful_area.unsqueeze(0)# upscale requires batch size
        useful_area = upscale_prediction_results_bilinear(useful_area, (orig_h, orig_w)) # already performs argmax
        useful_area=useful_area.squeeze(0)
    else:
         useful_area = torch.argmax(useful_area, dim=0)  # no upscaling so directy argmax
    # 6. Reconstruct the original frame
    if useful_area.ndim == 3:
        full_res_frame = torch.zeros((*original_res, useful_area.shape[2]), dtype=torch.uint8)
    else:
        full_res_frame = torch.zeros(original_res, dtype=torch.uint8)


    # 7. Final Shift: Paste back to original coordinates
    full_res_frame[y1:y2, x1:x2] = useful_area

    return full_res_frame


def print_iou_results(per_class_iou):
    """Prints the results formatted for your project."""
    class_names = ['Background', 'Spacecraft Body', 'Solar Panels']
    mean_iou = torch.mean(per_class_iou)
    
    print(f"{'Class Name':<20} | {'IoU Score':<10}")
    print("-" * 35)
    for i, score in enumerate(per_class_iou):
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"{name:<20} | {score.item():.4f}")
    print("-" * 35)
    print(f"{'Mean IoU (mIoU)':<20} | {mean_iou.item():.4f}")

    return per_class_iou, mean_iou


def save_evaluation_results(model_data, evaluation_data_list, save_path):
    """
    Saves results into a clean YAML format with named classes.
    """
    export_data = {
        "model_info": model_data.to_dict(),
        "evaluation_results": [eval_data.to_dict() for eval_data in evaluation_data_list]
    }
    
    file_path = os.path.join(save_path, "evaluation_results.yaml")
    
    with open(file_path, "w") as f:
        # sort_keys=False ensures the order (Background -> Body -> Panels) is preserved
        yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Results successfully saved with class names to: {file_path}")


