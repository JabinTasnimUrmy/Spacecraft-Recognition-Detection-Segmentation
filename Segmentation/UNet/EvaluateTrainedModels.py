import os
import subprocess
from pathlib import Path
import yaml
# Paths setup
CURRENT_DIR = Path(__file__).resolve().parent
MODELS_ROOT = CURRENT_DIR.parent .parent/ "models" / "segmentation" / "unet"
EVAL_SCRIPT = CURRENT_DIR.parent / "EvaluateModel.py" 
FRESH_EVAL = False
def run_evaluations():
    # 1. Iterate through all folders in the segformer directory\
    for category_dir in MODELS_ROOT.iterdir():
        if not category_dir.is_dir():
            continue

        # 2. Look for specific experiment folders inside categories
        for model_dir in category_dir.iterdir():
            cropped_mode = None
            model_res = 512 #default value (all the old models where trained at 512x512)
            if model_dir.is_dir():
                # Check if the folder contains the required weight file
                checkpoint_path = model_dir / "best_model.pth"
                
                if not checkpoint_path.exists():
                    print(f"Skipping {model_dir.name}: No 'best_model.pth' found.")
                    continue
                evaluation_file = model_dir / "evaluation_results.yaml"
                if evaluation_file.exists() and not FRESH_EVAL:
                    print(f"Skipping {model_dir.name}: Evaluation already exists.")
                    continue
                config_path = model_dir / "args.yaml"
                if not config_path.exists():
                    print(f"Skipping {model_dir.name}: No 'args.yaml' found.")
                    continue
                print(f"Evaluating model {model_dir.parent.name} {model_dir.name}")
            
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    if "cropped_dataset" in config and config["cropped_dataset"] == True: #Optional field -> initially i didnt include it, so for backward compatibility i check its existence
                        if "cropped_dataset_dir" in config:
                            print("Cropped PP Model Detected")
                            cropped_mode = "cropped_pp" # i might have to change this in the future
                        else:
                            print("Cropped Model Detected")
                            cropped_mode= "cropped"
                    if "image_size" in config: #Optional field -> initially i didnt include it, so for backward compatibility i check its existence
                        model_res = config["image_size"]
                        print(f"Model resolution set to {config['image_size']}")
                    

                relative_model_path = model_dir.relative_to(MODELS_ROOT)
                
                cmd = [
                    "python", str(EVAL_SCRIPT),
                    "--model_type", "unet",
                    "--model_path", str(relative_model_path),
                    "--fraction", "0.1",  # Optional: use a small fraction for speed
                    "--model_res", str(model_res)
                ]
                if cropped_mode is not None:
                    cmd.append("--crop_type")
                    cmd.append(str(cropped_mode))

                # 4. Execute the evaluation script as a separate process
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error evaluating {model_dir.name}: {e}")

if __name__ == "__main__":
    run_evaluations()