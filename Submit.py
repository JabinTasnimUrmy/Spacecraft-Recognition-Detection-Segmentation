from pathlib import Path
import argparse
import os
import yaml
import subprocess
import uuid
import yaml

class  DetectionSubmission:
    def __init__(self, model_path,model_name):
        self.model_path = model_path
        self.model_name = model_name
    def to_dict(self):
        return {
            "model_path": str(self.model_path),
            "model_name": self.model_name
        }
    def __hash__(self):
        return hash((self.model_path, self.model_name))
    def get_uuid(self):
        unique_string = f"{self.model_path}_{self.model_name}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

class SegmentationSubmission:
    def __init__(self, model_path,model_name):
        self.model_path = model_path
        self.model_name = model_name
    def to_dict(self):
        return {
            "model_path": str(self.model_path),
            "model_name": self.model_name
        }
    def __hash__(self):
        return hash((self.model_path, self.model_name))
    def get_uuid(self):
        unique_string = f"{self.model_path}_{self.model_name}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
def check_if_submission_exists(submission_segmentation_details, submission_details):
    if submission_details.is_file():
        with open(submission_details, 'r') as f:
            details = yaml.safe_load(f) 
            if details["submission_uuid"] == submission_segmentation_details.get_uuid():

                return True
    return False
def generate_detection_submission(submission_detection_details:DetectionSubmission,  output_dir):
    detection_model_path = submission_detection_details.model_path
    detection_model_architecture = None
    with open(detection_model_path/'args.yaml', 'r') as f:
        config = yaml.safe_load(f)
        detection_model_architecture = config.get('model').rsplit('.', 1)[0] # to get architecture like yolo11n -> used for output dir (rsplit to remove .pt if present)
    output_dir = output_dir/"detection"/detection_model_architecture # ex detection/yolo11n
    output_dir.mkdir(parents=True, exist_ok=True) # create dir if not exists
    submission_details = output_dir/'detection_submission_details.yaml'
    if check_if_submission_exists(submission_detection_details, submission_details):
        print("Detection submission already exists with the same model. Skipping generation.")
        return detection_model_architecture
    DETECTION_SCRIPT = DETECTION_SCRIPT_DIR / 'YOLO' / 'DetectionSubmissionYOLO.py'


    cmd = [
        "python", str(DETECTION_SCRIPT),
        "--model_path", str(detection_model_path),
        "--output_dir", str(output_dir)] 
    print(f"Generating detection submission in {output_dir}")
    try:
        print(f"detection model path: {detection_model_path}")
        subprocess.run(cmd, check=True)
        print("Saving submission details...")
        with open(submission_details, 'w') as f:
            yaml.dump({
                "submission_uuid": submission_detection_details.get_uuid(),
                "model_name": submission_detection_details.model_name,
                "model_path": str(submission_detection_details.model_path)
            }, f)
            return detection_model_architecture
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {detection_model_path.name}: {e}")
        exit(1)
    # Save submission details
    

def generate_segmentation_submission_segformer(submission_segmentation_details:SegmentationSubmission, output_dir, pixel_perfect:bool=False,crop:bool=False,sat_crop:bool=False):

    output_dir = output_dir/ "segmentation"
    suffix = "segformer"#for the masks folder
    SEGMENTATION_SCRIPT = SEGMENTATION_SCRIPT_DIR / 'SegFormer' 
    if crop:
       suffix += "_cropped"
       SEGMENTATION_SCRIPT = SEGMENTATION_SCRIPT / 'SegmentationSubmissionSegFormerCropped.py'
    elif sat_crop:
        suffix += "_sat"
        SEGMENTATION_SCRIPT = SEGMENTATION_SCRIPT / 'SegmentationSubmissionSegFormerCropped.py'
    else:
        SEGMENTATION_SCRIPT = SEGMENTATION_SCRIPT / 'SegmentationSubmissionSegFormer.py'
        
    output_dir = output_dir / suffix
    suffix+="-test"# for TEST PURPOSES only #TODO REMOVE
    raw_output_dir = CURRENT_DIR / 'predicted_masks' / suffix 

    raw_output_dir.mkdir(parents=True, exist_ok=True) # create dir if not exists
    output_dir.mkdir(parents=True, exist_ok=True) # create dir if not exists
    submission_details = output_dir/'segmentation_submission_details.yaml'
    if check_if_submission_exists(submission_segmentation_details, submission_details):
        print("Segmentation submission already exists with the same model. Skipping generation.")
        return
    

    segment_raw_cmd = [
        "python", str(SEGMENTATION_SCRIPT),
        "--model_path", str(segmentation_submission_details.model_path),
        "--output_dir", str(raw_output_dir)]
    if sat_crop:
        segment_raw_cmd.append("--satellite_crop")
    segment_prepare_cmd = ["python", str(SEGMENTATION_SCRIPT_DIR / 'PrepareImagesForSubmission.py'),"--input_dir", str(raw_output_dir), "--output_dir", str(output_dir)]
    if pixel_perfect:
        segment_raw_cmd.append("--pixel_perfect")
    print(f"Generating segmentation submission in {output_dir}")
    try:
        print(f"segmentation model path: {segmentation_submission_details.model_path}")
        subprocess.run(segment_raw_cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {segmentation_submission_details.model_path.name}: {e}")
        exit(1)
    # Save submission details
    try:
        print("Preparing segmentation submission...")
        print(f"Running command: {(SEGMENTATION_SCRIPT)}/PrepareImagesForSubmission.py --input_dir {str(raw_output_dir)} --output_dir {str(output_dir)}")
        subprocess.run(segment_prepare_cmd, check=True)
        print("Saving submission details...")
        with open(submission_details, 'w') as f:
            yaml.dump({
                "submission_uuid": submission_segmentation_details.get_uuid(),
                "model_name": submission_segmentation_details.model_name,
                "model_path": str(submission_segmentation_details.model_path)
            }, f)
    except subprocess.CalledProcessError as e:
        print(f"Error preparing segmentation submission: {e}")
        exit(1)

def generate_segmentation_submission_unet(submission_segmentation_details:SegmentationSubmission, output_dir):
    output_dir = output_dir/ "segmentation"
    suffix = "unet"
    SEGMENTATION_SCRIPT = SEGMENTATION_SCRIPT_DIR / 'UNet'/ 'SegmentationSubmission.py' 
    output_dir = output_dir / suffix
    raw_output_dir = CURRENT_DIR / 'predicted_masks' / suffix
    raw_output_dir.mkdir(parents=True, exist_ok=True) # create dir if not exists
    output_dir.mkdir(parents=True, exist_ok=True) # create dir if not exists
    submission_details = output_dir/'segmentation_submission_details.yaml' 

    if check_if_submission_exists(submission_segmentation_details, submission_details):
        print("Segmentation submission already exists with the same model. Skipping generation.")
        return  
    segment_raw_cmd = [
        "python", str(SEGMENTATION_SCRIPT),     
        "--model_path", str(segmentation_submission_details.model_path),
        "--output_dir", str(raw_output_dir)]
    segment_prepare_cmd = ["python", str(SEGMENTATION_SCRIPT_DIR / 'PrepareImagesForSubmission.py'),"--input_dir", str(raw_output_dir), "--output_dir", str(output_dir)]
    print(f"Generating segmentation submission in {output_dir}")
    try:
        print(f"segmentation model path: {segmentation_submission_details.model_path}")
        subprocess.run(segment_raw_cmd, check=True) 
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {segmentation_submission_details.model_path.name}: {e}")
        exit(1)
    try:
        print("Preparing segmentation submission...")
        print(f"Running command: {(SEGMENTATION_SCRIPT)}/PrepareImagesForSubmission.py --input_dir {str(raw_output_dir)} --output_dir {str(output_dir)}")
        subprocess.run(segment_prepare_cmd, check=True)
         # Save submission details
        print("Saving submission details...")
        with open(submission_details, 'w') as f:
            yaml.dump({
                "submission_uuid": submission_segmentation_details.get_uuid(),
                "model_name": submission_segmentation_details.model_name,
                "model_path": str(submission_segmentation_details.model_path)
            }, f)
    except subprocess.CalledProcessError as e:
        print(f"Error preparing segmentation submission: {e}")
        exit(1)
   


if __name__ == '__main__':
    CURRENT_DIR = Path(__file__).parent.resolve()
    #DIR DEFINITION
    SEGMENTATION_SCRIPT_DIR = CURRENT_DIR/'Segmentation'
    DETECTION_SCRIPT_DIR = CURRENT_DIR / 'Detection'
    MODELS_DIR = CURRENT_DIR / 'models'
    #Arguments Definition
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--detection_model_path", type=str, help="path to detection model dir (not the pth)", required=True)
    argparser.add_argument('--segmentation_model_path', type=str, help='path to segmentation model dir(not the pth)',required=True)
    #argparser.add_argument('--test_img_dir', type=str, default=TEST_IMG_DIR, help='test image directory')
    #argparser.add_argument('--test_csv_path', type=str, default=TEST_CSV_PATH, help='test csv path with bboxes')
    argparser.add_argument("--pixel_perfect", action='store_true', help="Use pixel-perfect resizing and cropping")


    args = argparser.parse_args()

    DETECTION_MODEL_PATH = MODELS_DIR /"detection"/ args.detection_model_path
    SEGMENTATION_MODEL_PATH = MODELS_DIR /"segmentation"/ args.segmentation_model_path
    SUBMISSION_OUTPUT_DIR = CURRENT_DIR / 'AutomatedSubmissionOutputs'#TODO REMOVE TEST SUFFIX(is just for development)
    os.makedirs(SUBMISSION_OUTPUT_DIR, exist_ok=True)
    detection_model_name = DETECTION_MODEL_PATH.name
    segmentation_model_name = SEGMENTATION_MODEL_PATH.name
    if DETECTION_MODEL_PATH.exists() is False:
        raise FileNotFoundError(f"Detection model file not found: {DETECTION_MODEL_PATH}")
    if SEGMENTATION_MODEL_PATH.exists() is False:
        raise FileNotFoundError(f"Segmentation model file not found: {SEGMENTATION_MODEL_PATH}")

    print(f"Using detection model: {detection_model_name}")
    print(f"Using segmentation model: {segmentation_model_name}")

    detection_submission_details = DetectionSubmission(
        model_path=DETECTION_MODEL_PATH,
        model_name=detection_model_name)
    segmentation_submission_details = SegmentationSubmission(
        model_path=SEGMENTATION_MODEL_PATH,
        model_name=segmentation_model_name)
    detection_model_family = generate_detection_submission(detection_submission_details, SUBMISSION_OUTPUT_DIR)
    segmentation_model_family = None # to be used in the zip creation Script
    if ("segformer" in str(SEGMENTATION_MODEL_PATH).lower()):
      segmentation_model_family = "segformer"
      crop = False
      sat_crop=False
      with open(SEGMENTATION_MODEL_PATH/'args.yaml', 'r') as f:
          config = yaml.safe_load(f)
          crop = config.get('cropped_dataset', False)
          sat_crop = config.get("cropped_satellite",False)
      if crop:
          segmentation_model_family += "_cropped"
      if sat_crop:
          segmentation_model_family += "_sat"
      print("Model family is SegFormer")
      print(f"Cropping enabled: {crop}")
      generate_segmentation_submission_segformer(segmentation_submission_details, SUBMISSION_OUTPUT_DIR, pixel_perfect=args.pixel_perfect,crop=crop,sat_crop=sat_crop)
    elif ("unet" in str(SEGMENTATION_MODEL_PATH).lower()):
      print("Model family is UNet")
      segmentation_model_family = "unet"
      generate_segmentation_submission_unet(segmentation_submission_details, SUBMISSION_OUTPUT_DIR)
    else:
      raise ValueError("Unknown segmentation model family. Supported: SegFormer, UNet")
    
    

    CREATE_ZIP_SCRIPT = CURRENT_DIR / 'Utils' / 'CreateSubmissionZip.py'
    zip_file_name = f"submission_{detection_model_name}_{segmentation_model_name}.zip"
    

        
    if "CustomLoss" in str(SEGMENTATION_MODEL_PATH):
        
        zip_file_name = f"submission_{detection_model_name}_{segmentation_model_name}_CL.zip"
    

    zip_cmd = [
        "python", str(CREATE_ZIP_SCRIPT),
        "--detection_model", detection_model_family,
        "--segmentation_model", segmentation_model_family,
        "--submission_dir", str(SUBMISSION_OUTPUT_DIR),
        "--output_zip", str(CURRENT_DIR / "AutomatedSubmissionZipArchive"/zip_file_name)

    ]
    try:
        print("Creating final submission zip...")
        subprocess.run(zip_cmd, check=True) 
    except subprocess.CalledProcessError as e:
        print(f"Error creating submission zip: {e}")
        exit(1)


