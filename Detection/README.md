# Detection

YOLO-based detection utilities and submission helper scripts.

Contents:

- `YOLO/TrainYolo.py` — quick training script using `YOLO`.
- `YOLO/DetectionSubmissionYOLO.py` — detection inference for test images. 
- `CheckSubmission.py` — validation of detection CSV outputs.

Quick usage:

- Train (quick): `python Detection/YOLO/TrainYolo.py`
- Inference: `python Detection/YOLO/DetectionSubmissionYOLO.py --model_path <models/detection/<name>> --output_dir <dir>`

