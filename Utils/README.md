# Utils

Helper scripts and dataset utilities used across the repo.

Content:

- `utils.py` — dataset helpers and PyTorch dataset classes (`PyTorchSPARKDataset`, `ResizedSPARKDataset`) and `CLASS_MAP`.
- `convert_dataset_to_YOLO.py` — helper to convert dataset to YOLO format.
- `yolo_conversion_checker.py` — checks YOLO-format dataset correctness.
- `CreateSubmissionZip.py` — packages `SubmissionOutputs/` into a zip; expects detection and segmentation subfolders.

Quick usage example:

- Create zip: `python Utils/CreateSubmissionZip.py --detection_model yolo11n --segmentation_model segformer --output_zip submission.zip`

