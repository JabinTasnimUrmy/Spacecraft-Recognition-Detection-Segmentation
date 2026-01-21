# Segmentation

Semantic segmentation scripts and helpers. Implements SegFormer and UNet-based flows.

Content:

- `SegFormer/TrainSegformer.py` — training script; saves `args.yaml` alongside outputs.
- `SegFormer/SegmentationSubmissionSegFormer.py` — inference for SegFormer; upsamples predictions to 1024x1024 and saves RGB PNG masks.
- `SegFormer/SegmentationSubmissionSegFormerCropped.py` — cropped variant (if trained with `--cropped_dataset`).
- `UNet/TrainUnet.py` and `UNet/SegmentationSubmission.py` — UNet training and inference.
- `PrepareImagesForSubmission.py` — converts PNG masks into NPZ files for submission.

Quick usage:

- Train SegFormer (example):
  - `python Segmentation/SegFormer/TrainSegformer.py --epochs 20 --fraction 0.1 --batch_size 4 --suffix=myexp`
- Run SegFormer inference:
  - `python Segmentation/SegFormer/SegmentationSubmissionSegFormer.py --model_path <models/segmentation/segformer/<exp>> --output_dir <raw_output_dir>`
  - Convert to submission: `python Segmentation/PrepareImagesForSubmission.py --input_dir <raw_output_dir> --output_dir <submission_dir>`
