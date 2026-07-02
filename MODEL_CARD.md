# Model Card: `models/yolo11n-best.pt`

## Intended Use

This weight is bundled for research annotation assistance in the Hip 22-Point Annotation Tool. It predicts up to two hip-side objects on AP pelvic X-ray images and returns 11 landmarks per side, which the tool converts into the 22-point JSON schema.

It is not a medical device and must not be used as a standalone clinical diagnosis system. Users should review and correct all outputs before use.

## Model Summary

- File: `models/yolo11n-best.pt`
- Base model family: `yolo11n-pose`
- Task format: `1 class = hip_side`, `11 keypoints per side`
- Output convention: `left_*` and `right_*` follow image-left/image-right, with no anatomical side swap.
- SHA256: `f9892b4aaa4395576b6eec5b0ee336d93bf94784e35e8d30ba34e465a40d2774`

## Training Data

The bundled yolo11n-best weight is trained from manually created research annotations on MTDDH-derived images by a non-medical annotator; users should review and correct all outputs before use.

The underlying demo image source is the MTDDH dataset, licensed under CC BY 4.0. The model was trained from local derived annotations and is distributed under this repository's AGPL-3.0 license unless a separate commercial license is obtained for the underlying model stack.

## Known Limitations

- The annotations used to train this weight were not produced by medical professionals.
- Abnormal anatomy, severe deformity, poor positioning, cropped images, or missing/indistinct anatomy may reduce accuracy.
- The model may still predict a point when a landmark is clinically absent or not clearly definable; the annotation tool allows users to mark such points as missing.
- The tool does not compute clinical measurements or make clinical decisions.

## Evaluation Snapshot

The archived source metadata reports the following internal evaluation summary:

- Evaluation images: 174
- Detected two sides: 174 / 174
- Side assignment violations: 0
- Overall mean pixel error: 14.557 px
- Overall median pixel error: 11.431 px
- PCK@5% side bbox diagonal: 0.958
- PCK@10% side bbox diagonal: 0.9963

These numbers are for internal engineering comparison only and should not be read as clinical validation.

## Recommended Review Workflow

1. Load an image folder in the annotation tool.
2. Let the bundled model generate initial points.
3. Manually inspect all 22 landmarks.
4. Correct misplaced points and mark genuinely absent/undefined points as missing.
5. Save the reviewed JSON and YOLO sidecar label.
