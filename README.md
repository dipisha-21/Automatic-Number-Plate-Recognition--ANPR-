# Automatic Number Plate Recognition (ANPR)

Computer-vision pipeline for **license-plate detection and text recognition** using YOLOv8, OpenCV and EasyOCR.

## Pipeline
```text
Image / Video
     ↓
YOLOv8 plate detector
     ↓
Bounding-box crop
     ↓
OpenCV preprocessing
     ↓
EasyOCR recognition
     ↓
Plate text + confidence
```

## Tech Stack
- Python
- YOLOv8 / Ultralytics
- EasyOCR
- OpenCV
- PyTorch

## What this upgrade adds
The original repository already contained a YOLOv8 dataset/training setup and a custom `predict_modified.py` integrating EasyOCR. The portfolio upgrade keeps that work while adding a clean modular inference layer, text normalization, evaluation utilities, tests, portable dataset configuration and interviewer-focused documentation.

## Repository Structure
```text
src/anpr/
  text.py          Plate-text normalization and matching
  pipeline.py      Reusable YOLOv8 + EasyOCR inference pipeline
  evaluate.py      Ground-truth vs prediction evaluation
scripts/run_anpr.py CLI entry point
tests/             Lightweight unit tests
docs/              Architecture and interview guide
data.yaml           Portable YOLO dataset paths
predict_modified.py Original/custom predictor retained for reference
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run Inference
```bash
python scripts/run_anpr.py --model path/to/best.pt --source path/to/image_or_video.jpg
```

For an image, the CLI prints structured JSON containing bounding boxes, detection confidence, recognized plate text and OCR confidence.

## Evaluation
Prepare a CSV containing:

```text
image,plate_text
image001.jpg,MH12AB1234
image002.jpg,KA01CD5678
```

Then run:
```bash
python -m src.anpr.evaluate --model path/to/best.pt --labels labels.csv
```

The evaluator reports:
- exact plate recognition accuracy
- character-level accuracy
- number of evaluated images

## Metric Integrity
A previous project description referenced **99.5% accuracy**. That number should only be presented as a measured result if the original evaluation protocol and outputs can reproduce it. This upgrade deliberately computes metrics from supplied ground truth instead of hard-coding a headline accuracy.

Detection metrics such as precision, recall and mAP are different from OCR exact-match accuracy; they should not be combined into one ambiguous “accuracy” number.

## Dataset Configuration
`data.yaml` now uses relative repository paths rather than a developer-specific Windows drive, making training portable after cloning the repository.

## Training
With a compatible Ultralytics installation:
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=100, imgsz=640)
```

## Interview Talking Points
Explain why ANPR is a two-stage problem, how detection and OCR errors propagate, why preprocessing helps OCR, the difference between IoU/mAP and text accuracy, confidence thresholds, failure cases such as blur/angle/lighting, and how you would monitor the pipeline after deployment.

## Author
Dipisha Shivangi
