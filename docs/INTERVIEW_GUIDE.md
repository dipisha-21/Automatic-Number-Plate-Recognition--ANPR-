# Interview Guide

## 60-second explanation
This is a two-stage ANPR pipeline. YOLOv8 first localizes the license plate and returns a bounding box and confidence. I crop that ROI, convert it to grayscale and apply light denoising before EasyOCR. OCR output is normalized so spaces and punctuation do not create artificial mismatches. I separated evaluation into exact plate match and character-level accuracy because detection quality and OCR quality are different problems.

## Key questions

### Why YOLOv8?
It provides a practical single-stage object detector with a straightforward training/inference workflow and a useful speed/accuracy trade-off for real-time-oriented applications.

### Why preprocess before OCR?
OCR receives only the detected ROI. Grayscale and edge-preserving denoising can reduce irrelevant colour/noise while preserving character structure. Preprocessing should still be validated because excessive thresholding can destroy characters.

### What metrics would you use?
For detection: precision, recall and mAP at stated IoU thresholds. For recognition: exact-match accuracy and character-level accuracy. For the complete application: end-to-end exact plate recognition on a held-out set.

### How do you prevent data leakage?
Split at the source/vehicle/sequence level where possible, rather than randomly putting near-identical video frames into train and validation sets.

### How would you improve video recognition?
Track a plate across frames and aggregate OCR predictions/confidences instead of treating every frame independently.

### What about the 99.5% resume number?
Only quote it if you can reproduce it and clearly state what it measures, the dataset size/split and evaluation method. Do not describe detection mAP as OCR accuracy or vice versa.
