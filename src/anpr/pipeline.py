"""Modern, reusable YOLOv8 + EasyOCR ANPR inference pipeline."""
from dataclasses import dataclass, asdict
from typing import List
import os
import site
import sys

# Keep CPU inference predictable on small hosted instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# This repository contains a legacy vendored ``ultralytics/`` tree from the
# original project. Prefer the current package installed in site-packages.
for _site_packages in reversed(site.getsitepackages()):
    if _site_packages in sys.path:
        sys.path.remove(_site_packages)
    sys.path.insert(0, _site_packages)

import cv2
from ultralytics import YOLO
import easyocr

from .text import normalize_plate


@dataclass
class PlateDetection:
    box: list[int]
    detection_confidence: float
    text: str
    ocr_confidence: float

    def to_dict(self):
        return asdict(self)


class ANPRPipeline:
    def __init__(self, model_path: str, detector_confidence: float = 0.25, gpu: bool = False):
        self.model = YOLO(model_path)
        self.detector_confidence = detector_confidence
        self.gpu = gpu
        # OCR is loaded only when YOLO actually finds a plate. EasyOCR's own
        # detector is disabled because YOLO already supplies the plate crop.
        self.reader = None

    def _get_reader(self):
        if self.reader is None:
            self.reader = easyocr.Reader(
                ["en"],
                gpu=self.gpu,
                detector=False,
                recognizer=True,
                model_storage_directory="/tmp/easyocr",
                user_network_directory="/tmp/easyocr-user",
            )
        return self.reader

    @staticmethod
    def preprocess_plate(crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        return gray

    def read_plate(self, crop) -> tuple[str, float]:
        processed = self.preprocess_plate(crop)
        candidates = self._get_reader().recognize(
            processed,
            detail=1,
            batch_size=1,
            workers=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        if not candidates:
            return "", 0.0
        best = max(candidates, key=lambda item: float(item[2]))
        return normalize_plate(best[1]), float(best[2])

    def predict_image(self, image_path: str) -> List[PlateDetection]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        result = self.model.predict(
            image,
            conf=self.detector_confidence,
            verbose=False,
            device="cpu",
            imgsz=640,
            max_det=5,
        )[0]
        detections: List[PlateDetection] = []
        height, width = image.shape[:2]

        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            text, ocr_conf = self.read_plate(crop)
            detections.append(PlateDetection(
                box=[x1, y1, x2, y2],
                detection_confidence=float(box.conf[0]),
                text=text,
                ocr_confidence=ocr_conf,
            ))
        return detections
