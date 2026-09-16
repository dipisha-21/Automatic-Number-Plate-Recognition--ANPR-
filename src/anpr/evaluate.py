"""Evaluate end-to-end plate recognition against a labelled CSV."""
import argparse
import csv
from pathlib import Path

from .pipeline import ANPRPipeline
from .text import normalize_plate, character_accuracy


def evaluate(model: str, labels_csv: str, gpu: bool = False) -> dict:
    pipeline = ANPRPipeline(model, gpu=gpu)
    exact = 0
    char_scores = []
    total = 0

    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            expected = normalize_plate(row["plate_text"])
            detections = pipeline.predict_image(row["image"])
            # For single-plate evaluation images, use the highest combined confidence.
            if detections:
                best = max(detections, key=lambda d: d.detection_confidence * max(d.ocr_confidence, 1e-6))
                predicted = normalize_plate(best.text)
            else:
                predicted = ""
            exact += int(predicted == expected)
            char_scores.append(character_accuracy(predicted, expected))

    return {
        "images": total,
        "exact_match_accuracy": exact / total if total else 0.0,
        "character_accuracy": sum(char_scores) / total if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    metrics = evaluate(args.model, args.labels, args.gpu)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
