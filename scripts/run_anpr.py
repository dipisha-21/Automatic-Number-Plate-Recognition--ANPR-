import argparse
import json

from src.anpr.pipeline import ANPRPipeline


def main():
    parser = argparse.ArgumentParser(description="Detect and recognize number plates in an image")
    parser.add_argument("--model", required=True, help="Path to trained YOLOv8 weights")
    parser.add_argument("--source", required=True, help="Path to an input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR")
    args = parser.parse_args()

    pipeline = ANPRPipeline(args.model, detector_confidence=args.conf, gpu=args.gpu)
    detections = pipeline.predict_image(args.source)
    print(json.dumps([d.to_dict() for d in detections], indent=2))


if __name__ == "__main__":
    main()
