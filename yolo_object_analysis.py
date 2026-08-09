"""
Run pretrained YOLO object detection on the curated image dataset.

This is an interpretable supporting analysis for the research project: it
records objects (especially people) visible in each advertisement frame. It
does NOT classify an advert as Promotional or Preventive; train.py remains
the experiment used for that research question.

Install once:
    py -m pip install ultralytics

Examples:
    py yolo_object_analysis.py
    py yolo_object_analysis.py --source Dataset_Split/test --save-annotated
    py yolo_object_analysis.py --model yolo26n.pt --conf 0.35

Outputs:
    yolo_analysis/detections.csv       one row per detected object
    yolo_analysis/image_summary.csv    one row per source image
    yolo_analysis/class_summary.csv    counts by dataset class and object
    yolo_analysis/annotated/           optional bounding-box images
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_label(path: Path, source: Path) -> str:
    """Use the immediate class directory when the source follows Dataset/<class>."""
    relative = path.relative_to(source)
    return relative.parts[0] if len(relative.parts) > 1 else "Unlabelled"


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect objects in dataset images with pretrained YOLO.")
    parser.add_argument("--source", default="Dataset", help="Image directory to analyse (default: Dataset).")
    parser.add_argument("--model", default="yolo26n.pt", help="Ultralytics detection weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--output", default="yolo_analysis", help="Output directory.")
    parser.add_argument("--save-annotated", action="store_true", help="Save images with detection boxes.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Missing package: install it with `py -m pip install ultralytics`.") from exc

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found under: {source}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    annotated_dir = output / "annotated"
    if args.save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)  # Ultralytics downloads pretrained weights on first use.
    detection_rows, image_rows, class_counts = [], [], Counter()

    for result in model.predict(
        source=[str(p) for p in images], conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False
    ):
        image_path = Path(result.path)
        label = image_label(image_path, source)
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_name = result.names[int(box.cls.item())]
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(class_name)
                class_counts[(label, class_name)] += 1
                detection_rows.append({
                    "image": image_path.as_posix(), "dataset_class": label,
                    "object_class": class_name, "confidence": f"{confidence:.6f}",
                    "x1": f"{x1:.1f}", "y1": f"{y1:.1f}", "x2": f"{x2:.1f}", "y2": f"{y2:.1f}",
                })
        image_rows.append({
            "image": image_path.as_posix(), "dataset_class": label,
            "detection_count": len(detections), "person_count": detections.count("person"),
            "detected_objects": "; ".join(detections) if detections else "None",
        })
        if args.save_annotated:
            relative = image_path.relative_to(source)
            destination = annotated_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.save(filename=str(destination))

    def write_csv(path, rows, fields):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "detections.csv", detection_rows,
              ["image", "dataset_class", "object_class", "confidence", "x1", "y1", "x2", "y2"])
    write_csv(output / "image_summary.csv", image_rows,
              ["image", "dataset_class", "detection_count", "person_count", "detected_objects"])
    summary_rows = [
        {"dataset_class": dataset_class, "object_class": object_class, "count": count}
        for (dataset_class, object_class), count in sorted(class_counts.items())
    ]
    write_csv(output / "class_summary.csv", summary_rows, ["dataset_class", "object_class", "count"])

    print(f"Analysed {len(images)} images with {args.model}.")
    print(f"Detected {len(detection_rows)} objects; {sum(r['person_count'] for r in image_rows)} are people.")
    print(f"Wrote results to: {output}")


if __name__ == "__main__":
    main()
