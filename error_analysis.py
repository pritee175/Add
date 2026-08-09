"""
error_analysis.py
-----------------
Runs both trained classifiers on Dataset_Split/test, exports per-image
predictions, collects misclassifications, copies error examples into a
review folder, and produces a hard-case annotation CSV for manual or
semi-manual failure analysis.

USAGE:
    python error_analysis.py

Outputs:
    error_analysis/predictions.csv
    error_analysis/errors.csv
    error_analysis/hard_cases.csv
    error_analysis/hard_cases_annotated.csv
    error_analysis/misclassified/<model>/<true>_predicted_as_<pred>/...
"""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset_Split" / "test"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "error_analysis"
MISCLASSIFIED_DIR = OUTPUT_DIR / "misclassified"

MODEL_SPECS = {
    "efficientnet_b0": {
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "build": efficientnet_b0,
        "checkpoint": CHECKPOINT_DIR / "efficientnet_b0_best.pt",
    },
    "resnet50": {
        "weights": ResNet50_Weights.IMAGENET1K_V2,
        "build": resnet50,
        "checkpoint": CHECKPOINT_DIR / "resnet50_best.pt",
    },
}


def build_model(name):
    spec = MODEL_SPECS[name]
    weights = spec["weights"]
    model = spec["build"](weights=weights)

    if name == "efficientnet_b0":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    return model, weights


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint


def get_test_loader(weights):
    transform = weights.transforms()
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    return dataset, loader


def parse_video_id(filename, category):
    name = Path(filename).stem
    prefix = category.upper()
    if name.startswith(prefix + "_"):
        rest = name[len(prefix) + 1 :]
        return rest.rsplit("_F", 1)[0]
    if "_F" in name:
        return name.rsplit("_F", 1)[0]
    return name


def infer(model, loader, device, class_names):
    records = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs)
            confs, preds = probs.max(dim=1)
            for i in range(len(labels)):
                records.append(
                    {
                        "true_idx": int(labels[i].item()),
                        "pred_idx": int(preds[i].item()),
                        "pred_class": class_names[int(preds[i].item())],
                        "confidence": float(confs[i].item()),
                        "probabilities": probs[i].cpu().tolist(),
                    }
                )
    return records


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def classify_failure_cause(image_path):
    """Return a likely failure cause tag for a hard case image."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            aspect = max(width, height) / max(1, min(width, height))
    except Exception:
        return "insufficient_information"

    name = image_path.name.lower()
    parent = image_path.parent.name.lower()

    if "text" in name or "subtitle" in name or "slide" in name:
        return "text_heavy_preventive_ad"
    if "soda" in name or "music" in name or "stag" in name or "choice" in name:
        return "surrogate_advertising"
    if "celeb" in name or "actor" in name:
        return "celebrity_only_no_product"
    if parent == "preventive" and aspect > 1.8:
        return "text_heavy_preventive_ad"
    if parent == "promotional" and aspect > 1.8:
        return "unclear_visual_context"
    return "insufficient_information"


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Missing test data directory: {DATA_DIR}")

    ensure_dir(OUTPUT_DIR)
    ensure_dir(MISCLASSIFIED_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_records = {}
    class_names = None
    sample_paths = None

    for model_name in ["efficientnet_b0", "resnet50"]:
        spec = MODEL_SPECS[model_name]
        checkpoint_path = spec["checkpoint"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {model_name}: {checkpoint_path}. "
                f"Retrain with: python train.py --model {model_name} --epochs 15 --freeze-backbone"
            )

        model, weights = build_model(model_name)
        checkpoint = load_checkpoint(model, checkpoint_path, device)

        dataset, loader = get_test_loader(weights)
        if class_names is None:
            class_names = dataset.classes
            sample_paths = [Path(p) for p, _ in dataset.samples]
        elif dataset.classes != class_names:
            raise RuntimeError(f"Class order mismatch for {model_name}: {dataset.classes} vs {class_names}")

        if "class_names" in checkpoint and list(checkpoint["class_names"]) != class_names:
            raise RuntimeError(
                f"Checkpoint class_names mismatch for {model_name}: {checkpoint['class_names']} vs {class_names}"
            )

        print(f"Running inference for {model_name} on {len(dataset)} test images")
        model_records[model_name] = infer(model, loader, device, class_names)

    rows = []
    for idx, image_path in enumerate(sample_paths):
        rel = image_path.relative_to(BASE_DIR)
        true_class = image_path.parent.name
        video_id = parse_video_id(image_path.name, true_class)

        row = {
            "filename": rel.as_posix(),
            "video_id": video_id,
            "true_class": true_class,
        }
        for model_name in ["efficientnet_b0", "resnet50"]:
            record = model_records[model_name][idx]
            row[f"{model_name}_predicted_class"] = record["pred_class"]
            row[f"{model_name}_confidence"] = f"{record['confidence']:.6f}"
            row[f"{model_name}_probabilities"] = json.dumps(record["probabilities"])
        rows.append(row)

    predictions_path = OUTPUT_DIR / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    error_rows = []
    hard_case_rows = []
    confusion_rows = {}
    summary = {}

    for model_name in ["efficientnet_b0", "resnet50"]:
        preds = [r[f"{model_name}_predicted_class"] for r in rows]
        truths = [r["true_class"] for r in rows]
        model_confusion = confusion_matrix(truths, preds, labels=class_names)
        confusion_rows[model_name] = model_confusion.tolist()
        summary[model_name] = {
            "errors": int(sum(p != t for p, t in zip(preds, truths))),
            "total": len(rows),
        }

        for row in rows:
            pred_class = row[f"{model_name}_predicted_class"]
            if pred_class != row["true_class"]:
                error_rows.append(
                    {
                        "model": model_name,
                        "filename": row["filename"],
                        "video_id": row["video_id"],
                        "true_class": row["true_class"],
                        "predicted_class": pred_class,
                        "confidence": row[f"{model_name}_confidence"],
                    }
                )

                src = BASE_DIR / Path(row["filename"])
                dst = MISCLASSIFIED_DIR / model_name / f"{row['true_class']}_predicted_as_{pred_class}"
                ensure_dir(dst)
                shutil.copy2(src, dst / src.name)

    errors_path = OUTPUT_DIR / "errors.csv"
    with errors_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "filename", "video_id", "true_class", "predicted_class", "confidence"])
        writer.writeheader()
        writer.writerows(error_rows)

    for row in rows:
        e_wrong = row["efficientnet_b0_predicted_class"] != row["true_class"]
        r_wrong = row["resnet50_predicted_class"] != row["true_class"]
        if e_wrong and r_wrong:
            hard_case_rows.append(
                {
                    "filename": row["filename"],
                    "video_id": row["video_id"],
                    "true_class": row["true_class"],
                    "efficientnet_b0_predicted_class": row["efficientnet_b0_predicted_class"],
                    "efficientnet_b0_confidence": row["efficientnet_b0_confidence"],
                    "resnet50_predicted_class": row["resnet50_predicted_class"],
                    "resnet50_confidence": row["resnet50_confidence"],
                }
            )

    hard_cases_path = OUTPUT_DIR / "hard_cases.csv"
    with hard_cases_path.open("w", newline="", encoding="utf-8") as fh:
        if hard_case_rows:
            writer = csv.DictWriter(fh, fieldnames=list(hard_case_rows[0].keys()))
            writer.writeheader()
            writer.writerows(hard_case_rows)
        else:
            fh.write("filename,video_id,true_class,efficientnet_b0_predicted_class,efficientnet_b0_confidence,resnet50_predicted_class,resnet50_confidence\n")

    annotated_rows = []
    for row in hard_case_rows:
        image_path = BASE_DIR / Path(row["filename"])
        row = dict(row)
        row["likely_failure_cause"] = classify_failure_cause(image_path)
        annotated_rows.append(row)

    annotated_path = OUTPUT_DIR / "hard_cases_annotated.csv"
    with annotated_path.open("w", newline="", encoding="utf-8") as fh:
        if annotated_rows:
            writer = csv.DictWriter(fh, fieldnames=list(annotated_rows[0].keys()))
            writer.writeheader()
            writer.writerows(annotated_rows)
        else:
            fh.write("filename,video_id,true_class,efficientnet_b0_predicted_class,efficientnet_b0_confidence,resnet50_predicted_class,resnet50_confidence,likely_failure_cause\n")

    both_wrong = len(hard_case_rows)
    only_eff_wrong = sum(
        row["efficientnet_b0_predicted_class"] != row["true_class"]
        and row["resnet50_predicted_class"] == row["true_class"]
        for row in rows
    )
    only_res_wrong = sum(
        row["resnet50_predicted_class"] != row["true_class"]
        and row["efficientnet_b0_predicted_class"] == row["true_class"]
        for row in rows
    )

    cause_counts = Counter(r["likely_failure_cause"] for r in annotated_rows)

    print("\n=== Error Analysis Summary ===")
    for model_name in ["efficientnet_b0", "resnet50"]:
        errs = summary[model_name]["errors"]
        total = summary[model_name]["total"]
        print(f"{model_name}: {errs}/{total} errors ({errs / total:.4f})")

    print(f"Both models wrong: {both_wrong}")
    print(f"Only EfficientNet-B0 wrong: {only_eff_wrong}")
    print(f"Only ResNet50 wrong: {only_res_wrong}")

    print("\nConfusion matrix: efficientnet_b0 (rows=true, cols=predicted)")
    print(confusion_rows["efficientnet_b0"])
    print("Confusion matrix: resnet50 (rows=true, cols=predicted)")
    print(confusion_rows["resnet50"])

    print("\nHard-case cause breakdown")
    for cause, count in cause_counts.most_common():
        print(f"  {cause}: {count}")

    print("\nWrote:")
    print(f"  {predictions_path}")
    print(f"  {errors_path}")
    print(f"  {hard_cases_path}")
    print(f"  {annotated_path}")
    print(f"  {MISCLASSIFIED_DIR}")


if __name__ == "__main__":
    main()