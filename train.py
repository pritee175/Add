"""
train.py
----------
Fine-tunes a pretrained image classifier (EfficientNet-B0 by default,
ResNet50 as a comparison option) on Dataset_Split/{train,val,test}.

Preprocessing rules followed:
    - Images are loaded from Dataset_Split at their ORIGINAL resolution.
      Resize to the model's expected input size happens here, at load
      time, via torchvision transforms -- never baked into the dataset.
    - Normalization uses the exact mean/std the pretrained model was
      trained with (via the model's own .transforms() from torchvision,
      so it can never drift out of sync with the backbone).
    - Data augmentation (flip, rotation, crop, color jitter) is applied
      ONLY to the training split. Val/test get resize + normalize only,
      nothing else -- augmenting eval data would make metrics unreliable.

USAGE:
    python train.py --model efficientnet_b0 --epochs 15
    python train.py --model resnet50 --epochs 15
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

DATA_DIR = "Dataset_Split"
CHECKPOINT_DIR = "checkpoints"
IMAGE_SIZE = 224


def get_model_and_weights(name):
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    elif name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model, weights


def build_transforms(weights):
    # Use the pretrained model's own preprocessing (resize/crop/normalize
    # stats) so eval-time preprocessing always matches what the backbone
    # was trained with.
    base = weights.transforms()

    eval_transform = base

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=base.mean, std=base.std),
    ])

    return train_transform, eval_transform


def unfreeze_last_blocks(model, model_name, n_blocks):
    """Unfreeze the last n_blocks of the backbone for stage-2 fine-tuning.
    The classifier head is assumed already trainable from stage 1."""
    if n_blocks <= 0:
        return
    if model_name == "efficientnet_b0":
        blocks = list(model.features.children())
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
    elif model_name == "resnet50":
        layer_names = ["layer4", "layer3", "layer2", "layer1"]
        for layer_name in layer_names[:n_blocks]:
            for param in getattr(model, layer_name).parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_epochs(model, train_loader, train_ds, val_loader, optimizer, criterion, device,
                class_names, n_epochs, stage_label, best_val_acc, best_ckpt_path,
                model_name, epoch_offset=0):
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device, class_names)
        global_epoch = epoch_offset + epoch
        print(f"[{stage_label}] Epoch {epoch:2d}/{n_epochs} (global {global_epoch})  "
              f"train_loss={train_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}")

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "class_names": class_names,
                "stage": stage_label,
                "global_epoch": global_epoch,
                "val_accuracy": best_val_acc,
            }, best_ckpt_path)
            print(f"  -> new best (val_acc={best_val_acc:.4f}), saved to {best_ckpt_path}")

    return best_val_acc


def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(class_names)))
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    return {
        "accuracy": acc,
        "per_class_precision": dict(zip(class_names, precision.tolist())),
        "per_class_recall": dict(zip(class_names, recall.tolist())),
        "per_class_f1": dict(zip(class_names, f1.tolist())),
        "confusion_matrix": cm.tolist(),
        "report_text": report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["efficientnet_b0", "resnet50"], default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=15,
                         help="Stage 1 epochs: head-only training (backbone frozen)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="Stage 1 learning rate (head only)")
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="Single-stage mode: freeze backbone for the whole run, no stage 2. "
                              "Omit this flag to run the default two-stage schedule instead.")
    parser.add_argument("--stage2-epochs", type=int, default=10,
                         help="Stage 2 epochs: fine-tune with last backbone blocks unfrozen. "
                              "Set to 0 to skip stage 2 even when --freeze-backbone is omitted.")
    parser.add_argument("--stage2-lr", type=float, default=1e-5,
                         help="Stage 2 learning rate (lower, since more parameters are now trainable)")
    parser.add_argument("--unfreeze-blocks", type=int, default=2,
                         help="Number of trailing backbone blocks/layers to unfreeze in stage 2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, weights = get_model_and_weights(args.model)
    train_transform, eval_transform = build_transforms(weights)

    # Stage 1 always starts with a fully frozen backbone -- only the head is trainable.
    for param in model.parameters():
        param.requires_grad = False
    head = model.classifier[1] if args.model == "efficientnet_b0" else model.fc
    for param in head.parameters():
        param.requires_grad = True

    run_stage2 = (not args.freeze_backbone) and args.stage2_epochs > 0

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_transform)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transform)

    class_names = train_ds.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.model}_best.pt")

    # ---- Stage 1: head-only, backbone frozen ----
    stage1_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(stage1_params, lr=args.lr)
    print(f"\n=== Stage 1: head-only ({sum(p.numel() for p in stage1_params):,} trainable params, "
          f"lr={args.lr}, {args.epochs} epochs) ===")
    best_val_acc = run_epochs(model, train_loader, train_ds, val_loader, optimizer, criterion, device,
                               class_names, args.epochs, "stage1-head", best_val_acc, best_ckpt_path, args.model)

    # ---- Stage 2: unfreeze last backbone blocks, fine-tune at a lower lr ----
    if run_stage2:
        unfreeze_last_blocks(model, args.model, args.unfreeze_blocks)
        stage2_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(stage2_params, lr=args.stage2_lr)
        print(f"\n=== Stage 2: unfreezing last {args.unfreeze_blocks} block(s) "
              f"({sum(p.numel() for p in stage2_params):,} trainable params, "
              f"lr={args.stage2_lr}, {args.stage2_epochs} epochs) ===")
        best_val_acc = run_epochs(model, train_loader, train_ds, val_loader, optimizer, criterion, device,
                                   class_names, args.stage2_epochs, "stage2-finetune", best_val_acc,
                                   best_ckpt_path, args.model, epoch_offset=args.epochs)
    else:
        print("\nStage 2 skipped (--freeze-backbone set or --stage2-epochs 0).")

    print("\nLoading best checkpoint for final test evaluation...")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\n=== VALIDATION SET ===")
    val_metrics = evaluate(model, val_loader, device, class_names)
    print(val_metrics["report_text"])

    print("\n=== TEST SET ===")
    test_metrics = evaluate(model, test_loader, device, class_names)
    print(test_metrics["report_text"])
    print("Confusion matrix (rows=true, cols=predicted):", class_names)
    for row in test_metrics["confusion_matrix"]:
        print(" ", row)

    results_path = os.path.join(CHECKPOINT_DIR, f"{args.model}_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": args.model,
            "stage1_epochs": args.epochs,
            "stage1_lr": args.lr,
            "stage2_ran": run_stage2,
            "stage2_epochs": args.stage2_epochs if run_stage2 else 0,
            "stage2_lr": args.stage2_lr if run_stage2 else None,
            "unfreeze_blocks": args.unfreeze_blocks if run_stage2 else 0,
            "best_val_accuracy": best_val_acc,
            "val_metrics": {k: v for k, v in val_metrics.items() if k != "report_text"},
            "test_metrics": {k: v for k, v in test_metrics.items() if k != "report_text"},
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
