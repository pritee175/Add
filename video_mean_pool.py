"""
video_mean_pool.py
--------------------
Experiment 1: Frozen ResNet50 feature extraction + mean pooling by video,
followed by a small linear classifier trained on video-level features.

This does NOT modify train.py, resnet50_best.pt, or any existing artifact.
It is a completely separate experiment.

Design (see conversation record for full rationale):
    1. Load the EXISTING checkpoints/resnet50_best.pt (already fine-tuned
       via train.py's two-stage schedule -- these are domain-adapted
       features, not raw ImageNet features).
    2. Replace model.fc with nn.Identity() to expose the 2048-D
       pre-classification feature vector.
    3. Extract that feature for every frame in Dataset_Split/{train,val,test}
       using split_manifest.csv to know each frame's video_id and split.
    4. Mean-pool features across all frames belonging to the same video_id
       (variable-length -- no padding, no fixed sequence length, since
       frame counts per video vary considerably and many videos have very
       few frames).
    5. Train a small linear classifier (2048 -> 2) on TRAIN video features
       only. Use VAL video features for model selection (best epoch by
       val accuracy, same convention as train.py). Evaluate exactly once
       on TEST video features.
    6. Also compute and report the majority-class baseline on the test
       videos, so the mean-pooled result has a meaningful floor to beat.

Outputs (does not touch existing checkpoints/ or error_analysis/):
    experiments/mean_pool_resnet50/
        results.json
        video_predictions.csv

Safety checks included:
    - Fixed random seed (42) for reproducibility.
    - Asserts the manifest matches the verified, authoritative CLEAN dataset
      (564 frames, after removing the Fy0HtLn0O2c cross-category
      contamination; 395/85/84 train/val/test frames; 52/23/23 videos).
      If these don't match, the script stops rather than silently running
      against a changed dataset.
    - Asserts no video_id spans more than one split or more than one label.
    - Majority-class baseline is computed from TRAIN labels only (no test
      information used), and test class distribution is reported separately.

This script assumes checkpoints/resnet50_best.pt is the CORRECTED run
(79.76% test accuracy on the clean 564-frame dataset), not the earlier
77.38% run on the contaminated 566-frame dataset. If you retrain, re-run
this script fresh -- do not mix checkpoints from different dataset states.

Preprocessing note: this script reuses weights.transforms() exactly as
train.py's build_transforms() does for its eval_transform (val/test) --
verified against train.py's actual code, not assumed.

USAGE:
    python video_mean_pool.py
    python video_mean_pool.py --epochs 50 --lr 1e-3
"""

import os
import argparse
import json
import csv
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

DATA_DIR = "Dataset_Split"
MANIFEST_PATH = os.path.join(DATA_DIR, "split_manifest.csv")
CHECKPOINT_PATH = os.path.join("checkpoints", "resnet50_best.pt")
OUT_DIR = os.path.join("experiments", "mean_pool_resnet50")
IMAGE_SIZE = 224
SEED = 42

# Expected values from the verified, authoritative split_manifest.csv,
# AFTER removing the Fy0HtLn0O2c cross-category contamination (2 frames
# that had been extracted into both Preventive and Promotional folders)
# and regenerating the split. Verified directly against the current
# manifest via PowerShell Group-Object, not inferred.
#
# History: the original 566-frame / 99-video dataset contained one video
# (Fy0HtLn0O2c) present under both categories, causing it to span both
# train and val with conflicting labels. That produced the now-superseded
# 77.38% ResNet50 result. After deleting the 2 contaminated frames and
# re-running build_split.py, this is the clean, leakage-verified dataset.
EXPECTED_TOTAL_FRAMES = 564
EXPECTED_SPLIT_FRAMES = {"train": 395, "val": 85, "test": 84}
EXPECTED_SPLIT_VIDEOS = {"train": 52, "val": 23, "test": 23}

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_frozen_feature_extractor(device):
    """Reconstruct the exact architecture train.py used for resnet50,
    load the existing best checkpoint, then strip the classification
    head so forward() returns the 2048-D pooled feature instead of logits."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Linear(in_features, 2)  # same shape train.py used to save

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_names = checkpoint["class_names"]
    print(f"Loaded checkpoint: stage={checkpoint.get('stage')}, "
          f"val_accuracy={checkpoint.get('val_accuracy'):.4f}, "
          f"class_names={class_names}")

    # Now strip the head to expose the 2048-D feature.
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model, weights, class_names, in_features


def build_eval_transform(weights):
    # Same deterministic eval-time transform train.py uses for val/test --
    # we want identical preprocessing so features are comparable to what
    # the classifier head was originally trained to see.
    return weights.transforms()


def read_manifest():
    """Returns list of dicts: filename, category, video_id, split.
    Asserts the manifest matches the authoritative, verified CLEAN dataset --
    stops immediately if the dataset has changed since the verified clean
    79.76% ResNet50 baseline."""
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == EXPECTED_TOTAL_FRAMES, (
        f"Manifest has {len(rows)} rows, expected {EXPECTED_TOTAL_FRAMES}. "
        f"The dataset has changed since the verified clean 79.76% ResNet50 "
        f"baseline -- stop and investigate before proceeding."
    )
    assert set(r["split"] for r in rows) == {"train", "val", "test"}, (
        f"Unexpected split values found: {set(r['split'] for r in rows)}"
    )

    split_counts = Counter(r["split"] for r in rows)
    for split, expected in EXPECTED_SPLIT_FRAMES.items():
        actual = split_counts[split]
        assert actual == expected, (
            f"Split '{split}' has {actual} frames, expected {expected}. "
            f"Dataset/split has changed since the verified baseline -- stop."
        )

    return rows


def extract_all_features(model, transform, device, manifest_rows, class_names):
    """Runs every frame through the frozen extractor once. Returns:
    features[video_id] -> list of (feature_tensor, split, category)
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    video_data = defaultdict(list)

    with torch.no_grad():
        for i, row in enumerate(manifest_rows):
            split = row["split"]
            category = row["category"]
            video_id = row["video_id"]
            filename = row["filename"]

            img_path = os.path.join(DATA_DIR, split, category, filename)
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            feat = model(x).squeeze(0).cpu()  # (2048,)
            label = class_to_idx[category]

            video_data[video_id].append({
                "feature": feat,
                "split": split,
                "label": label,
                "category": category,
            })

            if (i + 1) % 50 == 0:
                print(f"  extracted {i + 1}/{len(manifest_rows)} frames")

    return video_data


def mean_pool_by_video(video_data):
    """video_data: video_id -> list of frame dicts (must all share split/label
    for a given video_id, since split_manifest.csv assigns whole videos to
    one split). Returns dict split -> list of (video_id, pooled_feature, label).
    """
    pooled = defaultdict(list)
    for video_id, frames in video_data.items():
        splits = {f["split"] for f in frames}
        labels = {f["label"] for f in frames}
        assert len(splits) == 1, f"video {video_id} spans multiple splits -- should be impossible"
        assert len(labels) == 1, f"video {video_id} has multiple labels -- should be impossible"
        split = splits.pop()
        label = labels.pop()

        feats = torch.stack([f["feature"] for f in frames], dim=0)  # (n_frames, 2048)
        mean_feat = feats.mean(dim=0)  # (2048,)

        pooled[split].append((video_id, mean_feat, label, len(frames)))

    return pooled


def to_tensors(pooled_split_list):
    X = torch.stack([item[1] for item in pooled_split_list], dim=0)
    y = torch.tensor([item[2] for item in pooled_split_list], dtype=torch.long)
    video_ids = [item[0] for item in pooled_split_list]
    n_frames = [item[3] for item in pooled_split_list]
    return X, y, video_ids, n_frames


def majority_class_baseline(y_train, y_test, class_names):
    """What accuracy would a trivial always-predict-majority-class model get?"""
    counts = torch.bincount(y_train, minlength=len(class_names))
    majority_idx = int(counts.argmax())
    majority_name = class_names[majority_idx]
    preds = torch.full_like(y_test, majority_idx)
    acc = accuracy_score(y_test.tolist(), preds.tolist())
    return majority_name, acc


def evaluate(preds, labels, class_names):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(class_names)))
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    report = classification_report(labels, preds, target_names=class_names, digits=4)
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
    parser.add_argument("--epochs", type=int, default=100,
                         help="Epochs for the small linear classifier (video-level features, tiny dataset)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                         help="L2 regularization -- important given very few training videos")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Step 1: load frozen, already-domain-adapted feature extractor ----
    print("\n=== Loading frozen ResNet50 feature extractor from existing checkpoint ===")
    model, weights, class_names, feat_dim = load_frozen_feature_extractor(device)
    transform = build_eval_transform(weights)

    # ---- Step 2: extract features for every frame using the manifest ----
    print("\n=== Extracting frame-level features (frozen, no grad) ===")
    manifest_rows = read_manifest()
    print(f"Manifest rows (frames): {len(manifest_rows)}")
    video_data = extract_all_features(model, transform, device, manifest_rows, class_names)
    print(f"Unique videos: {len(video_data)}")

    # ---- Step 3: mean-pool per video, split into train/val/test ----
    print("\n=== Mean-pooling features by video ===")
    pooled = mean_pool_by_video(video_data)
    for split in ["train", "val", "test"]:
        n = len(pooled[split])
        expected = EXPECTED_SPLIT_VIDEOS[split]
        print(f"  {split}: {n} videos (expected {expected})")
        assert n == expected, (
            f"Split '{split}' produced {n} videos, expected {expected}. "
            f"Stop -- this no longer matches the verified baseline dataset."
        )

    X_train, y_train, vid_train, nf_train = to_tensors(pooled["train"])
    X_val, y_val, vid_val, nf_val = to_tensors(pooled["val"])
    X_test, y_test, vid_test, nf_test = to_tensors(pooled["test"])

    X_train, X_val, X_test = X_train.to(device), X_val.to(device), X_test.to(device)
    y_train, y_val, y_test = y_train.to(device), y_val.to(device), y_test.to(device)

    # ---- Step 4: majority-class baseline (the floor any real result must beat) ----
    # NOTE: majority class is chosen from TRAIN labels only -- never from test --
    # so this baseline uses no test information, exactly like a real model would.
    maj_name, maj_acc = majority_class_baseline(y_train.cpu(), y_test.cpu(), class_names)
    print(f"\n=== Majority-class baseline (majority computed from TRAIN, applied to TEST) ===")
    print(f"Always predict '{maj_name}': test accuracy = {maj_acc:.4f} ({maj_acc*100:.2f}%)")

    print("\n=== TEST VIDEO CLASS DISTRIBUTION (for reference only, not used to pick the baseline) ===")
    for idx, name in enumerate(class_names):
        n = int((y_test.cpu() == idx).sum().item())
        print(f"  {name}: {n}")

    print(f"\nNOTE: with {len(y_test)} test videos, accuracy can only move in "
          f"increments of 1/{len(y_test)} = {100/len(y_test):.2f} percentage points. "
          f"Treat this as a proof-of-concept result, not a definitive benchmark.")

    # ---- Step 5: train a small linear classifier on video-level features ----
    print(f"\n=== Training linear classifier ({feat_dim} -> {len(class_names)}) on "
          f"{len(y_train)} training videos ===")
    classifier = nn.Linear(feat_dim, len(class_names)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = accuracy_score(y_val.cpu().tolist(), val_preds.cpu().tolist())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{args.epochs}  train_loss={loss.item():.4f}  val_acc={val_acc:.4f}")

    print(f"\nBest val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    classifier.load_state_dict(best_state)

    # ---- Step 6: evaluate ONCE on test videos ----
    print("\n=== VIDEO-LEVEL TEST EVALUATION (primary result) ===")
    classifier.eval()
    with torch.no_grad():
        test_logits = classifier(X_test)
        test_preds = test_logits.argmax(dim=1).cpu().tolist()
    test_labels = y_test.cpu().tolist()

    video_metrics = evaluate(test_preds, test_labels, class_names)
    print(video_metrics["report_text"])
    print("Confusion matrix (rows=true, cols=predicted):", class_names)
    for row in video_metrics["confusion_matrix"]:
        print(" ", row)
    print(f"\nVideo-level accuracy: {video_metrics['accuracy']:.4f} "
          f"({video_metrics['accuracy']*100:.2f}%)  vs majority baseline {maj_acc*100:.2f}%")

    # ---- Step 7: secondary frame-level view (same prediction applied to every
    #      frame of its video) -- clearly labeled as NOT directly comparable
    #      to the clean 79.76% independent-frame ResNet50 baseline ----
    frame_preds, frame_labels = [], []
    vid_to_pred = dict(zip(vid_test, test_preds))
    for row in manifest_rows:
        if row["split"] != "test":
            continue
        vid = row["video_id"]
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        frame_preds.append(vid_to_pred[vid])
        frame_labels.append(class_to_idx[row["category"]])

    frame_metrics = evaluate(frame_preds, frame_labels, class_names)
    print("\n=== SECONDARY: 'frame accuracy under video-level prediction' ===")
    print("(This is NOT a new 84-frame classifier and is NOT directly comparable to")
    print(" the clean 79.76% independent-frame ResNet50 baseline -- it reuses one")
    print(" video-level prediction across every frame belonging to that video.)")
    print(frame_metrics["report_text"])

    # ---- Step 8: fair comparison -- majority-vote the ORIGINAL frame-level
    #      ResNet50's independent predictions up to video level, using the
    #      SAME 23 test videos and SAME prediction unit as the mean-pool
    #      experiment. This is the only apples-to-apples comparison; the
    #      raw 79.76% (84 independent frames) is a different unit entirely. ----
    print("\n=== FAIR COMPARISON: majority-voting the frame-level ResNet50's own")
    print("    predictions up to video level (same 23 test videos, same unit) ===")
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    frame_level_preds_by_video = defaultdict(list)
    for row in manifest_rows:
        if row["split"] != "test":
            continue
        frame_level_preds_by_video[row["video_id"]].append(row)

    weights_full = ResNet50_Weights.IMAGENET1K_V2
    full_model = resnet50(weights=weights_full)
    full_model.fc = nn.Linear(full_model.fc.in_features, len(class_names))
    full_checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    full_model.load_state_dict(full_checkpoint["model_state_dict"])
    full_model.eval().to(device)

    majority_vote_preds, majority_vote_labels, majority_vote_vids = [], [], []
    tie_count = 0
    with torch.no_grad():
        for vid, rows in frame_level_preds_by_video.items():
            frame_preds_this_vid = []
            true_label = class_to_idx[rows[0]["category"]]
            for row in rows:
                img_path = os.path.join(DATA_DIR, "test", row["category"], row["filename"])
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                logits = full_model(x)
                frame_preds_this_vid.append(int(logits.argmax(dim=1).item()))

            # Explicit, pre-defined majority-vote rule with a documented tie
            # policy -- with many videos having very few frames (some with
            # exactly 2), ties are a real possibility and must not depend on
            # arbitrary set()/dict iteration order.
            counts = Counter(frame_preds_this_vid)
            max_count = max(counts.values())
            winners = [cls for cls, c in counts.items() if c == max_count]
            if len(winners) == 1:
                vote = winners[0]
            else:
                # Deterministic tie-break: first frame's prediction (frames
                # are in chronological order per split_manifest.csv). Ties
                # are counted and reported separately, never silently hidden.
                vote = frame_preds_this_vid[0]
                tie_count += 1

            majority_vote_preds.append(vote)
            majority_vote_labels.append(true_label)
            majority_vote_vids.append(vid)

    if tie_count > 0:
        print(f"NOTE: {tie_count}/{len(majority_vote_vids)} test videos had a tied "
              f"frame-level vote; tie-break rule used = first frame's prediction "
              f"(chronological order). This is reported, not hidden.")

    majvote_metrics = evaluate(majority_vote_preds, majority_vote_labels, class_names)
    print(majvote_metrics["report_text"])
    print(f"Frame-baseline majority-vote video accuracy: {majvote_metrics['accuracy']:.4f} "
          f"({majvote_metrics['accuracy']*100:.2f}%) on {len(majority_vote_vids)} test videos")
    print(f"Mean-pool linear-classifier video accuracy:  {video_metrics['accuracy']:.4f} "
          f"({video_metrics['accuracy']*100:.2f}%) on {len(vid_test)} test videos")
    print("^ THIS is the fair, same-unit comparison for the paper -- not 79.76% vs the video number.")

    # ---- Save everything ----
    with open(os.path.join(OUT_DIR, "video_predictions.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_id", "split", "n_frames", "true_class", "predicted_class"])
        for vid, feat, label, nf in pooled["test"]:
            pred = vid_to_pred[vid]
            writer.writerow([vid, "test", nf, class_names[label], class_names[pred]])

    results = {
        "experiment": "mean_pool_resnet50",
        "description": "Frozen fine-tuned ResNet50 features, mean-pooled per video, "
                        "small linear classifier trained on video-level features.",
        "based_on_checkpoint": CHECKPOINT_PATH,
        "n_videos": {s: len(pooled[s]) for s in ["train", "val", "test"]},
        "classifier_epochs": args.epochs,
        "classifier_lr": args.lr,
        "classifier_weight_decay": args.weight_decay,
        "best_val_accuracy": best_val_acc,
        "majority_class_baseline": {"class": maj_name, "test_accuracy": maj_acc},
        "video_level_test_metrics": {k: v for k, v in video_metrics.items() if k != "report_text"},
        "secondary_frame_level_metrics": {k: v for k, v in frame_metrics.items() if k != "report_text"},
        "frame_baseline_majority_vote_video_metrics": {
            k: v for k, v in majvote_metrics.items() if k != "report_text"
        },
        "frame_baseline_majority_vote_ties": tie_count,
        "note": "PRIMARY fair comparison: mean-pool video accuracy vs frame-baseline "
                "majority-vote video accuracy -- both use the same 23 test videos as "
                "the prediction unit. The raw clean 79.76% frame-level ResNet50 number "
                "(84 independent frames) is a DIFFERENT prediction unit and should not "
                "be compared directly to either video-level number.",
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}/results.json")
    print(f"Video predictions saved to {OUT_DIR}/video_predictions.csv")
    print("\nExisting checkpoints/ and error_analysis/ were NOT modified.")


if __name__ == "__main__":
    main()