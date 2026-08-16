"""
video_attention_pool.py
-------------------------
Experiment 2: Frozen ResNet50 feature extraction + LEARNED attention
pooling by video (instead of Experiment 1's uniform mean pooling),
jointly trained end-to-end with a small linear classifier.

This does NOT modify train.py, resnet50_best.pt, video_mean_pool.py, or
any existing artifact. It is a completely separate, sibling experiment
to Experiment 1 (video_mean_pool.py), reusing the same verified dataset,
checkpoint, split, and fair-comparison methodology.

Motivation (from Experiment 1's result):
    Uniform mean pooling (82.61%, 19/23) UNDERPERFORMED the frame-level
    ResNet50's own majority vote (86.96%, 20/23) on the same 23 test
    videos, and collapsed almost entirely to predicting Promotional
    (Preventive recall 33.33% vs Promotional recall 100%). A plausible
    explanation: averaging dilutes a small number of highly-informative
    frames (e.g. a single warning slogan or graphic health-consequence
    shot) by mixing them with many visually neutral frames from the same
    video. Uniform averaging cannot express "this frame matters more."

This experiment tests a narrower, more specific hypothesis than "does
temporal context help": does letting the model LEARN which frames to
weight more heavily (still with no explicit ordering/sequence
information -- attention pooling is still ordering-agnostic, just not
uniform) recover the deficit or improve Preventive recall specifically?

Design:
    1. Load the EXISTING checkpoints/resnet50_best.pt, same as
       video_mean_pool.py (frozen, domain-adapted 2048-D features).
    2. For each video, extract all frame features (same as Experiment 1).
    3. A small attention scorer (2048 -> 1, i.e. a single linear layer)
       produces one scalar score per frame.
    4. Softmax over a video's frame scores gives attention weights that
       sum to 1 for that video.
    5. Weighted sum of frame features (NOT a simple average) produces
       one 2048-D video representation.
    6. A linear classifier (2048 -> 2) predicts the video's class.
    7. The attention scorer and classifier are trained JOINTLY,
       end-to-end, on TRAIN video features. VAL videos are used for
       model selection (same convention as train.py and Experiment 1).
       TEST is evaluated exactly once.
    8. Attention weights per frame are saved, so the result can be
       qualitatively checked: does the model actually attend to
       visually/semantically meaningful frames, or is it overfitting?

Kept identical to Experiment 1 for a fair comparison:
    - Same clean 564-frame manifest / 52-23-23 video split.
    - Same frozen checkpoint (resnet50_best.pt, 79.76% frame baseline).
    - Same random seed (42).
    - Same validation-based model selection convention.
    - Same frame-baseline majority-vote comparator (86.96%, 20/23),
      recomputed here (not hardcoded) so this script is self-contained
      and independently verifiable.
    - Same explicit, documented, counted tie-break policy.

Given only 52 training videos, the attention scorer is kept deliberately
tiny (a single linear layer, no hidden layers) and trained with L2
weight decay, to limit overfitting risk on such a small dataset.

Outputs (does not touch existing checkpoints/, error_analysis/, or
experiments/mean_pool_resnet50/):
    experiments/attention_pool_resnet50/
        results.json
        video_predictions.csv
        attention_weights.csv   -- per-frame attention weight, for
                                    qualitative inspection

USAGE:
    python video_attention_pool.py
    python video_attention_pool.py --epochs 100 --lr 1e-3 --weight-decay 1e-3
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
OUT_DIR = os.path.join("experiments", "attention_pool_resnet50")
IMAGE_SIZE = 224
SEED = 42

# Same verified, authoritative clean dataset as Experiment 1
# (video_mean_pool.py). See that script's header for the full history of
# the Fy0HtLn0O2c cross-category contamination and its fix.
EXPECTED_TOTAL_FRAMES = 564
EXPECTED_SPLIT_FRAMES = {"train": 395, "val": 85, "test": 84}
EXPECTED_SPLIT_VIDEOS = {"train": 52, "val": 23, "test": 23}

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AttentionPoolClassifier(nn.Module):
    """Learned attention over a variable-length set of frame features,
    followed by a linear classifier on the resulting weighted-sum vector.

    Deliberately minimal (single linear layer for scoring, single linear
    layer for classification) given only 52 training videos."""

    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.attn_scorer = nn.Linear(feat_dim, 1)
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, frame_feats):
        """frame_feats: (n_frames, feat_dim) for ONE video.
        Returns (logits, attn_weights) where attn_weights sum to 1."""
        scores = self.attn_scorer(frame_feats).squeeze(-1)  # (n_frames,)
        weights = torch.softmax(scores, dim=0)  # (n_frames,)
        video_repr = (weights.unsqueeze(-1) * frame_feats).sum(dim=0)  # (feat_dim,)
        logits = self.classifier(video_repr).unsqueeze(0)  # (1, n_classes)
        return logits, weights


def load_frozen_feature_extractor(device):
    """Identical to video_mean_pool.py's version -- reconstruct train.py's
    exact architecture, load the existing checkpoint, strip the head."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Linear(in_features, 2)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_names = checkpoint["class_names"]
    print(f"Loaded checkpoint: stage={checkpoint.get('stage')}, "
          f"val_accuracy={checkpoint.get('val_accuracy'):.4f}, "
          f"class_names={class_names}")

    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    return model, weights, class_names, in_features


def build_eval_transform(weights):
    return weights.transforms()


def read_manifest():
    """Same assertions as Experiment 1 -- stops if the dataset has drifted
    from the verified clean 564-frame baseline."""
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
    """Identical to Experiment 1: one 2048-D feature per frame, grouped by
    video_id, with split/category/filename retained for later use."""
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
                "filename": filename,
            })

            if (i + 1) % 50 == 0:
                print(f"  extracted {i + 1}/{len(manifest_rows)} frames")

    return video_data


def group_by_split(video_data):
    """video_id -> frame dicts, verified single split/label per video
    (same assertion as Experiment 1). Returns dict split -> list of
    (video_id, feature_tensor (n_frames, 2048), label, frame_rows)."""
    grouped = defaultdict(list)
    for video_id, frames in video_data.items():
        splits = {f["split"] for f in frames}
        labels = {f["label"] for f in frames}
        assert len(splits) == 1, f"video {video_id} spans multiple splits -- should be impossible"
        assert len(labels) == 1, f"video {video_id} has multiple labels -- should be impossible"
        split = splits.pop()
        label = labels.pop()

        feats = torch.stack([f["feature"] for f in frames], dim=0)  # (n_frames, 2048)
        grouped[split].append((video_id, feats, label, frames))

    return grouped


def majority_class_baseline(train_labels, test_labels, class_names):
    counts = Counter(train_labels)
    majority_idx = counts.most_common(1)[0][0]
    majority_name = class_names[majority_idx]
    preds = [majority_idx] * len(test_labels)
    acc = accuracy_score(test_labels, preds)
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


def compute_frame_majority_vote_baseline(model, transform, device, manifest_rows,
                                          class_names, class_to_idx):
    """Recomputes the SAME frame-level-ResNet50-majority-vote comparator used
    in Experiment 1, independently in this script, so this file is
    self-contained and doesn't silently depend on Experiment 1 having been
    run first. Uses the ORIGINAL (non-Identity) classifier head."""
    weights_full = ResNet50_Weights.IMAGENET1K_V2
    full_model = resnet50(weights=weights_full)
    full_model.fc = nn.Linear(full_model.fc.in_features, len(class_names))
    full_checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    full_model.load_state_dict(full_checkpoint["model_state_dict"])
    full_model.eval().to(device)

    by_video = defaultdict(list)
    for row in manifest_rows:
        if row["split"] != "test":
            continue
        by_video[row["video_id"]].append(row)

    preds, labels, vids = [], [], []
    tie_count = 0
    with torch.no_grad():
        for vid, rows in by_video.items():
            frame_preds = []
            true_label = class_to_idx[rows[0]["category"]]
            for row in rows:
                img_path = os.path.join(DATA_DIR, "test", row["category"], row["filename"])
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                logits = full_model(x)
                frame_preds.append(int(logits.argmax(dim=1).item()))

            counts = Counter(frame_preds)
            max_count = max(counts.values())
            winners = [cls for cls, c in counts.items() if c == max_count]
            if len(winners) == 1:
                vote = winners[0]
            else:
                vote = frame_preds[0]  # deterministic: chronological first frame
                tie_count += 1

            preds.append(vote)
            labels.append(true_label)
            vids.append(vid)

    return preds, labels, vids, tie_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                         help="L2 regularization on both the attention scorer "
                              "and classifier -- important given only 52 training videos")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Step 1: frozen feature extractor from the existing checkpoint ----
    print("\n=== Loading frozen ResNet50 feature extractor from existing checkpoint ===")
    model, weights, class_names, feat_dim = load_frozen_feature_extractor(device)
    transform = build_eval_transform(weights)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # ---- Step 2: extract per-frame features, grouped by video ----
    print("\n=== Extracting frame-level features (frozen, no grad) ===")
    manifest_rows = read_manifest()
    print(f"Manifest rows (frames): {len(manifest_rows)}")
    video_data = extract_all_features(model, transform, device, manifest_rows, class_names)
    print(f"Unique videos: {len(video_data)}")

    grouped = group_by_split(video_data)
    for split in ["train", "val", "test"]:
        n = len(grouped[split])
        expected = EXPECTED_SPLIT_VIDEOS[split]
        print(f"  {split}: {n} videos (expected {expected})")
        assert n == expected, (
            f"Split '{split}' produced {n} videos, expected {expected}. "
            f"Stop -- this no longer matches the verified baseline dataset."
        )

    train_videos = grouped["train"]
    val_videos = grouped["val"]
    test_videos = grouped["test"]

    # ---- Step 3: majority-class baseline (train-derived, applied to test) ----
    train_labels_flat = [label for (_, _, label, _) in train_videos]
    test_labels_flat = [label for (_, _, label, _) in test_videos]
    maj_name, maj_acc = majority_class_baseline(train_labels_flat, test_labels_flat, class_names)
    print(f"\n=== Majority-class baseline (majority computed from TRAIN, applied to TEST) ===")
    print(f"Always predict '{maj_name}': test accuracy = {maj_acc:.4f} ({maj_acc*100:.2f}%)")

    print("\n=== TEST VIDEO CLASS DISTRIBUTION ===")
    for idx, name in enumerate(class_names):
        n = sum(1 for l in test_labels_flat if l == idx)
        print(f"  {name}: {n}")

    print(f"\nNOTE: with {len(test_videos)} test videos, accuracy can only move in "
          f"increments of 1/{len(test_videos)} = {100/len(test_videos):.2f} percentage points. "
          f"Treat this as a proof-of-concept result, not a definitive benchmark.")

    # ---- Step 4: fair comparator -- frame-level ResNet50 majority vote,
    #      recomputed independently (self-contained, doesn't assume
    #      Experiment 1 was already run) ----
    print("\n=== Recomputing frame-level ResNet50 majority-vote comparator "
          "(same methodology as Experiment 1) ===")
    fmv_preds, fmv_labels, fmv_vids, fmv_ties = compute_frame_majority_vote_baseline(
        model, transform, device, manifest_rows, class_names, class_to_idx
    )
    fmv_metrics = evaluate(fmv_preds, fmv_labels, class_names)
    print(f"Frame-baseline majority-vote video accuracy: {fmv_metrics['accuracy']:.4f} "
          f"({fmv_metrics['accuracy']*100:.2f}%) on {len(fmv_vids)} test videos, "
          f"{fmv_ties} tie(s)")

    # ---- Step 5: train attention-pooling classifier JOINTLY, end-to-end ----
    print(f"\n=== Training attention-pooling classifier ({feat_dim} -> attn -> "
          f"{len(class_names)}) on {len(train_videos)} training videos ===")
    net = AttentionPoolClassifier(feat_dim, len(class_names)).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        net.train()
        optimizer.zero_grad()
        total_loss = 0.0
        # One video at a time (variable-length attention over that video's
        # own frames) -- accumulate gradients across the training videos
        # before stepping, i.e. full-batch gradient descent over 52 videos.
        for video_id, feats, label, _ in train_videos:
            feats = feats.to(device)
            logits, _ = net(feats)
            target = torch.tensor([label], device=device)
            loss = criterion(logits, target)
            loss.backward()
            total_loss += loss.item()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            val_preds = []
            for video_id, feats, label, _ in val_videos:
                feats = feats.to(device)
                logits, _ = net(feats)
                val_preds.append(int(logits.argmax(dim=1).item()))
            val_labels = [label for (_, _, label, _) in val_videos]
            val_acc = accuracy_score(val_labels, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_videos)
            print(f"  epoch {epoch:3d}/{args.epochs}  avg_train_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\nBest val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    net.load_state_dict(best_state)

    # ---- Step 6: evaluate ONCE on test videos, saving attention weights ----
    print("\n=== VIDEO-LEVEL TEST EVALUATION (primary result) ===")
    net.eval()
    test_preds, test_labels_out = [], []
    attention_rows = []
    with torch.no_grad():
        for video_id, feats, label, frame_rows in test_videos:
            feats_dev = feats.to(device)
            logits, attn_weights = net(feats_dev)
            pred = int(logits.argmax(dim=1).item())
            test_preds.append(pred)
            test_labels_out.append(label)

            attn_weights = attn_weights.cpu().tolist()
            for row, w in zip(frame_rows, attn_weights):
                attention_rows.append({
                    "video_id": video_id,
                    "filename": row["filename"],
                    "true_class": class_names[label],
                    "predicted_class": class_names[pred],
                    "attention_weight": w,
                })

    video_metrics = evaluate(test_preds, test_labels_out, class_names)
    print(video_metrics["report_text"])
    print("Confusion matrix (rows=true, cols=predicted):", class_names)
    for row in video_metrics["confusion_matrix"]:
        print(" ", row)
    print(f"\nAttention-pooling video accuracy: {video_metrics['accuracy']:.4f} "
          f"({video_metrics['accuracy']*100:.2f}%) on {len(test_videos)} test videos")

    # ---- Step 7: the fair, three-way, same-unit comparison ----
    print("\n=== FAIR COMPARISON (all on the same 23 test videos) ===")
    print(f"Majority-class baseline:               {maj_acc*100:.2f}%")
    print(f"Frame-level ResNet50 + majority vote:   {fmv_metrics['accuracy']*100:.2f}%  <- Experiment 0 comparator")
    print(f"Attention-pooling (this experiment):    {video_metrics['accuracy']*100:.2f}%")
    print("(Run video_mean_pool.py separately for the mean-pooling number, 82.61% "
          "in the prior run, for the full three-experiment table.)")

    # ---- Save everything ----
    with open(os.path.join(OUT_DIR, "video_predictions.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_id", "split", "n_frames", "true_class", "predicted_class"])
        for video_id, feats, label, frame_rows in test_videos:
            idx = [v[0] for v in test_videos].index(video_id)
            pred = test_preds[idx]
            writer.writerow([video_id, "test", len(frame_rows), class_names[label], class_names[pred]])

    with open(os.path.join(OUT_DIR, "attention_weights.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "video_id", "filename", "true_class", "predicted_class", "attention_weight"
        ])
        writer.writeheader()
        writer.writerows(attention_rows)

    results = {
        "experiment": "attention_pool_resnet50",
        "description": "Frozen fine-tuned ResNet50 features, LEARNED attention "
                        "pooling per video (joint end-to-end training with the "
                        "classifier), evaluated on the same clean 564-frame "
                        "dataset and 23 test videos as Experiment 1 (mean pooling).",
        "based_on_checkpoint": CHECKPOINT_PATH,
        "n_videos": {
            "train": len(train_videos), "val": len(val_videos), "test": len(test_videos)
        },
        "classifier_epochs": args.epochs,
        "classifier_lr": args.lr,
        "classifier_weight_decay": args.weight_decay,
        "best_val_accuracy": best_val_acc,
        "majority_class_baseline": {"class": maj_name, "test_accuracy": maj_acc},
        "frame_baseline_majority_vote_video_metrics": {
            k: v for k, v in fmv_metrics.items() if k != "report_text"
        },
        "frame_baseline_majority_vote_ties": fmv_ties,
        "attention_pooling_video_test_metrics": {
            k: v for k, v in video_metrics.items() if k != "report_text"
        },
        "note": "Fair comparison unit is 23 test videos throughout. Compare "
                "attention_pooling_video_test_metrics.accuracy against "
                "frame_baseline_majority_vote_video_metrics.accuracy (this script) "
                "and against Experiment 1's mean-pooling result (82.61%, see "
                "experiments/mean_pool_resnet50/results.json) for the full "
                "three-way table. The clean 79.76% frame-level ResNet50 number "
                "(84 independent frames) is a DIFFERENT prediction unit.",
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}/results.json")
    print(f"Video predictions saved to {OUT_DIR}/video_predictions.csv")
    print(f"Per-frame attention weights saved to {OUT_DIR}/attention_weights.csv")
    print("\nExisting checkpoints/, error_analysis/, and experiments/mean_pool_resnet50/ "
          "were NOT modified.")


if __name__ == "__main__":
    main()