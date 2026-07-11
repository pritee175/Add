"""
build_split.py
-----------------
Splits the dataset into train/val/test at the VIDEO level, not the frame
level. This is critical: frames extracted 1fps apart from the same video
are near-duplicates of each other. Splitting by frame would let the same
scene appear in both train and test, so the model "memorizes" videos
instead of learning to generalize -- accuracy numbers would look great
but be meaningless on unseen ads.

Every frame from a given source video goes entirely into ONE split.

Algorithm: for each class, group frames by video_id (parsed from the
filename), then greedily assign whole videos to train/val/test -- each
video goes to whichever split is currently furthest below its target
frame-count share. This approximates the target ratio at the image level
while keeping every video's frames together.

USAGE:
    python build_split.py --ratios 70 15 15 --seed 42
"""

import os
import re
import csv
import random
import shutil
import argparse

SOURCE_DIR = "Dataset"  # original, untouched images -- resize happens at load time in the training pipeline, not here
OUTPUT_DIR = "Dataset_Split"
CATEGORIES = ["Promotional", "Preventive"]


def parse_video_id(filename, category):
    """
    Filenames look like:
        PROMOTIONAL_<video_id>_F001.jpg
        PREVENTIVE_<video_id>_F001.jpg
        PREV004_F012.jpg              (teammate-contributed, no real video_id)
    Returns a grouping key that keeps all frames from the same source together.
    """
    name = os.path.splitext(filename)[0]
    prefix = category.upper()
    if name.startswith(prefix + "_"):
        rest = name[len(prefix) + 1:]
        video_id = rest.rsplit("_F", 1)[0]
        return video_id
    # teammate-style: PREV004_F012 -> PREV004
    m = re.match(r"^([A-Za-z]+\d+)_F\d+$", name)
    if m:
        return m.group(1)
    # fallback: strip trailing _F### if present, else whole name is its own group
    m2 = re.match(r"^(.*)_F\d+$", name)
    return m2.group(1) if m2 else name


def greedy_split(video_frame_counts, ratios, seed):
    """video_frame_counts: list of (video_id, count). Returns dict video_id -> split name."""
    rng = random.Random(seed)
    videos = list(video_frame_counts)
    rng.shuffle(videos)
    # sort by descending frame count so large videos get placed first (better packing)
    videos.sort(key=lambda x: -x[1])

    total = sum(c for _, c in videos)
    targets = {name: total * r for name, r in ratios.items()}
    assigned_count = {name: 0 for name in ratios}
    assignment = {}

    for video_id, count in videos:
        # pick the split currently furthest below its target (as a fraction)
        def deficit(name):
            return targets[name] - assigned_count[name]
        best = max(ratios.keys(), key=deficit)
        assignment[video_id] = best
        assigned_count[best] += count

    return assignment, assigned_count, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", type=float, nargs=3, default=[70, 15, 15],
                         help="train val test ratios (must sum to 100)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_r, val_r, test_r = [r / 100 for r in args.ratios]
    ratios = {"train": train_r, "val": val_r, "test": test_r}

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for split in ratios:
        for cat in CATEGORIES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cat), exist_ok=True)

    manifest_rows = []

    for cat in CATEGORIES:
        src_dir = os.path.join(SOURCE_DIR, cat)
        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))

        groups = {}
        for f in files:
            vid = parse_video_id(f, cat)
            groups.setdefault(vid, []).append(f)

        video_frame_counts = [(vid, len(fs)) for vid, fs in groups.items()]
        assignment, assigned_count, total = greedy_split(video_frame_counts, ratios, args.seed)

        print(f"\n{cat}: {len(files)} frames from {len(groups)} source videos")
        for split in ["train", "val", "test"]:
            pct = 100 * assigned_count[split] / total
            n_videos = sum(1 for v in assignment.values() if v == split)
            print(f"  {split:5s}: {assigned_count[split]:4d} frames ({pct:.1f}%) from {n_videos} videos")

        for vid, fs in groups.items():
            split = assignment[vid]
            for f in fs:
                shutil.copy2(os.path.join(src_dir, f), os.path.join(OUTPUT_DIR, split, cat, f))
                manifest_rows.append([f, cat, vid, split])

    manifest_path = os.path.join(OUTPUT_DIR, "split_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "category", "video_id", "split"])
        writer.writerows(manifest_rows)

    print(f"\nManifest written to {manifest_path}")
    print(f"Split folders written to {OUTPUT_DIR}/{{train,val,test}}/{{Promotional,Preventive}}/")


if __name__ == "__main__":
    main()
