"""
clean_frames.py
----------------
Automates Step 4 of the proposal ("Review all extracted images manually,
remove blurry / duplicate / low-quality frames"). This replaces most of
the manual review with automatic filtering. You'll still want to skim
the survivors, but this cuts the workload from "look at every frame" to
"look at a much smaller, already-curated set."

REQUIREMENTS:
    pip install opencv-python imagehash pillow

USAGE:
    python clean_frames.py --input ExtractedFrames/Promotional --output Dataset/Promotional

WHAT IT DOES:
    1. Blur detection -> drops frames below a sharpness threshold
       (variance of Laplacian -- standard, fast blur metric).
    2. Near-duplicate detection -> uses perceptual hashing (phash) so that
       consecutive near-identical frames (very common with 1 fps extraction
       from static ad shots) are collapsed to one representative frame.
    3. Copies the surviving "clean" frames into the output folder, keeping
       your Dataset/Promotional | Neutral | Preventive structure.
"""

import os
import cv2
import shutil
import argparse
from PIL import Image
import imagehash

BLUR_THRESHOLD = 100.0       # lower = more blurry frames allowed through
HASH_DISTANCE_THRESHOLD = 5  # lower = stricter duplicate matching


def is_blurry(image_path, threshold=BLUR_THRESHOLD):
    img = cv2.imread(image_path)
    if img is None:
        return True  # unreadable file -> treat as bad
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold


def get_hash(image_path):
    return imagehash.phash(Image.open(image_path))


def clean_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nScanning folder: {input_dir}")
    print(f"Output folder   : {output_dir}")

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    print(f"Found {len(files)} frame(s) to inspect")

    kept_hashes = []
    kept_count = 0
    blurry_count = 0
    dup_count = 0

    for fname in files:
        fpath = os.path.join(input_dir, fname)

        if is_blurry(fpath):
            blurry_count += 1
            continue

        h = get_hash(fpath)
        is_duplicate = any(
            (h - kh) <= HASH_DISTANCE_THRESHOLD for kh in kept_hashes
        )
        if is_duplicate:
            dup_count += 1
            continue

        kept_hashes.append(h)
        shutil.copy2(fpath, os.path.join(output_dir, fname))
        kept_count += 1

        print(f"  kept: {fname}")

    print(f"\n{input_dir}")
    print(f"  Total frames scanned : {len(files)}")
    print(f"  Removed (blurry)     : {blurry_count}")
    print(f"  Removed (duplicate)  : {dup_count}")
    print(f"  Kept (final)         : {kept_count}")
    print(f"  Saved to             : {output_dir}")
    print("Cleaning complete")


def main():
    global BLUR_THRESHOLD, HASH_DISTANCE_THRESHOLD

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of extracted frames")
    parser.add_argument("--output", required=True, help="Folder for cleaned dataset images")
    parser.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD)
    parser.add_argument("--hash-distance", type=int, default=HASH_DISTANCE_THRESHOLD)
    args = parser.parse_args()
    BLUR_THRESHOLD = args.blur_threshold
    HASH_DISTANCE_THRESHOLD = args.hash_distance

    print("Starting frame cleaning")
    print(f"Blur threshold  : {BLUR_THRESHOLD}")
    print(f"Hash distance   : {HASH_DISTANCE_THRESHOLD}")

    clean_folder(args.input, args.output)


if __name__ == "__main__":
    main()