"""
select_topN.py
----------------
Builds a clean, curated N-image dataset for any category folder
(Promotional, Neutral, Preventive) from raw extracted frames.

Pipeline:
    1. Blur filter -> drop frames below a sharpness threshold.
    2. Near-duplicate filter (perceptual hash) -> drop frames that are
       near-identical to one already kept.
    3. CLIP scoring with negative-prompt contrast -> score each surviving
       frame against product prompts, subtract the best match against a
       set of known false-positive patterns, so hands/faces/logos/talking
       heads with no real product visible get pushed down the ranking.
    4. Global ranking -> take top N by margin score.
    5. Write a manifest CSV + copy the top N into Dataset/<Category>.

USAGE:
    python select_topN.py --category Neutral --n 200
"""

import os
import csv
import shutil
import argparse
from PIL import Image
import cv2
import imagehash
import torch
from transformers import CLIPModel, CLIPProcessor

BASE = r"E:\ResearchF"
BLUR_THRESHOLD = 100.0
HASH_DISTANCE_THRESHOLD = 5
BATCH_SIZE = 16
MODEL_NAME = "openai/clip-vit-base-patch32"

PRODUCT_PROMPTS = [
    "a photo of a cigarette pack",
    "a person smoking a cigarette, cigarette clearly visible",
    "a pouch or sachet of gutka or pan masala with visible branding",
    "a sachet of chewing tobacco with visible branding",
    "a bottle of alcohol or whisky with visible label",
    "a glass of beer or wine, clearly visible",
    "a can of beer, clearly visible",
    "a hookah pipe",
    "a vape or e-cigarette device, clearly visible",
    "a cigar being smoked, clearly visible",
    "tobacco leaves or raw tobacco, clearly visible",
    "workers manufacturing cigarettes or pan masala in a factory, product visible",
]

NEGATIVE_PROMPTS = [
    "a close-up of a person's face with nothing in their hand",
    "a person holding a business card, ID card, or photograph",
    "a group of people talking or socializing with no product visible",
    "a blurry or out-of-focus image with no clear object",
    "a person holding jewelry, a ring, or a small trinket",
    "a wide shot of a bar or nightclub interior with no product in focus",
    "a warning text screen or subtitle card",
    "a vehicle, motorcycle, or car on a road",
    "a person's empty hand gesture with no object held",
    "a picture or poster being held up, not a real product photo",
    "a news anchor or reporter talking with no product visible",
    "a text-heavy news headline or graphic with no product visible",
    "a plain map, chart, or infographic with no product visible",
]


def blur_score(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def video_id_from_filename(fname, prefix):
    parts = fname.rsplit("_", 1)[0]
    return parts.replace(prefix + "_", "", 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="Promotional | Neutral | Preventive")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    category = args.category
    prefix = category.upper()
    input_dir = os.path.join(BASE, "ExtractedFrames", category)
    output_dir = os.path.join(BASE, "Dataset", category)
    manifest_path = os.path.join(BASE, "Dataset", f"{category.lower()}_selection_log.csv")
    all_ranked_path = os.path.join(BASE, "Dataset", f"{category.lower()}_all_candidates_ranked.csv")

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Found {len(files)} extracted frames in {input_dir}")

    print("\nStep 1-2: blur filtering + near-duplicate filtering...")
    survivors = []
    kept_hashes = []
    blurry_count = 0
    dup_count = 0

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(input_dir, fname)
        score = blur_score(fpath)
        if score is None or score < BLUR_THRESHOLD:
            blurry_count += 1
            continue

        h = imagehash.phash(Image.open(fpath))
        if any((h - kh) <= HASH_DISTANCE_THRESHOLD for kh in kept_hashes):
            dup_count += 1
            continue

        kept_hashes.append(h)
        survivors.append((fname, score))

        if i % 500 == 0:
            print(f"  ...scanned {i}/{len(files)}")

    print(f"  Blurry dropped     : {blurry_count}")
    print(f"  Near-dup dropped   : {dup_count}")
    print(f"  Survivors for CLIP : {len(survivors)}")

    print("\nStep 3: loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    pos_inputs = processor(text=PRODUCT_PROMPTS, return_tensors="pt", padding=True)
    neg_inputs = processor(text=NEGATIVE_PROMPTS, return_tensors="pt", padding=True)
    with torch.no_grad():
        pos_embeds = model.get_text_features(**pos_inputs)
        pos_embeds = pos_embeds / pos_embeds.norm(dim=-1, keepdim=True)
        neg_embeds = model.get_text_features(**neg_inputs)
        neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)

    print("Step 3: scoring survivors against product prompts (with negative-prompt contrast)...")
    scored = []
    for batch_start in range(0, len(survivors), BATCH_SIZE):
        batch = survivors[batch_start: batch_start + BATCH_SIZE]
        images = []
        valid_batch = []
        for fname, bscore in batch:
            fpath = os.path.join(input_dir, fname)
            try:
                images.append(Image.open(fpath).convert("RGB"))
                valid_batch.append((fname, bscore))
            except Exception as e:
                print(f"  [error] {fname}: {e}")

        if not images:
            continue

        image_inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            image_embeds = model.get_image_features(**image_inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            pos_sim = image_embeds @ pos_embeds.T
            neg_sim = image_embeds @ neg_embeds.T

        best_pos, best_pos_idx = pos_sim.max(dim=1)
        best_neg, _ = neg_sim.max(dim=1)
        margin = best_pos - best_neg
        for (fname, bscore), m, p, idx in zip(valid_batch, margin.tolist(), best_pos.tolist(), best_pos_idx.tolist()):
            scored.append((fname, m, PRODUCT_PROMPTS[idx], bscore, p))

        if (batch_start // BATCH_SIZE) % 10 == 0:
            print(f"  ...scored {min(batch_start + BATCH_SIZE, len(survivors))}/{len(survivors)}")

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:args.n]

    with open(all_ranked_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "filename", "video_id", "margin_score", "matched_prompt", "blur_score", "raw_positive_score"])
        for rank, (fname, margin, prompt, bscore, pos) in enumerate(scored, 1):
            writer.writerow([rank, fname, video_id_from_filename(fname, prefix), f"{margin:.4f}", prompt, f"{bscore:.1f}", f"{pos:.4f}"])
    print(f"Full ranked candidate list ({len(scored)} frames) written to {all_ranked_path}")

    print(f"\nStep 4: selected top {len(top)} frames by margin score")
    if top:
        print(f"  Highest margin: {top[0][1]:.4f} ({top[0][2]})")
        print(f"  Lowest margin in top {args.n}: {top[-1][1]:.4f} ({top[-1][2]})")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "filename", "video_id", "margin_score", "matched_prompt", "blur_score", "raw_positive_score"])
        for rank, (fname, margin, prompt, bscore, pos) in enumerate(top, 1):
            shutil.copy2(os.path.join(input_dir, fname), os.path.join(output_dir, fname))
            writer.writerow([rank, fname, video_id_from_filename(fname, prefix), f"{margin:.4f}", prompt, f"{bscore:.1f}", f"{pos:.4f}"])

    videos_represented = len({video_id_from_filename(f, prefix) for f, *_ in top})
    print(f"\nDone. {len(top)} images copied to {output_dir}")
    print(f"Drawn from {videos_represented} distinct source videos")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
