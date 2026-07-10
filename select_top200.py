"""
select_top200.py
-----------------
Builds the final 200-image Promotional class dataset from the raw
extracted frames (ExtractedFrames/Promotional).

Pipeline:
    1. Blur filter -> drop frames below a sharpness threshold.
    2. Near-duplicate filter (perceptual hash) -> drop frames that are
       near-identical to one already kept (collapses static/repeated shots).
    3. CLIP scoring -> score each surviving frame against a set of text
       prompts describing addiction-related products (cigarette pack,
       gutka pouch, alcohol bottle, hookah, etc). Score = best cosine
       similarity across all prompts.
    4. Global ranking -> sort all survivors by CLIP score, descending,
       and copy the top 200 into Dataset/Promotional.
    5. Write a manifest CSV (Dataset/promotional_selection_log.csv) with
       rank, filename, source video, clip score, matched prompt, and
       blur score, for documentation / spot-checking.

USAGE:
    python select_top200.py
"""

import os
import csv
import shutil
from PIL import Image
import cv2
import imagehash
import torch
from transformers import CLIPModel, CLIPProcessor

INPUT_DIR = r"E:\ResearchF\ExtractedFrames\Promotional"
OUTPUT_DIR = r"E:\ResearchF\Dataset\Promotional"
MANIFEST_PATH = r"E:\ResearchF\Dataset\promotional_selection_log.csv"

TOP_N = 200
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
]

# Prompts describing common FALSE POSITIVE patterns found during manual
# review (CLIP matching on hands/faces/logos/text with no actual product
# visible). We subtract the best negative match from the best positive
# match so these get pushed down in the ranking instead of polluting it.
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
]


def blur_score(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def video_id_from_filename(fname):
    # PROMOTIONAL_<video_id>_F001.jpg -> <video_id>
    parts = fname.rsplit("_", 1)[0]
    return parts.replace("PROMOTIONAL_", "", 1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Found {len(files)} extracted frames")

    # --- Step 1 + 2: blur filter + near-duplicate filter ---
    print("\nStep 1-2: blur filtering + near-duplicate filtering...")
    survivors = []
    kept_hashes = []
    blurry_count = 0
    dup_count = 0

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(INPUT_DIR, fname)
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

    # --- Step 3: CLIP scoring (batched) ---
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
            fpath = os.path.join(INPUT_DIR, fname)
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
            pos_sim = image_embeds @ pos_embeds.T  # [batch, num_pos_prompts]
            neg_sim = image_embeds @ neg_embeds.T  # [batch, num_neg_prompts]

        best_pos, best_pos_idx = pos_sim.max(dim=1)
        best_neg, _ = neg_sim.max(dim=1)
        margin = best_pos - best_neg
        for (fname, bscore), m, p, idx in zip(valid_batch, margin.tolist(), best_pos.tolist(), best_pos_idx.tolist()):
            scored.append((fname, m, PRODUCT_PROMPTS[idx], bscore, p))

        if (batch_start // BATCH_SIZE) % 10 == 0:
            print(f"  ...scored {min(batch_start + BATCH_SIZE, len(survivors))}/{len(survivors)}")

    # --- Step 4: global ranking by margin (positive match strength minus
    # best negative/false-positive-pattern match), take top N ---
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:TOP_N]

    # Save the FULL ranked candidate list (not just top 200) so weak picks
    # can be swapped for runners-up later without re-running CLIP scoring.
    all_ranked_path = r"E:\ResearchF\Dataset\promotional_all_candidates_ranked.csv"
    with open(all_ranked_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "filename", "video_id", "margin_score", "matched_prompt", "blur_score", "raw_positive_score"])
        for rank, (fname, margin, prompt, bscore, pos) in enumerate(scored, 1):
            writer.writerow([rank, fname, video_id_from_filename(fname), f"{margin:.4f}", prompt, f"{bscore:.1f}", f"{pos:.4f}"])
    print(f"Full ranked candidate list ({len(scored)} frames) written to {all_ranked_path}")

    print(f"\nStep 4: selected top {len(top)} frames by margin score (product match minus false-positive-pattern match)")
    if top:
        print(f"  Highest margin: {top[0][1]:.4f} ({top[0][2]})")
        print(f"  Lowest margin in top {TOP_N}: {top[-1][1]:.4f} ({top[-1][2]})")

    # --- Step 5: copy + manifest ---
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "filename", "video_id", "margin_score", "matched_prompt", "blur_score", "raw_positive_score"])
        for rank, (fname, margin, prompt, bscore, pos) in enumerate(top, 1):
            shutil.copy2(os.path.join(INPUT_DIR, fname), os.path.join(OUTPUT_DIR, fname))
            writer.writerow([rank, fname, video_id_from_filename(fname), f"{margin:.4f}", prompt, f"{bscore:.1f}", f"{pos:.4f}"])

    videos_represented = len({video_id_from_filename(f) for f, *_ in top})
    print(f"\nDone. {len(top)} images copied to {OUTPUT_DIR}")
    print(f"Drawn from {videos_represented} distinct source videos (out of 84)")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
