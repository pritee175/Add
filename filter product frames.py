"""
filter_product_frames.py
-------------------------
Automates the step you flagged: not every extracted frame actually
contains the addictive product (cigarette, gutka pouch, alcohol bottle,
etc). Many frames are just people talking, logos, transitions, or
unrelated scenes.

This uses CLIP (zero-shot, same idea as Paper 1 and Paper 6 in your
literature review) to score each frame against a set of text prompts
describing the product. Frames scoring above a similarity threshold for
ANY prompt are kept; the rest are moved aside for you to skip or
double-check.

REQUIREMENTS:
    pip install transformers torch pillow

USAGE:
    python filter_product_frames.py --input Dataset\\Promotional --output Dataset\\Promotional_filtered

This pairs AFTER clean_frames.py in the pipeline:
    extract -> clean_frames.py (remove blurry/dup) -> filter_product_frames.py (keep only product-visible frames)
"""

import os
import shutil
import argparse
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# ==========================================
# Text prompts describing what we're looking for.
# Add/remove prompts to match your specific Promotional class targets.
# Descriptive phrasing (per Paper 1's "Phrase Engineering" finding) works
# better than single words.
# ==========================================
PRODUCT_PROMPTS = [
    "a photo of a cigarette pack",
    "a person smoking a cigarette",
    "a pouch of gutka or pan masala",
    "a sachet of chewing tobacco",
    "a bottle of alcohol or whisky",
    "a glass of beer or wine",
    "a person drinking alcohol",
    "a hookah pipe",
]

# Frames must score above this similarity to be considered "product present"
SIMILARITY_THRESHOLD = 0.22  # tune this based on results, see notes at bottom

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_model():
    print("Loading CLIP model (first run downloads ~600MB, then cached)...")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def score_image(model, processor, image_path, prompts):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # cosine similarity between image and each text prompt, scaled
        logits_per_image = outputs.logits_per_image  # shape [1, num_prompts]
        probs = logits_per_image.softmax(dim=1)
        # also grab raw cosine similarity (more stable threshold than softmax)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        cos_sim = (image_embeds @ text_embeds.T).squeeze(0)  # [num_prompts]
    best_idx = cos_sim.argmax().item()
    return cos_sim[best_idx].item(), prompts[best_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of cleaned frames")
    parser.add_argument("--output", required=True, help="Folder for product-present frames")
    parser.add_argument("--rejected", default=None,
                         help="Optional folder to also save rejected frames for spot-checking")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.rejected:
        os.makedirs(args.rejected, exist_ok=True)

    model, processor = load_model()

    files = sorted(
        f for f in os.listdir(args.input)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    kept = 0
    rejected = 0

    for i, fname in enumerate(files, 1):
        fpath = os.path.join(args.input, fname)
        try:
            score, best_prompt = score_image(model, processor, fpath, PRODUCT_PROMPTS)
        except Exception as e:
            print(f"  [error] {fname}: {e}")
            continue

        if score >= args.threshold:
            shutil.copy2(fpath, os.path.join(args.output, fname))
            kept += 1
        else:
            rejected += 1
            if args.rejected:
                shutil.copy2(fpath, os.path.join(args.rejected, fname))

        if i % 50 == 0:
            print(f"  ...processed {i}/{len(files)}")

    print(f"\n{args.input}")
    print(f"  Total frames scanned : {len(files)}")
    print(f"  Kept (product likely present) : {kept}")
    print(f"  Rejected (no product detected): {rejected}")
    print(f"  Saved to              : {args.output}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# TUNING NOTES:
# - SIMILARITY_THRESHOLD around 0.20-0.25 is a reasonable starting point
#   for CLIP cosine similarity, but it depends heavily on your images.
# - Run once, check how many got kept vs rejected. If almost everything
#   gets rejected, lower the threshold (e.g. 0.18). If clearly irrelevant
#   frames are getting kept, raise it (e.g. 0.28).
# - Use --rejected to save the rejected frames to a separate folder so you
#   can quickly skim them and confirm the filter isn't throwing away good
#   frames (false negatives matter more than false positives here).
# ------------------------------------------------------------------