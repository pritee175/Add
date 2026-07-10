"""
preprocess_dataset.py
------------------------
Final preprocessing step before model training: takes the curated,
manually-verified images in Dataset/<Category>/ and produces a
model-ready version in Dataset_Processed/<Category>/.

Steps per image:
    1. Convert to RGB (guards against grayscale/RGBA/CMYK source frames).
    2. Resize with aspect ratio preserved, then pad (letterbox) to a
       square canvas -- avoids the stretching/distortion a plain resize
       would cause on the mixed 4:3 / 16:9 source footage.
    3. Save as .jpg at a fixed IMAGE_SIZE x IMAGE_SIZE resolution.

The curated originals in Dataset/<Category>/ are left untouched, so the
manifests / CSV logs there still map 1:1 to the source video_id.

USAGE:
    python preprocess_dataset.py --categories Promotional Neutral
"""

import os
import argparse
from PIL import Image

IMAGE_SIZE = 224
PAD_COLOR = (0, 0, 0)


def letterbox_resize(img, size=IMAGE_SIZE, pad_color=PAD_COLOR):
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), pad_color)
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def process_category(category):
    src_dir = os.path.join("Dataset", category)
    dst_dir = os.path.join("Dataset_Processed", category)
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"{category}: processing {len(files)} images -> {dst_dir}")

    ok, failed = 0, 0
    for fname in files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, os.path.splitext(fname)[0] + ".jpg")
        try:
            img = Image.open(src_path)
            out = letterbox_resize(img)
            out.save(dst_path, "JPEG", quality=95)
            ok += 1
        except Exception as e:
            print(f"  [error] {fname}: {e}")
            failed += 1

    print(f"  Done: {ok} processed, {failed} failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=["Promotional", "Neutral"])
    args = parser.parse_args()

    for category in args.categories:
        process_category(category)


if __name__ == "__main__":
    main()
