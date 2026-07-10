"""
make_contact_sheet.py
----------------------
Builds labeled thumbnail grid images ("contact sheets") from a range of
ranks in the full candidate manifest, so a human (or a vision-capable
reviewer) can quickly screen many candidate frames per image instead of
opening one file at a time.

USAGE:
    python make_contact_sheet.py --start 201 --end 400 --cols 8 --out-prefix contact_sheets/sheet
"""

import os
import csv
import argparse
from PIL import Image, ImageDraw, ImageFont

FRAMES_DIR = r"E:\ResearchF\ExtractedFrames\Promotional"
MANIFEST = r"E:\ResearchF\Dataset\promotional_all_candidates_ranked.csv"

THUMB_W, THUMB_H = 160, 120
LABEL_H = 18
PER_SHEET = 48


def load_candidates():
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_sheet(rows, cols, out_path):
    n = len(rows)
    rows_count = (n + cols - 1) // cols
    sheet_w = cols * THUMB_W
    sheet_h = rows_count * (THUMB_H + LABEL_H)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    for i, row in enumerate(rows):
        r, c = divmod(i, cols)
        x = c * THUMB_W
        y = r * (THUMB_H + LABEL_H)

        fpath = os.path.join(FRAMES_DIR, row["filename"])
        try:
            img = Image.open(fpath).convert("RGB")
            img.thumbnail((THUMB_W, THUMB_H))
            paste_x = x + (THUMB_W - img.width) // 2
            paste_y = y + (THUMB_H - img.height) // 2
            sheet.paste(img, (paste_x, paste_y))
        except Exception:
            pass

        label = f"#{row['rank']} {row['clip_score']}"
        draw.rectangle([x, y + THUMB_H, x + THUMB_W, y + THUMB_H + LABEL_H], fill=(0, 0, 0))
        draw.text((x + 2, y + THUMB_H + 2), label, fill=(255, 255, 0))

    sheet.save(out_path, quality=85)
    print(f"Saved {out_path} ({n} thumbnails)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="start rank (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="end rank (inclusive)")
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    all_rows = load_candidates()
    by_rank = {int(r["rank"]): r for r in all_rows}

    selected = [by_rank[r] for r in range(args.start, args.end + 1) if r in by_rank]

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    for sheet_idx, chunk_start in enumerate(range(0, len(selected), PER_SHEET), 1):
        chunk = selected[chunk_start: chunk_start + PER_SHEET]
        out_path = f"{args.out_prefix}_{sheet_idx}.jpg"
        build_sheet(chunk, args.cols, out_path)


if __name__ == "__main__":
    main()
