from PIL import Image, ImageDraw
from pathlib import Path
import csv, math

base = Path(".")
csv_path = base / "error_analysis" / "manual_hard_case_review.csv"

with csv_path.open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print("Columns:", list(rows[0].keys()))
print("Images:", len(rows))

cols = 4
thumb_w = 300
thumb_h = 220
label_h = 55

rows_per_page = math.ceil(len(rows) / cols)

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows_per_page * (thumb_h + label_h)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, row in enumerate(rows):

    filename = row.get("filename", "").strip()

    if not filename:
        print("Skipping row with no filename:", row)
        continue

    image_path = base / Path(filename)

    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((thumb_w - 10, thumb_h - 10))
    except Exception as e:
        print(f"Could not open: {image_path}")
        print(f"Reason: {e}")
        continue

    x = (i % cols) * thumb_w
    y = (i // cols) * (thumb_h + label_h)

    px = x + (thumb_w - img.width) // 2
    py = y + (thumb_h - img.height) // 2

    sheet.paste(img, (px, py))

    # Use filename as label
    label = Path(filename).stem

    draw.text(
        (x + 5, y + thumb_h + 5),
        label,
        fill="black"
    )

out = base / "error_analysis" / "shared_error_review.jpg"
sheet.save(out, quality=95)

print(f"Created: {out}")
