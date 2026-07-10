"""
download_videos_neutral.py
----------------------------
Same pipeline as download_videos.py (Section 4 of the proposal), but for
the "Neutral" class: factual / informational content that shows an
addiction-related product WITHOUT promotional framing (no jingles,
celebrity endorsement, lifestyle glamorization) and WITHOUT anti-use
messaging either.

Examples: news reports on tax/price/bans, documentaries on how the
product is made, factory footage, retail/wholesale/market footage.

REQUIREMENTS:
    pip install yt-dlp

USAGE:
    python download_videos_neutral.py
"""

import os
import csv
import yt_dlp

BASE_VIDEO_FOLDER = "Videos"
METADATA_FILE = "metadata.csv"

# Each entry: (search_query, category, brand/topic_tag)
VIDEOS = [
    # ---- News / policy / tax coverage ----
    ("ytsearch6:gutka pan masala GST tax news report india", "Neutral", "PanMasala_News"),
    ("ytsearch6:tobacco price hike news report india", "Neutral", "Tobacco_News"),
    ("ytsearch6:cigarette tax GST news india", "Neutral", "Cigarette_News"),
    ("ytsearch6:liquor policy news report india", "Neutral", "Alcohol_News"),
    ("ytsearch6:gutka ban news report india", "Neutral", "Gutka_News"),
    ("ytsearch6:pan masala company business news india", "Neutral", "PanMasala_Business_News"),

    # ---- Documentary / factual / how-it's-made ----
    ("ytsearch6:how pan masala is made factory documentary", "Neutral", "PanMasala_Factory"),
    ("ytsearch6:gutka manufacturing process documentary india", "Neutral", "Gutka_Factory"),
    ("ytsearch6:cigarette factory india documentary", "Neutral", "Cigarette_Factory"),
    ("ytsearch6:tobacco farming documentary india", "Neutral", "Tobacco_Farming"),
    ("ytsearch6:liquor distillery factory documentary india", "Neutral", "Alcohol_Factory"),

    # ---- Retail / market / wholesale footage ----
    ("ytsearch6:paan shop vlog india market", "Neutral", "PaanShop_Market"),
    ("ytsearch6:pan masala wholesale market india", "Neutral", "PanMasala_Wholesale"),
    ("ytsearch6:cigarette shop counter india vlog", "Neutral", "CigaretteShop"),
    ("ytsearch6:liquor shop india vlog market", "Neutral", "LiquorShop"),
    ("ytsearch6:hookah cafe india news report", "Neutral", "Hookah_News"),
    ("ytsearch6:vape shop india news report", "Neutral", "Vape_News"),
]

# ==========================================


def load_existing_ids():
    seen = set()
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["video_id"])
    return seen


def ensure_metadata_header():
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_id", "title", "channel", "upload_date",
                "duration_sec", "category", "brand_or_source",
                "source_url", "local_filename"
            ])


def download_one(query_or_url, category, tag, seen_ids):
    out_dir = os.path.join(BASE_VIDEO_FOLDER, category)
    os.makedirs(out_dir, exist_ok=True)

    ydl_opts = {
        "format": "mp4/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query_or_url, download=False)
        entries = info["entries"] if "entries" in info else [info]

        rows = []
        for entry in entries:
            if entry is None:
                continue
            vid = entry.get("id")
            if vid in seen_ids:
                print(f"  [skip] already have {vid}")
                continue

            print(f"  [download] {entry.get('title')}")
            try:
                ydl.download([entry["webpage_url"]])
            except Exception as e:
                print(f"    [ERROR downloading {vid}] {e}")
                continue

            local_name = f"{vid}.mp4"
            rows.append([
                vid,
                entry.get("title", ""),
                entry.get("uploader", ""),
                entry.get("upload_date", ""),
                entry.get("duration", ""),
                category,
                tag,
                entry.get("webpage_url", ""),
                os.path.join(out_dir, local_name),
            ])
            seen_ids.add(vid)
        return rows


def main():
    ensure_metadata_header()
    seen_ids = load_existing_ids()

    with open(METADATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for query_or_url, category, tag in VIDEOS:
            print(f"\nProcessing: {query_or_url}  [{category}]")
            try:
                rows = download_one(query_or_url, category, tag, seen_ids)
                writer.writerows(rows)
                f.flush()
            except Exception as e:
                print(f"  [ERROR] {query_or_url}: {e}")

    print("\nDone. Metadata saved to", METADATA_FILE)


if __name__ == "__main__":
    main()
