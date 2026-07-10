"""
download_videos_neutral_diverse.py
------------------------------------
Follow-up to download_videos_neutral.py. The first Neutral pass came out
86% cigarette/tobacco content (one video's slow factory pan alone
supplied 127/200 images). This pass specifically targets underrepresented
product categories -- alcohol retail/factual content above all, plus more
gutka/pan-masala and hookah -- to diversify the class.

REQUIREMENTS:
    pip install yt-dlp

USAGE:
    python download_videos_neutral_diverse.py
"""

import os
import csv
import yt_dlp

BASE_VIDEO_FOLDER = "Videos"
METADATA_FILE = "metadata.csv"

VIDEOS = [
    # ---- Alcohol: retail / factual / policy (heavily underrepresented) ----
    ("ytsearch6:liquor shop India shelf display vlog", "Neutral", "Alcohol_Retail"),
    ("ytsearch6:wine shop India retail store visit", "Neutral", "Alcohol_Retail"),
    ("ytsearch6:beer shop India shelf display", "Neutral", "Alcohol_Retail"),
    ("ytsearch6:Indian made foreign liquor factory tour", "Neutral", "Alcohol_Factory"),
    ("ytsearch6:brewery India factory tour documentary", "Neutral", "Alcohol_Factory"),
    ("ytsearch6:distillery India factory documentary", "Neutral", "Alcohol_Factory"),
    ("ytsearch6:duty free liquor shop India airport", "Neutral", "Alcohol_Retail"),
    ("ytsearch6:liquor price India news report", "Neutral", "Alcohol_News"),
    ("ytsearch6:alcohol shop India news GST tax", "Neutral", "Alcohol_News"),
    ("ytsearch6:country liquor shop India news", "Neutral", "Alcohol_News"),

    # ---- Gutka / pan masala: more retail diversity ----
    ("ytsearch4:pan masala shop India shelf display", "Neutral", "PanMasala_Retail"),
    ("ytsearch4:gutka shop India retail counter vlog", "Neutral", "Gutka_Retail"),

    # ---- Hookah: more diversity ----
    ("ytsearch4:hookah cafe India factual documentary", "Neutral", "Hookah_Factual"),
    ("ytsearch4:hookah bar India news report GST", "Neutral", "Hookah_News"),
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


def safe_title(t):
    return t.encode("ascii", "replace").decode("ascii")


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

    rows = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query_or_url, download=False)
        entries = info["entries"] if "entries" in info else [info]

        for entry in entries:
            if entry is None:
                continue
            vid = entry.get("id")
            if vid in seen_ids:
                print(f"  [skip] already have {vid}")
                continue

            print(f"  [download] {safe_title(entry.get('title',''))}")
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
            print(f"\nProcessing: {query_or_url}  [{category}/{tag}]")
            try:
                rows = download_one(query_or_url, category, tag, seen_ids)
                writer.writerows(rows)
                f.flush()
            except Exception as e:
                print(f"  [ERROR] {query_or_url}: {e}")

    print("\nDone. Metadata saved to", METADATA_FILE)


if __name__ == "__main__":
    main()
