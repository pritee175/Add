"""
download_videos.py
-------------------
Automates Step 1 of the dataset pipeline (Section 4 of the proposal):
"Collect commercial advertisement videos" + metadata logging.

Instead of manually downloading each YouTube video and typing out its
metadata, this script takes a list of YouTube URLs (or search queries),
downloads the video, and automatically writes a metadata.csv row for
each one: video_id, title, channel, upload_date, duration, category,
source_url, local_filename.

REQUIREMENTS:
    pip install yt-dlp

USAGE:
    1. Edit the VIDEOS list below (or load from a .txt file, see bottom).
    2. Run: python download_videos.py
    3. Videos land in Videos/<Category>/ and metadata.csv is updated.

This matches the folder structure already used in your project:
    Videos/
        Promotional/
        Neutral/
        Preventive/
"""

import os
import csv
import yt_dlp

# ==========================================
# CONFIGURATION
# ==========================================
BASE_VIDEO_FOLDER = "Videos"
METADATA_FILE = "metadata.csv"

# Each entry: (youtube_url_or_search_query, category, brand/source_tag)
# For an Indian gutka/tobacco/pan masala/alcohol/cigarette dataset,
# you can mix direct URLs and search queries (ytsearch5: pulls top 5 results).
VIDEOS = [
    # ---- Pan Masala ----
    ("ytsearch5:Vimal pan masala tv commercial", "Promotional", "Vimal"),
    ("ytsearch5:Rajnigandha pan masala ad", "Promotional", "Rajnigandha"),
    ("ytsearch5:Tulsi pan masala advertisement", "Promotional", "Tulsi"),
    ("ytsearch5:Pan Bahar advertisement", "Promotional", "PanBahar"),
    ("ytsearch5:Kamla Pasand pan masala ad", "Promotional", "KamlaPasand"),

    # ---- Gutka ----
    ("ytsearch5:Manikchand gutka old ad", "Promotional", "Manikchand"),
    ("ytsearch5:Goa 1000 gutka ad", "Promotional", "Goa1000"),
    ("ytsearch5:RMD gutka advertisement", "Promotional", "RMD"),

    # ---- Surrogate alcohol ads (music CDs, soda, club soda etc.) ----
    ("ytsearch5:Royal Stag music CD ad", "Promotional", "RoyalStag"),
    ("ytsearch5:McDowells soda ad", "Promotional", "McDowells"),
    ("ytsearch5:Bagpiper club soda ad", "Promotional", "Bagpiper"),
    ("ytsearch5:Imperial Blue music CD ad", "Promotional", "ImperialBlue"),
    ("ytsearch5:Kingfisher soda ad India", "Promotional", "Kingfisher"),
    ("ytsearch5:Officers Choice music ad", "Promotional", "OfficersChoice"),

    # ---- Old / vintage tobacco & cigarette ads (Doordarshan era, pre-ban) ----
    ("ytsearch5:Wills cigarette old tv commercial India", "Promotional", "Wills"),
    ("ytsearch5:Charminar cigarette old ad India", "Promotional", "Charminar"),
    ("ytsearch5:Four Square cigarette ad India", "Promotional", "FourSquare"),
    ("ytsearch5:Gold Flake cigarette old commercial", "Promotional", "GoldFlake"),
    ("ytsearch5:Marlboro cigarette commercial India vintage", "Promotional", "Marlboro"),

    # ---- Hookah / other addictive products ----
    ("ytsearch5:hookah lounge advertisement India", "Promotional", "Hookah"),
]

# ==========================================


def load_existing_ids():
    """Avoid re-downloading videos already logged."""
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


def download_one(query_or_url, category, brand_tag, seen_ids):
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
        # search queries return a 'entries' list; URLs return a single dict
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
            ydl.download([entry["webpage_url"]])

            local_name = f"{vid}.mp4"
            rows.append([
                vid,
                entry.get("title", ""),
                entry.get("uploader", ""),
                entry.get("upload_date", ""),
                entry.get("duration", ""),
                category,
                brand_tag,
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
        for query_or_url, category, brand_tag in VIDEOS:
            print(f"\nProcessing: {query_or_url}  [{category}]")
            try:
                rows = download_one(query_or_url, category, brand_tag, seen_ids)
                writer.writerows(rows)
                f.flush()
            except Exception as e:
                print(f"  [ERROR] {query_or_url}: {e}")

    print("\nDone. Metadata saved to", METADATA_FILE)


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# OPTIONAL: load VIDEOS from a plain text file instead of hardcoding.
# Create videos_list.txt with lines like:
#   ytsearch5:Pan Bahar advertisement|Promotional|PanBahar
#   https://youtu.be/XXXXXXXX|Preventive|MoHFW
# Then replace the VIDEOS list above with:
#
# VIDEOS = []
# with open("videos_list.txt", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if not line or line.startswith("#"):
#             continue
#         q, cat, tag = line.split("|")
#         VIDEOS.append((q, cat, tag))
# ------------------------------------------------------------------