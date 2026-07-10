"""
download_videos_neutral_tvc.py
---------------------------------
Strict-TVC rebuild of the Neutral class. After research showed pure
neutral-toned Indian consumer advertisements are rare (most brand TVCs
are persuasive by design -> Promotional; B2B/wholesale ads and news were
explicitly ruled out for this class), this targets the two genuine
categories found: airport duty-free catalog ads (informational,
price/range-focused, no lifestyle glamorization) and health-information
surrogate ads (product-adjacent, informational framing).

USAGE:
    python download_videos_neutral_tvc.py
"""

import os
import csv
import yt_dlp

BASE_VIDEO_FOLDER = "Videos"
METADATA_FILE = "metadata.csv"

VIDEOS = [
    ("4bHqg6Z4juE", "Neutral", "DutyFree_CIAL"),
    ("HJLjkGHvMl4", "Neutral", "DutyFree_CIAL"),
    ("skOGsdAQ6Xo", "Neutral", "DutyFree_Bengaluru"),
    ("V8D5E2m6GEI", "Neutral", "DutyFree_Hyderabad"),
    ("QzP7QM4LFTs", "Neutral", "DutyFree_Delhi"),
    ("LS-N2UQjELo", "Neutral", "DutyFree_Delhi"),
    ("q7BuXhuYR7g", "Neutral", "Manikchand_Health_Info"),
    ("VTYQOZRxADo", "Neutral", "Manikchand_Health_Info"),
    ("BbggSomzb_w", "Neutral", "Manikchand_Health_Info"),
    ("U9uJOcbAa-w", "Neutral", "Diageo_ProductLaunch"),
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


def main():
    ensure_metadata_header()
    seen_ids = load_existing_ids()

    with open(METADATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for vid, category, tag in VIDEOS:
            if vid in seen_ids:
                print(f"  [skip] already have {vid}")
                continue

            out_dir = os.path.join(BASE_VIDEO_FOLDER, category)
            os.makedirs(out_dir, exist_ok=True)
            url = f"https://www.youtube.com/watch?v={vid}"

            ydl_opts = {
                "format": "mp4/best",
                "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    print(f"  [download] {safe_title(info.get('title',''))}")
                    ydl.download([url])
                    row = [
                        vid, info.get("title", ""), info.get("uploader", ""),
                        info.get("upload_date", ""), info.get("duration", ""),
                        category, tag, url,
                        os.path.join(out_dir, f"{vid}.mp4"),
                    ]
                    writer.writerow(row)
                    f.flush()
            except Exception as e:
                print(f"  [ERROR] {vid}: {e}")

    print("\nDone. Metadata saved to", METADATA_FILE)


if __name__ == "__main__":
    main()
