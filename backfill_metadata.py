"""
backfill_metadata.py
----------------------
Some video downloads succeeded but never got a metadata.csv row because
yt-dlp's progress printer hit a Windows console encoding error mid-run
(cp1252 can't print some Unicode title characters), which aborted that
query's row-writing before it reached the successful download.

This scans Videos/<Category>/*.mp4, finds any video_id missing from
metadata.csv, re-fetches just its info (no re-download) via yt-dlp, and
appends the row. Print statements are guarded against non-ASCII titles
so this script itself doesn't hit the same crash.

USAGE:
    python backfill_metadata.py --category Neutral
"""

import os
import csv
import argparse
import yt_dlp

METADATA_FILE = "metadata.csv"


def safe_print(s):
    print(s.encode("ascii", "replace").decode("ascii"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    video_dir = os.path.join("Videos", args.category)

    known_ids = set()
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                known_ids.add(row["video_id"])

    mp4_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    missing = [f for f in mp4_files if os.path.splitext(f)[0] not in known_ids]
    safe_print(f"{len(mp4_files)} videos on disk, {len(missing)} missing metadata rows")

    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    rows = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for fname in missing:
            vid = os.path.splitext(fname)[0]
            url = f"https://www.youtube.com/watch?v={vid}"
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as e:
                safe_print(f"  [ERROR] {vid}: {e}")
                continue
            safe_print(f"  [ok] {vid} -> {info.get('title','')}")
            rows.append([
                vid,
                info.get("title", ""),
                info.get("uploader", ""),
                info.get("upload_date", ""),
                info.get("duration", ""),
                args.category,
                args.tag,
                info.get("webpage_url", url),
                os.path.join(video_dir, fname),
            ])

    with open(METADATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    safe_print(f"Backfilled {len(rows)} metadata rows for {args.category}")


if __name__ == "__main__":
    main()
