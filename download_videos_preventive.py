"""
download_videos_preventive.py
--------------------------------
Sources genuine Indian Preventive-class advertisements (government/NGO
anti-tobacco, anti-alcohol, anti-drug PSAs) with full metadata tracking,
to close the class-imbalance gap against Promotional (200 images).

USAGE:
    python download_videos_preventive.py
"""

import os
import csv
import yt_dlp

BASE_VIDEO_FOLDER = "Videos"
METADATA_FILE = "metadata.csv"

VIDEOS = [
    ("uWimehBOIHM", "MoHFW"),
    ("0kHeW6_X6K4", "MoHFW"),
    ("qNBCiDWjtEA", "MoHFW"),
    ("8WQQLDuEJMc", "PHFI_RahulDravid"),
    ("eH1T8s00F4s", "PHFI_RahulDravid"),
    ("ZDTWVDRbgQc", "PIB_NashaMuktBharat"),
    ("V_fSL4xlEdo", "PIB_NashaMuktBharat"),
    ("T7P-MgQLIPw", "PIB_NashaMuktBharat"),
    ("IHR158IF6YU", "NMBA_Campaign"),
    ("xM2Qcp4ke9k", "VitalStrategies_India"),
    ("NSbzJ63aYT4", "VitalStrategies_India"),
    ("SC-3td5-pdY", "VitalStrategies_India"),
    ("l8K_f4DVvVc", "VitalStrategies_India"),
    ("RTc5NgyiGTs", "ISFC_Hyderabad"),
    ("tjFu-LqtiE0", "NGO_Campaign"),
    ("LVKLxfNguLw", "NGO_Campaign"),
    ("wAH0z3cIAL0", "NGO_Campaign"),
    ("D4hCpB5nAx0", "NGO_Campaign"),
    ("n1Q2CgaJp7Y", "NGO_Campaign"),
    ("S9B-zJcXCwM", "Other"),
]

CATEGORY = "Preventive"

# ==========================================


def load_existing_ids():
    seen = set()
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["video_id"])
    return seen


def safe_title(t):
    return t.encode("ascii", "replace").decode("ascii")


def main():
    seen_ids = load_existing_ids()
    out_dir = os.path.join(BASE_VIDEO_FOLDER, CATEGORY)
    os.makedirs(out_dir, exist_ok=True)

    with open(METADATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for vid, tag in VIDEOS:
            if vid in seen_ids:
                print(f"  [skip] already have {vid}")
                continue

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
                        CATEGORY, tag, url,
                        os.path.join(out_dir, f"{vid}.mp4"),
                    ]
                    writer.writerow(row)
                    f.flush()
            except Exception as e:
                print(f"  [ERROR] {vid}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
