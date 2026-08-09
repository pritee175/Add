"""
download_preventive_tv_ads.py
------------------------------
Downloads ONLY Indian TV advertisements (15-60 seconds) for preventive 
anti-tobacco/anti-alcohol campaigns. Filters out documentaries, long videos,
street plays, and YouTube content.

USAGE:
    python download_preventive_tv_ads.py --max-videos 50
"""

import os
import csv
import argparse
import yt_dlp

BASE_VIDEO_FOLDER = "Videos"
CATEGORY_NEW = "Preventive_NEW"

# Search queries optimized for TV ADVERTISEMENTS only
SEARCH_QUERIES = [
    # Hindi TV ads
    "anti smoking ad india tv hindi 30 seconds",
    "tobacco warning ad india doordarshan",
    "anti gutka ad india tv commercial",
    "pan masala warning tv ad india",
    "cigarette warning indian tv commercial",
    
    # English TV ads
    "anti tobacco tv advertisement india",
    "no smoking tv commercial india",
    "quit smoking ad india television",
    
    # Government TV campaigns
    "ministry of health anti tobacco tv ad",
    "government anti smoking commercial india",
    "COTPA tv advertisement india",
    
    # Celebrity TV ads
    "amitabh bachchan anti tobacco ad",
    "shahrukh khan anti smoking commercial",
    "akshay kumar tobacco warning ad",
    
    # Alcohol TV ads
    "dont drink and drive tv ad india",
    "anti alcohol commercial india tv",
    
    # Regional language TV ads
    "तंबाकू विरोधी विज्ञापन टीवी",  # Anti-tobacco TV ad Hindi
    "धूम्रपान विरोधी विज्ञापन",  # Anti-smoking ad Hindi
    
    # Old Doordarshan ads (typically TV format)
    "anti smoking doordarshan old ad",
    "tobacco warning doordarshan commercial",
]


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def load_existing_ids():
    """Load already downloaded video IDs"""
    seen = set()
    
    # Check main metadata
    if os.path.exists("metadata.csv"):
        with open("metadata.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["video_id"])
    
    # Check NEW metadata
    if os.path.exists("metadata_preventive_NEW.csv"):
        with open("metadata_preventive_NEW.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["video_id"])
    
    return seen


def search_youtube(query, max_results=5):
    """Search YouTube and return video info"""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
    }
    
    search_query = f"ytsearch{max_results}:{query}"
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_query, download=False)
            
            if "entries" in result:
                videos = []
                for entry in result["entries"]:
                    if entry and "id" in entry:
                        videos.append({
                            "id": entry["id"],
                            "title": entry.get("title", "Unknown"),
                            "duration": entry.get("duration", 0),
                            "channel": entry.get("channel", "Unknown"),
                        })
                return videos
    except Exception as e:
        safe_print(f"  [SEARCH ERROR] {query}: {str(e)[:80]}")
    
    return []


def is_tv_ad(video):
    """Filter for TV advertisement format (15-60 seconds, ad-like content)"""
    duration = video.get("duration", 0)
    title_lower = video["title"].lower()
    
    # Duration check: TV ads are typically 15-60 seconds
    if duration < 15 or duration > 60:
        return False
    
    # Exclude obvious non-ads
    bad_keywords = [
        "documentary", "full video", "interview", "news", "debate",
        "webinar", "conference", "play", "natak", "drama", "movie",
        "song", "music video", "tutorial", "how to", "review",
        "reaction", "vlog", "podcast", "speech", "lecture"
    ]
    
    if any(bad in title_lower for bad in bad_keywords):
        return False
    
    # Prefer videos with ad/commercial keywords
    good_keywords = ["ad", "advertisement", "commercial", "tvc", "psa", "campaign"]
    has_ad_keyword = any(good in title_lower for good in good_keywords)
    
    # Or keywords indicating preventive content
    preventive_keywords = [
        "anti smoking", "anti tobacco", "quit", "warning", "danger",
        "cancer", "health", "no smoking", "gutka", "pan masala",
        "dont drink", "alcohol awareness"
    ]
    
    has_preventive = any(kw in title_lower for kw in preventive_keywords)
    
    return has_ad_keyword or has_preventive


def download_video(video_info):
    """Download video and return metadata"""
    vid = video_info["id"]
    url = f"https://www.youtube.com/watch?v={vid}"
    out_dir = os.path.join(BASE_VIDEO_FOLDER, CATEGORY_NEW)
    
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            safe_print(f"  [fetching] {video_info['title'][:60]}...")
            info = ydl.extract_info(url, download=False)
            
            if info is None:
                return None
            
            # Double-check duration after full fetch
            actual_duration = info.get("duration", 0)
            if actual_duration < 15 or actual_duration > 60:
                safe_print(f"  [SKIP] Duration {actual_duration}s (not TV ad format)")
                return None
            
            safe_print(f"  [downloading] ...")
            ydl.download([url])
            
            row = [
                vid,
                info.get("title", video_info["title"]),
                info.get("uploader", video_info.get("channel", "")),
                info.get("upload_date", ""),
                info.get("duration", video_info.get("duration", "")),
                "Preventive",
                "TVAd_SearchDiscovered",
                "Not identified",
                url,
                os.path.join(out_dir, f"{vid}.mp4"),
            ]
            
            safe_print(f"  [✓] Downloaded {actual_duration}s TV ad")
            return row
            
    except Exception as e:
        safe_print(f"  [✗] Error: {str(e)[:80]}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--per-query", type=int, default=5)
    args = parser.parse_args()
    
    safe_print("=" * 80)
    safe_print("INDIAN TV ADVERTISEMENT DOWNLOADER - PREVENTIVE ONLY")
    safe_print("=" * 80)
    safe_print("Filtering: 15-60 second TV commercials only")
    safe_print("Excluding: Documentaries, plays, long videos, music videos")
    safe_print("")
    
    existing_ids = load_existing_ids()
    safe_print(f"✓ {len(existing_ids)} videos already downloaded (will skip)")
    
    out_dir = os.path.join(BASE_VIDEO_FOLDER, CATEGORY_NEW)
    os.makedirs(out_dir, exist_ok=True)
    
    # Search
    safe_print(f"\n🔍 Searching {len(SEARCH_QUERIES)} TV ad queries...\n")
    
    all_candidates = []
    for i, query in enumerate(SEARCH_QUERIES, 1):
        safe_print(f"[{i}/{len(SEARCH_QUERIES)}] {query}")
        videos = search_youtube(query, args.per_query)
        safe_print(f"   → Found {len(videos)} results")
        all_candidates.extend(videos)
    
    safe_print(f"\n✓ Total found: {len(all_candidates)}")
    
    # Filter for TV ads
    safe_print("🔍 Filtering for TV ad format (15-60s)...")
    tv_ads = [v for v in all_candidates if is_tv_ad(v)]
    safe_print(f"✓ {len(tv_ads)} pass TV ad filter")
    
    # Remove duplicates
    seen_this_search = set()
    unique_ads = []
    for video in tv_ads:
        if video["id"] not in existing_ids and video["id"] not in seen_this_search:
            seen_this_search.add(video["id"])
            unique_ads.append(video)
    
    unique_ads = unique_ads[:args.max_videos]
    safe_print(f"✓ {len(unique_ads)} NEW unique TV ads to download")
    
    if not unique_ads:
        safe_print("\n✓ No new TV ads found")
        return
    
    # Show list
    safe_print("\n" + "=" * 80)
    safe_print("TV ADS TO DOWNLOAD:")
    safe_print("=" * 80)
    for i, v in enumerate(unique_ads, 1):
        safe_print(f"{i:2d}. [{v['id']}] {v['title'][:60]} ({v['duration']}s)")
    
    # Confirm
    safe_print("\n" + "=" * 80)
    response = input(f"Download {len(unique_ads)} TV ads? (y/n): ")
    if response.lower() != 'y':
        safe_print("Cancelled")
        return
    
    # Download
    safe_print("\n📥 Downloading...\n")
    successful = []
    failed = []
    
    for i, video in enumerate(unique_ads, 1):
        safe_print(f"\n[{i}/{len(unique_ads)}] {video['id']}")
        row = download_video(video)
        
        if row:
            successful.append(row)
        else:
            failed.append(video)
    
    # Save metadata
    if successful:
        metadata_file = "metadata_preventive_NEW.csv"
        file_exists = os.path.exists(metadata_file)
        
        with open(metadata_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["video_id", "title", "uploader", "upload_date", 
                               "duration", "category", "tag", "actor", "url", "local_filename"])
            writer.writerows(successful)
        
        safe_print(f"\n✓ Saved metadata to {metadata_file}")
    
    # Summary
    safe_print("\n" + "=" * 80)
    safe_print("DOWNLOAD COMPLETE")
    safe_print("=" * 80)
    safe_print(f"✓ Downloaded: {len(successful)} TV ads")
    safe_print(f"✗ Failed: {len(failed)}")
    safe_print(f"\n📂 Location: {out_dir}/")
    safe_print("\n✅ All videos are 15-60 second TV advertisements")
    safe_print("   Review and merge: python merge_reviewed_videos.py")


if __name__ == "__main__":
    main()
