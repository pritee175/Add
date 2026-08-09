"""
Download Indian PROMOTIONAL TV ads (15-60s)
Cigarettes, Pan Masala, Gutka, Tobacco, Alcohol brands
"""
import os, csv, yt_dlp

CATEGORY_NEW = "Promotional_NEW"

# Indian promotional ad searches
SEARCH_QUERIES = [
    # Pan Masala/Gutka brands
    "vimal pan masala ad ajay devgan",
    "rajnigandha pan masala tv ad",
    "pan bahar advertisement india",
    "kamla pasand pan masala ad",
    "bolo zubaan kesari vimal ad",
    
    # Alcohol (surrogate ads)
    "royal stag tv commercial india",
    "mcdowell's no 1 soda ad",
    "bagpiper club soda ad india",
    "imperial blue advertisement india",
    "royal challenge advertisement",
    
    # Cigarette brands (old ads)
    "four square cigarette ad india",
    "gold flake cigarette ad doordarshan",
    "wills cigarette advertisement india",
    
    # Celebrity pan masala ads
    "shahrukh khan vimal ad",
    "amitabh bachchan kamla pasand",
    "akshay kumar vimal elaichi",
    "pierce brosnan pan bahar",
    
    # Regional
    "pan masala ad hindi",
    "gutka advertisement india",
]

def safe_print(text):
    try:
        print(text)
    except:
        print(text.encode("ascii", "replace").decode("ascii"))

def load_existing_ids():
    seen = set()
    if os.path.exists("metadata.csv"):
        with open("metadata.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["video_id"])
    return seen

def search_youtube(query, max_results=5):
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            if "entries" in result:
                return [{"id": e["id"], "title": e.get("title", ""), 
                        "duration": e.get("duration", 0), "channel": e.get("channel", "")}
                       for e in result["entries"] if e and "id" in e]
    except:
        pass
    return []

def is_promotional_tv_ad(video):
    duration = video.get("duration", 0)
    title = video["title"].lower()
    
    # Must be TV ad length
    if duration < 15 or duration > 60:
        return False
    
    # Exclude non-ads
    bad = ["documentary", "interview", "news", "reaction", "review", "making", "song", "full video"]
    if any(b in title for b in bad):
        return False
    
    # Must be promotional content
    brands = ["vimal", "rajnigandha", "pan bahar", "kamla pasand", "royal stag", 
              "mcdowell", "bagpiper", "imperial blue", "pan masala", "gutka", 
              "cigarette", "elaichi", "tobacco"]
    
    return any(brand in title for brand in brands)

def download_video(video_info):
    vid = video_info["id"]
    out_dir = os.path.join("Videos", CATEGORY_NEW)
    os.makedirs(out_dir, exist_ok=True)
    
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "quiet": False,
        "ignoreerrors": True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            safe_print(f"  [fetching] {video_info['title'][:60]}...")
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
            if not info:
                return None
            
            duration = info.get("duration", 0)
            if duration < 15 or duration > 60:
                safe_print(f"  [SKIP] {duration}s (not TV ad)")
                return None
            
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
            
            row = [vid, info.get("title", ""), info.get("uploader", ""),
                   info.get("upload_date", ""), duration, "Promotional",
                   "TVAd_SearchDiscovered", "Not identified",
                   f"https://www.youtube.com/watch?v={vid}",
                   os.path.join(out_dir, f"{vid}.mp4")]
            
            safe_print(f"  [✓] Downloaded {duration}s ad")
            return row
    except Exception as e:
        safe_print(f"  [✗] Error: {str(e)[:80]}")
        return None

def main():
    safe_print("=" * 80)
    safe_print("INDIAN PROMOTIONAL TV ADS DOWNLOADER")
    safe_print("Cigarettes, Pan Masala, Gutka, Tobacco, Alcohol")
    safe_print("=" * 80)
    
    existing = load_existing_ids()
    safe_print(f"\n✓ {len(existing)} videos already downloaded")
    
    safe_print(f"\n🔍 Searching {len(SEARCH_QUERIES)} queries...\n")
    
    all_candidates = []
    for i, query in enumerate(SEARCH_QUERIES, 1):
        safe_print(f"[{i}/{len(SEARCH_QUERIES)}] {query}")
        videos = search_youtube(query, 5)
        safe_print(f"   → {len(videos)} results")
        all_candidates.extend(videos)
    
    safe_print(f"\n✓ Found {len(all_candidates)} videos")
    
    # Filter for promotional TV ads
    tv_ads = [v for v in all_candidates if is_promotional_tv_ad(v)]
    safe_print(f"✓ {len(tv_ads)} are promotional TV ads (15-60s)")
    
    # Remove duplicates
    seen = set()
    unique = []
    for v in tv_ads:
        if v["id"] not in existing and v["id"] not in seen:
            seen.add(v["id"])
            unique.append(v)
    
    unique = unique[:40]
    safe_print(f"✓ {len(unique)} NEW unique ads to download")
    
    if not unique:
        safe_print("\n✓ No new ads found")
        return
    
    # Show list
    safe_print("\n" + "=" * 80)
    for i, v in enumerate(unique, 1):
        safe_print(f"{i:2d}. [{v['id']}] {v['title'][:60]} ({v['duration']}s)")
    
    response = input(f"\nDownload {len(unique)} ads? (y/n): ")
    if response.lower() != 'y':
        return
    
    # Download
    safe_print("\n📥 Downloading...\n")
    successful = []
    
    for i, v in enumerate(unique, 1):
        safe_print(f"\n[{i}/{len(unique)}] {v['id']}")
        row = download_video(v)
        if row:
            successful.append(row)
    
    # Save metadata
    if successful:
        meta_file = "metadata_promotional_NEW.csv"
        with open(meta_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "title", "uploader", "upload_date", "duration",
                           "category", "tag", "actor", "url", "local_filename"])
            writer.writerows(successful)
        safe_print(f"\n✓ Saved to {meta_file}")
    
    safe_print("\n" + "=" * 80)
    safe_print(f"✓ Downloaded: {len(successful)} promotional TV ads")
    safe_print(f"📂 Location: Videos/{CATEGORY_NEW}/")
    safe_print("\n✅ Review and process next!")

if __name__ == "__main__":
    main()
