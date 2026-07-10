import cv2
import os

# ==========================================
# CONFIGURATION -- update this path to match your actual researchf location
# ==========================================
BASE_VIDEO_FOLDER = r"E:\researchf\Videos"
BASE_OUTPUT_FOLDER = r"E:\researchf\ExtractedFrames"
FRAME_INTERVAL_SECONDS = 1   # extract 1 frame every N seconds
# ==========================================


def extract_frames(video_path, output_folder, prefix, video_id):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        print(f"  [skip] could not read FPS for {video_path}")
        cap.release()
        return

    frame_interval = max(1, int(fps * FRAME_INTERVAL_SECONDS))
    frame_count = 0
    saved_count = 1

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            filename = f"{prefix}_{video_id}_F{saved_count:03}.jpg"
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"{video_id} --> {saved_count - 1} frames extracted")


# ==========================================
# Run extraction for each category folder found under Videos\
# ==========================================
for category in ["Promotional", "Neutral", "Preventive"]:
    video_folder = os.path.join(BASE_VIDEO_FOLDER, category)
    output_folder = os.path.join(BASE_OUTPUT_FOLDER, category)

    if not os.path.isdir(video_folder):
        print(f"[skip] no folder found: {video_folder}")
        continue

    videos = [v for v in os.listdir(video_folder) if v.lower().endswith(".mp4")]
    if not videos:
        print(f"[skip] no .mp4 files in: {video_folder}")
        continue

    for video in videos:
        video_path = os.path.join(video_folder, video)
        # video_id.mp4 -> use the filename (without extension) as video_id
        video_id = os.path.splitext(video)[0]
        # use the category as the prefix (e.g. PROMOTIONAL_<id>_F001.jpg)
        prefix = category.upper()

        extract_frames(video_path, output_folder, prefix, video_id)

print("\n===================================")
print("Dataset Generation Completed")
print("===================================")