import cv2
import os
from config import Config

def extract_frames_from_video(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    count = 0
    saved = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // frame_rate) if fps > 0 else 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{saved:05}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"[{video_name}] Extracted {saved} frames.")

def extract_all_videos(video_dir, output_dir, frame_rate=1):
    for fname in os.listdir(video_dir):
        if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path_
