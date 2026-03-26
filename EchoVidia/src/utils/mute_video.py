import os
from tqdm import tqdm
from moviepy import VideoFileClip

video_dir = "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data"
video_paths = []
for video in os.listdir(video_dir):
    if video.endswith(".mp4"):
        video_paths.append(os.path.join(video_dir, video))

new_dir = "/mnt/bn/icvg-xinyao-llm2/bingxuan/interleaved_generation/data/muted"
os.makedirs(new_dir, exist_ok=True)

for video_path in tqdm(video_paths):
    video_clip = VideoFileClip(video_path)
    video_clip = video_clip.without_audio()
    video_clip.write_videofile(os.path.join(new_dir, os.path.basename(video_path)))

