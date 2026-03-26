import os
import json

name = "mmaudio_small_16k"
dir = f"/home/jovyan/workspace/eval/evaluation/sound_generation/results/{name}"
out_path = f"/home/jovyan/workspace/eval/evaluation/sound_generation/metrics/AES/inputs/{name}.jsonl"

path_list = []
for file in os.listdir(dir):
    if file.endswith(".flac") or file.endswith(".wav") or file.endswith(".mp3"):
        path_list.append({"path":os.path.join(dir, file)})

with open(out_path, "w") as f:
    for path in path_list:
        f.write(json.dumps(path) + "\n")