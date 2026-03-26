import os
import json

INPUT_DIR="/home/jovyan/workspace/eval/evaluation/sound_generation/metrics/AES/inputs"

OUTPUT_DIR="/home/jovyan/workspace/eval/evaluation/sound_generation/metrics/AES/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.endswith(".jsonl"):
        name = file.split(".")[0]
        output_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        with open(os.path.join(INPUT_DIR, file), "r") as f:
            input_list = [json.loads(line) for line in f]
        length = len(input_list)
        os.system(f"audio-aes {os.path.join(INPUT_DIR, file)} --batch-size {length} > {output_path}")