import os
import json

input_dir = "/home/jovyan/workspace/eval/evaluation/sound_generation/metrics/AES/outputs"

for file in os.listdir(input_dir):
    if file.endswith(".jsonl"):
        with open(os.path.join(input_dir, file), "r") as f:
            data = [json.loads(line) for line in f]
        avg_data = {
            "CE": sum([item["CE"] for item in data]) / len(data),
            "CU": sum([item["CU"] for item in data]) / len(data),
            "PC": sum([item["PC"] for item in data]) / len(data),
            "PQ": sum([item["PQ"] for item in data]) / len(data)
        }
        print(file)
        print(json.dumps(avg_data, indent=4))