import os
import time
import json
import base64
import shutil
import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor

API_KEY = ""
AZURE_ENDPOINT = "https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl"
API_VERSION = "2024-03-01"
# MODEL_NAME = "gemini-2.5-flash"
MODEL_NAME = "gemini-2.5-pro-preview-06-05"

MAX_WORKERS = 5
DATA_PATH = "./sounding_event_aware_data.csv"
RESULT_DIR = f"./1110_results/{MODEL_NAME}"
CACHE_DIR = f"./1110_cache/{MODEL_NAME}"
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)
if os.path.exists(RESULT_DIR):
    shutil.rmtree(RESULT_DIR)
os.makedirs(RESULT_DIR, exist_ok=True)

##### Utils Functions#####
def video_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def v2t_generate(prompt, video_url, model_name=MODEL_NAME, sleep=60):
    content = [{"type": "text", "text": prompt}]
    if "http" not in video_url:
        video_url = video_to_base64(video_url)
    content.append({
        "type": "image_url",
        "image_url": {
            "url": video_url,
            "mime_type": "video/mp4",
        },
        "video_metadata": {"videoMetadata": {"fps": "1"}}
    })
    time_s = time.time()
    client = AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=API_VERSION,
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            timeout=600,
            max_tokens=65536,
            temperature=0,
            extra_body={
                "generationConfig": {
                    "audioTimestamp": True,
                    "thinkingConfig": {"includeThoughts": True}
                }
            }
        )
        duration = time.time() - time_s
        if duration < sleep:
            time.sleep(sleep - duration)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error:", e)
        return None

VIDEO_OVERVIEW_PROMPT = """***Goal:**
Generate a detailed log of inferred sounds for the provided silent video. Analyze the video chronologically and identify all significant diegetic sounds (sounds originating from within the video's world).
Focusing on the non-speech sounds, and sound that caused by action or movement.
Ignore the ambient sounds.
Please try to include as many sound effects as possible.
**Instructions:**
1.  **Output Format:** Present the sounds in json list with the keys "sound", "start_time", and "end_time".
2.  **Be Specific:** Do not use generic descriptions. Instead of "car sound," specify "a car engine starting," "tires screeching on asphalt," or "a distant car horn."
3.  **Be Self-Contained:** The sound description should be self-contained and not rely on other sound descriptions. Each sound description should be isolated. Don't mention other sounds (e.g. the second sound is from the first sound).
4.  **Identify Concurrent Sounds:** If multiple sounds occur simultaneously, list them all at the same timestamp.
5.  **Note Pauses:** If there is a clear break or silence between distinct sounds, represent that with separate timestamp entries. If some sound happen more than once, break them into two.
6.  **Describe Sound Quality:** Include adjectives to describe the nature of the sound (e.g. "distant", "sudden").
**Example of Desired Output:**
```json
[
    {
        "sound": "A heavy wooden door creaks open slowly",
        "start_time": "00:02",
        "end_time": "00:04"
    },
    {
        "sound": "Soft footsteps on a polished floor",
        "start_time": "00:04",
        "end_time": "00:07"
    },
    {
        "sound": "The sharp click of a light switch",
        "start_time": "00:07",
        "end_time": "00:07"
    }
]
```
Please only output the exact same format as the example. Wrap the json with ```json and ```. Mak sure the names of the keys are exactly the same as the example.
Don't include any other text.
"""

def parse_response(response):
    if "```json" in response:
        response = response.replace("```json", "").replace("```", "")
    try:
        return json.loads(response)
    except Exception as e:
        print(f"Error parsing response: {e}")
        return response

def process_single_data(model_name, row):
    id = row["object_id"]
    video_url = row["video"]
    ground_truth = row["original_sounding_events"]
    response = v2t_generate(VIDEO_OVERVIEW_PROMPT, video_url, model_name)
    parsed_response = parse_response(response)
    return id, video_url, parsed_response, ground_truth

def process_data_parallel(model_name, data):
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for _, row in tqdm(data.iterrows(), total=len(data)):
            futures.append(executor.submit(process_single_data, model_name, row))
        for future in futures:
            id, video_url, parsed_response, ground_truth = future.result()
            results[id] = {
                "response": parsed_response,
                "ground_truth": ground_truth
            }
            with open(os.path.join(CACHE_DIR, f"{id}.json"), "w") as f:
                json.dump(results[id], f)
    return results

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    results = process_data_parallel(MODEL_NAME, data)
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f)