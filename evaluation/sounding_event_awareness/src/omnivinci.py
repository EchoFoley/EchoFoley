import json
import time
import torch
from transformers import AutoProcessor, AutoModel, AutoConfig,AutoModelForCausalLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

generation_kwargs = {"max_new_tokens": 1024, "max_length": 99999999}
load_audio_in_video = False
num_video_frames = 128
audio_length = "max_3600"

parent_result_dir = "/home/jovyan/workspace/results_omnivinci_zero_shot"
os.makedirs(parent_result_dir, exist_ok=True)

supported_models = [
    "/home/jovyan/workspace/eval/evaluation/sounding_event_awareness/env/omnivinci",
    "/home/jovyan/workspace/eval/evaluation/sounding_event_awareness/env/omnivinci",
    "/home/jovyan/workspace/eval/evaluation/sounding_event_awareness/env/omnivinci"
]


def parse_response(response):
    if "thinking" in response:
        thinking, res = response.split("</think>")
        res = res.strip()
    else:
        res = response.strip()
    res = res.replace("```json", "").replace("```", "")
    try:
        result = json.loads(res)
        return result
    except Exception as e:
        print(f"Error parsing response: {e}")
        return res


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

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj

def generate(model_path, data):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path,
                                  trust_remote_code=True,
                                  torch_dtype=torch.float16,
                                  device_map="auto").to("cuda")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    generation_config = model.default_generation_config
    generation_config.update(**generation_kwargs)
    model.config.load_audio_in_video = load_audio_in_video
    processor.config.load_audio_in_video = load_audio_in_video
    if num_video_frames > 0:
        model.config.num_video_frames = num_video_frames
        processor.config.num_video_frames = num_video_frames
    if audio_length != -1:
        model.config.audio_chunk_length = audio_length
        processor.config.audio_chunk_length = audio_length

    results = []
    for index, row in data.iterrows():
        try:
            print(f"Processing video {index} of {len(data)}")
            conversation = [{
            "role": "user",
            "content": [
                {"type": "video", "video":row["video"]},
                {"type": "text", "text": VIDEO_OVERVIEW_PROMPT}
            ]
            }]
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor([text])
            try:
                target_device = next(model.parameters()).device
            except StopIteration:
                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if hasattr(inputs, "to"):
                inputs = inputs.to(target_device)
            else:
                inputs = move_to_device(inputs, target_device)
            output_ids = model.generate(
                input_ids=inputs.input_ids,
                media=getattr(inputs, 'media', None),
                media_config=getattr(inputs, 'media_config', None),
                generation_config=generation_config,
            )
            response = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            response = response[0]
            print(response)
            results.append({
                "video": row["video"],
                "gt": row["original_sounding_events"],
                "pred": parse_response(response),
                "raw": response
            })
        except Exception as e:
            print(f"Error processing video {index}: {e}")
    return results


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv("/home/jovyan/workspace/eval/data/sounding_event_aware_data.csv")
    for model in supported_models:
        print(f"Generating results for {model}...")
        results = generate(model, data)
        with open(os.path.join(parent_result_dir, f"{model.replace('/', '-') }_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(results, f)