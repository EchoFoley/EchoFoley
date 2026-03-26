import os
import time
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ['VLLM_USE_V1'] = '0'

parent_result_dir = "/home/jovyan/workspace/results_omini_zero_shot"
os.makedirs(parent_result_dir, exist_ok=True)

supported_models = [
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Qwen/Qwen3-Omni-30B-A3B-Thinking"
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

def prepare_inputs(row):
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": row["video"]
            },
            {"type": "text", "text": VIDEO_OVERVIEW_PROMPT},
        ]
    }]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": False,
        },
    }
    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios
    return inputs

def generate(checkpoint_path, data, processor, llm, sampling_params):
    results = []
    inputs = []
    for index, row in data.iterrows():
        print(f"Processing video {index} of {len(data)}")
        inputs.append(prepare_inputs(row))
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i in range(len(outputs)):
        response = outputs[i].outputs[0].text
        results.append({
                "video": data.iloc[i]["video"],
                "gt": data.iloc[i]["original_sounding_events"],
                "pred": parse_response(outputs[i].outputs[0].text),
                "raw": outputs[i].outputs[0].text
            })
    return results


if __name__ == '__main__':
  

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    import pandas as pd
    data = pd.read_csv("/home/jovyan/workspace/eval/data/sounding_event_aware_data.csv")
    for model in tqdm(supported_models):
        print(f"Generating results for {model}...")
        processor = Qwen3OmniMoeProcessor.from_pretrained(model)
        llm = LLM(
                model=model, trust_remote_code=True, gpu_memory_utilization=0.95,
                tensor_parallel_size=torch.cuda.device_count(),
                limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3},
                max_num_seqs=8,
                max_model_len=20000,
                seed=1234,
        )
        results = generate(model, data, processor, llm, sampling_params)
        with open(os.path.join(parent_result_dir, f"{model.replace('/', '-') }_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(results, f)
