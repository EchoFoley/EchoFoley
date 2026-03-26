import json
import time
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

parent_result_dir = "/home/jovyan/workspace/results_qwen3vl_zero_shot"
os.makedirs(parent_result_dir, exist_ok=True)

supported_models = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Instruct"
]

def prepare_inputs_for_vllm(messages, processor):
    print(f"Processing messages: {messages[0]['content'][0]['video']}")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


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


def generate(checkpoint_path, data):
    messages_list = []
    for index, row in data.iterrows() :
        messages_list.append([{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": row["video"]
                },
                {"type": "text", "text": VIDEO_OVERVIEW_PROMPT},
            ]}])
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    inputs = [prepare_inputs_for_vllm(message, processor)
              for message in messages_list]
    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.7,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
        max_model_len=100000
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2000,
        top_k=-1,
        stop_token_ids=[],
    )
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        results.append({
            "video": data.iloc[i]["video"],
            "gt": data.iloc[i]["original_sounding_events"],
            "pred": parse_response(generated_text),
            "raw": generated_text
        })
    return results


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv("/home/jovyan/workspace/eval/data/sounding_event_aware_data.csv")
    for model in supported_models:
        print(f"Generating results for {model}...")
        results = generate(model, data)
        with open(os.path.join(parent_result_dir, f"{model.replace('/', '-') }_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(results, f)