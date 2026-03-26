import os
import re
import time
import json
import logid
import base64
import concurrent.futures

import math
from tqdm import tqdm
from openai import AzureOpenAI
from moviepy import VideoFileClip, vfx

FAST_MODEL_NAME="gemini-2.5-pro-preview-06-05"
SLOW_MODEL_NAME="gemini-2.5-flash"


##########
# Gemini #
##########

def t2t_generate(prompt, model= "gpt-4o-2024-08-06"):
    max_tokens = 4095
    client = AzureOpenAI(
        azure_endpoint="https://search-va.byteintl.net/gpt/openapi/online/v2/crawl",
        api_version="2024-03-01-preview",
        api_key="",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                         "text": prompt
                     }
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content

def video_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def v2t_generate(prompt, video_file_path, model_name='gemini-2.5-flash', sleep=60):
    content = [{"type": "text", "text": prompt}]
    video_url = video_to_base64(video_file_path)
    content.append({
        "type": "image_url",
        "image_url": {
            "url": video_url,
            "mime_type": "video/mp4",
        },
        "video_metadata" : {"videoMetadata": { "fps": "1"}}
    })

    time_s = time.time()
    client = AzureOpenAI(
        api_key="",
        azure_endpoint="https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl",
        api_version="2024-03-01",
    )
    log_id = logid.generate()
    # print("🔗 Request ID:", log_id)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            timeout=600,
            max_tokens=65536,
            temperature=0,
            extra_headers={"X-TT-LOGID": log_id},
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


#########
# Utils #
#########

def preprocess_video_to_1fps(video_path, output_dir):
    with VideoFileClip(video_path) as video:
        video_clip = video.without_audio()
        video_name = os.path.basename(video_clip.filename).split(".")[0]
        video_fps = video_clip.fps
        video_duration = video_clip.duration * 1000
        video_output_path = os.path.join(output_dir, f"{video_name}_1fps.mp4")
        try:
            video_clip = video_clip.with_effects([vfx.MultiplySpeed(1.0 / video_clip.fps)])
            video_clip.write_videofile(video_output_path, fps=1, codec='libx264')
        except Exception as e:
            print(f"\n{e}")
    return video_fps, video_duration, video_output_path

def extract_event_meta_data(text):
    pattern = re.compile(r'\[(\d{2}:\d{2})\] - (.*)')
    matches = pattern.findall(text)
    return matches

def convert_timestamp_to_1fps(original_timestamp, original_fps=30.0):
    try:
        minutes_str, seconds_str = original_timestamp.split(':')
        minutes = int(minutes_str)
        seconds = int(seconds_str)
        total_original_seconds = (minutes * 60) + seconds
        frame_number = math.floor(total_original_seconds * original_fps)
        new_total_seconds = frame_number
        new_minutes = new_total_seconds // 60
        new_seconds = new_total_seconds % 60
        return f"{new_minutes:02d}:{new_seconds:02d}"

    except (ValueError, IndexError):
        print(f"Error: Invalid timestamp format provided: '{original_timestamp}'. Please use 'MM:SS'.")
        return None

def cut_video_from_timestamp(input_path, start_time_str, output_directory):
    try:
        try:
            minutes, seconds = map(int, start_time_str.split(':'))
            start_time_in_seconds = (minutes * 60) + seconds
        except ValueError:
            print("Error: Invalid start_time format. Please use 'MM:SS'.")
            return None

        if not os.path.exists(input_path):
            print(f"Error: Input file not found at '{input_path}'")
            return None
        if not os.path.exists(output_directory):
            print(f"Output directory not found. Creating it at '{output_directory}'")
            os.makedirs(output_directory)

        with VideoFileClip(input_path) as video:
            if start_time_in_seconds >= video.duration:
                print(f"Error: Start time ({start_time_in_seconds}s) is after the video ends ({video.duration}s).")
                return None

            subclip = video.subclipped(start_time_in_seconds, video.duration)
            base_filename = os.path.basename(input_path)
            filename, file_extension = os.path.splitext(base_filename)
            safe_timestamp = start_time_str.replace(':', '-')
            output_filename = f"{filename}_cut_from_{safe_timestamp}{file_extension}"
            output_path = os.path.join(output_directory, output_filename)
            subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")
 
            return output_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def add_time(timestamp, add_time_stamp):
    try:
        minutes, seconds = map(int, timestamp.split(':'))
        add_minutes, add_seconds = map(int, add_time_stamp.split(':'))
        total_seconds = (minutes * 60) + seconds
        total_add_seconds = (add_minutes * 60) + add_seconds
        new_total_seconds = total_seconds + total_add_seconds
        new_minutes = new_total_seconds // 60
        new_seconds = new_total_seconds % 60
        return f"{new_minutes:02d}:{new_seconds:02d}"
    except ValueError:
        print("Error: Invalid timestamp format. Please use 'MM:SS'.")
        return None


def convert_event_timestamps(data, original_fps):    
    def mmss_to_seconds(mmss_string):
        """Helper function to convert 'MM:SS' string to seconds."""
        minutes, seconds = map(int, mmss_string.split(':'))
        return minutes * 60 + seconds

    processed_data = []
    for event in data:
        new_event = event.copy()

        start_1fps_seconds = mmss_to_seconds(event['start_timestamp'])
        original_start_seconds = start_1fps_seconds / original_fps
        new_event['start_timestamp'] = round(original_start_seconds * 1000)
        
        end_1fps_seconds = mmss_to_seconds(event['end_timestamp'])
        original_end_seconds = end_1fps_seconds / original_fps
        new_event['end_timestamp'] = round(original_end_seconds * 1000)
        
        processed_data.append(new_event)
        
    return processed_data


###########################
# Understanding Functions #
###########################

def video_overview(input_video_path):
    
    video_overview_prompt = """***Goal:**
Generate a detailed log of inferred sounds for the provided silent video. Analyze the video chronologically and identify all significant diegetic sounds (sounds originating from within the video's world).
Focusing on the non-speech sounds, and sound that caused by action or movement.
Ignore the ambient sounds.
Please try to include as many sound effects as possible.

**Instructions:**
1.  **Output Format:** Present the sounds in a timestamped list. Use the format: `[Timestamp] - [Sound Description]`
2.  **Be Specific:** Do not use generic descriptions. Instead of "car sound," specify "a car engine starting," "tires screeching on asphalt," or "a distant car horn."
3.  **Be Self-Contained:** The sound description should be self-contained and not rely on other sound descriptions. Each sound description should be isolated. Don't mention other sounds (e.g. the second sound is from the first sound).
4.  **Identify Concurrent Sounds:** If multiple sounds occur simultaneously, list them all at the same timestamp.
5.  **Note Pauses:** If there is a clear break or silence between distinct sounds, represent that with separate timestamp entries. If some sound happen more than once, break them into two.
6.  **Describe Sound Quality:** Include adjectives to describe the nature of the sound (e.g. "distant", "sudden").

**Example of Desired Output:**
[00:02] - A heavy wooden door creaks open slowly.
[00:04] - Soft footsteps on a polished floor.
[00:07] - The sharp click of a light switch.
[00:07] - The low hum of a fluorescent light turning on.
[00:12] - A piece of paper rustles as it's picked up.

Please only output timestamped list. Don't include any other text."""
    response = v2t_generate(video_overview_prompt, input_video_path, model_name=FAST_MODEL_NAME, sleep=60)
    if response:
        try:
            matches = extract_event_meta_data(response)
            return response, matches
        except Exception as e:
            print(f"Error: {e}")
            return response, None
    else:
        return None, None


def start_timestamp_localization(input_video_path, formated_overview):
    start_timestamp_localization_prompt = f"""# Goal:
Act as a video analysis expert. Your task is to analyze a silent video to pinpoint the precise start time of a specific event, based on a descriptive prompt and a rough timestamp that serves as a search hint.

# Context:
The user provides a silent video and a log of sound events with approximate timestamps. 
You must correct this timestamp by finding the exact moment the visual action corresponding to the sound begins.

# Instructions:
1. Establish a Search Window: The provided timestamp is an approximation. Focus your analysis on the video segment from 10 seconds before to 10 seconds after this timestamp.
2. Identify Key Visual Cues: Read the "Event Description" and determine the specific physical actions that would create that sound. For example, for a "meow," the key visual cue is the initial opening of the cat's mouth. For a "thud," it might be the moment an object makes impact.
3. Pinpoint the Start Frame: Meticulously review the frames within the search window to find the very first frame where the key visual cue occurs. The timestamp of this frame is the precise start time.
4. Format the Output: Structure your response clearly with the precise timestamp and your justification.

# Output format:
[00:02] - A heavy wooden door creaks open slowly.
[00:04] - Soft footsteps on a polished floor.
[00:07] - The sharp click of a light switch.

# Input:
""" 
    start_timestamp_localization_prompt += formated_overview
    response = v2t_generate(start_timestamp_localization_prompt, input_video_path,  model_name=FAST_MODEL_NAME, sleep=60)
    if response:
        try:
            matches = extract_event_meta_data(response)
            return response, matches
        except Exception as e:
            print(f"Extract Error: {e}")
            return response, None
    else:
        return None, None

def single_event_end_timestamp_localization(slowed_video_path, cache_dir, start_timestampe, description):

    if start_timestampe is None or description is None:
        return None, None
    
    edited_video_path = cut_video_from_timestamp(slowed_video_path, start_timestampe, cache_dir)
    end_timestamp_localization_prompt = f"""# Goal:
Act as a video analysis expert. Your task is to determine the precise end time of an event in a silent video using a two-step analysis.

# Instructions:
You will be given an Event Description.

1. Step 1: Define the End Cue
Based on the Event Description, first determine the specific visual action that signals the end of the event. This is the "End Cue."
- Example for "a meow": The End Cue is "The cat's mouth is fully closed."
- Example for "a door opening": The End Cue is "The door stops moving."

2. Step 2: Locate the End Cue
Analyze the video frame-by-frame. 
Find the exact timestamp (MM:SS) when the End Cue you defined in Step 1 first occurs.
Scan the video chronologically from the Start Search Time, identify the first instance where the End Cue is completed. The timestamp of that moment is your answer.

# Output Format (Make Sure to follow the exact format, including the row names but not the square brackets):
End Sign Description: <the description of action/scene that indicte the end of the sound/event>
Reasoning on how to locate the end time: <the reasoning process on how to locate the end time>
End Time: <MM:SS>

# Input to process:
- Search Event Description: {description}
"""
    end_timestamp_localization_response = v2t_generate(end_timestamp_localization_prompt, edited_video_path, SLOW_MODEL_NAME, sleep=20)
    try:
        if not end_timestamp_localization_response:
            print(f"Error: No response found.")
            return end_timestamp_localization_response, None

        if "End Time: " in end_timestamp_localization_response:
            end_time = end_timestamp_localization_response.split("End Time: ")[1].split("\n")[0].strip()
            if end_time:
                end_time = add_time(start_timestampe, end_time)
                return end_timestamp_localization_response, end_time
            else:
                print(f"Error: Invalid response format. Please use 'End Time: (MM:SS)' format.")
            return end_timestamp_localization_response, None
        else:
            print(f"Error: Invalid response format. Please use 'End Time: (MM:SS)' format.")
            return end_timestamp_localization_response, None
    except Exception as e:
        print(f"Error: {e}")
        return end_timestamp_localization_response, None
    

def end_timestamp_localization(slowed_video_path, cache_dir, start_timestamps, intermediate_dir):
    end_timestamps = []
    end_timestamp_responses = []
    for start_timestamp, description in tqdm(start_timestamps):
        end_timestamp_localization_response, end_time = single_event_end_timestamp_localization(slowed_video_path, cache_dir, start_timestamp, description)
        end_timestamps.append((start_timestamp, end_time, description))
        end_timestamp_responses.append(end_timestamp_localization_response)
        with open(os.path.join(intermediate_dir, "end_timestamp_responses.txt"), "w") as f:
            f.write(end_timestamp_localization_response)
        with open(os.path.join(intermediate_dir, "end_timestamps.json"), "w") as f:
            json.dump(end_timestamps, f, indent=4)
    return end_timestamps, end_timestamp_responses


def _process_single_event(args):
    """
    Worker function to process a single event.
    This function is designed to be run by the thread pool.
    """
    slowed_video_path, cache_dir, start_timestamp, description = args
    response, end_time = single_event_end_timestamp_localization(
        slowed_video_path,
        cache_dir,
        start_timestamp,
        description
    )
    return (start_timestamp, end_time, description), response


def end_timestamp_localization_parallel(slowed_video_path, cache_dir, start_timestamps, intermediate_dir):
    end_timestamps = []
    end_timestamp_responses = []
    task_args = [
        (slowed_video_path, cache_dir, start_timestamp, description)
        for start_timestamp, description in start_timestamps
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(executor.map(_process_single_event, task_args), total=len(task_args)))

    for end_timestamp_tuple, response in results:
        end_timestamps.append(end_timestamp_tuple)
        end_timestamp_responses.append(response)

    os.makedirs(intermediate_dir, exist_ok=True)
    with open(os.path.join(intermediate_dir, "end_timestamp_responses.txt"), "w") as f:
        f.write("\n".join(end_timestamp_responses))
    with open(os.path.join(intermediate_dir, "end_timestamps.json"), "w") as f:
        json.dump(end_timestamps, f, indent=4)

    return end_timestamps, end_timestamp_responses



#######
# Run #
#######

def VideoUnderstanding(video_path, output_dir, intermediate_dir, video_cache_dir):

    time_record = {}
    start_time_s = time.time()

    print("[1/3] video overview")
    overview_response, overview = video_overview(video_path)
    with open(os.path.join(intermediate_dir, "overview_response.txt"), "w") as f:
        f.write(overview_response)
    with open(os.path.join(intermediate_dir, "overview.json"), "w") as f:
        json.dump(overview, f, indent=4)

    overview_end_time_s = time.time()
    time_record["overview_generation"] = overview_end_time_s - start_time_s

    print("[2/3] convert video to 1fps")
    video_fps, video_duration, slowed_video_path = preprocess_video_to_1fps(video_path, video_cache_dir)

    preprocess_end_time_s = time.time()
    time_record["preprocess"] = preprocess_end_time_s - start_time_s

    print("[3/3] start timestamp localization")
    formated_str = ""
    for timestamp, description in overview:
        formated_str += f"[{convert_timestamp_to_1fps(timestamp, float(video_fps))}] - {description}\n"
    start_timestamps_response, start_timestamps = start_timestamp_localization(slowed_video_path, formated_str)
    with open(os.path.join(intermediate_dir, "start_timestamp_localization_response.txt"), "w") as f:
        f.write(start_timestamps_response)
    with open(os.path.join(intermediate_dir, "start_timestamp_localization.json"), "w") as f:
        json.dump(start_timestamps, f, indent=4)
    
    start_timestamp_localization_end_time_s = time.time()
    time_record["start_timestamp_localization"] = start_timestamp_localization_end_time_s - preprocess_end_time_s

    print("[4/4] end timestamp localization")
    end_timestamps, end_timestamp_responses = end_timestamp_localization_parallel(slowed_video_path, video_cache_dir, start_timestamps, intermediate_dir)
    with open(os.path.join(intermediate_dir, "end_timestamp_localization.json"), "w") as f:
        json.dump(end_timestamps, f, indent=4)
    with open(os.path.join(intermediate_dir, "end_timestamp_localization_response.txt"), "w") as f:
        for response in end_timestamp_responses:
            f.write(response + "\n")

    end_timestamp_localization_end_time_s = time.time()
    time_record["end_timestamp_localization"] = end_timestamp_localization_end_time_s - preprocess_end_time_s
    
    print("finalize sounding events metadata")
    sounding_events_meta_data = []
    for start_timestamp, end_timestamp, description in tqdm(end_timestamps):
        if end_timestamp is not None:
            sounding_events_meta_data.append({"start_timestamp": start_timestamp, "end_timestamp": end_timestamp, "description": description})
    original_sounding_events_meta_data = convert_event_timestamps(sounding_events_meta_data, float(video_fps))

    finalize_end_time_s = time.time()
    time_record["finalize"] = finalize_end_time_s - end_timestamp_localization_end_time_s

    with open(os.path.join(output_dir, "original_sounding_events_meta_data.json"), "w") as f:
        json.dump(original_sounding_events_meta_data, f, indent=4)
    with open(os.path.join(output_dir, "sounding_events_meta_data.json"), "w") as f:
        json.dump(sounding_events_meta_data, f, indent=4)

    total_time_s = time.time()
    time_record["total"] = total_time_s - start_time_s
    
    with open(os.path.join(output_dir, "time_record_video_reasoning_.json"), "w") as f:
        json.dump(time_record, f, indent=4)

    return video_duration, original_sounding_events_meta_data

if __name__ == "__main__":
    video_path = "/mnt/bn/icvg-xinyao-llm2/bingxuan/new_pipeline/data/test_video_10.mp4"
    output_dir = "test_4"

    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    intermediate_dir = os.path.join(cache_dir, "intermediate_results")
    os.makedirs(intermediate_dir, exist_ok=True)
    video_cache_dir = os.path.join(cache_dir, "video_cache")
    os.makedirs(video_cache_dir, exist_ok=True)

    video_duration, original_sounding_events_meta_data = VideoUnderstanding(video_path, output_dir, intermediate_dir, video_cache_dir)
    

   

    
    
