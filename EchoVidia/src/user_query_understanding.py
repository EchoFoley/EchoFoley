import os
import re
import time
import json
import logid
import base64

import math
from tqdm import tqdm
from openai import AzureOpenAI
from moviepy import VideoFileClip, vfx

##########
# Gemini #
##########

def t2t_generate(prompt, model= "gpt-4.1-2025-04-14"):
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

def v2t_generate(prompt, video_file_path, model_name='gemini-2.5-pro-preview-06-05', sleep=60):
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
        api_version="2025-09-27",
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
3.  **Be Self-Contained:** The sound description should be self-contained and not rely on other sound descriptions.
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
    response = v2t_generate(video_overview_prompt, input_video_path)
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
    response = v2t_generate(start_timestamp_localization_prompt, input_video_path)
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
    end_timestamp_localization_response = v2t_generate(end_timestamp_localization_prompt, edited_video_path)
    try:
        if "End Time: " in end_timestamp_localization_response:
            end_time = end_timestamp_localization_response.split("End Time: ")[1].split("\n")[0]
            end_time = add_time(start_timestampe, end_time)
            return end_timestamp_localization_response, end_time.strip()
        else:
            print(f"Error: Invalid response format. Please use 'End Time: (MM:SS)' format.")
            return end_timestamp_localization_response, None
    except Exception as e:
        print(f"Error: {e}")
        return end_timestamp_localization_response, None
    
def end_timestamp_localization(slowed_video_path, cache_dir, start_timestamps):
    end_timestamps = []
    end_timestamp_responses = []
    for start_timestamp, description in tqdm(start_timestamps):
        end_timestamp_localization_response, end_time = single_event_end_timestamp_localization(slowed_video_path, cache_dir, start_timestamp, description)
        end_timestamps.append((start_timestamp, end_time, description))
        end_timestamp_responses.append(end_timestamp_localization_response)
    return end_timestamps, end_timestamp_responses


###################################
# InstructionGeneration Functions #
###################################

def intention_detection(sounding_events_meta_data, user_query):
    intention_detection_prompt = f"""***GOAL***: 
    Your primary goal is to interpret a user's request (`user_query`) and translate it into a structured JSON command to manipulate a list of existing sound events.

    ***INPUT SCHEMA***:
    The `sound_events` input will be a JSON array of objects. Each object represents a sound event with the following schema:
    ```json
    {{
    "id": "string",       // Unique identifier for the event
    "start_timestamp": "float", // Start time in seconds
    "start_timestamp": "float",   // End time in seconds
    "description": "string"     // A description of the sound (e.g., "dog barking", "heavy rain")
    }}
    ```

    ***OUTPUT SCHEMA***:
    You MUST respond with a single, valid JSON object. The object must have two keys: "action" and "parameters".
    The "action" key must be one of the following strings:
    1. "ADD_EVENT": To add one or more new sound events.
    2. "DELETE_EVENTS": To remove one or more existing events.
    3. "MODIFY_TIME": To change the start and/or end time of one or more existing events.
    4. "MODIFY_DESCRIPTION": To change the description (e.g timbre) of one or more existing events.
    5. "CHECK_VIDEO": If the an action required to reference the visual content, you should respond with "CHECK_VIDEO" and corresponding extracted description on the visual content to check, and the action object.
    6. "NO_OP": If the user's query is not a valid command, is ambiguous, or is a simple question.

    ***PARAMETER SCHEMA (Varies based on the action)***:

    1. For "ADD_EVENT":
    The `parameters` object contains one key, `events`, which is an array of new event objects to be created. 
    Please reason where the new sound should be inserted. previous_event and next_event are the id of the sound before and after the new sound. Previous and next event can be None if the new sound is the first or last sound.
    If the new sound is a background sound that last for entire video, please set background_flag to True. Otherwise, set background_flag to False.
    ```json
    {{
        "action": "ADD_EVENT",
        "parameters": {{
        "events": [
            {{ "background_flag": boolean, "previous_event": id, "next_event": id, "description": "cat meowing" }}
        ]
        }}
    }}
    ```

    2. For "DELETE_EVENTS":
    The `parameters` object contains one key, `id`, which is an array of strings corresponding to the `id` of the sounds to be removed.
    ```json
    {{
        "action": "DELETE_EVENTS",
        "parameters": {{ "id": ["evt_002"] }}
    }}
    ```

    3. For "MODIFY_TIME":
    The `parameters` object contains one key, `events`, which is an array of objects. Each object must have an `id` and the extracted `instruction`.
    ```json
    {{
        "action": "MODIFY_TIME",
        "parameters": {{
        "events": [
            {{"id": "rain_sound_1", "instruction": "make the sound last 10 seconds"}}
        ]
        }}
    }}
    ```

    4. For "MODIFY_DESCRIPTION":
    The `parameters` object contains one key, `events`, which is an array of objects. Each object must have an `id` and a extracted `instruction`.
    ```json
    {{
        "action": "MODIFY_DESCRIPTION",
        "parameters": {{
        "events": [
            {{ "id": "dog_1", "instruction": "change to puppy yelp" }}
        ]
        }}
    }}
    ```

    5. For "CHECK_VIDEO":
    The `parameters` object contains `event_query_description`, which is a string, and `action`  which is an object.
    ```json
    {{
        "action": "CHECK_VIDEO",
        "parameters": {{ "event_query_description": "check the video for the dog first turn the head to the left." ,
        "action": the action object (ADD_EVENT, DELETE_EVENTS, MODIFY_TIME, or MODIFY_DESCRIPTION)}}
    }}


    6. For "NO_OP":
    The `parameters` object contains one key, `reason`, with a string explaining why no action could be determined.
    ```json
    {{
        "action": "NO_OP",
        "parameters": {{ "reason": "The user query is too ambiguous." }}
    }}
    ```

    ***INSTRUCTIONS & HEURISTICS***

    - Analyze the user's query in the context of the provided `sound_events`.
    - Identify the specific event(s) the user is referring to by matching their description, timing, or relative position (e.g., "the first sound").
    - When modifying or deleting, it is CRITICAL to use the correct `id` from the input list.
    - If the user wants to add a sound but does not specify a duration, assume a reasonable default duration (e.g., 2-3 seconds).
    - If a query is ambiguous (e.g., "make it better") or is just a question ("what sounds are there?"), use the "NO_OP" action.
    - Your entire output must be only the JSON object, with no other text before or after it.

    ***TASK***

    - Input (User Query)
    {user_query}

    - Input (Sound Events)
    {sounding_events_meta_data}
    """
    intention_detection_response = t2t_generate(intention_detection_prompt)
    if intention_detection_response:
        try:
            if "```json" in intention_detection_response:
                extracted_json = intention_detection_response.split("```json")[1].split("```")[0]
                extracted_json = json.loads(extracted_json)
            else:
                extracted_json = json.loads(intention_detection_response)
            return intention_detection_response, extracted_json
        except Exception as e:
            print(f"Error: {e}")
            return intention_detection_response, None
    else:
        return None, None


################
# Actions Pool #
################

def delete_events(sound_events_meta_data, action):
    ids_to_delete = action["parameters"]["id"]
    sound_events_meta_data = [event for event in sound_events_meta_data if event["id"] not in ids_to_delete]
    return sound_events_meta_data

def modify_description(sound_events_meta_data, action):
    events_to_modify = action["parameters"]["events"]
    for event in events_to_modify:
        for existing_event in sound_events_meta_data:
            if existing_event["id"] == event["id"]:
                existing_event["description"] = event["description"]

def add_event(sound_events_meta_data, action):
    new_event = action["parameters"]["events"][0]
    sound_events_meta_data.append(new_event)
    return sound_events_meta_data

def modify_time(sound_events_meta_data, action):
    events_to_modify = action["parameters"]["events"]
    for event in events_to_modify:
        for existing_event in sound_events_meta_data:
            if existing_event["id"] == event["id"]:
                existing_event["start_timestamp"] = event["start_timestamp"]
                existing_event["end_timestamp"] = event["end_timestamp"]


###############################
# EventPlan atomic edit helpers
###############################

def apply_atomic_edit_to_event_plan(event_plan_dicts, atomic_edit):
    """
    Apply one atomic edit action to an event_plan list.

    event_plan_dicts must follow ActionState format:
      {id, t:[t_start_sec, t_end_sec], d:{subject,action,object}, p:{pitch,volume,intensity,spatial}}
    """
    if not atomic_edit:
        return event_plan_dicts
    action = atomic_edit.get("action")
    params = atomic_edit.get("parameters") or {}

    plan = [dict(e) for e in event_plan_dicts]
    by_id = {str(e.get("id")): e for e in plan if e.get("id") is not None}

    if action == "NO_OP":
        return plan

    if action == "DELETE_EVENT":
        ids = [str(i) for i in params.get("id", [])]
        plan = [e for e in plan if str(e.get("id")) not in set(ids)]

    elif action == "ADD_EVENT":
        events = params.get("events") or []
        for e in events:
            # Ensure required structure exists.
            t = e.get("t") or e.get("time") or [0.0, 0.0]
            d = e.get("d") or {}
            p = e.get("p") or {}
            plan.append(
                {
                    "id": str(e.get("id") or f"evt_{len(plan)}"),
                    "t": [float(t[0]), float(t[1])],
                    "d": {
                        "subject": str(d.get("subject", "unknown")),
                        "action": str(d.get("action", "unknown")),
                        "object": str(d.get("object", "")),
                    },
                    "p": {
                        "pitch": str(p.get("pitch", "DEFAULT")),
                        "volume": str(p.get("volume", "DEFAULT")),
                        "intensity": str(p.get("intensity", "DEFAULT")),
                        "spatial": str(p.get("spatial", "DEFAULT")),
                    },
                }
            )

    elif action == "MODIFY_DESCRIPTION":
        events = params.get("events") or []
        for e in events:
            event_id = str(e.get("id"))
            if event_id in by_id:
                d = e.get("d") or {}
                by_id[event_id].setdefault("d", {})
                by_id[event_id]["d"]["subject"] = str(d.get("subject", by_id[event_id]["d"].get("subject", "unknown")))
                by_id[event_id]["d"]["action"] = str(d.get("action", by_id[event_id]["d"].get("action", "unknown")))
                by_id[event_id]["d"]["object"] = str(d.get("object", by_id[event_id]["d"].get("object", "")))

    elif action == "MODIFY_TIME":
        events = params.get("events") or []
        for e in events:
            event_id = str(e.get("id"))
            if event_id in by_id:
                t = e.get("t") or []
                if len(t) == 2:
                    by_id[event_id]["t"] = [float(t[0]), float(t[1])]

    elif action == "MODIFY_PROPERTIES":
        events = params.get("events") or []
        for e in events:
            event_id = str(e.get("id"))
            if event_id in by_id:
                p = e.get("p") or {}
                by_id[event_id].setdefault("p", {})
                by_id[event_id]["p"]["pitch"] = str(p.get("pitch", by_id[event_id]["p"].get("pitch", "DEFAULT")))
                by_id[event_id]["p"]["volume"] = str(p.get("volume", by_id[event_id]["p"].get("volume", "DEFAULT")))
                by_id[event_id]["p"]["intensity"] = str(p.get("intensity", by_id[event_id]["p"].get("intensity", "DEFAULT")))
                by_id[event_id]["p"]["spatial"] = str(p.get("spatial", by_id[event_id]["p"].get("spatial", "DEFAULT")))

    # Normalize ordering and remove invalid spans.
    normalized = []
    for e in plan:
        t = e.get("t") or []
        if len(t) != 2:
            continue
        t_start, t_end = float(t[0]), float(t[1])
        if t_end <= t_start:
            continue
        e["t"] = [t_start, t_end]
        normalized.append(e)
    normalized.sort(key=lambda e: e["t"][0])
    return normalized


def edit_event_plan_once_via_llm_atomic(event_plan_dicts, user_query):
    """
    Optional: produce one atomic edit command (ADD/DELETE/MODIFY_*) and apply it.
    Not used by the current controller (which relies on `sound_design_actions`), but kept
    here to match the action-pool semantics and for future iterative refinement loops.
    """
    prompt = f"""SYSTEM ROLE:
You are a Sound Design Controller. Decide which SINGLE atomic edit action to apply to
the CURRENT_EVENT_PLAN based on USER_INSTRUCTION.

INPUT:
USER_INSTRUCTION: {user_query}
CURRENT_EVENT_PLAN: {json.dumps(event_plan_dicts)}

TASK:
Return exactly one action. Allowed actions:
ADD_EVENT, DELETE_EVENT, MODIFY_DESCRIPTION, MODIFY_TIME, MODIFY_PROPERTIES, NO_OP

OUTPUT FORMAT (STRICT):
Return valid JSON only:
{{"action": "...", "parameters": {{...}}}}

Schema:
- ADD_EVENT: parameters={{"events":[{{"id":"...","t":[x.xx,y.yy],"d":{{"subject":"...","action":"...","object":"..."}}, "p":{{"pitch":"...","volume":"...","intensity":"...","spatial":"..."}}}}]}}
- DELETE_EVENT: parameters={{"id":["evt_0", "..."]}}
- MODIFY_DESCRIPTION: parameters={{"events":[{{"id":"...","d":{{"subject":"...","action":"...","object":"..."}}}}]}}
- MODIFY_TIME: parameters={{"events":[{{"id":"...","t":[x.xx,y.yy]}}]}}
- MODIFY_PROPERTIES: parameters={{"events":[{{"id":"...","p":{{"pitch":"...","volume":"...","intensity":"...","spatial":"..."}}}}]}}
- NO_OP: parameters={{"reason":"..."}}
"""
    resp = t2t_generate(prompt)
    atomic = json.loads(resp)
    return apply_atomic_edit_to_event_plan(event_plan_dicts, atomic)


#######
# Run #
#######

def VideoUnderstanding(video_path, output_dir, intermediate_dir, video_cache_dir):

    print("[1/3] video overview")
    overview_response, overview = video_overview(video_path)
    with open(os.path.join(intermediate_dir, "overview_response.txt"), "w") as f:
        f.write(overview_response)
    with open(os.path.join(intermediate_dir, "overview.json"), "w") as f:
        json.dump(overview, f, indent=4)


    print("[2/3] convert video to 1fps")
    video_fps, video_duration, slowed_video_path = preprocess_video_to_1fps(video_path, video_cache_dir)
    

    print("[3/3] start timestamp localization")
    formated_str = ""
    for timestamp, description in overview:
        formated_str += f"[{convert_timestamp_to_1fps(timestamp, float(video_fps))}] - {description}\n"
    start_timestamps_response, start_timestamps = start_timestamp_localization(slowed_video_path, formated_str)
    with open(os.path.join(intermediate_dir, "start_timestamp_localization_response.txt"), "w") as f:
        f.write(start_timestamps_response)
    with open(os.path.join(intermediate_dir, "start_timestamp_localization.json"), "w") as f:
        json.dump(start_timestamps, f, indent=4)

    print("[4/4] end timestamp localization")
    end_timestamps, end_timestamp_responses = end_timestamp_localization(slowed_video_path, video_cache_dir, start_timestamps)
    with open(os.path.join(intermediate_dir, "end_timestamp_localization.json"), "w") as f:
        json.dump(end_timestamps, f, indent=4)
    with open(os.path.join(intermediate_dir, "end_timestamp_localization_response.txt"), "w") as f:
        for response in end_timestamp_responses:
            f.write(response + "\n")

    print("finalize sounding events metadata")
    sounding_events_meta_data = []
    for start_timestamp, end_timestamp, description in end_timestamps:
        if end_timestamp is not None:
            sounding_events_meta_data.append({"start_timestamp": start_timestamp, "end_timestamp": end_timestamp, "description": description})
    original_sounding_events_meta_data = convert_event_timestamps(sounding_events_meta_data, float(video_fps))
    
    with open(os.path.join(intermediate_dir, "original_sounding_events_meta_data.json"), "w") as f:
        json.dump(original_sounding_events_meta_data, f, indent=4)
    with open(os.path.join(output_dir, "sounding_events_meta_data.json"), "w") as f:
        json.dump(sounding_events_meta_data, f, indent=4)

    return video_duration, original_sounding_events_meta_data

def InstructionalVideoUnderstanding(video_path, output_dir, intermediate_dir, video_cache_dir, audio_clip_data, instruction):

    print("[1/3] intention detection")
    intention_detection_response, extracted_intention_detection= intention_detection(audio_clip_data, instruction)
    with open(os.path.join(intermediate_dir, "intention_detection_response.txt"), "w") as f:
        f.write(intention_detection_response)
    with open(os.path.join(intermediate_dir, "intention_detection.json"), "w") as f:
        json.dump(extracted_intention_detection, f, indent=4)


if __name__ == "__main__":

    video_path = "/mnt/bn/icvg-xinyao-llm2/bingxuan/new_pipeline/data/test_video_10.mp4"
    output_dir = "/mnt/bn/icvg-xinyao-llm2/bingxuan/new_pipeline/editing_results/test/take_1"
    audio_clip_data = [
    {
        "id": "0_A_sharp,_high-pitched_meow_from_a_cat.",
        "prompt": "cat meowing softly",
        "description": "A sharp, high-pitched meow from a cat.",
        "start_timestamp": 1133,
        "end_timestamp": 1933
    },
    {
        "id": "1_A_second,_drawn-out,_high-pitched_meow_from_a_cat.",
        "prompt": "cat meowing softly and slowly",
        "description": "A second, drawn-out, high-pitched meow from a cat.",
        "start_timestamp": 4767,
        "end_timestamp": 7133
    }
    ]

    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    intermediate_dir = os.path.join(cache_dir, "intermediate_results")
    os.makedirs(intermediate_dir, exist_ok=True)
    video_cache_dir = os.path.join(cache_dir, "video_cache")
    os.makedirs(video_cache_dir, exist_ok=True)

    instruction = "change the second cat meowing to a lion roaring"
    instruction = "add a backgroud sounds of a park environment"
    instruction = "make the cat make purr sound when it not look at the camera."
    intention_detection_response, extracted_intention = intention_detection(audio_clip_data, instruction)
    print(intention_detection_response)
    

   

    
    
