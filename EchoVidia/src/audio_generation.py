import os
import json
import re
from typing import Optional
from tqdm import tqdm
from openai import OpenAI
from moviepy import VideoFileClip, AudioFileClip

import warnings
warnings.filterwarnings("ignore")

##############
# Model Init #
##############

# audio model
import torch
AUDIO_MODEL_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
audio_model_name = "stabilityai/stable-audio-open-small"
audio_model, audio_model_config = get_pretrained_model(audio_model_name)
sample_rate = audio_model_config["sample_rate"]
sample_size = audio_model_config["sample_size"]

import time
audio_model = audio_model.to(AUDIO_MODEL_DEVICE)

# text model
def t2t_generate(prompt, model= "openai/gpt-4.1-mini"):
    max_tokens = 4095
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API"),
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
    time.sleep(5)
    return response.choices[0].message.content


############################
# Audio Planning Functions #
############################

def audio_planing(description):
    audio_prompt_design_prompt = f"""**Task:** You are a professional sound designer. Your task is to transform a given event description into a concise and effective audio generation prompt for an AI model.

**Instructions:**
1.  Deconstruct into Core Sounds (Foley Principle):
    - AI models struggle with complex, poetic, or abstract concepts like "raindrop on a lotus.
    - Break down ideas into their simplest, most physically reproducible sound components, just like a foley artist would.
    - Instead of: "wind blowing through ancient ruins," think **"wind gusting," "gravel shifting,"** or **"door creaking."**
    - Focus on what is tangibly making the sound.
2.  Strict Conciseness:
    - The prompt must be **8 words or less**.
    - Remove all unnecessary adjectives, adverbs, and descriptive fluff. Your goal is to provide a single, clear idea.
        - Good examples: "Cat purring," "car passing by," "gentle rain."
        - Bad examples: "A fluffy cat happily purring," "a fast red car passing by on the street."
3.  Clarity and Structure:
    - Structure your prompt with a simple, caption-like style. The ideal formula is `[Subject] [Action]` or `[Subject] [Action] [Environment]`.
4.  Audio Profile Restrictions:
    - NO SPEECH: Do not include any form of human speech, singing, or vocalizations that form discernible words.
    - Avoid Harsh Sounds: Do not generate prompts for sounds that are excessively loud, jarring, high-pitched, or startling (e.g., "gunshot," "siren," "screaming").
5.  Final Output Format: Your entire response must consist of only the generated prompt text.

**Input to process:**
{description}"""
    response = t2t_generate(audio_prompt_design_prompt)
    return response


##############################
# Audio Generation Functions #
##############################

def audio_generation(prompt, second, output_file_path, model=audio_model, sample_rate=sample_rate, sample_size=sample_size):
    conditioning = [{
        "prompt": prompt,
        "seconds_total": second
    }]
    output = generate_diffusion_cond(
        model,
        steps=8,
        cfg_scale=1.0,
        conditioning=conditioning,
        sample_size=sample_size,
        sampler_type="pingpong",
        device=AUDIO_MODEL_DEVICE
    )
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    output = output[:, :int(sample_rate * second)]
    torchaudio.save(f"{output_file_path}.wav", output, sample_rate)

from pydub import AudioSegment
import numpy as np
def mix_audios(total_length_ms, audio_clips, output_path):
    """
    Backwards-compatible mixing helper (legacy ms-based overlay).
    Prefer `mix_audios_crossfade_normalized` for the new action-pool pipeline.
    """
    final_mix = AudioSegment.silent(duration=total_length_ms)
    for clip in audio_clips:
        path = clip.get("path")
        start_time = clip.get("start_timestamp")
        end_time = clip.get("end_timestamp")
        volume_change = clip.get("volume_db", 0)

        if not all([path, isinstance(start_time, int), isinstance(end_time, int)]):
            print(f"Warning: Skipping invalid clip data: {clip}")
            continue

        if not os.path.exists(path):
            print(f"Warning: Audio file not found at {path}. Skipping.")
            continue

        try:
            segment = AudioSegment.from_file(path)
            segment = segment + volume_change
            clip_duration = end_time - start_time
            if clip_duration <= 0:
                continue
            trimmed_segment = segment[:clip_duration]
            final_mix = final_mix.overlay(trimmed_segment, position=start_time)
        except Exception as e:
            print(f"An error occurred with {path}: {e}. Skipping.")

    try:
        file_format = output_path.split(".")[-1]
        final_mix.export(output_path, format=file_format)
    except Exception as e:
        print(f"Error exporting file: {e}")

    return final_mix


def mix_audios_crossfade_normalized(
    *,
    total_duration_s,
    event_clips,
    output_wav_path,
    output_mp3_path=None,
    crossfade_s=0.05,
    peak_normalize=0.98,
):
    """
    Crossfade-capable event overlay + peak normalization.

    event_clips entries:
      - path: str
      - start_s: float
      - end_s: float
      - volume_db: float (optional)
    """
    if not event_clips:
        raise ValueError("event_clips must be non-empty")

    # Determine sample rate from first clip.
    first_seg = AudioSegment.from_file(event_clips[0]["path"])
    sr = int(first_seg.frame_rate)

    total_s = float(total_duration_s)
    final_len = int(total_s * sr)
    final = np.zeros((final_len,), dtype=np.float32)

    clips_sorted = sorted(event_clips, key=lambda c: float(c["start_s"]))
    crossfade_samples = max(1, int(float(crossfade_s) * sr))

    def seg_to_mono_float(seg: AudioSegment) -> np.ndarray:
        s = seg.set_frame_rate(sr).set_channels(1)
        arr = np.array(s.get_array_of_samples()).astype(np.float32)
        # Likely int16.
        if np.max(np.abs(arr)) > 1.5:
            arr = arr / 32768.0
        return arr

    prev_end_idx = 0
    for i, clip in enumerate(clips_sorted):
        seg = AudioSegment.from_file(clip["path"])
        vol_db = float(clip.get("volume_db", 0.0))
        if vol_db:
            seg = seg + vol_db

        samples = seg_to_mono_float(seg)
        start_idx = int(float(clip["start_s"]) * sr)
        end_idx = min(start_idx + len(samples), final_len)
        if end_idx <= start_idx:
            continue

        if i == 0:
            final[start_idx:end_idx] += samples[: end_idx - start_idx]
        else:
            overlap_start = start_idx
            overlap_end = min(int(prev_end_idx), end_idx)
            overlap_len = overlap_end - overlap_start

            if overlap_len > 0:
                # Apply linear crossfade in the overlap region.
                alpha = np.linspace(0.0, 1.0, overlap_len, endpoint=False, dtype=np.float32)
                prev_slice = final[overlap_start:overlap_end].copy()
                new_slice = prev_slice * (1.0 - alpha) + samples[:overlap_len] * alpha
                final[overlap_start:overlap_end] = new_slice

                remaining_start_in_clip = overlap_len
                remaining_start_in_final = overlap_end
                if end_idx > remaining_start_in_final:
                    final[remaining_start_in_final:end_idx] += samples[remaining_start_in_clip:]
            else:
                final[start_idx:end_idx] += samples[: end_idx - start_idx]

        prev_end_idx = int(float(clip["end_s"]) * sr)

    peak = float(np.max(np.abs(final))) if final.size else 0.0
    if peak > 0:
        final = final * (float(peak_normalize) / peak)

    out_int16 = np.clip(final, -1.0, 1.0) * 32767.0
    out_int16 = out_int16.astype(np.int16)
    out_seg = AudioSegment(
        data=out_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )
    out_seg.export(output_wav_path, format="wav")
    if output_mp3_path:
        out_seg.export(output_mp3_path, format="mp3")
    return output_wav_path


def _volume_str_to_db(volume: str) -> float:
    v = str(volume).strip().upper()
    if v in {"", "DEFAULT", "NONE", "NULL"}:
        return 0.0
    # Try numeric parse first.
    m = re.search(r"[-+]?\d*\.?\d+", v) if "re" in globals() else None
    if m:
        return float(m.group(0))
    if "LOUD" in v:
        return 3.0
    if "SOFT" in v:
        return -6.0
    if "MEDIUM" in v:
        return 0.0
    return 0.0


def synthesize_from_event_plan(
    event_plan,
    *,
    output_dir: str,
    cache_dir: str,
    video_duration_s: Optional[float] = None,
    raw_video_path: Optional[str] = None,
    crossfade_s: float = 0.05,
):
    """
    High-level synthesis helper that consumes a symbolic event plan.

    event_plan:
      - list[Event] (from action_state) OR list[dict] in ActionState format.
    """
    import re as _re
    from moviepy import AudioFileClip, VideoFileClip

    # Small local override so we don't depend on module-level re imports.
    def _vol(volume: str) -> float:
        v = str(volume).strip().upper()
        if v in {"", "DEFAULT", "NONE", "NULL"}:
            return 0.0
        m = _re.search(r"[-+]?\d*\.?\d+", v)
        if m:
            return float(m.group(0))
        if "LOUD" in v:
            return 3.0
        if "SOFT" in v:
            return -6.0
        if "MEDIUM" in v:
            return 0.0
        return 0.0

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    audio_dir = os.path.join(cache_dir, "soundtracks")
    os.makedirs(audio_dir, exist_ok=True)

    clips = []
    last_end = 0.0

    for idx, ev in enumerate(event_plan):
        if isinstance(ev, dict):
            ev_id = str(ev.get("id", f"evt_{idx}"))
            t = ev.get("t") or [0.0, 0.0]
            t_start_s, t_end_s = float(t[0]), float(t[1])
            d = ev.get("d") or {}
            p = ev.get("p") or {}
            subject = str(d.get("subject", "unknown"))
            action = str(d.get("action", "unknown"))
            obj = str(d.get("object", ""))
            volume = str(p.get("volume", "DEFAULT"))
        else:
            ev_id = str(getattr(ev, "id", f"evt_{idx}"))
            t_start_s, t_end_s = float(ev.t[0]), float(ev.t[1])
            subject = str(ev.d.subject)
            action = str(ev.d.action)
            obj = str(ev.d.object)
            volume = str(ev.p.volume)

        if t_end_s <= t_start_s:
            continue
        last_end = max(last_end, t_end_s)
        duration_s = t_end_s - t_start_s
        if duration_s < 0.05:
            continue

        description = " ".join([subject.strip(), action.strip(), obj.strip()]).strip()
        audio_prompt = audio_planing(description)

        output_path_no_ext = os.path.join(audio_dir, f"audio_{ev_id}")
        audio_generation(audio_prompt, duration_s + 0.1, output_path_no_ext)

        clips.append(
            {
                "path": f"{output_path_no_ext}.wav",
                "start_s": t_start_s,
                "end_s": t_end_s,
                "volume_db": _vol(volume),
            }
        )

    if video_duration_s is None:
        video_duration_s = last_end

    mixed_wav_path = os.path.join(audio_dir, "mixed_audio.wav")
    mixed_mp3_path = os.path.join(audio_dir, "mixed_audio.mp3")

    mix_audios_crossfade_normalized(
        total_duration_s=video_duration_s,
        event_clips=clips,
        output_wav_path=mixed_wav_path,
        output_mp3_path=mixed_mp3_path,
        crossfade_s=crossfade_s,
        peak_normalize=0.98,
    )

    if raw_video_path is None:
        return mixed_mp3_path

    video_clip = VideoFileClip(raw_video_path)
    audio_clip = AudioFileClip(mixed_mp3_path)
    video_clip.audio = audio_clip

    final_video_path = os.path.join(output_dir, "final_video.mp4")
    video_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac")
    return final_video_path


#######################
# Audio   Generation  #
#######################

def AudioGeneration(sounding_events_metdata, output_dir, cache_dir, total_duration_ms=4000, raw_video_path=None):

    time_record = {}
    start_time_s = time.time()

    print("[1/3] preprocessing audio clips data...")
    audio_clips_data = []
    id_ = 0
    for event in tqdm(sounding_events_metdata):
        start_ms = event.get("start_timestamp")
        end_ms = event.get("end_timestamp")
        clip_description = event.get("description")
        clip_id = f"{id_}_{clip_description.replace(' ', '_')[:10]}"
        audio_prompt = audio_planing(clip_description)
        #audio_prompt = event.get("prompt")
        audio_clips_data.append({
            "id": clip_id,
            "prompt": audio_prompt,
            "description": clip_description,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms
        })
        id_ += 1


    preprocess_time_s = time.time()
    time_record["preprocess"] = preprocess_time_s - start_time_s

    with open(os.path.join(output_dir, f"audio_clips_data.json"), "w") as f:
        json.dump(audio_clips_data, f, indent=4)

    print(f"\nGenerating {len(audio_clips_data)} audio clips...")
    audio_dir = os.path.join(cache_dir, "soundtracks")
    os.makedirs(audio_dir, exist_ok=True)

    print("[2/3] generating audio clips...")

    for i, clip in enumerate(audio_clips_data):
        clip_prompt = clip.get("prompt")
        start_ms = clip.get("start_timestamp")
        end_ms = clip.get("end_timestamp")
        clip_id = clip.get("id", f"clip_{i}").replace(" ", "_")

        if not all([clip_prompt, isinstance(start_ms, int), isinstance(end_ms, int)]):
            print(f"Skipping invalid audio clip data: {clip}")
            continue

        duration_s = end_ms - start_ms
        if duration_s <= 0:
            print(f"Skipping clip '{clip_id}' with non-positive duration.")
            continue
        if duration_s < 250:
            duration_s = 250
        duration_s = duration_s / 1000.0

        output_path_no_ext = os.path.join(audio_dir, f"audio_{clip_id}")
        print(f"  - Generating '{clip_id}' ({duration_s:.2f}s): '{clip_prompt}'")
        audio_generation(clip_prompt, duration_s + 0.1, output_path_no_ext)

        clip['path'] = f"{output_path_no_ext}.wav"

    with open(os.path.join(audio_dir, f"metadata.json"), "w") as f:
        json.dump(audio_clips_data, f, indent=4)
    
    generation_time_s = time.time()
    time_record["generate"] = generation_time_s - preprocess_time_s
        
    print("\n[3/3] Mixing audio tracks...")
    mixed_audio_path = os.path.join(audio_dir, "mixed_audio.mp3")
    mix_audios(total_duration_ms, audio_clips_data, mixed_audio_path)

    mix_time_s = time.time()
    time_record["mix"] = mix_time_s - generation_time_s

    time_record["total"] = time.time() - start_time_s

    with open(os.path.join(output_dir, f"time_record_audio_generation.json"), "w") as f:
        json.dump(time_record, f, indent=4)

    if raw_video_path is not None:
        print("\nCombining video and audio...")
        final_video_path = os.path.join(output_dir, "final_video.mp4")

        try:
            if not os.path.exists(raw_video_path) or not os.path.exists(mixed_audio_path):
                print("Error: Raw video or mixed audio file not found for final compilation.")
                return

            video_clip = VideoFileClip(raw_video_path)
            audio_clip = AudioFileClip(mixed_audio_path)

            video_clip.audio = audio_clip

            video_clip.write_videofile(
                final_video_path,
                codec="libx264",
                audio_codec="aac"
            )
            
            print(f"\n--- Success! Final video saved to: {final_video_path} ---")
            return final_video_path

        except Exception as e:
            print(f"\nAn error occurred during the final compilation: {e}")


def mmss_to_seconds(mmss_string):
    """Helper function to convert 'MM:SS' string to seconds."""
    minutes, seconds = map(int, mmss_string.split(':'))
    return (minutes * 60 + seconds) * 1000

def convert_to_metadata(audio_clip_data):
    """
    Convert audio clip data to metadata format.
    """
    metadata = []
    for clip in audio_clip_data:
        clip_id = clip.get("id", f"clip_{len(metadata)}").replace(" ", "_")
        start_ms = clip.get("start_timestamp")
        end_ms = clip.get("end_timestamp")
        clip_description = clip.get("description")
        if not all([clip_id, start_ms, end_ms, clip_description]):
            print(f"Skipping invalid metadata entry: {clip}")
            continue
        metadata.append({
            "id": clip_id,
            "description": clip_description,
            "start_timestamp": mmss_to_seconds(start_ms),
            "end_timestamp": mmss_to_seconds(end_ms)
        })
    return metadata
            

if __name__ == "__main__":

#     sounding_events_metdata = [
#     {
#         "id": "0_Sneakers_scuffing_against_sandy_ground.",
#         "prompt": "Sneakers scuffing on dry sandy ground",
#         "description": "Sneakers scuffing against sandy ground.",
#         "start_timestamp": 938,
#         "end_timestamp": 2438
#     },
#     {
#         "id": "1_A_quick_thud_of_feet_pushing_off_a_wooden_plank.",
#         "prompt": "feet pushing off wooden plank, quick thud",
#         "description": "A quick thud of feet pushing off a wooden plank.",
#         "start_timestamp": 2000,
#         "end_timestamp": 4000
#     },
#     {
#         "id": "2_A_slight,_metallic_squeak_from_the_seesaw's_pivot_point_under_sudden_force.",
#         "prompt": "metal seesaw pivot squeaking softly under pressure",
#         "description": "A slight, metallic squeak from the seesaw's pivot point under sudden force.",
#         "start_timestamp": 2000,
#         "end_timestamp": 4062
#     },
#     {
#         "id": "3_A_heavy,_solid_thud_as_the_wooden_seesaw_seat_hits_the_sandy_ground.",
#         "prompt": "wooden seesaw seat thudding onto sandy ground",
#         "description": "A heavy, solid thud as the wooden seesaw seat hits the sandy ground.",
#         "start_timestamp": 2062,
#         "end_timestamp": 4250
#     },
#     {
#         "id": "33_A_quick_thud_of_feet_pushing_off_a_wooden_plank.",
#         "prompt": "Children laughing, happy",
#         "description": "A quick thud of feet pushing off a wooden plank.",
#         "start_timestamp": 2000,
#         "end_timestamp": 6000
#     },
#     {
#         "id": "4_A_soft_thud_of_sneakers_landing_on_grass.",
#         "prompt": "sneakers softly landing on grass",
#         "description": "A soft thud of sneakers landing on grass.",
#         "start_timestamp": 4238,
#         "end_timestamp": 4638
#     },
#     {
#         "id": "5_A_light_rustle_of_denim_clothing_from_the_landing_motion.",
#         "prompt": "denim fabric softly rustling with gentle movement",
#         "description": "A light rustle of denim clothing from the landing motion.",
#         "start_timestamp": 4238,
#         "end_timestamp": 5012
#     }
# ]
    total_duration_ms = 23000
    
    audio_metadata = [
  {
      "id": "0",
      "prompt": "car dirving through",
      "description": "car dirving through.",
      "start_timestamp": "00:03",
      "end_timestamp": "00:05"
  },
  {
    "id": "1",
    "prompt": "car tires on the sand",
    "description": "car tires on the sand.",
    "start_timestamp": "00:05",
    "end_timestamp": "00:06"
  },
  {
    "id": "2",
    "prompt": "engine sound",
    "description": "engine sound.",
    "start_timestamp": "00:05",
    "end_timestamp": "00:06"
  },
  {
    "id": "3",
    "prompt": "car dirving away",
    "description": "car dirving away.",
    "start_timestamp": "00:06",
    "end_timestamp": "00:08"
  },
  {
    "id": "13",
    "prompt": "The mechanical sound of the cannon being elevated",
    "description": "The mechanical sound of the cannon being elevated.",
    "start_timestamp": "00:08",
    "end_timestamp": "00:12"
  },
  {
    "id": "4",
    "prompt": "The sound of loading a shell into a cannon",
    "description": "The sound of loading a shell into a cannon.",
    "start_timestamp": "00:15",
    "end_timestamp": "00:16"
  },
  {
    "id": "5",
    "prompt": "The sound of loading a shell into a cannon",
    "description": "The sound of loading a shell into a cannon.",
    "start_timestamp": "00:16",
    "end_timestamp": "00:17"
  },
  {
    "id": "6",
    "prompt": "The sound of cannon firing",
    "description": "The sound of cannon firing.",
    "start_timestamp": "00:17",
    "end_timestamp": "00:18"
  },
  {
    "id": "7",
    "prompt": "The sound of cannon firing",
    "description": "The sound of cannon firing.",
    "start_timestamp": "00:18",
    "end_timestamp": "00:19"
  },
  {
    "id": "8",
    "prompt": "The sound of cannon firing",
    "description": "The sound of cannon firing.",
    "start_timestamp": "00:20",
    "end_timestamp": "00:21"
  },
  {
    "id": "9",
    "prompt": "The sound of cannon firing",
    "description": "The sound of cannon firing.",
    "start_timestamp": "00:21",
    "end_timestamp": "00:22"
  },
  {
    "id": "10",
    "prompt": "The sound of cannon firing",
    "description": "The sound of cannon firing.",
    "start_timestamp": "00:22",
    "end_timestamp": "00:23"
  }
]
    sounding_events_metdata = convert_to_metadata(audio_metadata)

    video_path = "/home/jovyan/workspace/dev_vt2a/new_test_data/test3.mp4"
    output_dir = "/home/jovyan/workspace/results/take2/test_video_3/4"

    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    AudioGeneration(sounding_events_metdata, output_dir, cache_dir, total_duration_ms, raw_video_path=video_path)
