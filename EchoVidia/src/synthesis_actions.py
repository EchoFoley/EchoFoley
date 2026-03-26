from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from action_pool import ActionRegistry
from action_state import ActionState
from text_llm import t2t_generate


def _parse_json_maybe(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass

    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [p for p in [start_obj, start_arr] if p != -1]
    if not starts:
        raise ValueError("No JSON object/array found in model output")
    start = min(starts)
    end_obj = text.rfind("}")
    end_arr = text.rfind("]")
    ends = [p for p in [end_obj, end_arr] if p != -1]
    end = max(ends)
    return json.loads(text[start : end + 1])


def _volume_str_to_db(volume: str) -> float:
    v = str(volume).strip().upper()
    if v in {"", "DEFAULT", "NONE", "NULL"}:
        return 0.0
    # Try numeric parse (supports "+3", "-2.5", "3", "3.0 dB").
    m = re.search(r"[-+]?\d*\.?\d+", v)
    if m:
        return float(m.group(0))
    if "LOUD" in v:
        return 3.0
    if "SOFT" in v:
        return -6.0
    if "MEDIUM" in v:
        return 0.0
    return 0.0


def _build_default_generation_commands(state: ActionState) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    commands: List[Dict[str, Any]] = []
    for ev in state.event_plan:
        d = ev.d
        p = ev.p
        prompt_parts = [d.subject.strip(), d.action.strip(), (d.object or "").strip()]
        prompt = " ".join([pp for pp in prompt_parts if pp])
        # Add subtle property hints for controllability.
        if str(p.pitch).upper() != "DEFAULT":
            prompt = f"{prompt}, pitch: {p.pitch}"
        if str(p.intensity).upper() != "DEFAULT":
            prompt = f"{prompt}, intensity: {p.intensity}"

        commands.append(
            {
                "event_id": ev.id,
                "synthesis_prompt": prompt.strip()[:250],
                "t_start": float(ev.t[0]),
                "t_end": float(ev.t[1]),
                "properties": {
                    "volume": str(p.volume),
                    "pitch": str(p.pitch),
                    "intensity": str(p.intensity),
                    "spatial": str(p.spatial),
                },
            }
        )

    mixing_instructions = {
        "layering": "overlay",
        "crossfade_s": 0.05,
        "loudness_normalization": "peak_normalize_0.98",
        "global_effects": "",
    }
    return commands, mixing_instructions


def register_synthesis_actions(registry: ActionRegistry) -> None:
    @registry.register(
        name="generation_planner",
        category="synthesis",
        description="Plan per-event synthesis prompts and mixing parameters.",
    )
    def generation_planner(state: ActionState) -> ActionState:
        event_plan_dicts = state.event_plan_to_dicts()

        use_llm = bool(state.visual.get("use_generation_planner_llm", True))
        if not use_llm:
            commands, mixing_instructions = _build_default_generation_commands(state)
            state.audio["generation_commands"] = commands
            state.audio["mixing_instructions"] = mixing_instructions
            return state

        prompt = f"""SYSTEM ROLE:
You are the Audio Generation Planner. You convert symbolic events
into commands for the audio synthesis backend.

INPUT:
FINAL_EVENT_PLAN:
{json.dumps(event_plan_dicts)}

TASK:
For each event, produce a generator command block with:
- event_id
- synthesis_prompt (derived from d and p)
- t_start / t_end
- acoustic properties

Then produce mixing instructions specifying:
- layering
- crossfades
- loudness normalization
- global effects

OUTPUT FORMAT (STRICT):
Return valid JSON only with this exact shape:
{{
  "GENERATION_COMMANDS": [
    {{
      "event_id": "evt_0",
      "synthesis_prompt": "...",
      "t_start": 0.00,
      "t_end": 1.23,
      "properties": {{"volume":"...","pitch":"...","intensity":"...","spatial":"..."}}
    }}
  ],
  "MIXING_INSTRUCTIONS": {{"layering":"...","crossfade_s":0.05,"loudness_normalization":"...","global_effects":"..."}}
}}
No other text.
"""

        try:
            gen_text = t2t_generate(prompt, sleep_s=0)
            gen_json = _parse_json_maybe(gen_text)
            commands = gen_json.get("GENERATION_COMMANDS", [])
            mixing_instructions = gen_json.get("MIXING_INSTRUCTIONS", {})
        except Exception:
            commands, mixing_instructions = _build_default_generation_commands(state)

        if not commands:
            commands, mixing_instructions = _build_default_generation_commands(state)

        state.audio["generation_commands"] = commands
        state.audio["mixing_instructions"] = mixing_instructions
        return state

    @registry.register(
        name="generate_audio",
        category="synthesis",
        description="Generate audio wav files for each planned event.",
    )
    def generate_audio_action(state: ActionState) -> ActionState:
        generation_commands: List[Dict[str, Any]] = state.audio.get("generation_commands") or []
        if not generation_commands:
            raise ValueError("Missing generation_commands; run generation_planner first.")

        cache_dir = state.cache_dir or state.output_dir or "."
        audio_dir = os.path.join(cache_dir, "soundtracks")
        os.makedirs(audio_dir, exist_ok=True)

        # Import lazily to avoid loading stable-audio when only reasoning/sound-design is needed.
        from audio_generation import audio_generation as st_generate_audio

        clips: List[Dict[str, Any]] = []
        for cmd in generation_commands:
            event_id = str(cmd.get("event_id"))
            synthesis_prompt = str(cmd.get("synthesis_prompt", "")).strip()
            t_start = float(cmd.get("t_start"))
            t_end = float(cmd.get("t_end"))
            if t_end <= t_start:
                continue
            dur_s = t_end - t_start
            if dur_s < 0.05:
                continue

            duration_for_model = dur_s + 0.1  # small buffer to avoid cut-off
            output_path_no_ext = os.path.join(audio_dir, f"audio_{event_id}")

            # Generate wav segment.
            st_generate_audio(synthesis_prompt, duration_for_model, output_path_no_ext)

            clips.append(
                {
                    "event_id": event_id,
                    "path": f"{output_path_no_ext}.wav",
                    "start_s": t_start,
                    "end_s": t_end,
                    "duration_s": dur_s,
                    "properties": cmd.get("properties") or {},
                }
            )

        state.audio["event_audio_clips"] = clips
        return state

    @registry.register(
        name="tune_audio_volume",
        category="synthesis",
        description="Convert symbolic volume to a numeric gain for each audio clip.",
    )
    def tune_audio_volume_action(state: ActionState) -> ActionState:
        clips: List[Dict[str, Any]] = state.audio.get("event_audio_clips") or []
        for clip in clips:
            props = clip.get("properties") or {}
            clip["volume_db"] = _volume_str_to_db(props.get("volume", "DEFAULT"))
        state.audio["event_audio_clips"] = clips
        return state

    @registry.register(
        name="mix_audio_tracks",
        category="synthesis",
        description="Crossfade + overlay all event tracks into a final soundtrack.",
    )
    def mix_audio_tracks_action(state: ActionState) -> ActionState:
        clips: List[Dict[str, Any]] = state.audio.get("event_audio_clips") or []
        if not clips:
            raise ValueError("No generated audio clips found; run generate_audio first.")

        # Determine output length: use video duration if known, else use max end.
        if state.video_duration_s is not None:
            total_s = float(state.video_duration_s)
        else:
            total_s = max(float(c["end_s"]) for c in clips)

        crossfade_s = float((state.audio.get("mixing_instructions") or {}).get("crossfade_s", 0.05))

        from audio_generation import mix_audios_crossfade_normalized

        audio_dir = os.path.join(state.cache_dir or state.output_dir or ".", "soundtracks")
        mixed_wav_path = os.path.join(audio_dir, "mixed_audio.wav")
        mixed_mp3_path = os.path.join(audio_dir, "mixed_audio.mp3")

        mix_audios_crossfade_normalized(
            total_duration_s=total_s,
            event_clips=clips,
            output_wav_path=mixed_wav_path,
            output_mp3_path=mixed_mp3_path,
            crossfade_s=crossfade_s,
            peak_normalize=0.98,
        )

        state.audio["mixed_audio_wav_path"] = mixed_wav_path
        state.audio["mixed_audio_mp3_path"] = mixed_mp3_path
        return state

    @registry.register(
        name="render_video_with_audio",
        category="synthesis",
        description="Mux the mixed audio back into the original video.",
    )
    def render_video_with_audio_action(state: ActionState) -> ActionState:
        if not state.video_path:
            raise ValueError("state.video_path required for render_video_with_audio")

        mixed_audio_path = state.audio.get("mixed_audio_mp3_path") or state.audio.get("mixed_audio_wav_path")
        if not mixed_audio_path:
            raise ValueError("Missing mixed audio path")

        from moviepy import AudioFileClip, VideoFileClip

        output_dir = state.output_dir or "."
        final_video_path = os.path.join(output_dir, "final_video.mp4")

        video_clip = VideoFileClip(state.video_path)
        audio_clip = AudioFileClip(mixed_audio_path)
        video_clip.audio = audio_clip

        video_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac")

        state.audio["final_video_path"] = final_video_path
        return state

