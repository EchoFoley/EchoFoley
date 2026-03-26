from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from action_pool import ActionRegistry
from action_state import Event, EventDescription, EventProperties, ActionState
from text_llm import t2t_generate

from video_understanding import (
    convert_event_timestamps,
    convert_timestamp_to_1fps,
    end_timestamp_localization_parallel,
    preprocess_video_to_1fps,
    start_timestamp_localization,
    video_overview,
)


def _ensure_dirs(state: ActionState) -> Tuple[str, str]:
    intermediate_dir = os.path.join(state.cache_dir or state.output_dir or "", "intermediate_results")
    video_cache_dir = os.path.join(state.cache_dir or state.output_dir or "", "video_cache")
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(video_cache_dir, exist_ok=True)
    return intermediate_dir, video_cache_dir


def _event_plan_from_slow_end_metadata(sounding_events_meta_data_ms: List[Dict[str, Any]]) -> List[Event]:
    events: List[Event] = []
    for idx, evt in enumerate(sounding_events_meta_data_ms):
        start_ms = evt["start_timestamp"]
        end_ms = evt["end_timestamp"]
        desc = str(evt.get("description", "unknown"))
        # Placeholder decomposition; verification action will refine if LLM is enabled.
        subject = "unknown"
        action = desc
        obj = ""
        events.append(
            Event(
                id=f"evt_{idx}",
                t=(float(start_ms) / 1000.0, float(end_ms) / 1000.0),
                d=EventDescription(subject=subject, action=action, object=obj),
                p=EventProperties(),
            )
        )
    return events


def _parse_json_maybe(text: str) -> Any:
    # Best-effort: try direct JSON first, then extract first JSON object/array region.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt extraction of outermost JSON.
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [p for p in [start_obj, start_arr] if p != -1]
    if not starts:
        raise ValueError("No JSON object/array found in model output")
    start = min(starts)
    # Heuristic: find last closing brace/bracket.
    end_obj = text.rfind("}")
    end_arr = text.rfind("]")
    ends = [p for p in [end_obj, end_arr] if p != -1]
    end = max(ends) if ends else None
    if end is None or end <= start:
        raise ValueError("Could not isolate JSON region")

    candidate = text[start : end + 1]
    return json.loads(candidate)


def register_reasoning_actions(registry: ActionRegistry) -> None:
    @registry.register(name="video_overview", category="reasoning", description="Infer timestamped sound events from video.")
    def video_overview_action(state: ActionState) -> ActionState:
        if not state.video_path:
            raise ValueError("state.video_path is required for video overview")

        response, matches = video_overview(state.video_path)
        state.visual["video_overview_response"] = response
        state.visual["overview_matches"] = matches or []
        return state

    @registry.register(
        name="start_timestamp_localization",
        category="reasoning",
        description="Refine event start times using a 1fps resample + fast video LLM.",
    )
    def start_timestamp_localization_action(state: ActionState) -> ActionState:
        if not state.video_path:
            raise ValueError("state.video_path is required for start timestamp localization")

        intermediate_dir, video_cache_dir = _ensure_dirs(state)

        overview_matches: List[Tuple[str, str]] = state.visual.get("overview_matches") or []
        if not overview_matches:
            raise ValueError("No overview matches available; run 'video_overview' first.")

        video_fps, video_duration_ms, slowed_video_path = preprocess_video_to_1fps(state.video_path, video_cache_dir)
        state.visual["video_fps"] = float(video_fps)
        state.visual["video_duration_ms"] = float(video_duration_ms)
        state.video_duration_s = float(video_duration_ms) / 1000.0
        state.visual["slowed_video_path"] = slowed_video_path

        formatted_overview = ""
        for timestamp, description in overview_matches:
            converted = convert_timestamp_to_1fps(timestamp, float(video_fps))
            formatted_overview += f"[{converted}] - {description}\n"

        start_resp, start_matches = start_timestamp_localization(slowed_video_path, formatted_overview)
        state.visual["start_timestamp_localization_response"] = start_resp
        state.visual["start_timestamp_matches"] = start_matches or []

        with open(os.path.join(intermediate_dir, "start_timestamp_localization_response.txt"), "w") as f:
            f.write(start_resp or "")
        with open(os.path.join(intermediate_dir, "start_timestamp_localization.json"), "w") as f:
            json.dump(start_matches or [], f, indent=2)

        return state

    @registry.register(
        name="end_timestamp_localization",
        category="reasoning",
        description="Refine event end times using a slower/stronger video LLM.",
    )
    def end_timestamp_localization_action(state: ActionState) -> ActionState:
        intermediate_dir, video_cache_dir = _ensure_dirs(state)

        start_timestamps: List[Tuple[str, str]] = state.visual.get("start_timestamp_matches") or []
        slowed_video_path: Optional[str] = state.visual.get("slowed_video_path")
        if not slowed_video_path:
            raise ValueError("Missing slowed_video_path; run 'start_timestamp_localization' first.")
        if not start_timestamps:
            raise ValueError("No start timestamps; run 'start_timestamp_localization' first.")

        end_timestamps, end_timestamp_responses = end_timestamp_localization_parallel(
            slowed_video_path,
            video_cache_dir,
            start_timestamps,
            intermediate_dir,
        )

        state.visual["end_timestamp_tuples_1fps"] = end_timestamps

        with open(os.path.join(intermediate_dir, "end_timestamp_localization.json"), "w") as f:
            json.dump(end_timestamps, f, indent=2)

        with open(os.path.join(intermediate_dir, "end_timestamp_localization_response.txt"), "w") as f:
            for response in end_timestamp_responses:
                f.write((response or "") + "\n")

        # Convert 1fps MM:SS timestamps to original-ms timestamps and keep as meta-data.
        original_fps = float(state.visual.get("video_fps"))
        sounding_events_meta_data: List[Dict[str, Any]] = []
        for start_timestamp, end_timestamp, description in end_timestamps:
            if end_timestamp is not None:
                sounding_events_meta_data.append(
                    {"start_timestamp": start_timestamp, "end_timestamp": end_timestamp, "description": description}
                )

        sounding_events_meta_data_ms = convert_event_timestamps(sounding_events_meta_data, original_fps)
        state.visual["sounding_events_meta_data_ms"] = sounding_events_meta_data_ms

        return state

    @registry.register(
        name="slow_fast_fusion",
        category="reasoning",
        description="Fuse fast and slow event timelines into a consistent merged timeline.",
    )
    def slow_fast_fusion_action(state: ActionState) -> ActionState:
        # We already have precise start/end from localization; fusion is mainly to remove duplicates.
        slow_events_ms: List[Dict[str, Any]] = state.visual.get("sounding_events_meta_data_ms") or []
        if not slow_events_ms:
            raise ValueError("Missing sounding_events_meta_data_ms; run end localization first.")

        slow_view_events = [
            {"label": str(e.get("description", "unknown")), "t_start": float(e["start_timestamp"]) / 1000.0, "t_end": float(e["end_timestamp"]) / 1000.0}
            for e in slow_events_ms
        ]

        # Fast view approximation: bound by next event start.
        slow_view_events_sorted = sorted(slow_view_events, key=lambda x: x["t_start"])
        fast_view_events: List[Dict[str, Any]] = []
        for i, e in enumerate(slow_view_events_sorted):
            t_start = e["t_start"]
            if i + 1 < len(slow_view_events_sorted):
                approx_end = min(e["t_end"], max(t_start + 0.05, slow_view_events_sorted[i + 1]["t_start"] - 0.02))
            else:
                approx_end = e["t_end"]
            fast_view_events.append({"label": e["label"], "t_start": t_start, "t_end": float(max(t_start + 0.05, approx_end))})

        # LLM fusion (best-effort; if it fails, we fall back to the slow/localized events).
        use_llm = bool(state.visual.get("use_fusion_llm", True))
        if not use_llm:
            merged = slow_view_events_sorted
        else:
            fusion_prompt = f"""SYSTEM ROLE:
You are a Temporal Fusion Expert. Your job is to merge two event streams into one consistent timeline.

INPUT:
FAST_VIEW_EVENTS:
{json.dumps(fast_view_events)}

SLOW_VIEW_EVENTS:
{json.dumps(slow_view_events_sorted)}

TASK:
1. Merge events with similar semantics from both lists.
2. Refine event timing using SLOW_VIEW when available.
3. Assign timestamps (t_start, t_end) in seconds.
4. Remove duplicates and ensure chronological ordering.

OUTPUT FORMAT (STRICT):
Return valid JSON only with this exact shape:
{{"MERGED_EVENTS": [{{"label": "...", "t_start": x.xx, "t_end": y.yy}}]}}
"""
            try:
                fusion_text = t2t_generate(fusion_prompt, sleep_s=0)
                fusion_json = _parse_json_maybe(fusion_text)
                merged = fusion_json.get("MERGED_EVENTS", [])
            except Exception:
                merged = slow_view_events_sorted

        merged = sorted(merged, key=lambda x: float(x["t_start"]))
        state.visual["merged_events"] = merged
        return state

    @registry.register(
        name="verification_to_event_plan",
        category="reasoning",
        description="Convert merged events into symbolic (t,d,p) event plan.",
    )
    def verification_action(state: ActionState) -> ActionState:
        merged_events = state.visual.get("merged_events") or []
        if not merged_events:
            raise ValueError("Missing merged_events; run slow_fast_fusion first.")

        verification_prompt = f"""SYSTEM ROLE:
You are the Sounding Event Structuring Agent. You convert events
into symbolic representations for controllable audio generation.

INPUT:
MERGED_EVENTS:
{json.dumps(merged_events)}

TASK:
For each event, construct e = (t, d, p):
 - t = (t_start, t_end) in seconds
 - d = {{subject, action, object}}
 - p = {{pitch, volume, intensity, spatial}}

Rules:
- Infer d from visual cues only (use the event label as the only hint).
- Use DEFAULT for p attributes if uncertain.
- All fields must exist.

OUTPUT FORMAT (STRICT):
Return valid JSON only:
{{"EVENT_PLAN": [
  {{"t": [x.xx, y.yy], "d": {{"subject": "...", "action": "...", "object": "..."}}, "p": {{"pitch": "...", "volume": "...", "intensity": "...", "spatial": "..."}}}}
]}}
"""

        try:
            verification_text = t2t_generate(verification_prompt, sleep_s=0)
            verification_json = _parse_json_maybe(verification_text)
            event_plan_raw = verification_json.get("EVENT_PLAN", [])
        except Exception:
            event_plan_raw = []
        # Assign ids deterministically; keep LLM output fields as source of truth.
        events: List[Event] = []
        for idx, e in enumerate(event_plan_raw):
            t = e.get("t")
            t_start, t_end = float(t[0]), float(t[1])
            d = e.get("d") or {}
            p = e.get("p") or {}
            events.append(
                Event(
                    id=f"evt_{idx}",
                    t=(t_start, t_end),
                    d=EventDescription(
                        subject=str(d.get("subject", "unknown")),
                        action=str(d.get("action", "unknown")),
                        object=str(d.get("object", "")),
                    ),
                    p=EventProperties(
                        pitch=str(p.get("pitch", "DEFAULT")),
                        volume=str(p.get("volume", "DEFAULT")),
                        intensity=str(p.get("intensity", "DEFAULT")),
                        spatial=str(p.get("spatial", "DEFAULT")),
                    ),
                )
            )
        state.event_plan = events

        # Fallback if the LLM output is empty: keep placeholders derived from localized events.
        if not state.event_plan:
            state.event_plan = _event_plan_from_slow_end_metadata(state.visual.get("sounding_events_meta_data_ms") or [])

        return state

