from __future__ import annotations

import json
from typing import Any, Dict, List

from action_pool import ActionRegistry
from action_state import ActionState, Event, EventDescription, EventProperties
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

    candidate = text[start : end + 1]
    return json.loads(candidate)


def _validate_and_normalize_event_plan(state: ActionState, events_raw: List[Dict[str, Any]]) -> List[Event]:
    events: List[Event] = []
    for idx, e in enumerate(events_raw):
        t = e.get("t")
        if not isinstance(t, (list, tuple)) or len(t) != 2:
            raise ValueError("Each event must include t=[t_start,t_end]")
        t_start = float(t[0])
        t_end = float(t[1])
        if t_end <= t_start:
            raise ValueError("Each event must satisfy t_end > t_start")

        d = e.get("d") or {}
        p = e.get("p") or {}
        events.append(
            Event(
                id=str(e.get("id") or f"evt_{idx}"),
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

    # Sort for chronological consistency.
    events = sorted(events, key=lambda ev: ev.t[0])

    # Optional: enforce within video duration (soft constraint).
    if state.video_duration_s is not None:
        max_t = float(state.video_duration_s) + 0.5
        for ev in events:
            if ev.t[0] > max_t:
                raise ValueError("Event start time exceeds video duration")
    return events


def register_sound_design_actions(registry: ActionRegistry) -> None:
    @registry.register(
        name="sound_design_edit_plan",
        category="sound_design",
        description="Update event plan using user instruction (add/delete/modify events).",
    )
    def sound_design_edit_plan(state: ActionState) -> ActionState:
        current_plan_dicts = state.event_plan_to_dicts()

        # Editing Prompt (adapted to JSON-only output for robust parsing).
        editing_prompt = f"""SYSTEM ROLE:
You are the Sound Design Controller. You update an existing symbolic
event plan based on user instructions.

INPUT:
USER_INSTRUCTION:
{state.user_instruction}

CURRENT_EVENT_PLAN:
{json.dumps(current_plan_dicts)}

TASK:
1. Identify all referenced events.
2. Apply required edits using ONLY:
   - ADD_EVENT
   - DELETE_EVENT
   - MODIFY_DESCRIPTION
   - MODIFY_TIME
   - MODIFY_PROPERTIES
3. Validate chronological ordering and value ranges.

OUTPUT FORMAT (STRICT):
Return valid JSON only:
{{"EVENT_PLAN": [
  {{"t":[x.xx,y.yy], "d":{{"subject":"...","action":"...","object":"..."}}, "p":{{"pitch":"...","volume":"...","intensity":"...","spatial":"..."}}}}
]}}
"""

        try:
            updated_text = t2t_generate(editing_prompt, sleep_s=0)
            updated_json = _parse_json_maybe(updated_text)
            updated_plan_raw = updated_json.get("EVENT_PLAN", [])
            if not updated_plan_raw:
                raise ValueError("Empty EVENT_PLAN from sound-design editor")

            state.event_plan = _validate_and_normalize_event_plan(state, updated_plan_raw)
        except Exception as e:
            # NO_OP: sound design is optional; preserve current plan if editing fails.
            state.debug.setdefault("sound_design_errors", []).append({"error": repr(e)})
        return state

