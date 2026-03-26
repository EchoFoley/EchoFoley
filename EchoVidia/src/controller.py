from __future__ import annotations

import copy
import os
from typing import List, Optional

from action_pool import ActionRegistry
from action_state import ActionState


class VideoLLMController:
    """
    Agentic multi-stage controller:
    1) reasoning: video -> initial symbolic event plan (t,d,p)
    2) sound design: iteratively edit event plan via user instruction
    3) synthesis: render event plan to audio and mix (global temporal consistency)
    """

    def __init__(
        self,
        *,
        registry: Optional[ActionRegistry] = None,
        max_sound_design_rounds: int = 2,
    ) -> None:
        self.registry = registry or ActionRegistry()
        self.max_sound_design_rounds = max_sound_design_rounds

    def run(
        self,
        *,
        video_path: str,
        user_instruction: str,
        output_dir: str,
    ) -> str:
        cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        state = ActionState(
            user_instruction=user_instruction,
            video_path=video_path,
            output_dir=output_dir,
            cache_dir=cache_dir,
        )

        # Phase 1: reasoning
        state = self._run_reasoning(state)

        # Phase 2: sound design edits (optional, can be NO_OP if instruction empty)
        state = self._run_sound_design(state)

        # Phase 3: synthesis
        state = self._run_synthesis(state)

        final_video_path = state.audio.get("final_video_path")
        if not final_video_path:
            raise RuntimeError("Synthesis completed but 'final_video_path' missing in state.audio")
        return final_video_path

    def _run_sequence(self, action_names: List[str], state: ActionState, *, rollback_on_error: bool) -> ActionState:
        for action_name in action_names:
            snapshot = copy.deepcopy(state)
            try:
                state = self.registry.run(action_name, state)
            except Exception as e:
                if rollback_on_error:
                    state = snapshot
                state.debug.setdefault("errors", []).append(
                    {"action": action_name, "error": repr(e)}
                )
                raise
        return state

    def _run_reasoning(self, state: ActionState) -> ActionState:
        action_names = [
            "video_overview",
            "start_timestamp_localization",
            "end_timestamp_localization",
            "slow_fast_fusion",
            "verification_to_event_plan",
        ]
        return self._run_sequence(action_names, state, rollback_on_error=True)

    def _run_sound_design(self, state: ActionState) -> ActionState:
        if not state.user_instruction or not state.user_instruction.strip():
            state.debug["sound_design"] = "NO_OP (empty user_instruction)"
            return state

        # Loop: apply the controller edit prompt and accept updated plan if valid.
        prev_signature = None
        for round_idx in range(self.max_sound_design_rounds):
            snapshot = copy.deepcopy(state)
            state = self.registry.run("sound_design_edit_plan", state)

            signature = repr([e.to_dict() for e in state.event_plan])
            state.debug.setdefault("sound_design_rounds", []).append(
                {"round": round_idx, "events": len(state.event_plan)}
            )
            if prev_signature is not None and signature == prev_signature:
                break
            prev_signature = signature

            # Basic validation: chronological ordering must hold.
            if not self._event_plan_is_temporally_valid(state.event_plan):
                state = snapshot
                state.debug.setdefault("sound_design_rollbacks", []).append(
                    {"round": round_idx, "reason": "temporal ordering invalid"}
                )
                continue
        return state

    def _event_plan_is_temporally_valid(self, event_plan) -> bool:
        # Must be strictly non-decreasing, and each event must be t_start < t_end.
        times = [(e.t[0], e.t[1]) for e in event_plan]
        for t_start, t_end in times:
            if t_end <= t_start:
                return False
        starts = [t[0] for t in times]
        if starts != sorted(starts):
            return False
        return True

    def _run_synthesis(self, state: ActionState) -> ActionState:
        action_names = [
            "generation_planner",
            "generate_audio",
            "tune_audio_volume",
            "mix_audio_tracks",
            "render_video_with_audio",
        ]
        return self._run_sequence(action_names, state, rollback_on_error=True)

