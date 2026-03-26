from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EventDescription:
    subject: str
    action: str
    object: str = ""


@dataclass
class EventProperties:
    pitch: str = "DEFAULT"
    volume: str = "DEFAULT"
    intensity: str = "DEFAULT"
    spatial: str = "DEFAULT"


@dataclass
class Event:
    id: str
    t: Tuple[float, float]  # (t_start_sec, t_end_sec)
    d: EventDescription
    p: EventProperties = field(default_factory=EventProperties)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "t": [float(self.t[0]), float(self.t[1])],
            "d": {"subject": self.d.subject, "action": self.d.action, "object": self.d.object},
            "p": {
                "pitch": self.p.pitch,
                "volume": self.p.volume,
                "intensity": self.p.intensity,
                "spatial": self.p.spatial,
            },
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Event":
        if "id" not in d:
            raise ValueError("Event dict must include 'id'")
        if "t" not in d:
            raise ValueError("Event dict must include 't'")

        t = d["t"]
        if isinstance(t, (list, tuple)) and len(t) == 2:
            t_start, t_end = float(t[0]), float(t[1])
        else:
            raise ValueError("Event 't' must be a 2-element list/tuple of seconds")

        desc = d.get("d", {})
        props = d.get("p", {})
        return Event(
            id=str(d["id"]),
            t=(t_start, t_end),
            d=EventDescription(
                subject=str(desc.get("subject", "unknown")),
                action=str(desc.get("action", "unknown")),
                object=str(desc.get("object", "")),
            ),
            p=EventProperties(
                pitch=str(props.get("pitch", "DEFAULT")),
                volume=str(props.get("volume", "DEFAULT")),
                intensity=str(props.get("intensity", "DEFAULT")),
                spatial=str(props.get("spatial", "DEFAULT")),
            ),
        )


@dataclass
class ActionState:
    """
    Shared state object passed between all action-pool steps.

    Actions are expected to treat this as immutable-in-practice and return an updated instance
    (but most actions will mutate and return the same object for efficiency).
    """

    user_instruction: str
    video_path: Optional[str] = None
    video_duration_s: Optional[float] = None

    # IO locations used by actions to write intermediate artifacts / final outputs
    output_dir: Optional[str] = None
    cache_dir: Optional[str] = None

    # Visual latent state (resampled/cropped metadata, etc.)
    visual: Dict[str, Any] = field(default_factory=dict)

    # Symbolic sounding-event plan (event latent state)
    event_plan: List[Event] = field(default_factory=list)

    # Audio latent state (per-event wav paths, gains, and final mix metadata)
    audio: Dict[str, Any] = field(default_factory=dict)

    # Debug/trace data
    debug: Dict[str, Any] = field(default_factory=dict)

    def event_plan_to_dicts(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.event_plan]

    def set_event_plan_from_dicts(self, events: List[Dict[str, Any]]) -> None:
        self.event_plan = [Event.from_dict(e) for e in events]

