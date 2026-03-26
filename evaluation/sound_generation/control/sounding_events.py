from dataclasses import dataclass, field
from __future__ import annotations
from typing import Any, Dict, MutableMapping, Optional

@dataclass
class SoundingEvent:
    id: str
    start_timestamp: int
    end_timestamp: int
    description: str = ""
    volume: Optional[Any] = None
    attributes: MutableMapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_timestamp - self.start_timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "description": self.description,
            "volume": self.volume,
            "attributes": dict(self.attributes),
        }

__all__ = ["SoundingEvent"]

