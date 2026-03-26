from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from action_state import ActionState


ActionFn = Callable[[ActionState], ActionState]


@dataclass(frozen=True)
class RegisteredAction:
    name: str
    category: str
    fn: ActionFn
    description: str = ""


class ActionRegistry:
    """
    Minimal action-pool registry: maps action names to callables operating on ActionState.
    """

    def __init__(self) -> None:
        self._actions: Dict[str, RegisteredAction] = {}

    def register(
        self,
        *,
        name: str,
        category: str,
        description: str = "",
    ) -> Callable[[ActionFn], ActionFn]:
        def decorator(fn: ActionFn) -> ActionFn:
            if name in self._actions:
                raise ValueError(f"Action '{name}' already registered")
            self._actions[name] = RegisteredAction(
                name=name, category=category, fn=fn, description=description
            )
            return fn

        return decorator

    def get(self, name: str) -> RegisteredAction:
        if name not in self._actions:
            raise KeyError(f"Unknown action '{name}'")
        return self._actions[name]

    def run(self, name: str, state: ActionState) -> ActionState:
        action = self.get(name)
        return action.fn(state)

    def list_actions(self, category: Optional[str] = None) -> Dict[str, RegisteredAction]:
        if category is None:
            return dict(self._actions)
        return {k: v for k, v in self._actions.items() if v.category == category}

