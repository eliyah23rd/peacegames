from __future__ import annotations

from typing import Any, Dict, List, Optional


class DefaultAgent:
    """Reference agent that takes no actions and preserves summary."""

    def __init__(self, max_summary_len: int = 512) -> None:
        self.summary = ""
        self.max_summary_len = max_summary_len

    def act(self, agent_input: dict) -> dict:
        action = {
            "purchase_mils": 0,
            "attacks": {},
            "cede_territories": {},
            "money_grants": {},
            "messages": {},
            "summary": self.summary[: self.max_summary_len],
        }
        self.summary = action["summary"]
        return action


class ScriptedAgent:
    """Deterministic agent that replays a fixed list of actions."""

    def __init__(
        self,
        actions: Optional[List[Dict[str, Any]]] = None,
        *,
        default_action: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._actions = list(actions or [])
        self._default_action = default_action or {}
        self._turn = 0

    def act(self, agent_input: dict) -> dict:
        if self._turn < len(self._actions):
            action = self._actions[self._turn]
        else:
            action = self._default_action
        self._turn += 1
        return action
