"""
Interface between a larvaworld simulation and a remote model
"""

from __future__ import annotations

from typing import Any

__displayname__ = "Client-Server"

__all__: list[str] = ["Client", "Server", "BrianInterfaceMessage"]

from .ipc import Client, Server, Message


class BrianInterfaceMessage(Message):
    def __init__(self, sim_id: str, model_id: str, step: int, **params: Any) -> None:
        self.sim_id: str = sim_id
        self.model_id: str = model_id
        self.step: int = step
        self.params: dict[str, Any] = params

    def _get_args(self) -> tuple[list[Any], dict[str, Any]]:
        return [self.sim_id, self.model_id, self.step], self.params

    def with_params(self, **params: Any) -> "BrianInterfaceMessage":
        return BrianInterfaceMessage(self.sim_id, self.model_id, self.step, **params)

    def param(self, key: str) -> Any | None:
        try:
            return self.params[key]
        except Exception:
            return None
