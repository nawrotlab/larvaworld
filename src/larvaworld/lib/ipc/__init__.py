"""
Interface between a larvaworld simulation and a remote model
"""

from __future__ import annotations

from typing import Any

__displayname__ = "Client-Server"

__all__: list[str] = ["Client", "Server", "BrianInterfaceMessage"]

from .ipc import Client, Server, Message


class BrianInterfaceMessage(Message):
    """A message exchanged with a remotely simulated Brian model."""

    def __init__(self, sim_id: str, model_id: str, step: int, **params: Any) -> None:
        """Build a message to the remote Brian model.

        Args:
            sim_id: The simulation identifier.
            model_instance_id: The agent's model instance.
            **kwargs: The message payload.
        """
        self.sim_id: str = sim_id
        self.model_id: str = model_id
        self.step: int = step
        self.params: dict[str, Any] = params

    def _get_args(self) -> tuple[list[Any], dict[str, Any]]:
        """Return the arguments this message serializes."""
        return [self.sim_id, self.model_id, self.step], self.params

    def with_params(self, **params: Any) -> "BrianInterfaceMessage":
        """Return a copy carrying the given parameters.

        Args:
            **kwargs: The parameters to attach.

        Returns:
            The updated message.
        """
        return BrianInterfaceMessage(self.sim_id, self.model_id, self.step, **params)

    def param(self, key: str) -> Any | None:
        """Read one parameter from the message.

        Args:
            key: The parameter name.
            default: The value returned when it is absent.

        Returns:
            The parameter value.
        """
        try:
            return self.params[key]
        except Exception:
            return None
