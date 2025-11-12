"""
The larvaworld Command Line Interface (CLI)
"""

from __future__ import annotations

__displayname__ = "CLI"

__all__: list[str] = ["SimModeParser"]

from .argparser import SimModeParser
