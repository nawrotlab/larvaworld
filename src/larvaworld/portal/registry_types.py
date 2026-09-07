"""
Types describing the entries on the portal's landing page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["panel_app", "external_link", "placeholder"]
Status = Literal["ready", "planned", "hidden"]
Level = Literal["core", "advanced", "demo"]
Lane = Literal["simulate", "data", "models", "eval", "demos"]
QuickStartModeId = Literal["user", "modeler", "experimentalist"]


@dataclass(frozen=True)
class LearnMore:
    """A documentation link shown on a landing card."""

    issue_url: str | None = None
    docs_url: str | None = None


@dataclass(frozen=True)
class LandingItem:
    """One app as presented on the portal's landing page."""

    id: str
    kind: Kind
    status: Status
    lane: Lane
    level: Level

    title: str
    subtitle: str
    cta: str

    # For panel_app only: must match id in strict mode.
    panel_app_id: str | None = None

    # For external_link only.
    url: str | None = None

    learn_more: LearnMore | None = None
    prereq_hint: str | None = None
    preview_md: str | None = None

    badges: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LaneSpec:
    """A titled row grouping related landing cards."""

    title: str
    lane: Lane
    item_ids: list[str]
    collapsed_by_default: bool = False


@dataclass(frozen=True)
class QuickStartModeSpec:
    """One quick-start entry point offered by an app."""

    mode_id: QuickStartModeId
    title: str
    color: str
    item_ids: list[str]


@dataclass(frozen=True)
class PrimaryAction:
    """The main button a landing card offers."""

    label: str
    href: str | None
    enabled: bool
