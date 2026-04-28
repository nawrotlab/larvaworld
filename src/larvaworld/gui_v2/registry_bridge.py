from __future__ import annotations

from dataclasses import dataclass

from larvaworld.portal.landing_registry import ITEMS, LANES, QUICK_START_MODES
from larvaworld.portal.registry_logic import compute_badges, validate_registry
from larvaworld.portal.registry_types import LandingItem, QuickStartModeSpec


@dataclass(frozen=True)
class GuiEntry:
    entry_id: str
    title: str
    subtitle: str
    lane_title: str
    lane_key: str
    level: str
    status: str
    kind: str
    badges: tuple[str, ...]
    cta: str
    panel_app_id: str | None
    docs_url: str | None


@dataclass(frozen=True)
class GuiQuickStartMode:
    mode_id: str
    title: str
    color: str
    entries: tuple[GuiEntry, ...]


@dataclass(frozen=True)
class GuiLane:
    lane_id: str
    title: str
    entries: tuple[GuiEntry, ...]


@dataclass(frozen=True)
class GuiNavigationModel:
    quick_start_modes: tuple[GuiQuickStartMode, ...]
    lanes: tuple[GuiLane, ...]
    entry_index: dict[str, GuiEntry]


def _build_entry(item: LandingItem) -> GuiEntry:
    lane_title = next(
        (lane.title for lane in LANES if lane.lane == item.lane), item.lane
    )
    learn_more = item.learn_more.docs_url if item.learn_more else None
    return GuiEntry(
        entry_id=item.id,
        title=item.title,
        subtitle=item.subtitle,
        lane_title=lane_title,
        lane_key=item.lane,
        level=item.level,
        status=item.status,
        kind=item.kind,
        badges=tuple(compute_badges(item)),
        cta=item.cta,
        panel_app_id=item.panel_app_id,
        docs_url=learn_more,
    )


def _build_quick_start_mode(
    mode: QuickStartModeSpec, entry_index: dict[str, GuiEntry]
) -> GuiQuickStartMode:
    return GuiQuickStartMode(
        mode_id=mode.mode_id,
        title=mode.title,
        color=mode.color,
        entries=tuple(entry_index[item_id] for item_id in mode.item_ids),
    )


def build_navigation_model() -> GuiNavigationModel:
    validate_registry(strict=False)

    entry_index = {item_id: _build_entry(item) for item_id, item in ITEMS.items()}
    quick_start_modes = tuple(
        _build_quick_start_mode(mode, entry_index) for mode in QUICK_START_MODES
    )
    lanes = tuple(
        GuiLane(
            lane_id=lane.lane,
            title=lane.title,
            entries=tuple(entry_index[item_id] for item_id in lane.item_ids),
        )
        for lane in LANES
    )

    return GuiNavigationModel(
        quick_start_modes=quick_start_modes,
        lanes=lanes,
        entry_index=entry_index,
    )
