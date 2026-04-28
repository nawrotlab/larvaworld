from __future__ import annotations

import warnings

from larvaworld.portal.landing_registry import (
    ITEMS,
    LANES,
    PINNED_QUICK_START,
    QUICK_START_DEFAULT_MODE,
    QUICK_START_MODES,
)
from larvaworld.portal.registry_types import LandingItem, PrimaryAction


def validate_registry(*, strict: bool = True) -> None:
    def _err(msg: str) -> None:
        raise ValueError(f"[larvaworld.portal] {msg}")

    referenced: list[str] = []
    referenced.extend(PINNED_QUICK_START)
    for mode in QUICK_START_MODES:
        referenced.extend(mode.item_ids)
    for lane in LANES:
        referenced.extend(lane.item_ids)

    missing = [item_id for item_id in referenced if item_id not in ITEMS]
    if missing:
        _err(f"Missing item definitions for: {missing}")

    if len(PINNED_QUICK_START) != len(set(PINNED_QUICK_START)):
        _err("Duplicate IDs in PINNED_QUICK_START")

    mode_ids = [mode.mode_id for mode in QUICK_START_MODES]
    if len(mode_ids) != len(set(mode_ids)):
        _err("Duplicate quick-start mode ids")
    if QUICK_START_DEFAULT_MODE not in set(mode_ids):
        _err(f"Unknown QUICK_START_DEFAULT_MODE='{QUICK_START_DEFAULT_MODE}'")

    for mode in QUICK_START_MODES:
        if len(mode.item_ids) != len(set(mode.item_ids)):
            _err(f"Duplicate IDs in quick-start mode '{mode.mode_id}'")
        if strict and len(mode.item_ids) != 3:
            _err(f"Quick-start mode '{mode.mode_id}' must include exactly 3 items")

    hidden_in_refs = [
        item_id for item_id in referenced if ITEMS[item_id].status == "hidden"
    ]
    if hidden_in_refs:
        _err(f"Hidden items must not be referenced in lanes/pinned: {hidden_in_refs}")

    # Validate that each item listed in a lane matches the lane enum in the item itself.
    lane_item_counts: dict[str, int] = {}
    for lane in LANES:
        for item_id in lane.item_ids:
            lane_item_counts[item_id] = lane_item_counts.get(item_id, 0) + 1
            item = ITEMS[item_id]
            if item.lane != lane.lane:
                _err(
                    f"Item '{item_id}' has lane='{item.lane}' but is listed under lane='{lane.lane}'"
                )

    duplicate_lane_items = [item_id for item_id, n in lane_item_counts.items() if n > 1]
    if duplicate_lane_items:
        _err(f"Items must not appear in multiple lanes: {duplicate_lane_items}")

    for item in ITEMS.values():
        # Basic non-empty copy checks.
        if not item.title.strip():
            _err(f"Item '{item.id}' has empty title")
        if not item.subtitle.strip():
            _err(f"Item '{item.id}' has empty subtitle")
        if not item.cta.strip():
            _err(f"Item '{item.id}' has empty cta")

        # Kind-specific invariants.
        if item.kind == "panel_app":
            if not item.panel_app_id:
                _err(f"panel_app item '{item.id}' missing panel_app_id")
            if strict and item.panel_app_id != item.id:
                _err(
                    f"panel_app item '{item.id}' must have panel_app_id == id (got '{item.panel_app_id}')"
                )
        elif item.kind == "external_link":
            if not item.url:
                _err(f"external_link item '{item.id}' missing url")
        elif item.kind == "placeholder":
            if strict and item.url is not None:
                _err(f"placeholder item '{item.id}' must not define url")

        # Optional warning: planned items without any learn_more links.
        if item.status == "planned":
            has_links = bool(
                item.learn_more
                and (item.learn_more.issue_url or item.learn_more.docs_url)
            )
            if not has_links:
                warnings.warn(
                    f"[larvaworld.portal] planned item '{item.id}' has no learn_more links",
                    stacklevel=2,
                )


def resolve_target(item: LandingItem) -> str | None:
    if item.kind == "panel_app":
        return f"/{item.panel_app_id}"
    if item.kind == "external_link":
        return item.url
    return None


def compute_badges(item: LandingItem) -> list[str]:
    badges: list[str] = []

    if item.level == "core":
        badges.append("Core")
    elif item.level == "advanced":
        badges.append("Advanced")
    elif item.level == "demo":
        badges.append("Demo")

    if item.status == "planned":
        badges.append("Under construction")

    for extra in item.badges:
        if extra not in badges:
            badges.append(extra)

    return badges


def compute_primary_action(item: LandingItem) -> PrimaryAction:
    if item.status == "hidden":
        return PrimaryAction(label="Hidden", href=None, enabled=False)

    if item.learn_more and item.learn_more.docs_url:
        return PrimaryAction(
            label=item.cta, href=item.learn_more.docs_url, enabled=True
        )

    if item.kind == "panel_app":
        return PrimaryAction(label=item.cta, href=resolve_target(item), enabled=True)

    if item.kind == "external_link":
        return PrimaryAction(label=item.cta, href=item.url, enabled=True)

    # Placeholders / planned workflows.
    if item.learn_more and (item.learn_more.issue_url or item.learn_more.docs_url):
        href = item.learn_more.issue_url or item.learn_more.docs_url
        return PrimaryAction(label="Learn more", href=href, enabled=True)

    return PrimaryAction(label="Under construction", href=None, enabled=False)
