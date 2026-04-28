from __future__ import annotations

from larvaworld.gui_v2.apps.models_environments.environment_builder import (
    build_environment_builder_text,
    build_environment_builder_widget,
)
from larvaworld.gui_v2.registry_bridge import GuiEntry, GuiNavigationModel
from PySide6.QtWidgets import QWidget


def build_group_text(
    title: str, description: str, model: GuiNavigationModel
) -> tuple[str, str, list[tuple[str, str]]]:
    blocks = [
        (
            "Overview",
            description,
        ),
        (
            "M1 behavior",
            "Groups currently act as desktop navigation containers. Individual entries open "
            "in-shell detail pages or placeholders until embedded content is wired.",
        ),
        (
            "Shared-source note",
            f"The current shell uses {len(model.entry_index)} shared registry entries sourced through a "
            "temporary bridge. Extraction to a portal-neutral registry remains post-M1 work.",
        ),
    ]
    return title, "Group-level desktop shell view.", blocks


def build_entry_text(entry: GuiEntry) -> tuple[str, str, list[tuple[str, str]]]:
    badges = ", ".join(entry.badges) if entry.badges else "None"
    docs = entry.docs_url or "No documentation link registered."

    if entry.status == "ready":
        state_body = (
            "This entry is marked ready in the shared registry. The gui_v2 shell currently renders "
            "its metadata in-window while embedded local app integration is implemented incrementally."
        )
    else:
        state_body = (
            "This entry is not ready yet. The desktop shell exposes it as an in-window placeholder "
            "so the final GUI structure is already visible without sending the user to an external browser."
        )

    blocks = [
        (
            "Summary",
            entry.subtitle.replace("\n", " "),
        ),
        (
            "Metadata",
            f"Lane: {entry.lane_title}\n"
            f"Kind: {entry.kind}\n"
            f"Status: {entry.status}\n"
            f"Level: {entry.level}\n"
            f"CTA: {entry.cta}\n"
            f"Badges: {badges}",
        ),
        (
            "Desktop rendering state",
            state_body,
        ),
        (
            "Documentation",
            docs,
        ),
    ]

    return entry.title, "In-window app detail view.", blocks


def build_entry_widget(entry: GuiEntry) -> QWidget | None:
    if entry.entry_id == "wf.environment_builder":
        return build_environment_builder_widget(entry)
    return None
