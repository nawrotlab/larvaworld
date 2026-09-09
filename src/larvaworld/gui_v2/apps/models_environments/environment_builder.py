from __future__ import annotations

from larvaworld.gui_v2.apps.models_environments.env_builder_widget import (
    EnvBuilderWidget,
    build_environment_builder_widget,
)


def build_environment_builder_text(entry) -> tuple[str, str, list[tuple[str, str]]]:
    return (
        "Environment Builder",
        "PySide6-native desktop Environment Builder for Larvaworld.",
        [
            (
                "Scope",
                "Arena, source units, source groups, borders, and explicit workspace/registry load-save flows.",
            ),
            (
                "M1 boundaries",
                "No Body drawing, no Panel/Bokeh embedding, and no Qt-only schema.",
            ),
        ],
    )


__all__ = [
    "EnvBuilderWidget",
    "build_environment_builder_text",
    "build_environment_builder_widget",
]
