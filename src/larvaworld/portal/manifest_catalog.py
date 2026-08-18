from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import panel as pn

from larvaworld.lib.sim.manifest import (
    ManifestCatalogRecord,
    discover_run_manifests,
    load_run_manifest,
    rerun_from_manifest,
)


class ManifestCatalogController:
    """Small reusable manifest browser for simulation Portal apps."""

    def __init__(self, *, modes: Iterable[str], title: str = "Run Manifests") -> None:
        self.modes = tuple(modes)
        self.title = title
        self.records: list[ManifestCatalogRecord] = []
        self._record_by_path: dict[str, ManifestCatalogRecord] = {}
        self.manifest_select = pn.widgets.Select(name="Manifest", options={})
        self.refresh_button = pn.widgets.Button(name="Refresh", button_type="default")
        self.inspect_button = pn.widgets.Button(name="Inspect", button_type="primary")
        self.reproducibility = pn.widgets.Select(
            name="Reproducibility",
            options={"Strict": "strict", "Parameters only": "parameters"},
            value="strict",
        )
        self.with_media = pn.widgets.Checkbox(name="Repeat media", value=False)
        self.rerun_button = pn.widgets.Button(name="Rerun", button_type="success")
        self.summary = pn.pane.HTML(margin=0)
        self.inspector = pn.pane.JSON(
            object={}, depth=4, sizing_mode="stretch_width", margin=0
        )
        self.refresh_button.on_click(self._refresh)
        self.inspect_button.on_click(self._inspect)
        self.rerun_button.on_click(self._rerun)
        self.manifest_select.param.watch(self._inspect, "value")
        self.refresh()

    def refresh(self) -> None:
        self.records = discover_run_manifests(modes=self.modes)
        valid = [
            record
            for record in self.records
            if record.valid and record.manifest_path is not None
        ]
        self._record_by_path = {
            str(record.manifest_path): record
            for record in valid
            if record.manifest_path
        }
        options = {
            (
                f"{record.workspace_name} / {record.run_id or record.manifest_id} "
                f"[{record.status}]"
            ): str(record.manifest_path)
            for record in valid
        }
        previous = self.manifest_select.value
        self.manifest_select.options = options
        values = list(options.values())
        self.manifest_select.value = (
            previous if previous in values else (values[0] if values else None)
        )
        invalid = len([record for record in self.records if not record.valid])
        unavailable = len([record for record in self.records if not record.available])
        self.summary.object = (
            f"<p><strong>{len(valid)}</strong> valid manifest(s); "
            f"<strong>{invalid}</strong> invalid; "
            f"<strong>{unavailable}</strong> unavailable workspace(s).</p>"
        )
        self.inspect_button.disabled = not valid
        self.rerun_button.disabled = not valid
        self._inspect()

    def _refresh(self, *_events: object) -> None:
        self.refresh()

    def _inspect(self, *_events: object) -> None:
        raw_path = self.manifest_select.value
        if not raw_path:
            self.inspector.object = {}
            return
        try:
            self.inspector.object = load_run_manifest(Path(raw_path))
        except Exception as exc:
            self.summary.object = (
                f"<p>Manifest inspection failed: {escape(str(exc))}</p>"
            )

    def _rerun(self, *_events: object) -> None:
        raw_path = self.manifest_select.value
        if not raw_path:
            return
        self.rerun_button.disabled = True
        self.summary.object = "<p>Rerun in progress…</p>"
        try:
            result = rerun_from_manifest(
                raw_path,
                reproducibility=self.reproducibility.value,
                with_media=bool(self.with_media.value),
            )
        except Exception as exc:
            self.summary.object = f"<p>Rerun failed: {escape(str(exc))}</p>"
        else:
            self.summary.object = (
                "<p>Rerun completed. Manifest: "
                f"<code>{escape(str(result.manifest_path))}</code></p>"
            )
            self.refresh()
        finally:
            self.rerun_button.disabled = not bool(self._record_by_path)

    def view(self) -> pn.Card:
        actions = pn.Row(
            self.refresh_button,
            self.inspect_button,
            self.rerun_button,
            sizing_mode="stretch_width",
        )
        return pn.Card(
            pn.Column(
                self.summary,
                self.manifest_select,
                self.reproducibility,
                self.with_media,
                actions,
                self.inspector,
                sizing_mode="stretch_width",
                margin=0,
            ),
            title=self.title,
            collapsed=True,
            sizing_mode="stretch_width",
        )


__all__ = ["ManifestCatalogController"]
