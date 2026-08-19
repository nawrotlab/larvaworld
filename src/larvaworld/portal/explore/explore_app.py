"""Explore - the zero-configuration entry point to Larvaworld.

Three states in one page: a gallery of curated scenarios, a live run, and a
result view. The promise is one click and no decisions between arriving and
watching virtual larvae behave.

Everything here is display-only: runs are capped, nothing is written to the
workspace, and no simulation semantics are altered. Users who want control
graduate to the Single Experiment app via a deep link.
"""

from __future__ import annotations

import tempfile
from html import escape
from pathlib import Path
from typing import Any

import panel as pn

from larvaworld.portal.buttons import run_button
from larvaworld.portal.explore.scenarios import (
    CATEGORY_TITLES,
    SCENARIOS,
    Scenario,
    scenario_by_id,
    scenarios_by_category,
)
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header

EXPLORE_RAW_CSS = """
.lw-explore-intro {
  font-size: 14px;
  line-height: 1.5;
  margin: 0 0 6px 0;
}

.lw-explore-category-title {
  margin: 20px 0 8px 0;
  font-size: 17px;
  font-weight: 650;
}

.lw-explore-card {
  display: flex;
  flex-direction: column;
  border: 1px solid rgba(0,0,0,0.12);
  border-top: 3px solid #b5c2b0;
  border-radius: 14px;
  padding: 14px;
  background: rgba(255,255,255,0.96);
  box-shadow: 0 1px 8px rgba(0,0,0,0.05);
  height: 100%;
}

.lw-explore-card-title {
  font-size: 15px;
  font-weight: 650;
  margin: 0 0 6px 0;
}

.lw-explore-card-teaser {
  font-size: 13px;
  line-height: 1.4;
  color: rgba(0,0,0,0.72);
  margin: 0 0 8px 0;
  min-height: calc(1.4em * 2);
}

.lw-explore-card-meta {
  font-size: 11px;
  color: rgba(0,0,0,0.55);
  margin: 0 0 10px 0;
}

.lw-explore-stage-title {
  font-size: 20px;
  font-weight: 650;
  margin: 0 0 4px 0;
}

.lw-explore-stage-teaser {
  font-size: 14px;
  line-height: 1.5;
  color: rgba(0,0,0,0.75);
  margin: 0 0 10px 0;
}

.lw-explore-summary {
  font-size: 14px;
  line-height: 1.55;
  padding: 12px 16px;
  border-radius: 10px;
  border: 1px solid rgba(181,194,176,0.7);
  background: rgba(181,194,176,0.16);
}

.lw-explore-watchfor {
  font-size: 13px;
  line-height: 1.6;
  padding: 10px 14px;
  border-radius: 10px;
  border: 1px solid rgba(181,194,176,0.7);
  background: rgba(181,194,176,0.16);
}

.lw-explore-preview-placeholder {
  font-size: 14px;
  line-height: 1.5;
  padding: 16px;
  border-radius: 10px;
  border: 1px solid rgba(181,194,176,0.7);
  background: rgba(181,194,176,0.16);
}

.lw-explore-result-grid {
  align-items: flex-start;
  gap: 18px;
}

.lw-explore-playback {
  flex: 1 1 560px;
  min-width: 0;
}

.lw-explore-analysis {
  flex: 1 1 380px;
  min-width: 300px;
  box-sizing: border-box;
  padding: 14px;
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 12px;
  background: rgba(0,0,0,0.025);
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.lw-explore-analysis-title {
  font-size: 16px;
  font-weight: 650;
  margin: 0 0 3px 0;
}

.lw-explore-analysis-note {
  font-size: 12px;
  line-height: 1.4;
  color: rgba(0,0,0,0.65);
  margin: 0 0 10px 0;
}

.lw-explore-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 8px;
  margin-bottom: 10px;
}

.lw-explore-dataset-tables-title {
  font-size: 13px;
  font-weight: 650;
  margin: 4px 0 4px 0;
}

.lw-explore-metric {
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 8px;
  background: rgba(0,0,0,0.03);
  padding: 8px 10px;
}

.lw-explore-metric-label {
  font-size: 11px;
  color: rgba(0,0,0,0.62);
}

.lw-explore-metric-value {
  font-size: 16px;
  font-weight: 650;
  margin-top: 2px;
}

.lw-explore-analysis-figure-title {
  font-size: 13px;
  font-weight: 650;
  margin: 0;
}

.lw-explore-analysis-figure + .lw-explore-analysis-figure {
  margin-top: 0;
  padding-top: 0;
}

.lw-explore-analysis-warning {
  font-size: 12px;
  line-height: 1.4;
  margin-top: 8px;
  color: #7a4a00;
}

@media (max-width: 900px) {
  .lw-explore-result-grid {
    flex-direction: column;
  }

  .lw-explore-analysis {
    min-width: 0;
    width: 100%;
  }
}

.lw-explore-explanation {
  font-size: 14px;
  line-height: 1.6;
  padding: 12px 16px;
  border-radius: 10px;
  border: 1px solid rgba(0,0,0,0.10);
  background: rgba(0,0,0,0.03);
}

.lw-explore-literature {
  font-size: 12px;
  color: rgba(0,0,0,0.6);
  margin-top: 8px;
}

.lw-explore-error {
  font-size: 13px;
  line-height: 1.5;
  padding: 12px 16px;
  border-radius: 10px;
  border: 1px solid rgba(160,40,40,0.24);
  background: rgba(160,40,40,0.08);
}
""".strip()

#: Larva count is capped so individual animals stay distinguishable.
MAX_AGENTS = 20


def _estimated_seconds(scenario: Scenario, dt: float) -> float:
    return scenario.step_cap * dt


def _preview_dataset_tables_view(datasets: list[Any]) -> pn.viewable.Viewable:
    """Build Step/Endpoint actions for one selected in-memory preview dataset."""
    from larvaworld.portal.datasets import LarvaDatasetTablesWidget

    tables = LarvaDatasetTablesWidget(datasets[0])
    title = pn.pane.HTML(
        '<div class="lw-explore-dataset-tables-title">Inspect preview data</div>',
        margin=0,
    )
    if len(datasets) == 1:
        return pn.Column(title, tables.view(), sizing_mode="stretch_width", margin=0)

    options: dict[str, int] = {}
    for index, dataset in enumerate(datasets):
        config = getattr(dataset, "config", None)
        label = str(
            getattr(config, "id", None)
            or getattr(config, "group_id", None)
            or f"Dataset {index + 1}"
        )
        if label in options:
            label = f"{label} ({index + 1})"
        options[label] = index
    selector = pn.widgets.Select(
        name="Larva group",
        options=options,
        value=0,
        sizing_mode="stretch_width",
        margin=(0, 0, 4, 0),
    )
    selector.param.watch(
        lambda event: tables.set_dataset(datasets[event.new]),
        "value",
    )
    return pn.Column(
        title,
        selector,
        tables.view(),
        sizing_mode="stretch_width",
        margin=0,
    )


class _ExploreController:
    """Drives the gallery -> running -> result flow."""

    def __init__(self) -> None:
        self.body = pn.Column(sizing_mode="stretch_width", margin=0)
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._analysis_figures: list[Any] = []
        self.show_gallery()

    # ---- gallery -------------------------------------------------------

    def _scenario_card(self, scenario: Scenario) -> pn.viewable.Viewable:
        watch = run_button(name="Watch", width=110)
        watch.on_click(lambda _event, s=scenario: self.start_scenario(s))
        seconds = _estimated_seconds(scenario, dt=0.1)
        meta = (
            f"{scenario.n_agents} larva{'e' if scenario.n_agents != 1 else ''} "
            f"&middot; about {seconds:.0f}s of simulated time"
        )
        return pn.Column(
            pn.pane.HTML(
                f'<div class="lw-explore-card-title">{escape(scenario.title)}</div>'
                f'<div class="lw-explore-card-teaser">{escape(scenario.teaser)}</div>'
                f'<div class="lw-explore-card-meta">{meta}</div>',
                margin=0,
            ),
            watch,
            css_classes=["lw-explore-card"],
            margin=0,
            sizing_mode="stretch_width",
        )

    def show_gallery(self) -> None:
        self._release_preview_resources()
        sections: list[pn.viewable.Viewable] = [
            pn.pane.HTML(
                '<div class="lw-explore-intro">'
                "Each scenario below is a complete behavioral experiment that is "
                "ready to run. Pick one and watch what the virtual larvae do - "
                "there is nothing to configure."
                "</div>",
                margin=0,
            )
        ]
        for category, scenarios in scenarios_by_category().items():
            sections.append(
                pn.pane.HTML(
                    '<div class="lw-explore-category-title">'
                    f"{escape(CATEGORY_TITLES[category])}</div>",
                    margin=0,
                )
            )
            sections.append(
                pn.GridBox(
                    *[self._scenario_card(s) for s in scenarios],
                    ncols=3,
                    sizing_mode="stretch_width",
                )
            )
        self.body[:] = sections

    # ---- running -------------------------------------------------------

    @staticmethod
    def _scenario_summary(scenario: Scenario) -> pn.pane.HTML:
        """Return the stable explanation shown before a scenario runs."""
        teaser = " ".join(scenario.teaser.split())
        explanation = " ".join(scenario.explanation.split())
        literature = ""
        if scenario.literature:
            literature = (
                '<div class="lw-explore-literature">Paradigm after '
                f"{escape(scenario.literature)}.</div>"
            )
        return pn.pane.HTML(
            '<div class="lw-explore-summary">'
            "<strong>Experiment summary</strong><br/>"
            f"{escape(teaser)}<br/><br/>"
            f"{escape(explanation)}"
            f"{literature}"
            "</div>",
            margin=(10, 0, 0, 0),
        )

    def start_scenario(self, scenario: Scenario) -> None:
        """Show a static scenario canvas and offer an offline simulation preview."""
        # Imported lazily: these pull in the registry and the simulation stack,
        # which must not be paid for at portal startup.
        from larvaworld.lib import reg, util
        from larvaworld.portal.canvas_widgets.environment_canvas import (
            EnvironmentCanvas,
        )
        from larvaworld.portal.canvas_widgets.environment_mapping import (
            env_params_to_canvas_state,
        )

        self._release_preview_resources()
        try:
            parameters = util.AttrDict(reg.conf.Exp.expand(scenario.exp_id))
            parameters = self._apply_scenario_overrides(parameters, scenario)
            canvas = EnvironmentCanvas(editable=False, show_larva_groups=True)
            canvas.set_state(
                env_params_to_canvas_state(
                    parameters.env_params,
                    larva_groups=parameters.get("larva_groups", {}),
                    show_group_shapes=False,
                )
            )
        except Exception as exc:
            self._show_error(scenario, exc)
            return

        back = pn.widgets.Button(
            name="< All scenarios", button_type="default", width=140
        )
        back.on_click(lambda _event: self.show_gallery())
        generate_preview = run_button(name="Generate simulation preview", width=240)
        generate_preview.on_click(
            lambda _event: self._request_simulation_preview(scenario, parameters)
        )
        self.body[:] = [
            pn.Row(back, generate_preview, margin=0),
            pn.pane.HTML(
                f'<div class="lw-explore-stage-title">{escape(scenario.title)}</div>'
                f'<div class="lw-explore-stage-teaser">{escape(scenario.teaser)}</div>',
                margin=(10, 0, 0, 0),
            ),
            self._scenario_summary(scenario),
            canvas.view(),
        ]

    def _request_simulation_preview(self, scenario: Scenario, parameters: Any) -> None:
        """Render the preparation state before generating all preview frames."""
        back = pn.widgets.Button(
            name="< All scenarios", button_type="default", width=140
        )
        back.on_click(lambda _event: self.show_gallery())
        self.body[:] = [
            back,
            pn.pane.HTML(
                f'<div class="lw-explore-stage-title">{escape(scenario.title)}</div>',
                margin=(10, 0, 0, 0),
            ),
            self._scenario_summary(scenario),
            pn.pane.HTML(
                (
                    '<div class="lw-explore-preview-placeholder">'
                    "Generating simulation preview. The environment and agents are being initialized."
                    "</div>"
                ),
                margin=0,
            ),
        ]
        document = pn.state.curdoc
        if document is None:
            self._generate_simulation_preview(scenario, parameters)
            return
        document.add_next_tick_callback(
            lambda: self._generate_simulation_preview(scenario, parameters)
        )

    def _generate_simulation_preview(self, scenario: Scenario, parameters: Any) -> None:
        """Generate all frames before exposing the interactive canvas playback."""
        from larvaworld.portal.canvas_widgets.environment_canvas import (
            EnvironmentCanvas,
        )
        from larvaworld.portal.canvas_widgets.environment_mapping import (
            env_params_to_canvas_state,
        )
        from larvaworld.portal.explore.analysis import (
            PreviewAnalysisResult,
            build_preview_analysis,
            preview_enrichment,
        )
        from larvaworld.portal.simulation.preview_frames import generate_preview_frames
        from larvaworld.portal.simulation.run_playback import (
            build_bounded_launcher,
            finalize_preview_datasets,
        )

        launcher = None
        datasets: list[Any] = []
        analysis = PreviewAnalysisResult()
        try:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="lw_explore_")
            run_dir = Path(self._temp_dir.name) / scenario.id
            run_dir.mkdir(parents=True, exist_ok=True)
            launcher, note = build_bounded_launcher(
                scenario.exp_id,
                parameters,
                run_dir,
                step_cap=scenario.step_cap,
                analysis_enrichment=preview_enrichment(scenario.id),
            )
            frames = generate_preview_frames(
                launcher,
                preview_steps=scenario.step_cap,
            )
            if not frames:
                raise ValueError("No preview frames were generated.")
            try:
                datasets = finalize_preview_datasets(launcher)
            except Exception as exc:
                analysis.warnings.append(
                    f"Preview datasets were unavailable: {type(exc).__name__}: {exc}"
                )
            else:
                try:
                    analysis = build_preview_analysis(scenario.id, datasets)
                    self._analysis_figures = [item.figure for item in analysis.figures]
                except Exception as exc:
                    analysis.warnings.append(
                        f"Preview analysis was unavailable: {type(exc).__name__}: {exc}"
                    )
            canvas = EnvironmentCanvas(editable=False, show_larva_groups=False)
            canvas.set_state(
                env_params_to_canvas_state(
                    parameters.env_params,
                    larva_groups=None,
                    show_group_shapes=False,
                )
            )
            self.show_result(
                scenario,
                canvas,
                frames,
                dt=float(launcher.dt),
                note=note,
                analysis=analysis,
                datasets=datasets,
            )
        except Exception as exc:
            self._show_error(scenario, exc)
        finally:
            try:
                if launcher is not None and getattr(launcher, "screen_manager", None):
                    launcher.screen_manager.close()
            except Exception:
                pass

    @staticmethod
    def _apply_scenario_overrides(parameters: Any, scenario: Scenario) -> Any:
        """Shrink the group sizes so individual larvae stay legible.

        Only the number of agents is touched. Model parameters, arena geometry
        and every other scientific setting stay exactly as the stored config
        defines them.
        """
        groups = parameters.get("larva_groups", {})
        if not groups:
            return parameters
        per_group = max(1, min(MAX_AGENTS, scenario.n_agents) // max(1, len(groups)))
        for group in groups.values():
            distribution = group.get("distribution")
            if distribution is not None and "N" in distribution:
                distribution["N"] = per_group
        return parameters

    # ---- result --------------------------------------------------------

    def show_result(
        self,
        scenario: Scenario,
        canvas: Any,
        frames: list[Any],
        *,
        dt: float,
        note: str | None,
        analysis: Any | None = None,
        datasets: list[Any] | None = None,
    ) -> None:
        from larvaworld.portal.simulation.run_playback import FramePlayback

        try:
            playback = FramePlayback(canvas=canvas, frames=frames, dt=dt)
            playback_view: pn.viewable.Viewable = playback.view()
        except ValueError:
            playback_view = canvas.view()

        again = run_button(name="Run it again", width=150)
        again.on_click(lambda _event: self.start_scenario(scenario))
        back = pn.widgets.Button(
            name="< All scenarios", button_type="default", width=150
        )
        back.on_click(lambda _event: self.show_gallery())
        advanced = pn.pane.HTML(
            '<a class="lw-portal-btn" href="/wf.run_experiment">'
            "Open in Single Experiment</a>",
            margin=0,
        )

        note_pane = pn.pane.HTML(
            f'<div class="lw-explore-literature">{escape(note)}</div>' if note else "",
            margin=0,
            visible=bool(note),
        )
        observations = (
            "This animation is one simulated run. Replay it to inspect the "
            "movement pattern in more detail."
        )
        if scenario.watch_for:
            bullets = "".join(f"<li>{escape(hint)}</li>" for hint in scenario.watch_for)
            observations = (
                "This animation is one simulated run. Look for these visual cues "
                "in the paths and behavior:"
                f"<ul style='margin:6px 0 0 0;padding-left:20px;'>{bullets}</ul>"
            )
        observation_pane = pn.pane.HTML(
            '<div class="lw-explore-watchfor"><strong>What you just saw</strong><br/>'
            f"{observations}"
            "</div>",
            margin=(10, 0, 0, 0),
        )
        result_view: pn.viewable.Viewable = playback_view
        if analysis is not None:
            result_view = pn.Row(
                pn.Column(
                    playback_view,
                    css_classes=["lw-explore-playback"],
                    sizing_mode="stretch_width",
                    margin=0,
                ),
                self._analysis_view(analysis, datasets=datasets or []),
                css_classes=["lw-explore-result-grid"],
                sizing_mode="stretch_width",
                margin=(0, 0, 0, 0),
            )

        self.body[:] = [
            pn.Row(back, again, advanced, margin=0),
            pn.pane.HTML(
                f'<div class="lw-explore-stage-title">{escape(scenario.title)}</div>',
                margin=(10, 0, 0, 0),
            ),
            self._scenario_summary(scenario),
            result_view,
            observation_pane,
            note_pane,
        ]

    @staticmethod
    def _analysis_view(
        analysis: Any,
        *,
        datasets: list[Any],
    ) -> pn.viewable.Viewable:
        """Render transient metrics and Matplotlib plots beside the playback."""
        cards = "".join(
            '<div class="lw-explore-metric">'
            f'<div class="lw-explore-metric-label">{escape(metric.label)}</div>'
            '<div class="lw-explore-metric-value">'
            f"{metric.value:.3g}{(' ' + escape(metric.unit)) if metric.unit else ''}"
            "</div></div>"
            for metric in analysis.metrics
        )
        contents: list[pn.viewable.Viewable] = [
            pn.pane.HTML(
                '<div class="lw-explore-analysis-title">Preview analysis</div>'
                '<div class="lw-explore-analysis-note">'
                "Calculated from this shortened simulation run; it is not a full experiment analysis."
                "</div>"
                f'<div class="lw-explore-metrics">{cards}</div>',
                margin=0,
            )
        ]
        if datasets:
            contents.append(_preview_dataset_tables_view(datasets))
        for item in analysis.figures:
            contents.append(
                pn.Column(
                    pn.pane.HTML(
                        '<div class="lw-explore-analysis-figure-title">'
                        f"{escape(item.title)}</div>",
                        margin=0,
                    ),
                    pn.pane.Matplotlib(
                        item.figure,
                        tight=True,
                        sizing_mode="scale_width",
                        margin=0,
                    ),
                    css_classes=["lw-explore-analysis-figure"],
                    sizing_mode="stretch_width",
                    margin=0,
                )
            )
        if analysis.warnings:
            warnings = "<br/>".join(escape(message) for message in analysis.warnings)
            contents.append(
                pn.pane.HTML(
                    f'<div class="lw-explore-analysis-warning">{warnings}</div>',
                    margin=0,
                )
            )
        return pn.Column(
            *contents,
            css_classes=["lw-explore-analysis"],
            sizing_mode="stretch_width",
            margin=0,
        )

    # ---- errors and teardown -------------------------------------------

    def _show_error(self, scenario: Scenario, exc: Exception) -> None:
        back = pn.widgets.Button(
            name="< All scenarios", button_type="default", width=150
        )
        back.on_click(lambda _event: self.show_gallery())
        self.body[:] = [
            back,
            pn.pane.HTML(
                '<div class="lw-explore-error">'
                f"<strong>Could not run &quot;{escape(scenario.title)}&quot;.</strong>"
                f"<br/>{escape(f'{type(exc).__name__}: {exc}')}"
                "<br/>Try another scenario, or open this experiment in Single "
                "Experiment for full diagnostics."
                "</div>",
                margin=(10, 0, 0, 0),
            ),
        ]

    def _release_preview_resources(self) -> None:
        if self._analysis_figures:
            from larvaworld.portal.explore.analysis import (
                PreviewFigure,
                close_preview_figures,
            )

            close_preview_figures(
                [PreviewFigure("", figure) for figure in self._analysis_figures]
            )
            self._analysis_figures = []
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except OSError:
                pass
            self._temp_dir = None

    def view(self) -> pn.viewable.Viewable:
        return pn.Column(
            self.body,
            css_classes=["lw-explore-root"],
            sizing_mode="stretch_width",
        )


def explore_app() -> pn.viewable.Viewable:
    pn.extension("tabulator", raw_css=[PORTAL_RAW_CSS, EXPLORE_RAW_CSS])
    controller = _ExploreController()
    template = pn.template.MaterialTemplate(
        title="",
        header_background="#b5c2b0",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Explore"))
    template.main.append(controller.view())
    return template


__all__ = ["SCENARIOS", "Scenario", "explore_app", "scenario_by_id"]
