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

.lw-explore-watchfor {
  font-size: 13px;
  line-height: 1.6;
  padding: 10px 14px;
  border-radius: 10px;
  border: 1px solid rgba(181,194,176,0.7);
  background: rgba(181,194,176,0.16);
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


class _ExploreController:
    """Drives the gallery -> running -> result flow."""

    def __init__(self) -> None:
        self.body = pn.Column(sizing_mode="stretch_width", margin=0)
        self._runner: Any = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
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
        self._stop_runner()
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

    def start_scenario(self, scenario: Scenario) -> None:
        """Build the run and start streaming frames into the canvas."""
        # Imported lazily: these pull in the registry and the simulation stack,
        # which must not be paid for at portal startup.
        from larvaworld.lib import reg, util
        from larvaworld.portal.canvas_widgets.environment_canvas import (
            EnvironmentCanvas,
        )
        from larvaworld.portal.canvas_widgets.environment_mapping import (
            env_params_to_canvas_state,
        )
        from larvaworld.portal.simulation.run_playback import (
            ChunkedFrameRunner,
            build_bounded_launcher,
        )

        self._stop_runner()
        try:
            parameters = util.AttrDict(reg.conf.Exp.expand(scenario.exp_id))
            parameters = self._apply_scenario_overrides(parameters, scenario)
            canvas = EnvironmentCanvas(editable=False)
            canvas.set_state(
                env_params_to_canvas_state(
                    parameters.env_params,
                    larva_groups=parameters.get("larva_groups", {}),
                    show_group_shapes=False,
                )
            )
            self._temp_dir = tempfile.TemporaryDirectory(prefix="lw_explore_")
            run_dir = Path(self._temp_dir.name) / scenario.id
            run_dir.mkdir(parents=True, exist_ok=True)
            launcher, note = build_bounded_launcher(
                scenario.exp_id,
                parameters,
                run_dir,
                step_cap=scenario.step_cap,
            )
        except Exception as exc:
            self._show_error(scenario, exc)
            return

        progress = pn.indicators.Progress(
            value=0, max=scenario.step_cap, sizing_mode="stretch_width"
        )
        status = pn.pane.HTML("Starting...", margin=(4, 0, 0, 0))
        dt = float(parameters.get("dt", 0.1))

        def _on_progress(done: int, total: int) -> None:
            progress.value = done
            status.object = (
                f"Running... {done}/{total} steps "
                f"({done * dt:.1f}s of simulated time)"
            )

        def _on_complete(frames: list[Any]) -> None:
            self.show_result(scenario, canvas, frames, dt=dt, note=note)

        self._runner = ChunkedFrameRunner(
            launcher,
            total_steps=scenario.step_cap,
            on_frame=canvas.set_larva_frame,
            on_progress=_on_progress,
            on_complete=_on_complete,
            on_error=lambda exc: self._show_error(scenario, exc),
        )

        watch_for = ""
        if scenario.watch_for:
            bullets = "".join(f"<li>{escape(hint)}</li>" for hint in scenario.watch_for)
            watch_for = (
                '<div class="lw-explore-watchfor"><strong>What to watch for</strong>'
                f"<ul style='margin:6px 0 0 0;padding-left:20px;'>{bullets}</ul></div>"
            )

        back = pn.widgets.Button(
            name="< All scenarios", button_type="default", width=140
        )
        back.on_click(lambda _event: self.show_gallery())

        self.body[:] = [
            back,
            pn.pane.HTML(
                f'<div class="lw-explore-stage-title">{escape(scenario.title)}</div>'
                f'<div class="lw-explore-stage-teaser">{escape(scenario.teaser)}</div>',
                margin=(10, 0, 0, 0),
            ),
            canvas.view(),
            progress,
            status,
            pn.pane.HTML(watch_for, margin=(10, 0, 0, 0), visible=bool(watch_for)),
        ]
        self._runner.start()

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
    ) -> None:
        from larvaworld.portal.simulation.run_playback import FramePlayback

        try:
            playback = FramePlayback(canvas=canvas, frames=frames, dt=dt)
            playback_view: pn.viewable.Viewable = playback.view()
        except ValueError:
            playback_view = canvas.view()

        literature = ""
        if scenario.literature:
            literature = (
                '<div class="lw-explore-literature">Paradigm after '
                f"{escape(scenario.literature)}.</div>"
            )
        explanation = pn.pane.HTML(
            '<div class="lw-explore-explanation">'
            "<strong>What you just saw</strong><br/>"
            f"{escape(scenario.explanation)}"
            f"{literature}"
            "</div>",
            margin=(12, 0, 0, 0),
        )

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

        self.body[:] = [
            pn.Row(back, again, advanced, margin=0),
            pn.pane.HTML(
                f'<div class="lw-explore-stage-title">{escape(scenario.title)}</div>',
                margin=(10, 0, 0, 0),
            ),
            playback_view,
            note_pane,
            explanation,
        ]

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

    def _stop_runner(self) -> None:
        if self._runner is not None:
            self._runner.stop()
            self._runner = None
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
    pn.extension(raw_css=[PORTAL_RAW_CSS, EXPLORE_RAW_CSS])
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
