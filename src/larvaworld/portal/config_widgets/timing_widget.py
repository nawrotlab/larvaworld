"""Reusable timing/duration panel widget for portal apps.

Displays and controls: duration, steps, timestep, framerate (derived).
Reduces duplication across Module Inspector, Model Inspector, and simulation apps.
"""

from __future__ import annotations

import panel as pn
import param

__all__ = ["build_timing_widget"]


def build_timing_widget(
    owner: param.Parameterized,
    *,
    include_framerate: bool = False,
    title: str = "Simulation Timing",
    collapsible: bool = False,
) -> pn.viewable.Viewable:
    """Build a timing/duration panel widget.

    Parameters
    ----------
    owner : param.Parameterized
        Object with timing parameters (duration, steps, dt/timestep).
    include_framerate : bool, optional
        If True, display calculated framerate (read-only). Default False.
    title : str, optional
        Title for the timing section. Default "Simulation Timing".
    collapsible : bool, optional
        If True, wrap in collapsible Card. Default False.

    Returns
    -------
    pn.viewable.Viewable
        Panel widget displaying timing controls.

    Notes
    -----
    Expected parameters on owner:
    - duration: total simulation time (seconds)
    - steps: number of steps / number of steps
    - dt (or timestep): integration timestep (seconds)
    - framerate (optional, read-only): calculated from duration/steps
    """
    widgets = []

    # Duration input (if available)
    if "duration" in owner.param:
        widgets.append(
            pn.widgets.FloatInput(
                name="Duration (s)",
                value=owner.param.duration.default,
                start=0.1,
                step=0.1,
                tooltip="Total simulation time in seconds",
            )
        )

    # Steps input (if available)
    if "steps" in owner.param:
        widgets.append(
            pn.widgets.IntInput(
                name="Steps",
                value=owner.param.steps.default,
                start=1,
                step=1,
                tooltip="Number of integration steps",
            )
        )

    # Timestep / dt input (if available)
    dt_param = None
    if "dt" in owner.param:
        dt_param = "dt"
    elif "timestep" in owner.param:
        dt_param = "timestep"

    if dt_param:
        param_obj = getattr(owner.param, dt_param)
        widgets.append(
            pn.widgets.FloatInput(
                name="Timestep (s)",
                value=param_obj.default,
                start=0.0001,
                step=0.0001,
                tooltip="Integration timestep in seconds",
            )
        )

    # Framerate display (derived, read-only)
    if include_framerate:
        widgets.append(
            pn.widgets.FloatInput(
                name="Framerate (Hz)",
                value=30.0,
                disabled=True,
                tooltip="Calculated framerate (steps / duration)",
            )
        )

    if not widgets:
        return pn.pane.HTML(
            '<div style="color:#999;font-size:12px;">No timing parameters available.</div>'
        )

    body = pn.Column(*widgets, sizing_mode="stretch_width", margin=0)

    if collapsible:
        return pn.Card(
            body,
            title=title,
            collapsed=False,
            collapsible=True,
            sizing_mode="stretch_width",
        )

    return pn.Column(
        pn.pane.Markdown(f"### {title}", margin=(0, 0, 8, 0)),
        body,
        sizing_mode="stretch_width",
        margin=0,
    )
