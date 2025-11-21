from __future__ import annotations
from typing import Any

import holoviews as hv
import numpy as np
import panel as pn
from panel.template import DarkTheme

pn.extension()

from larvaworld.lib import reg, screen, sim, util


__all__: list[str] = [
    "ExperimentViewer",
    "experiment_viewer_app",
]

w, h = 800, 500


class ExperimentViewer:
    def __init__(self) -> None:
        self.size = 600
        self.draw_ops = screen.AgentDrawOps(draw_centroid=True, draw_segs=False)

    def get_tank_plot(self) -> hv.element.Overlay:
        a = self.env.arena
        if a.geometry == "circular":
            tank = hv.Ellipse(0, 0, a.dims[0]).opts(line_width=5, bgcolor="lightgrey")
        elif a.geometry == "rectangular":
            tank = hv.Box(0, 0, spec=a.dims).opts(line_width=5, bgcolor="lightgrey")
        else:
            raise ValueError("Not implemented")
        return tank

    def draw_imgs(self) -> hv.Overlay:
        agents = self.launcher.agents
        sources = self.launcher.sources
        d = util.AttrDict(
            {
                "draw_segs": hv.Overlay(
                    [
                        hv.Polygons([seg.vertices for seg in a.segs]).opts(
                            color=a.color
                        )
                        for a in agents
                    ]
                ),
                "draw_centroid": hv.Points(agents.get_position()).opts(
                    size=5, color="black"
                ),
                "draw_head": hv.Points(agents.head.front_end).opts(size=5, color="red"),
                "draw_midline": hv.Overlay(
                    [
                        hv.Path(a.midline_xy).opts(color="blue", line_width=2)
                        for a in agents
                    ]
                ),
                "visible_trails": hv.Contours(
                    [a.trajectory[-self.Nfade :] for a in agents]
                ).opts(color="black"),
            }
        )
        source_imgs = [
            hv.Ellipse(s.pos[0], s.pos[1], s.radius * 2).opts(
                line_width=5, color=s.color, bgcolor=s.color
            )
            for s in sources
        ]
        agent_imgs = [img for k, img in d.items() if getattr(self.draw_ops, k)]

        return hv.Overlay([self.tank_plot] + source_imgs + agent_imgs).opts(
            responsive=False, **self.image_kws
        )

    def get_app(self, experiment: str = "dish", duration: int = 1, **kwargs: Any):
        self.launcher = sim.ExpRun(experiment=experiment, duration=duration, **kwargs)
        self.Nfade = int(self.draw_ops.trail_dt / self.launcher.dt)
        self.env = self.launcher.p.env_params
        x, y = self.env.arena.dims
        self.image_kws = {
            "title": "Arena viewer",
            "xlim": (-x / 2, x / 2),
            "ylim": (-y / 2, y / 2),
            "width": self.size,
            "height": int(self.size * y / x),
            "xlabel": "X (m)",
            "ylabel": "Y (m)",
        }
        self.launcher.sim_setup(steps=self.launcher.p.steps)
        slider_kws = {
            "width": int(self.size / 2),
            "start": 0,
            "end": self.launcher.Nsteps - 1,
            "interval": int(1000 * self.launcher.dt),
            "value": 1,
            # 'step': 5,
            # 'loop_policy': 'loop',
        }
        progress_kws = {
            "width": int(self.size / 2),
            "max": self.launcher.Nsteps - 1,
            "value": self.launcher.t,
        }
        self.progress_bar = pn.widgets.Progress(bar_color="primary", **progress_kws)
        time_slider = pn.widgets.Player(**slider_kws)
        self.tank_plot = self.get_tank_plot()

        @pn.depends(i=time_slider)
        def get_image(i: int):
            while i > self.launcher.t:
                self.launcher.sim_step()
                self.progress_bar.value = self.launcher.t
            return self.draw_imgs()

            # overlay = self.tank_plot
            # agents=self.launcher.agents
            # if draw_ops.draw_segs:
            #     for a in agents:
            #         segpolys = hv.Polygons([seg.vertices for seg in a.segs]).opts(color=a.color)
            #         overlay *= segpolys
            # if draw_ops.draw_centroid:
            #     points = hv.Points(agents.get_position()).opts(size=5, color='black')
            #     overlay*=points
            # if draw_ops.draw_head:
            #     hpoints = hv.Points(agents.head.front_end).opts(size=5, color='red')
            #     overlay *= hpoints
            # if draw_ops.draw_midline:
            #     for a in agents:
            #         mid = hv.Path(a.midline_xy).opts(color='blue',line_width=2)
            #         overlay *= mid
            # if draw_ops.trails:
            #     Nfade = int(draw_ops.trajectory_dt / self.launcher.dt)
            #
            #     _paths = [a.trajectory[-Nfade:] for a in agents]
            #     paths = hv.Contours(_paths).opts(color='black')
            #     overlay *= paths
            #
            # for s in self.launcher.sources:
            #     source = hv.Ellipse(s.pos[0], s.pos[1], s.radius*2).opts(line_width=5,color=s.color, bgcolor=s.color)
            #     overlay *= source

            # overlay.opts(responsive=False, **self.image_kws)
            #
            # return overlay

        img_dmap = hv.DynamicMap(get_image)
        app = pn.Row(
            img_dmap,
            pn.Column(
                pn.Row(pn.Column("Tick", time_slider)),
                pn.Row(pn.Column("Simulation timestep", self.progress_bar)),
                pn.Param(self.draw_ops),
            ),
        )
        return app


v = ExperimentViewer()

CT = reg.conf.Exp
Msel = pn.widgets.Select(value="dish", name="experiment", options=CT.confIDs)
Mrun = pn.widgets.Button(name="Run")

experiment_viewer_app: "pn.template.MaterialTemplate" = pn.template.MaterialTemplate(
    title="larvaworld : Experiment viewer", theme=DarkTheme, sidebar_width=w
)
experiment_viewer_app.sidebar.append(pn.Row(Msel, Mrun, width=300, height=80))
experiment_viewer_app.main.append(pn.bind(v.get_app, Msel))

experiment_viewer_app.servable()
