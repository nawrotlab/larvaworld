from __future__ import annotations
from typing import Any

import pandas as pd
import holoviews as hv
import panel as pn
from panel.template import DarkTheme

pn.extension()

from larvaworld.lib import reg, util
from larvaworld.lib.process import LarvaDataset, LarvaDatasetCollection

__all__: list[str] = ["TrackViewer", "track_viewer_app"]

w, h = 800, 500


class TrackViewer:
    def __init__(self, size: int = 600) -> None:
        self.size = size

    def dataset_as_dict(self, d: LarvaDataset) -> util.AttrDict:
        xy = d.load_traj()
        xy_origin = pd.concat(
            [g - g.dropna().iloc[0] for id, g in xy.groupby("AgentID")]
        ).sort_index()
        dsp_mu = {
            i: ((ttt.dropna() ** 2).sum(axis=1) ** 0.5).mean()
            for i, ttt in xy_origin.groupby("Step")
        }
        return util.AttrDict({"default": xy, "origin": xy_origin, "dispersal": dsp_mu})

    def build_data(self, id: str) -> util.AttrDict:
        # TODO: This has been modified to work with both LarvaDataset and LarvaDatasetCollection. Needs to be simplified!
        if id in reg.conf.Ref.RefGroupIDs:
            d = reg.conf.Ref.loadRefGroup(id)
            data = util.AttrDict(
                {l: self.dataset_as_dict(d) for l, d in d.data_dict.items()}
            )
            x, y = d.arena_dims
            self.labels = d.labels
            self.Nticks = d.Nticks
            self.dt = d.dt
            self.color_palette = d.color_palette
        elif id in reg.conf.Ref.confIDs:
            d = reg.conf.Ref.loadRef(id)
            data = util.AttrDict({d.id: self.dataset_as_dict(d)})
            x, y = d.env_params.arena.dims
            self.labels = [d.id]
            self.Nticks = d.config.Nticks
            self.dt = d.config.dt
            self.color_palette = util.AttrDict({d.id: d.config.color})
        else:
            raise ValueError("Invalid ID! No single or group dataset found.")

        self.image_kws = {
            "title": "Trajectory viewer",
            "xlim": (-x / 2, x / 2),
            "ylim": (-y / 2, y / 2),
            "width": self.size,
            "height": self.size,
            "xlabel": "X (m)",
            "ylabel": "Y (m)",
        }
        return data

    def get_app(self, id: str):
        self.xy_data = self.build_data(id=id)
        cb_IDs = pn.widgets.CheckBoxGroup(value=self.labels, options=self.labels)
        cb_vis = pn.widgets.CheckBoxGroup(
            value=["Positions", "Disperal circle"],
            options=["Positions", "IDs", "Tracks", "Disperal circle"],
        )
        cb_rnd_col = pn.widgets.Checkbox(
            name="Random colors", value=False, disabled=True
        )
        cb_dispersal = pn.widgets.Checkbox(name="Align tracks to origin", value=True)

        slider_kws = {
            "width": self.size,
            "start": 0,
            "end": self.Nticks - 1,
            "interval": int(1000 * self.dt),
            "value": 1,
            "step": 5,
            "loop_policy": "loop",
        }
        time_slider = pn.widgets.Player(**slider_kws)

        @pn.depends(
            i=time_slider,
            valid_gIDs=cb_IDs,
            dispersal_on=cb_dispersal,
            vis_ops=cb_vis,
            rnd_cols=cb_rnd_col,
        )
        def get_image(valid_gIDs, i, dispersal_on, vis_ops, rnd_cols):
            pos_on = "Positions" in vis_ops
            ids_on = "IDs" in vis_ops
            paths_on = "Tracks" in vis_ops
            circle_on = "Disperal circle" in vis_ops
            mode = "origin" if dispersal_on else "default"

            goverlay = None
            for gID in valid_gIDs:
                gdata = self.xy_data[gID]
                grouped_xy = gdata[mode].loc[:i].groupby("AgentID")

                track_kws = {
                    "color": None if rnd_cols else self.color_palette[gID],
                }
                _points = grouped_xy.last()

                points = hv.Points(_points, label=gID).opts(size=2, **track_kws)
                overlay = points

                if ids_on:
                    labels = hv.Labels(_points.reset_index(), ["x", "y"]).opts(
                        text_font_size="8pt", xoffset=0.015, visible=ids_on
                    )
                    overlay *= labels
                if paths_on:
                    _paths = [xyi for id, xyi in grouped_xy]
                    paths = hv.Path(_paths).opts(**track_kws)
                    overlay *= paths
                if circle_on and mode == "origin":
                    r = gdata["dispersal"][i]
                    circle = hv.Ellipse(0, 0, r).opts(line_width=5, **track_kws)
                    overlay *= circle
                    # r = ((_points.dropna() ** 2).sum(axis=1) ** 0.5).mean()
                    # circle2 = hv.Ellipse(0, 0, r).opts(line_width=4,line_dash='dotted', **track_kws)
                    # overlay *= circle2
                goverlay = overlay if goverlay is None else goverlay * overlay

            goverlay.opts(
                hv.opts.Points(size=5, visible=pos_on),
                # hv.opts.Labels(text_font_size='8pt', xoffset=0.015,visible=ids_on),
            )

            goverlay.opts(responsive=False, **self.image_kws)

            return goverlay

        img_dmap = hv.DynamicMap(get_image)
        app = pn.Row(
            img_dmap,
            pn.Column(
                pn.Row(
                    pn.Column("Datasets", cb_IDs, sizing_mode="stretch_width"),
                    pn.Column("Visibility", cb_vis, sizing_mode="stretch_width"),
                    pn.Column(
                        "Settings",
                        cb_rnd_col,
                        cb_dispersal,
                        sizing_mode="stretch_width",
                    ),
                    width=self.size,
                ),
                pn.Row(pn.Column("Tick", time_slider)),
            ),
        )
        return app


v = TrackViewer()

CT = reg.conf.Ref
Msel = pn.widgets.Select(
    value=reg.default_refID,
    name="Reference datasets (single or grouped)",
    options=reg.conf.Ref.confIDs + reg.conf.Ref.RefGroupIDs,
)
# Mrun = pn.widgets.Button(name="Run")

track_viewer_app: "pn.template.MaterialTemplate" = pn.template.MaterialTemplate(
    title="larvaworld : Dataset track viewer", theme=DarkTheme, sidebar_width=w
)
track_viewer_app.sidebar.append(pn.Row(Msel, width=300, height=80))
track_viewer_app.main.append(pn.bind(v.get_app, Msel))


track_viewer_app.servable()
