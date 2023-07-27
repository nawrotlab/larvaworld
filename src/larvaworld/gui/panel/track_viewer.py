import pandas as pd
import panel as pn
import numpy as np
import holoviews as hv

pn.extension()

from larvaworld.lib import reg, aux
from larvaworld.lib.process.dataset import LarvaDatasetCollection


class TrackViewer(LarvaDatasetCollection):

    def __init__(self, size=600, **kwargs):
        super().__init__(**kwargs)
        self.size = size

        self.xy_data = self.build_data()

        x, y = self.arena_dims
        self.image_kws = {
            'title': f'Trajectory viewer',
            'xlim': (-x / 2, x / 2),
            'ylim': (-y / 2, y / 2),
            'width': self.size,
            'height': self.size,
            'xlabel': 'X (m)',
            'ylabel': 'Y (m)',
        }
        self.app = self.get_app()


    def build_data(self):
        data = aux.AttrDict()
        for l, d in self.data_dict.items():
            xy = d.load_traj()
            xy_origin = pd.concat([g - g.dropna().iloc[0] for id, g in xy.groupby('AgentID')]).sort_index()
            dsp_mu={i: ((ttt.dropna() ** 2).sum(axis=1) ** 0.5).mean() for i, ttt in xy_origin.groupby('Step')}
            data[l] = aux.AttrDict({'default': xy, 'origin': xy_origin, 'dispersal': dsp_mu})
        return data

    def get_app(self):

        cb_IDs = pn.widgets.CheckBoxGroup(value=self.labels, options=self.labels)
        cb_vis = pn.widgets.CheckBoxGroup(value=['Positions', 'Disperal circle'],
                                          options=['Positions', 'IDs', 'Tracks', 'Disperal circle'])
        cb_rnd_col = pn.widgets.Checkbox(name='Random colors', value=False, disabled=True)
        cb_dispersal = pn.widgets.Checkbox(name='Align tracks to origin', value=True)

        slider_kws = {
            'width': self.size,
            'start': 0,
            'end': self.Nticks - 1,
            'interval': int(1000 * self.dt),
            'value': 0,
            'step': 5,
            'loop_policy': 'loop',

        }
        time_slider = pn.widgets.Player(**slider_kws)

        @pn.depends(i=time_slider, valid_gIDs=cb_IDs, dispersal_on=cb_dispersal, vis_ops=cb_vis, rnd_cols=cb_rnd_col)
        def get_image(valid_gIDs, i, dispersal_on, vis_ops, rnd_cols):
            pos_on = 'Positions' in vis_ops
            ids_on = 'IDs' in vis_ops
            paths_on = 'Tracks' in vis_ops
            circle_on = 'Disperal circle' in vis_ops
            mode = 'origin' if dispersal_on else 'default'

            goverlay = None
            for gID in valid_gIDs:
                gdata=self.xy_data[gID]
                grouped_xy=gdata[mode].loc[:i].groupby('AgentID')

                track_kws = {
                    'color': None if rnd_cols else self.color_palette[gID],
                }
                _points = grouped_xy.last()

                points = hv.Points(_points, label=gID).opts(size=2, **track_kws)
                overlay = points

                if ids_on:
                    labels = hv.Labels(_points.reset_index(), ['x', 'y']).opts(text_font_size='8pt', xoffset=0.015,
                                                                               visible=ids_on)
                    overlay *= labels
                if paths_on:
                    _paths = [xyi for id, xyi in grouped_xy]
                    paths = hv.Path(_paths).opts(**track_kws)
                    overlay *= paths
                if circle_on and mode == 'origin':
                    r = gdata['dispersal'][i]
                    circle = hv.Ellipse(0, 0, r).opts(line_width=5,  **track_kws)
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
        app = pn.Row(img_dmap, pn.Column(
            pn.Row(
                pn.Column('Datasets', cb_IDs, sizing_mode='stretch_width'),
                pn.Column('Visibility', cb_vis, sizing_mode='stretch_width'),
                pn.Column('Settings', cb_rnd_col, cb_dispersal, sizing_mode='stretch_width'),
                width=self.size),
            pn.Row(pn.Column('Tick', time_slider))
        ))
        return app


# if __name__ == "__main__":
#     refIDs = reg.conf.Ref.confIDs[10:13]
#     # d=reg.loadRef(refID)
#     v = TrackViewer(refIDs=refIDs)
#     # from larvaworld.gui.tabs.larvaworld_gui import LarvaworldGui
#     # larvaworld_gui = LarvaworldGui()
#     # v=TrackViewer()
#     app = v.get_app()
#     # v.get_app()
#     # app = v.get_app()
#     app.servable()

