import dvc.cli.parser
import pandas as pd
import panel as pn
import numpy as np
import holoviews as hv

from larvaworld.lib.process.dataset import LarvaDatasetCollection

pn.extension()

import larvaworld
from larvaworld.lib import reg, aux


class TrackViewer(LarvaDatasetCollection):

    def __init__(self, size=600, **kwargs):
        super().__init__(**kwargs)
        self.size = size

        self.xy_data=self.get_xy()

        x, y = self.arena_dims
        self.image_kws = {
            'title': f'Trajectory viewer',
            'xlim': (-x / 2, x / 2),
            'ylim': (-y / 2, y / 2),
            'width': self.size,
            'height': self.size,
            'xlabel': 'X (m)',
            'ylabel': 'Y (m)',
            # 'legend': 'True',
        }

        slider_kws = {
            'width': self.size,
            'start': 0,
            'end': self.Nticks - 1,
            'interval': 1,
            'value': 0,
            # 'show_value': True,
            'loop_policy': 'loop',

        }

        # self.track_kws={
        #     'color' : d.color
        # }

        self.time_slider = pn.widgets.Player(name='Tick', **slider_kws)
        self.visible_ops = pn.widgets.CheckBoxGroup(name='Visibility', value=['Positions'],
                                                    options=['Positions', 'IDs', 'Tracks'])
        self.cb_rnd_colors = pn.widgets.Checkbox(name='Random colors', value=False, disabled=True)
        self.cb_dispersal = pn.widgets.Checkbox(name='Align tracks to origin', value=True)
        self.valid_gIDs = pn.widgets.CheckBoxGroup(name='Datasets', value=self.labels, options=self.labels)


        self.app=self.get_app()


    # @property
    # def xydata_palette(self):
    #
    #     @pn.depends(dispersal_on=self.cb_dispersal)
    #     def mode(dispersal_on):
    #         return 'origin' if dispersal_on else 'default'
    #
    #     d=self.xy_data[mode(self.cb_dispersal.value)]
    #
    #     return zip(self.labels, d, self.colors)
    # @property
    # # @pn.depends(dispersal_on=self.cb_dispersal)
    # def mode(self):
    #     return 'origin' if self.cb_dispersal.value else 'default'
    #
    # @property
    # def xydata_valid(self):
    #     d = dict(zip(self.labels, self.xy_data[self.mode]))
    #     valid_gIDs=self.valid_gIDs.value
    #     i=self.time_slider.value
    #     return {gID : d[gID].loc[:i].groupby('AgentID') for gID in valid_gIDs}
    # @property
    # def xydata_dict(self):
    #     @pn.depends(dispersal_on=self.cb_dispersal)
    #     def mode(dispersal_on):
    #         return 'origin' if dispersal_on else 'default'
    #
    #     d = self.xy_data[mode(self.cb_dispersal.value)]
    #     return dict(zip(self.labels, d))


    def get_xydata_valid(self, valid_gIDs, i, dispersal_on):
        mode= 'origin' if dispersal_on else 'default'
        d = dict(zip(self.labels,self.xy_data[mode]))
        return {gID : d[gID].loc[:i].groupby('AgentID') for gID in valid_gIDs}

    # @property
    # def xydata_valid(self):
    #
    #     @pn.depends(dispersal_on=self.cb_dispersal)
    #     def mode(dispersal_on):
    #         return 'origin' if dispersal_on else 'default'
    #
    #     d = dict(zip(self.labels, self.xy_data[mode]))
    #
    #
    #
    #
    #     return self.get_xydata_valid(valid_gIDs, i, dispersal_on)

    #



    def get_xy(self):
        xy_data=aux.AttrDict({'default': [], 'origin': []})
        for d in self.datasets :
            xy = d.load_traj()
            xy_grouped = xy.groupby('AgentID')
            # ids = xy.index.get_level_values('AgentID').unique()
            # Nids = len(ids)
            xy0s = xy_grouped.first()
            xy_origin = pd.concat([g - xy0s.loc[id] for id, g in xy_grouped]).sort_index()
            # Ncolors = dict(zip(ids, aux.N_colors(Nids)))
            xy_data.default.append(xy)
            xy_data.origin.append(xy_origin)
            # xy_data.default[d.id]=xy
            # xy_data.origin[d.id]=xy_origin
        return xy_data


    def get_app(self):



        # @pn.depends(dispersal_on=self.cb_dispersal)
        # def mode(dispersal_on):
        #     return 'origin' if dispersal_on else 'default'
        #
        # @pn.depends(i=self.time_slider, valid_gIDs=self.valid_gIDs)
        # def xydata_valid(i,valid_gIDs):
        #     d = dict(zip(self.labels, self.xy_data[mode]))
        #     return {gID : d[gID].loc[:i].groupby('AgentID') for gID in valid_gIDs}


        @pn.depends(i=self.time_slider, valid_gIDs=self.valid_gIDs,dispersal_on=self.cb_dispersal,visible_ops=self.visible_ops, rnd_cols=self.cb_rnd_colors)
        def get_image(valid_gIDs, i, dispersal_on,visible_ops, rnd_cols):
            pos_on='Positions' in visible_ops
            ids_on='IDs' in visible_ops
            paths_on='Tracks' in visible_ops
            dic=self.get_xydata_valid(valid_gIDs, i, dispersal_on)
            # for gID, data in dic.items()


            goverlay = None
            for gID, data in dic.items():
                track_kws = {
                    'color': None if rnd_cols else self.color_palette[gID],
                    # 'color_by' :
                }
                _points = data.last()
                points = hv.Points(_points).opts(**track_kws)
                labels = hv.Labels(_points.reset_index(), ['x', 'y'])
                overlay = points * labels
                if paths_on:
                    _paths = [xyi for id, xyi in data]
                    paths = hv.Path(_paths).opts(**track_kws)
                    overlay*=paths
                goverlay = overlay if goverlay is None else goverlay * overlay

            goverlay.opts(
                hv.opts.Points(size=5, visible=pos_on),
                hv.opts.Labels(text_font_size='8pt', xoffset=0.015,visible=ids_on),
                          )
            goverlay.opts(responsive=False, **self.image_kws)
            return goverlay

        img_dmap = hv.DynamicMap(get_image)
        app = pn.Column(img_dmap,
                             pn.Row(
                                 pn.Column(self.valid_gIDs),
                                 pn.Column(self.visible_ops),
                                    pn.Column(self.cb_rnd_colors, self.cb_dispersal),
                                    width=self.size),
                             self.time_slider
                             )
        return app


    # def image_per_dataset(self, gID, i):
    #     gdata=self.xydata_dict[gID]
    #     data = gdata.loc[:i].groupby('AgentID')
    #     track_kws = {
    #         'color': self.color_palette[gID]
    #     }
    #     _points = data.last()
    #     _paths = [xyi for id, xyi in data]
    #     points = hv.Points(_points).opts(**track_kws)
    #     labels = hv.Labels(_points.reset_index(), ['x', 'y'])
    #
    #     if paths_on:
    #         _paths = [xyi for id, xyi in data]
    #         paths = hv.Path(_paths).opts(**track_kws)
    #         overlay = (paths * points * labels)
    #     else:
    #         overlay = (points * labels)
    #     return overlay



