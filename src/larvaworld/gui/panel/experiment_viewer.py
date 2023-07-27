import pandas as pd
import panel as pn
import numpy as np
import holoviews as hv
from panel.template import DarkTheme

pn.extension()

from larvaworld.lib import reg, aux, model, sim
from larvaworld.lib.param import SimOps

class ArenaViewer:

    def __init__(self, size=600, experiment='dish',**kwargs):
        # super().__init__(**kwargs)
        self.size = size

        self.launcher=sim.ExpRun(experiment=experiment, **kwargs)
        self.env=self.launcher.p.env_params
        # self.conf=reg.conf.Exp.expand(experiment)
        # self.conf.update(SimOps().nestedConf)
        # self.xy_data = self.build_data()

        x, y = self.env.arena.dims
        self.image_kws = {
            'title': f'Arena viewer',
            'xlim': (-x / 2, x / 2),
            'ylim': (-y / 2, y / 2),
            'width': self.size,
            'height': int(self.size*y/x),
            'xlabel': 'X (m)',
            'ylabel': 'Y (m)',
        }
        self.app=self.get_app()
        # self.app.servable()

    def get_tank_plot(self):
        a = self.env.arena
        if a.geometry == 'circular':
            tank = hv.Ellipse(0, 0, a.dims[0]).opts(line_width=5, bgcolor='lightgrey')
            # arena = plt.Circle((0, 0), x / 2, edgecolor='black', facecolor='lightgrey', lw=3)
        elif a.geometry == 'rectangular':
            tank = hv.Box(0, 0, spec=a.dims).opts(line_width=5, bgcolor='lightgrey')
        else:
            raise ValueError('Not implemented')
        return tank

    def get_app(self):
        # pass
        # cb_IDs = pn.widgets.CheckBoxGroup(value=self.labels, options=self.labels)
        # cb_vis = pn.widgets.CheckBoxGroup(value=['Positions', 'Disperal circle'],
        #                                   options=['Positions', 'IDs', 'Tracks', 'Disperal circle'])
        # cb_rnd_col = pn.widgets.Checkbox(name='Random colors', value=False, disabled=True)
        # cb_dispersal = pn.widgets.Checkbox(name='Align tracks to origin', value=True)
        #
        self.launcher.sim_setup()
        slider_kws = {
            'width': int(self.size/2),
            'start': 0,
            'end': self.launcher.Nsteps-1,
            'interval':  int(1000*self.launcher.dt),
            'value': 0,
            # 'step': 5,
            # 'loop_policy': 'loop',

        }
        progress_kws = {
            'width': int(self.size / 2),
            # 'start': 0,
            'max': self.launcher.Nsteps - 1,
            # 'interval': int(1000 * self.launcher.dt),
            'value': self.launcher.t,
            # 'step': 5,
            # 'loop_policy': 'loop',

        }
        self.progress_bar = pn.widgets.Progress(bar_color="primary",**progress_kws)
        time_slider = pn.widgets.Player(**slider_kws)
        self.tank_plot=self.get_tank_plot()
        @pn.depends(i=time_slider)
        def get_image(i):
            while i>self.launcher.t :
                self.launcher.sim_step()
                self.progress_bar.value=self.launcher.t

            # progress_bar=i
            # pos_on = 'Positions' in vis_ops
            # ids_on = 'IDs' in vis_ops
            # paths_on = 'Tracks' in vis_ops
            # circle_on = 'Disperal circle' in vis_ops
            # mode = 'origin' if dispersal_on else 'default'

            # goverlay = None

            overlay = self.tank_plot
            ps=self.launcher.agents.get_position()
            colors = self.launcher.agents.color
            points = hv.Points(ps).opts(size=5, color='black')
            overlay*=points
            hps = self.launcher.agents.head.front_end
            hpoints = hv.Points(hps).opts(size=5, color='red')
            overlay *= hpoints
            for s in self.launcher.sources:
                source = hv.Ellipse(s.pos[0], s.pos[1], s.radius*2).opts(line_width=5,color=s.color, bgcolor=s.color)
                overlay *= source

            # for gID in valid_gIDs:
            #     gdata=self.xy_data[gID]
            #     grouped_xy=gdata[mode].loc[:i].groupby('AgentID')
            #
            #     track_kws = {
            #         'color': None if rnd_cols else self.color_palette[gID],
            #     }
            #     _points = grouped_xy.last()
            #
            #     points = hv.Points(_points, label=gID).opts(size=2, **track_kws)
            #     overlay = points
            #
            #     if ids_on:
            #         labels = hv.Labels(_points.reset_index(), ['x', 'y']).opts(text_font_size='8pt', xoffset=0.015,
            #                                                                    visible=ids_on)
            #         overlay *= labels
            #     if paths_on:
            #         _paths = [xyi for id, xyi in grouped_xy]
            #         paths = hv.Path(_paths).opts(**track_kws)
            #         overlay *= paths
            #     if circle_on and mode == 'origin':
            #         r = gdata['dispersal'][i]
            #         circle = hv.Ellipse(0, 0, r).opts(line_width=5,  **track_kws)
            #         overlay *= circle
            #         r = ((_points.dropna() ** 2).sum(axis=1) ** 0.5).mean()
            #         circle2 = hv.Ellipse(0, 0, r).opts(line_width=4,line_dash='dotted', **track_kws)
            #         overlay *= circle2
                # goverlay = overlay if goverlay is None else goverlay * overlay
            #
            # goverlay.opts(
            #     hv.opts.Points(size=5, visible=pos_on),
            #     hv.opts.Labels(text_font_size='8pt', xoffset=0.015,visible=ids_on),
            # )
            #
            overlay.opts(responsive=False, **self.image_kws)

            return overlay

        img_dmap = hv.DynamicMap(get_image)
        app = pn.Row(img_dmap, pn.Column(
            # pn.Row(
            #     pn.Column('Datasets', cb_IDs, sizing_mode='stretch_width'),
            #     pn.Column('Visibility', cb_vis, sizing_mode='stretch_width'),
            #     pn.Column('Settings', cb_rnd_col, cb_dispersal, sizing_mode='stretch_width'),
            #     width=self.size),
            pn.Row(pn.Column('Tick', time_slider)),
            pn.Row(pn.Column('Simulation timestep', self.progress_bar)),
        ))
        # from my_template import DarkTheme
        # template = pn.template.MaterialTemplate(title='Material Dark', theme=DarkTheme)

        # template.sidebar.append(A_in)
        # template.sidebar.append(turner_conf)

        # template.main.append(
        #     app
        # )
        # template.servable();

        return app


# if __name__ == "__main__":
#     # from larvaworld.gui.tabs.larvaworld_gui import LarvaworldGui
#     # larvaworld_gui = LarvaworldGui()
#     v=ArenaViewer()
#     app = v.get_app()
#     # v.get_app()
#     # app = v.get_app()
#     app.servable()

