import pandas as pd
import panel as pn
import numpy as np
import holoviews as hv
from panel.template import DarkTheme

pn.extension()

from larvaworld.lib import reg, aux, model, sim, screen
from larvaworld.lib.param import SimOps

class ArenaViewer:

    def __init__(self, size=600, experiment='dish',**kwargs):
        # super().__init__(**kwargs)
        self.size = size

        self.launcher=sim.ExpRun(experiment=experiment,**kwargs)
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
        draw_ops=screen.AgentDrawOps(draw_centroid=True, draw_segs=False)
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
            agents=self.launcher.agents
            if draw_ops.draw_segs:
                for a in agents:
                    segpolys = hv.Polygons([seg.vertices for seg in a.segs]).opts(color=a.color)
                    overlay *= segpolys
            if draw_ops.draw_centroid:
                # ps=agents.get_position()
                # colors = agents.color
                points = hv.Points(agents.get_position()).opts(size=5, color='black')
                overlay*=points
            if draw_ops.draw_head:
                hpoints = hv.Points(agents.head.front_end).opts(size=5, color='red')
                overlay *= hpoints
            if draw_ops.draw_midline:
                for a in agents:
                    mid = hv.Path(a.midline_xy).opts(color='blue',line_width=2)
                    overlay *= mid
            if draw_ops.trails:
                Nfade = int(draw_ops.trajectory_dt / self.launcher.dt)

                _paths = [a.trajectory[-Nfade:] for a in agents]
                paths = hv.Contours(_paths).opts(color='black')
                overlay *= paths

                # segpolys = hv.Polygons(aux.flatten_list([[seg.vertices for seg in a.segs]for a in agents])).opts(color='black')
                # overlay *= segpolys

            for s in self.launcher.sources:
                source = hv.Ellipse(s.pos[0], s.pos[1], s.radius*2).opts(line_width=5,color=s.color, bgcolor=s.color)
                overlay *= source


            overlay.opts(responsive=False, **self.image_kws)

            return overlay

        img_dmap = hv.DynamicMap(get_image)
        app = pn.Row(img_dmap, pn.Column(

            #     pn.Column('Datasets', cb_IDs, sizing_mode='stretch_width'),
            #     pn.Column('Visibility', cb_vis, sizing_mode='stretch_width'),
            #     pn.Column('Settings', cb_rnd_col, cb_dispersal, sizing_mode='stretch_width'),
            #     width=self.size),
            pn.Row(pn.Column('Tick', time_slider)),
            pn.Row(pn.Column('Simulation timestep', self.progress_bar)),
            pn.Param(draw_ops),
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

