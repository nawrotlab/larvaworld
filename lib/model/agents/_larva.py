from copy import deepcopy
import numpy as np

from lib.model.agents._agent import LarvaworldAgent


class Larva(LarvaworldAgent):
    def __init__(self, unique_id, model, pos=None, radius=None, default_color=None, **kwargs):
        if default_color is None:
            default_color = model.generate_larva_color()
        super().__init__(unique_id=unique_id, model=model, default_color=default_color, pos=pos, radius=radius,
                         **kwargs)
        self.behavior_pars = ['stride_stop', 'stride_id', 'pause_id', 'feed_id', 'Lturn_id', 'Rturn_id']
        self.null_behavior_dict = dict(zip(self.behavior_pars, [False] * len(self.behavior_pars)))
        self.carried_objects = []

    def update_color(self, default_color, dic, mode='lin'):
        color = deepcopy(default_color)
        if mode == 'lin':
            if dic.stride_id or dic.run_id:
                color = np.array([0, 150, 0])
            elif dic.pause_id:
                color = np.array([255, 0, 0])
            elif dic.feed_id:
                color = np.array([0, 0, 255])
        elif mode == 'ang':
            if dic.Lturn_id:
                color[2] = 150
            elif dic.Rturn_id:
                color[2] = 50
        return color

    @property
    def dt(self):
        return self.model.dt

    @property
    def scaled_amount_eaten(self):
        return self.amount_eaten / self.get_real_mass()

    @property
    def x(self):
        return self.pos[0] / self.model.scaling_factor

    @property
    def y(self):
        return self.pos[1] / self.model.scaling_factor

    @property
    def x0(self):
        return self.initial_pos[0] / self.model.scaling_factor

    @property
    def y0(self):
        return self.initial_pos[1] / self.model.scaling_factor
