import agentpy
import numpy as np
import param
from scipy.stats import multivariate_normal
from shapely import geometry

from larvaworld.lib.model import Odor
from larvaworld.lib.screen.rendering import InputBox
from larvaworld.lib import aux




class LarvaworldAgent(param.Parameterized, agentpy.Agent):
    """LarvaworldAgent class that inherits from agentpy.Agent."""

    group = param.String(None, doc='The unique ID of the agent group')
    default_color = param.Color('black', doc='The default color of the agent')
    radius = param.Number(0.003, bounds=(0, None), softbounds=(0, 0.1), step=0.001,
                    doc='The spatial radius of the source in meters')
    pos = param.NumericTuple(default=(0.0, 0.0), doc='The xy spatial position coordinates')
    odor = param.ClassSelector(class_=Odor, default=Odor(), doc='The odor of the agent')


    def __init__(self, unique_id: str, odor={},model=None, visible=True, regeneration=False, regeneration_pos=None, **kwargs):
        """
            Initialize a LarvaworldAgent instance.

            Args:
            - unique_id: str representing the unique ID of the agent.
            - model: optional model instance.
            - pos: optional tuple representing the position of the agent.
            - default_color: optional str or tuple representing the default color of the agent.
            - radius: optional float representing the radius of the agent.
            - visible: optional boolean indicating whether the agent is visible or not.
            - odor: optional dictionary containing odor information of the agent.
            - regeneration: optional boolean indicating whether the agent is regenerating or not.
            - regeneration_pos: optional tuple representing the position of the regeneration.
            - group: optional str representing the group of the agent.
            - *args, **kwargs: optional arguments to be passed to the super().__init__() method.
        """


        param.Parameterized.__init__(self,odor=Odor(**odor), **kwargs)
        agentpy.Agent.__init__(self, model=model)
        # super().__init__(model, **kwargs)
        self.visible = visible
        self.selected = False
        self.unique_id = unique_id
        # self.group = group
        self.base_odor_id = f'{self.group}_base_odor'
        self.gain_for_base_odor = 100

        # self.pos = pos
        # if type(default_color) == str:
        #     default_color = aux.colorname2tuple(default_color)
        # self.default_color = default_color
        self.color = self.default_color


        self.odor_id = self.odor.id
        self.set_odor_dist(self.odor.intensity, self.odor.spread)

        self.regeneration = regeneration
        self.regeneration_pos = regeneration_pos

        self.id_box = InputBox(text=self.unique_id, color_inactive=self.default_color,
                                   color_active=self.default_color,
                                   agent=self)

    def nest_record(self, reporter_dic):


        # Connect log to the model's dict of logs
        if self.group not in self.model._logs:
            self.model._logs[self.group] = {}
        self.model._logs[self.group][self.unique_id] = self.log
        self.log['t'] = [self.model.t]  # Initiate time dimension

        # Perform initial recording
        for name, codename in reporter_dic.items():
            v = aux.rgetattr(self, codename)
            self.log[name] = [v]

        # Set default recording function from now on
        self.nest_record = self._nest_record  # noqa

    def _nest_record(self, reporter_dic):

        for name, codename in reporter_dic.items():

            # Create empty lists
            if name not in self.log:
                self.log[name] = [None] * len(self.log['t'])

            if self.model.t != self.log['t'][-1]:

                # Create empty slot for new documented time step
                for v in self.log.values():
                    v.append(None)

                # Store time step
                self.log['t'][-1] = self.model.t
            self.log[name][-1] = aux.rgetattr(self, codename)


    def get_position(self):
        return tuple(self.pos)

    def set_id(self, id):
        self.unique_id = id
        self.id_box.text = self.unique_id

    def get_shape(self, scale=1):
        p = self.get_position()
        return geometry.Point(p).buffer(self.radius * scale) if not np.isnan(p).all() else None

    def set_color(self, color):
        self.color = color

    def contained(self, point):
        return geometry.Point(self.get_position()).distance(geometry.Point(point)) <= self.radius

    def step(self):
        pass

    def set_default_color(self, color):
        self.default_color = color
        self.set_color(color)

    def set_odor_dist(self, intensity=None, spread=None):
        self.odor_intensity = intensity
        self.odor_spread = spread
        if intensity is not None and spread is not None:
            self.odor_dist = multivariate_normal([0, 0], [[self.odor_spread, 0], [0, self.odor_spread]])
            self.odor_peak_value = self.odor_intensity / self.odor_dist.pdf([0, 0])

    def get_gaussian_odor_value(self, pos):
        return self.odor_dist.pdf(pos) * self.odor_peak_value

    def draw(self, viewer, filled=True):
        p, c, r = self.get_position(), self.color, self.radius
        if np.isnan(p).all():
            return
        viewer.draw_circle(p, r, c, filled, r / 5)

        if self.odor_id is not None:
            if self.odor_intensity > 0:
                if self.model.screen_manager.odor_aura:
                    viewer.draw_circle(p, r * 1.5, c, False, r / 10)
                    viewer.draw_circle(p, r * 2.0, c, False, r / 15)
                    viewer.draw_circle(p, r * 3.0, c, False, r / 20)
        if self.selected:
            viewer.draw_circle(p, r * 1.1, self.model.screen_manager.selection_color, False, r / 5)



