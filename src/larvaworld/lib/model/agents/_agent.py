import agentpy
import numpy as np
import param
from shapely import geometry


from larvaworld.lib import aux
from larvaworld.lib.model.object import ModelEntity
from larvaworld.lib.model.composition import Odor



class LarvaworldAgent(ModelEntity):
    """LarvaworldAgent class that inherits from agentpy.Agent."""
    radius = aux.PositiveNumber(0.003, softmax=0.1, step=0.001,doc='The spatial radius of the source in meters')
    pos = param.NumericTuple((0.0, 0.0), doc='The xy spatial position coordinates')
    odor = aux.ClassAttr(Odor, doc='The odor of the agent')


    def __init__(self, model=None,odor={},  regeneration=False, regeneration_pos=None, **kwargs):
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


        super().__init__(model=model,odor=Odor(**odor),**kwargs)
        self.base_odor_id = f'{self.group}_base_odor'
        self.gain_for_base_odor = 100

        # self.odor_id = self.odor.id
        # self.set_odor_dist(self.odor.intensity, self.odor.spread)

        self.regeneration = regeneration
        self.regeneration_pos = regeneration_pos
        self.setup(**kwargs)


    def get_position(self):
        return tuple(self.pos)



    def get_shape(self, scale=1):
        p = self.get_position()
        return geometry.Point(p).buffer(self.radius * scale) if not np.isnan(p).all() else None



    def contained(self, point):
        return geometry.Point(self.get_position()).distance(geometry.Point(point)) <= self.radius

    def step(self):
        pass



    def draw(self, viewer, filled=True):
        p, c, r = self.get_position(), self.color, self.radius
        if np.isnan(p).all():
            return
        viewer.draw_circle(p, r, c, filled, r / 5)

        if self.odor.peak_value > 0:
            if self.model.screen_manager.odor_aura:
                viewer.draw_circle(p, r * 1.5, c, False, r / 10)
                viewer.draw_circle(p, r * 2.0, c, False, r / 15)
                viewer.draw_circle(p, r * 3.0, c, False, r / 20)
        if self.selected:
            viewer.draw_circle(p, r * 1.1, self.model.screen_manager.selection_color, False, r / 5)



