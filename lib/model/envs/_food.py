import mesa
import numpy as np
from Box2D import Box2D, b2ChainShape
from scipy.stats import multivariate_normal

from lib.aux.functions import circle_to_polygon


class Food(mesa.Agent):
    # _friction = 1.0
    # _linear_damping = 15.  # * _world_scale
    # _angular_damping = 60.  # * _world_scale

    def __init__(self, unique_id, model, position,
                 shape_radius, amount=1.0,
                 odor_id=None, odor_intensity=None, odor_spread=None):
        super().__init__(unique_id=unique_id, model=model)
        # CAUTION : Applying the same scaling factor to the odor distributions
        # The rest will be set externally from the _create_odor_layers method
        self.radius = shape_radius * self.model.scaling_factor
        self.initial_amount = amount
        self.amount = self.initial_amount
        self.odor_id = odor_id
        self.odor_intensity = odor_intensity
        self.odor_spread = odor_spread
        # print(self.odor_spread, odor_intensity)

        # self.__dict__.update(_food_params)
        # if self.static_food:
        # We bypass the init function of super because it creates a dynamic body. We need a static
        shape = circle_to_polygon(60, self.radius)

        if self.model.physics_engine:
            self._body: Box2D.b2Body = self.model.space.CreateStaticBody(position=position)
            # else:
            #     # all parameters in real space units
            #     self._body: Box2D.b2Body = self.space.CreateDynamicBody(position=self.position, orientation=None)
            #     # super().__init__(space=self.space, position=self.position, orientation=None, radius=self.shape_radius)
            #     self._body.bullet = True

            self.food_shape = b2ChainShape(vertices=shape.tolist())
            self._body.CreateFixture(shape=self.food_shape)
            self._body.fixtures[0].filterData.groupIndex = -1
        else:
            self.model.space.place_agent(self, position)

        self.initial_color = np.array((100, 200, 120))
        self._color = self.initial_color

        # will be set by the environment
        self._id = None

        # # put all agents into same group (negative so that no collisions are detected)
        # self._fixtures[0].filterData.groupIndex = -1

        # self._body_color = (100, 100, 100)
        # self._highlight_color = (100, 100, 100)

    def set_id(self, id):
        self._id = id

    @property
    def id(self):
        return self._id

    def get_odor_id(self):
        return self.odor_id

    def get_gaussian_odor_value(self, pos):
        return self.odor_dist.pdf(pos) * self.odor_peak_value

    def get_position(self):
        if not self.model.physics_engine:
            return self.pos
        else:
            return np.asarray(self._body.position)

    def get_radius(self):
        return self.radius

    def get_amount(self):
        return self.amount

    def subtract_amount(self, amount):
        # print(amount, self.amount)
        prev_amount = self.amount
        self.amount -= amount
        if self.amount <= 0.0:
            self.amount=0.0
            self.model.delete(self)
        else :
            r=(self.initial_amount - self.amount)/self.initial_amount
            self._color=r*np.array((255,255,255)) + (1-r)*self.initial_color
            # self._color=self.initial_color + (self.initial_amount - self.amount)*(np.array((255,255,255))-self.initial_color)
        return np.min([amount, prev_amount])

    def get_odor_intensity(self):
        return self.odor_intensity

    def get_odor_spread(self):
        return self.odor_spread

    def set_scaled_odor_intensity(self, intensity):
        self.odor_intensity = intensity * self.model.scaling_factor

    def set_scaled_odor_spread(self, spread):
        self.odor_spread = spread * self.model.scaling_factor

    def set_odor_dist(self):
        self.odor_dist = multivariate_normal([0, 0], [[self.odor_spread, 0], [0, self.odor_spread]])
        self.odor_peak_value = self.odor_intensity / self.odor_dist.pdf([0, 0])

    def set_odor_id(self, odor_id):
        self.odor_id = odor_id

    def set_color(self, color):
        self._color = color

    # @abc.abstractmethod
    def step(self):
        # print('ff')
        pass

    def draw(self, viewer):
        viewer.draw_circle(position=self.get_position(), radius=self.radius, color=self._color)
