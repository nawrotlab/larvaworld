import numpy as np
from Box2D import Box2D, b2ChainShape
from scipy.stats import multivariate_normal
from matplotlib.patches import Circle

from lib.aux.functions import circle_to_polygon
from lib.aux.rendering import InputBox
from lib.model.larva._larva import LarvaworldAgent


class Food(LarvaworldAgent):
    # _friction = 1.0
    # _linear_damping = 15.  # * _world_scale
    # _angular_damping = 60.  # * _world_scale

    def __init__(self, unique_id, model, position,
                 shape_radius, amount=1.0, quality=1,
                 odor_id=None, odor_intensity=None, odor_spread=None):

        super().__init__(unique_id=unique_id, model=model)
        # CAUTION : Applying the same scaling factor to the odor distributions
        # The rest will be set externally from the _create_odor_layers method
        self.default_color = np.array((100, 200, 120))
        self.radius = shape_radius * self.model.scaling_factor
        self.initial_amount = amount
        self.quality = quality
        self.amount = self.initial_amount
        self.odor_id = odor_id
        self.odor_intensity = odor_intensity
        self.odor_spread = odor_spread
        if self.odor_intensity>0 :
            self.set_odor_dist()
        # print(self.odor_spread, odor_intensity)

        # self.__dict__.update(_food_params)
        # if self.static_food:
        # We bypass the init function of super because it creates a dynamic body. We need a static
        shape = circle_to_polygon(60, self.radius)

        if self.model.physics_engine:
            self._body: Box2D.b2Body = self.model.space.CreateStaticBody(position=position)
            # else:
            #     # all parameters in real space units
            #     self._body: Box2D.b2Body = self.space.CreateDynamicBody(pos=self.pos, orientation=None)
            #     # super().__init__(space=self.space, pos=self.pos, orientation=None, radius=self.shape_radius)
            #     self._body.bullet = True

            self.food_shape = b2ChainShape(vertices=shape.tolist())
            self._body.CreateFixture(shape=self.food_shape)
            self._body.fixtures[0].filterData.groupIndex = -1
        else:
            self.model.space.place_agent(self, position)


        self._color = self.default_color

        # will be set by the environment
        self._id = None
        self.circle = Circle(self.get_position(), radius=self.radius)
        # # put all agents into same group (negative so that no collisions are detected)
        # self._fixtures[0].filterData.groupIndex = -1

        # self._body_color = (100, 100, 100)
        # self._highlight_color = (100, 100, 100)
        self.id_box = InputBox(visible=False, text=self.unique_id,
                               color_inactive=self.default_color, color_active=self.default_color,
                               screen_pos=self.model.space2screen_pos(self.get_position()),
                               linewidth=1)



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
            self._color=r*np.array((255,255,255)) + (1-r)*self.default_color
            # self._color=self.default_color + (self.initial_amount - self.amount)*(np.array((255,255,255))-self.default_color)
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

    def set_odor(self, id, intensity=2, spread=0.0002):
        self.set_odor_id(id)
        self.set_scaled_odor_intensity(intensity)
        self.set_scaled_odor_spread(spread)
        self.set_odor_dist()

    def set_color(self, color):
        self._color = color

    # @abc.abstractmethod
    def step(self):
        # print('ff')
        pass

    def draw(self, viewer):
        self.id_box.set_shape(self.model.space2screen_pos(self.get_position()))
        if self.amount>0:
            filled=True
        else :
            filled=False
        viewer.draw_circle(position=self.get_position(), radius=self.radius, color=self._color, filled=filled, width=self.radius/5)
        if self.selected :
            viewer.draw_circle(position=self.get_position(), radius=self.radius+self.radius / 5, color=self.model.selection_color, filled=False,
                               width=self.radius / 5)


