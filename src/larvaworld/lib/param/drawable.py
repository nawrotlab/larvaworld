import param

from larvaworld.lib import aux
from larvaworld.lib.param import NestedConf, Named, RadiallyExtended, BoundedArea, LineExtended, LineClosed, \
    RandomizedColor, Grid


class Viewable(NestedConf) :
    '''
        Basic Parameterized Class for all visible Objects in simulation

            Args:
            - default_color: optional str or tuple representing the default color of the agent.
            - visible: optional boolean indicating whether the agent is visible or not.


    '''


    default_color = RandomizedColor(default='black', doc='The default color of the entity',instantiate=True)
    visible = param.Boolean(True, doc='Whether the entity is visible or not')
    selected = param.Boolean(False, doc='Whether the entity is selected or not')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.color = self.default_color

    def set_color(self, color):
        self.color = color

    def set_default_color(self, color):
        self.default_color = color
        self.color=color

    def invert_default_color(self):
        c00, c01 = aux.invert_color(self.default_color)
        self.set_default_color(c01)


    def _draw(self,v,**kwargs):
        if self.visible :
            self.draw(v,**kwargs)
            if self.selected:
                # raise
                self.draw_selected(v, **kwargs)


    def draw_selected(self, v, **kwargs):
        pass

    def draw(self, v, **kwargs):
        pass

    #@property
    def toggle_vis(self):
        self.visible = not self.visible

class ViewableToggleable(Viewable):
    active = param.Boolean(False, doc='Whether entity is active')
    active_color = param.Color('lightblue',doc='The color of the entity when active')
    inactive_color = param.Color('lightgreen',doc='The color of the entity when inactive')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.active_color is None:
            self.active_color = self.default_color
        if self.inactive_color is None:
            self.inactive_color = self.default_color
        self.update_color()

    @param.depends('active', watch=True)
    def update_color(self):
        self.color= self.active_color if self.active else self.inactive_color


    def toggle(self):
        self.active = not self.active


class ViewableNamed(Viewable,Named): pass







class ViewableLine(Viewable,LineExtended):



    def draw(self, v, **kwargs):
        try :
           v.draw_polyline(vertices=self.vertices,color=self.color,width=self.width,closed=self.closed)
        except :
            for ver in self.vertices:
                v.draw_polyline(ver, color=self.color, width=self.width, closed=self.closed)


class Contour(Viewable,LineClosed):

    def draw(self, v, **kwargs):
        v.draw_polygon(self.vertices, filled=True, color=self.color)


class ViewableNamedBoundedArea(Viewable,BoundedArea,Named): pass

class ViewableNamedGrid(Viewable,Grid,Named): pass

class ViewableCircle(Viewable,RadiallyExtended):

    def draw(self, v, filled=True, radius_coeff=1, color=None, width_as_radius_fraction=5):
        if color is None :
            color = self.color
        v.draw_circle(position=self.get_position(), radius=self.radius*radius_coeff, color=color, filled=filled, width=self.radius/width_as_radius_fraction)



