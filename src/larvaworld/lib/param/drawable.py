import param

from larvaworld.lib import aux
from larvaworld.lib.param import NestedConf, Named, RadiallyExtended, BoundedArea, LineExtended, LineClosed


class Viewable(NestedConf) :
    '''
        Basic Parameterized Class for all visible Objects in simulation

            Args:
            - default_color: optional str or tuple representing the default color of the agent.
            - visible: optional boolean indicating whether the agent is visible or not.


    '''


    default_color = param.Color('black', doc='The default color of the entity',instantiate=True)
    visible = param.Boolean(True, doc='Whether the entity is visible or not')

    def __init__(self,default_color ='black',**kwargs):
        if isinstance(default_color,tuple):
            default_color=aux.colortuple2str(default_color)
        super().__init__(default_color =default_color,**kwargs)
        self.color = self.default_color

    def set_color(self, color):
        self.color = color

    def set_default_color(self, color):
        self.default_color = color
        self.color=color

    def _draw(self,v,**kwargs):
        if self.visible :
            self.draw(v,**kwargs)

    def draw(self, v, **kwargs):
        pass

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
        if self.vertices is not None and len(self.vertices)>1:
            v.draw_polygon(self.vertices, filled=True, color=self.color)


class ViewableNamedBoundedArea(Viewable,BoundedArea,Named): pass

class ViewableCircle(Viewable,RadiallyExtended):

    def draw(self, v, filled=True, radius_coeff=1, color=None, width_as_radius_fraction=5):
        if color is None :
            color = self.color
        v.draw_circle(position=self.get_position(), radius=self.radius*radius_coeff, color=color, filled=filled, width=self.radius/width_as_radius_fraction)



