import param

from larvaworld.lib import aux



class Viewable(aux.NestedConf) :
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


class ViewableNamed(Viewable,aux.Named):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ViewableSingleObject(Viewable,aux.NamedObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ViewableGroupedObject(Viewable,aux.GroupedObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)





class ViewableLine(Viewable,aux.LineExtended):


    def draw(self, v, **kwargs):
        try :
           v.draw_polyline(vertices=self.vertices,color=self.color,width=self.width,closed=self.closed)
        except :
            for ver in self.vertices:
                v.draw_polyline(ver, color=self.color, width=self.width, closed=self.closed)

class ViewableBoundedArea(Viewable,aux.BoundedArea): pass



class ViewableNamedLine(ViewableLine,aux.Named): pass

class ViewableNamedBoundedArea(ViewableBoundedArea,aux.Named): pass

class ViewableCircle(Viewable,aux.RadiallyExtended):

    def draw(self, v, filled=True, radius_coeff=1, color=None, width_as_radius_fraction=5):
        if color is None :
            color = self.color
        v.draw_circle(position=self.get_position(), radius=self.radius*radius_coeff, color=color, filled=filled, width=self.radius/width_as_radius_fraction)



class SpatialEntity(ViewableSingleObject):
    default_color = param.Color(default='white')
    # visible = param.Boolean(default=False)
    # def __init__(self, visible=False,default_color='white', **kwargs):
    #     super().__init__(visible=visible,default_color=default_color,**kwargs)

    def record_positions(self, label='p'):
        """ Records the positions of each agent.

        Arguments:
            label (string, optional):
                Name under which to record each position (default p).
                A number will be added for each coordinate (e.g. p1, p2, ...).
        """
        for agent, pos in self.positions.items():
            for i, p in enumerate(pos):
                agent.record(label+str(i), p)