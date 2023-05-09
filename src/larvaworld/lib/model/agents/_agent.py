import agentpy
import numpy as np
import param
from shapely import geometry


from larvaworld.lib import aux
from larvaworld.lib.model.drawable import LabelledGroupedObject
from larvaworld.lib.model.composition import Odor
from larvaworld.lib.model.spatial import RadiallyExtended, OrientedPoint


class NonSpatialAgent(LabelledGroupedObject):
    """
                LarvaworldAgent base class for all agent types

                Note that the setup() method is called right after initialization as in the agentpy.Agent class
                This is contrary to the parent class

                Args:
                - odor: optional dictionary containing odor information of the agent.


            """


    odor = aux.ClassAttr(Odor, doc='The odor of the agent')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.setup(**kwargs)

    @property
    def dt(self):
        return self.model.dt

    def step(self):
        pass



class PointAgent(RadiallyExtended,NonSpatialAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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


class OrientedAgent(OrientedPoint,NonSpatialAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



def AgentConf(agent_class, mode='Group') :
    class A(aux.NestedConf):
        def __init__(self, **kwargs):
            if hasattr(A,'distribution'):
                D=A.distribution.__class__
                ks=list(D.param.objects().keys())
                existing=[k for k in ks if k in kwargs.keys()]
                if len(existing)>0:
                    d={}
                    for k in existing :
                        d[k]=kwargs[k]
                        kwargs.pop(k)
                    kwargs['distribution']=D(**d)
            if 'c' in kwargs.keys():
                kwargs['default_color']=kwargs['c']
                kwargs.pop('c')
            if 'or' in kwargs.keys():
                kwargs['orientation']=kwargs['or']
                kwargs.pop('or')
            if 'id' in kwargs.keys():
                kwargs['unique_id']=kwargs['id']
                kwargs.pop('id')
            if 'r' in kwargs.keys():
                kwargs['radius']=kwargs['r']
                kwargs.pop('r')
            if 'a' in kwargs.keys():
                kwargs['amount']=kwargs['a']
                kwargs.pop('a')
            if 'o' in kwargs.keys():
                assert 'odor' not in kwargs.keys()
                assert len(kwargs['o'])==3
                kwargs['odor']=dict(zip(['id', 'intensity','spread'], kwargs['o']))
                kwargs.pop('o')
            if 'sub' in kwargs.keys():
                assert 'substrate' not in kwargs.keys()
                assert len(kwargs['sub'])==2
                kwargs['substrate']=dict(zip(['quality', 'type'], kwargs['sub']))
                kwargs.pop('sub')

            super().__init__(**kwargs)


    A.__name__=f'{agent_class.__name__}{mode}'
    invalid = ['name', 'closed', 'visible']
    if mode=='Group':
        if issubclass(agent_class, PointAgent):
            distro=aux.Spatial_Distro
        elif issubclass(agent_class, OrientedAgent):
            distro = aux.Larva_Distro

        A.param._add_parameter('distribution',aux.ClassAttr(distro, doc='The spatial distribution of the group agents'))
        invalid+=['unique_id', 'pos', 'orientation']
    elif mode=='Unit':
        pass
    for k, p in agent_class.param.params().items():
        if k not in invalid:
            A.param._add_parameter(k,p)
    return A


