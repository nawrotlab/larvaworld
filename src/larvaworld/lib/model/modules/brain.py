import numpy as np

from larvaworld.lib import reg, aux
from larvaworld.lib.model import modules


class Brain:
    def __init__(self, agent=None, dt=None):
        self.agent = agent
        self.A_olf = 0
        self.A_touch = 0
        self.A_thermo = 0
        self.A_wind = 0
        self.olfactor, self.toucher, self.windsensor, self.thermosensor = [None]*4
        # A dictionary of the possibly existing sensors along with the sensing functions and the possibly existing memory modules
        self.sensor_dict = aux.AttrDict({
            'olfactor': {'func': self.sense_odors, 'A': 0.0, 'mem': 'memory'},
            'toucher': {'func': self.sense_food_multi, 'A': 0.0, 'mem': 'touch_memory'},
            'thermosensor': {'func': self.sense_thermo, 'A': 0.0, 'mem': None},
            'windsensor': {'func': self.sense_wind, 'A': 0.0, 'mem': None}
        })

        if dt is None:
            dt = self.agent.model.dt
        self.dt = dt

    def sense_odors(self, pos=None):
        if self.agent is None:
            return {}
        if pos is None:
            pos = self.agent.olfactor_pos

        cons = {}
        for id, layer in self.agent.model.odor_layers.items():
            cons[id] = layer.get_value(pos)

        return cons

    def sense_food_multi(self,**kwargs):
        a = self.agent
        if a is None:
            return {}
        kws={
            'sources' : a.model.sources, 'grid' : a.model.food_grid, 'radius' : a.radius
        }
        return {s: int(aux.sense_food(pos=a.get_sensor_position(s), **kws) is not None) for s in list(a.sensors.keys())}




    def sense_wind(self,**kwargs):
        if self.agent is None:
            v = 0.0
        else:
            w = self.agent.model.windscape
            if w is None:
                v = 0.0
            else:
                v = w.get_value(self.agent)
        return {'windsensor': v}

    def sense_thermo(self, pos=None):
        a = self.agent
        if a is None:
            return {'cool': 0, 'warm': 0}
        if pos is None:
            pos = a.pos
        ad = a.model.space.dims
        pos_adj = [(pos[0] + (ad[0] * 0.5)) / ad[0], (pos[1] + (ad[1] * 0.5)) / ad[1]]
        try:
            cons = a.model.thermoscape.get_thermo_value(pos_adj)
        except AttributeError:
            return {'cool': 0, 'warm': 0}
        return cons






    @ property
    def A_in(self):
        return self.A_olf + self.A_touch + self.A_thermo + self.A_wind


class DefaultBrain(Brain):
    def __init__(self, conf,agent=None, **kwargs):
        super().__init__(agent=agent)
        self.locomotor = modules.DefaultLocomotor(conf=conf, **kwargs)

        kws = {"brain": self, "dt": self.dt}
        self.touch_memory = None
        self.memory = None

        mods = conf.modules
        memory_modes = {
            'RL': modules.RLOlfMemory,
            'MB': modules.RemoteBrianModelMemory,
            'touchRL': modules.RLTouchMemory,
        }
        if mods['memory']:
            mm = conf['memory_params']
            class_func=memory_modes[mm['mode']]




        if mods.olfactor:
            self.olfactor=modules.Olfactor(**kws,**conf['olfactor_params'])
            if mods['memory']:
                mm.gain = self.olfactor.gain
                self.memory = class_func(**mm, **kws)
        if mods.toucher:
            self.toucher=modules.Toucher(**kws,**conf['toucher_params'])
            self.toucher.init_sensors()
            if mods['memory']:
                mm.gain = self.toucher.gain
                self.touch_memory = class_func(**mm, **kws)
        if mods.windsensor:
            self.windsensor=modules.WindSensor(**kws,**conf['windsensor_params'])
        if mods.thermosensor:
            self.thermosensor=modules.Thermosensor(**kws,**conf['thermosensor_params'])





    def sense(self, pos=None, reward=False):

        if self.olfactor :
            if self.memory:
                dx = self.olfactor.get_dX()
                self.olfactor.gain = self.memory.step(dx, reward)
            self.A_olf = self.olfactor.step(self.sense_odors(pos))
        if self.toucher :
            if self.touch_memory:
                dx = self.toucher.get_dX()
                self.toucher.gain = self.touch_memory.step(dx, reward)
            self.A_touch = self.toucher.step(self.sense_food_multi())
        if self.thermosensor :
            self.A_thermo = self.thermosensor.step(self.sense_thermo(pos))
        if self.windsensor :
            self.A_wind = self.windsensor.step(self.sense_wind())

    def step(self, pos, length, on_food=False, **kwargs):
        self.sense(pos=pos, reward=on_food)
        return self.locomotor.step(A_in=self.A_in, length=length, on_food=on_food)
