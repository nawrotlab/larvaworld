import random
import numpy as np

from lib.model.modules.locomotor import Locomotor, DefaultLocomotor
from lib.model.modules.memory import RLOlfMemory, RLTouchMemory
from lib.model.modules.sensor import Olfactor, Thermosensor, Toucher, WindSensor


class Brain():
    def __init__(self, conf, agent=None, dt=None):
        self.conf = conf
        self.agent = agent
        self.modules = conf.modules

        self.olfactory_activation = 0
        self.touch_activation = 0
        self.wind_activation = 0
        self.thermo_activation = 0
        self.olfactor, self.memory, self.toucher, self.touch_memory, self.windsensor, self.thermosensor = [None] * 6

        if dt is None:
            dt = self.agent.model.dt
        self.dt = dt
        m = self.modules
        c = self.conf
        if m['windsensor']:
            self.windsensor = WindSensor(brain=self, dt=dt, gain_dict={'windsensor': 1.0}, **c['windsensor_params'])
        if m['olfactor']:
            self.olfactor = Olfactor(brain=self, dt=dt, **c['olfactor_params'])
        if m['thermosensor']:
            self.thermosensor = Thermosensor(brain=self, dt=dt, **c['thermosensor_params'])

        # self.crawler, self.turner, self.feeder, self.olfactor, self.intermitter = None, None, None, None, None

    @ property
    def activation(self):
        return self.touch_activation + self.wind_activation + self.olfactory_activation + self.thermo_activation

    def sense_odors(self, pos=None):
        if pos is None:
            pos = self.agent.pos
        cons = {}
        for id, layer in self.agent.model.odor_layers.items():
            v = layer.get_value(pos)
            cons[id] = v + np.random.normal(scale=v * self.olfactor.noise)
        return cons

    def sense_food(self):
        a = self.agent
        sensors = a.get_sensors()
        return {s: int(a.detect_food(a.get_sensor_position(s))[0] is not None) for s in sensors}

    def sense_wind(self):
        from lib.aux.ang_aux import angle_dif
        w = self.agent.model.windscape
        if w is None:
            v = 0.0
        else:
            v = w.get_value(self.agent)
            # wo, wv = w['wind_direction'], w['wind_speed']
            # if a.wind_obstructed(wo):
            #     v = 0
            # else:
            #     o = np.rad2deg(a.head.get_orientation())
            #     v = np.abs(angle_dif(o, wo)) / 180 * wv
        return {'windsensor': v}

    def sense_thermo(self, pos=None):
        if pos is None:
            pos = self.agent.pos
        # cons = {}
        # made it so dimensions are automatically put in here - as arena is xxy long with 0,0 centre (bottom left being -0.5x,-0.5y). thermoscape is 0.5,0.5 in the centre, and bottom left is 0,0 and top right is 1,1)
        ad = self.agent.model.arena_dims
        pos_adj = [(pos[0]+(ad[0]*0.5))/ad[0], (pos[1]+(ad[1]*0.5))/ad[1]]
        try:
            cons = self.agent.model.thermo_layers.get_thermo_value(pos_adj)
        except AttributeError:
            return {'cool': 0, 'warm': 0}
        # for id, layer in self.agent.model.odor_layers.items():
        #     v = layer.get_value(pos)
        #     cons[id] = v + np.random.normal(scale=v * self.olfactor.noise)
        # print("Sensing temp")
        
        # print(f'{cons} and {pos_adj}')
        return cons

class DefaultBrain(Brain):
    def __init__(self, conf, agent=None,dt=None,**kwargs):
        super().__init__(conf=conf, agent=agent, dt=dt)
        m = self.modules
        c = self.conf

        self.locomotor=DefaultLocomotor(dt=self.dt, conf=self.conf,**kwargs)

        if m['memory'] and c['memory_params']['modality'] == 'olfaction':
            self.memory = RLOlfMemory(brain=self, dt=self.dt, gain=self.olfactor.gain, **c['memory_params'])
        if m['toucher']:
            t = self.toucher = Toucher(brain=self, dt=self.dt, **c['toucher_params'])
        if m['memory'] and c['memory_params']['modality'] == 'touch':
            self.touch_memory = RLTouchMemory(brain=self, dt=self.dt, gain=t.gain, **c['memory_params'])

    def step(self, pos, reward=False,**kwargs):
        if self.memory:
            self.olfactor.gain = self.memory.step(self.olfactor.get_dX(), reward)
        if self.olfactor:
            self.olfactory_activation = self.olfactor.step(self.sense_odors(pos))
        if self.touch_memory:
            self.toucher.gain = self.touch_memory.step(self.toucher.get_dX(), reward)
        if self.toucher:
            self.touch_activation = self.toucher.step(self.sense_food())
        if self.windsensor:
            self.wind_activation = self.windsensor.step(self.sense_wind())
        if self.thermosensor:
            # print("Have a thermoactivation!")
            self.thermo_activation = self.thermosensor.step(self.sense_thermo())
        # A_in=self.touch_activation + self.wind_activation + self.olfactory_activation
        # print(self.activation)
        return self.locomotor.step(A_in=self.activation, length = self.agent.real_length)