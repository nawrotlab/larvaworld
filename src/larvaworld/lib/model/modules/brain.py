import numpy as np

from larvaworld.lib.model.modules.locomotor import DefaultLocomotor
from larvaworld.lib import reg, aux


class Brain:
    def __init__(self, agent=None, dt=None):
        self.agent = agent
        self.A_olf = 0
        self.A_touch = 0
        self.A_thermo = 0
        self.A_wind = 0

        # A dictionary of the possibly existing sensors along with the sensing functions and the possibly existing memory modules
        self.sensor_dict = aux.AttrDict({
            'olfactor': {'func': self.sense_odors, 'A': 0.0, 'mem': 'memory'},
            'toucher': {'func': self.sense_food, 'A': 0.0, 'mem': 'touch_memory'},
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
            v = layer.get_value(pos)
            cons[id] = v + np.random.normal(scale=v * self.olfactor.noise)
        return cons

    def sense_food(self,**kwargs):
        a = self.agent
        if a is None:
            return {}
        sensors = a.get_sensors()
        return {s: int(a.detect_food(a.get_sensor_position(s))[0] is not None) for s in sensors}

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

    def sense(self, reward=False, **kwargs):
        if self.olfactor :
            if self.memory:
                dx = self.olfactor.get_dX()
                self.olfactor.gain = self.memory.step(dx, reward)
            self.A_olf = self.olfactor.step(self.sense_odors(**kwargs), brain=self)
        if self.toucher :
            if self.touch_memory:
                dx = self.toucher.get_dX()
                self.toucher.gain = self.touch_memory.step(dx, reward)
            self.A_touch = self.toucher.step(self.sense_food(**kwargs), brain=self)
        if self.thermosensor :
            self.A_thermo = self.thermosensor.step(self.sense_thermo(**kwargs), brain=self)
        if self.windsensor :
            self.A_wind = self.windsensor.step(self.sense_wind(**kwargs), brain=self)


        # for k in self.sensor_dict.keys():
        #     sensor = getattr(self, k)
        #     if sensor:
        #         mem = self.sensor_dict[k]['mem']
        #         if mem is not None:
        #             sensor_memory = getattr(self, mem)
        #             if sensor_memory:
        #                 dx = sensor.get_dX()
        #                 sensor.gain = sensor_memory.step(dx, reward)
        #
        #         func = self.sensor_dict[k]['func']
        #         self.sensor_dict[k]['A'] = sensor.step(func(**kwargs), brain=self)

    @ property
    def A_in(self):
        return self.A_olf + self.A_touch + self.A_thermo + self.A_wind
        # return np.sum([v['A'] for v in self.sensor_dict.values()])


class DefaultBrain(Brain):
    def __init__(self, conf, agent=None, dt=None, **kwargs):
        super().__init__(agent=agent, dt=dt)
        self.locomotor = DefaultLocomotor(dt=self.dt, conf=conf, **kwargs)

    # def init_brain(self, conf, B):
        D = reg.model.dict.model.m
        for k in ['olfactor', 'toucher', 'windsensor', 'thermosensor']:
            if conf.modules[k]:
                m = conf[f'{k}_params']
                if k == 'windsensor':
                    m.gain_dict = {'windsensor': 1.0}
                mode = 'default'
                kws = {kw: getattr(self, kw) for kw in D[k].kwargs.keys()}
                M = D[k].mode[mode].class_func(**m, **kws)
                if k == 'toucher':
                    M.init_sensors(brain=self)


            else:
                M = None
            setattr(self, k, M)
        self.touch_memory = None
        self.memory = None
        if conf.modules['memory']:
            mm = conf['memory_params']
            # modality = mm['modality']
            mode = mm['mode']
            kws = {"brain" : self, "dt" : self.dt}
            # kws = {kw: getattr(self, kw) for kw in D['memory'].kwargs.keys()}
            if self.olfactor:
            # if modality == 'olfaction' and self.olfactor:
                mm.gain = self.olfactor.gain
                self.memory = D['memory'].mode[mode].class_func(**mm, **kws)
            if self.toucher:
                mm.gain = self.toucher.gain
                self.touch_memory = D['memory'].mode[mode].class_func(**mm, **kws)
        # return B


        # if m.memory and c.memory_params.modality == 'olfaction':
        #     mode = c.memory_params.mode if 'mode' in c.memory_params.keys() else 'RL'
        #     if mode == 'RL':
        #         self.memory = RLOlfMemory(brain=self, dt=self.dt, gain=self.olfactor.gain, **c['memory_params'])
        #         # raise
        #     elif mode == 'MB':
        #         # raise
        #         self.memory = RemoteBrianModelMemory(sim_id=self.agent.model.id, brain=self, dt=self.dt, gain=self.olfactor.gain,**c['memory_params'])
        #
        # if m['toucher']:
        #     t = self.toucher = Toucher(brain=self, dt=self.dt, **c['toucher_params'])
        # if m.memory and c.memory_params.modality == 'touch':
        #     self.touch_memory = RLTouchMemory(brain=self, dt=self.dt, gain=t.gain, **c['memory_params'])



    def step(self, pos=(0.0, 0.0), reward=False, **kwargs):
        self.sense(pos=pos, reward=reward)

        length = self.agent.real_length if self.agent is not None else 1
        return self.locomotor.step(A_in=self.A_in, length=length)
