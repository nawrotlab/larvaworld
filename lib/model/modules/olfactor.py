import numpy as np

from lib.model.modules.basic import Effector

class Sensor(Effector) :
    def __init__(self,brain,gain_dict={}, decay_coef=1, noise=0, **kwargs):
        super().__init__(**kwargs)
        self.brain = brain
        self.decay_coef = decay_coef
        self.noise = noise
        self.A0, self.A1 = [-1.0, 1.0]
        self.activation = 0

        self.init_gain(gain_dict)

    def compute_dif(self, input):
        pass

    def step(self, input):
        if len(input) == 0:
            self.activation = 0
        else:
            self.compute_dif(input)
            self.activation *= 1 - self.dt * self.decay_coef
            self.activation += self.dt * np.sum([self.gain[id] * self.dCon[id] for id in self.gain_ids])

            if self.activation > self.A1:
                self.activation = self.A1
            elif self.activation < self.A0:
                self.activation = self.A0

        return self.activation

    def init_gain(self, gain_dict):
        if gain_dict in [None, 'empty_dict']:
            gain_dict = {}
        self.base_gain = {}
        self.Con = {}

        self.dCon={}
        self.Ngains = len(gain_dict)
        self.gain_ids = list(gain_dict.keys())
        self.gain = {}

    def get_dCon(self):
        return self.dCon

    def get_gain(self):
        return self.gain

class Olfactor(Sensor):
    def __init__(self, odor_dict={}, perception='log', olfactor_noise=0, **kwargs):
        super().__init__(gain_dict=odor_dict, noise=olfactor_noise, **kwargs)

        self.perception = perception
        # self.init_gain(odor_dict)

    def set_gain(self, value, odor_id):
        self.gain[odor_id] = value

    def reset_gain(self, odor_id):
        self.gain[odor_id] = self.base_gain[odor_id]

    def reset_all_gains(self):
        self.gain = self.base_gain

    def compute_dif(self, input):
        for id, cur in input.items():
            if id not in self.gain_ids:
                self.add_novel_odor(id, con=cur, gain=0.0)
            else:
                prev = self.Con[id]
                if self.perception == 'log':
                    self.dCon[id] = cur / prev - 1 if prev != 0 else 0
                elif self.perception == 'linear':
                    self.dCon[id] = cur - prev
        self.Con = input





    def init_gain(self, gain_dict):
        if gain_dict in [None, 'empty_dict']:
            gain_dict = {}
        self.base_gain = {}
        self.Con = {}
        self.dCon = {}
        self.Ngains = len(gain_dict)
        self.gain_ids = list(gain_dict.keys())
        # print(odor_dict)
        for id, p in gain_dict.items():
            if type(p) == dict:
                m, s = p['mean'], p['std']
                self.base_gain[id] = float(np.random.normal(m, s, 1))
            else:
                self.base_gain[id] = p
            self.Con[id] = 0.0
            self.dCon[id] = 0.0
        self.gain = self.base_gain

    def add_novel_odor(self, id, con=0.0, gain=0.0):
        self.Ngains += 1
        self.gain_ids.append(id)
        self.base_gain[id] = gain
        self.gain[id] = gain
        self.dCon[id] = 0.0
        self.Con[id] = con

class Toucher(Sensor):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def compute_dif(self, input):
        for id, cur in input.items():
            prev = self.Con[id]
            self.dCon[id] = cur - prev
        self.Con = input

    def init_gain(self, gain_dict):
        if gain_dict in [None, 'empty_dict']:
            gain_dict = {}
        self.base_gain = {}
        self.Con = {}
        self.dCon = {}
        self.Ngains = len(gain_dict)
        self.gain_ids = list(gain_dict.keys())
        for id, p in gain_dict.items():
            self.base_gain[id] = p
            self.Con[id] = 0.0
            self.dCon[id] = 0.0
        self.gain = self.base_gain