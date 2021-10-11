import numpy as np

from lib.model.modules.basic import Effector


class Sensor(Effector):
    def __init__(self, brain, perception='linear', gain_dict={}, decay_coef=1, input_noise=0, **kwargs):
        super().__init__(**kwargs)
        self.brain = brain
        self.perception = perception
        self.decay_coef = decay_coef
        self.noise = input_noise
        self.A0, self.A1 = [-1.0, 1.0]
        self.activation = 0
        self.exp_decay_coef = np.exp(- self.dt * self.decay_coef)

        self.init_gain(gain_dict)

    def compute_dif(self, input):
        pass

    def step(self, input):
        if len(input) == 0:
            self.activation = 0
        else:
            self.compute_dif(input)
            self.activation *= self.exp_decay_coef
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

    def get_dCon(self):
        return self.dCon

    def get_gain(self):
        return self.gain

    def set_gain(self, value, gain_id):
        self.gain[gain_id] = value

    def reset_gain(self, gain_id):
        self.gain[gain_id] = self.base_gain[gain_id]

    def reset_all_gains(self):
        self.gain = self.base_gain

    def compute_dif(self, input):
        for id, cur in input.items():
            if id not in self.gain_ids:
                self.add_novel_gain(id, con=cur, gain=0.0)
            else:
                prev = self.Con[id]
                if self.perception == 'log':
                    self.dCon[id] = cur / prev - 1 if prev != 0 else 0
                elif self.perception == 'linear':
                    self.dCon[id] = cur - prev
        self.Con = input

    def add_novel_gain(self, id, con=0.0, gain=0.0):
        self.Ngains += 1
        self.gain_ids.append(id)
        self.base_gain[id] = gain
        self.gain[id] = gain
        self.dCon[id] = 0.0
        self.Con[id] = con


class Olfactor(Sensor):
    def __init__(self, odor_dict={}, **kwargs):
        super().__init__(gain_dict=odor_dict, **kwargs)

class Toucher(Sensor):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

