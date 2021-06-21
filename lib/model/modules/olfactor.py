import numpy as np

from lib.model.modules.basic import Effector


class Olfactor(Effector):
    def __init__(self,
                 odor_dict={}, perception='log', decay_coef=1, olfactor_noise=0, **kwargs):
        super().__init__(**kwargs)

        self.perception = perception
        self.decay_coef = decay_coef
        self.noise = olfactor_noise
        self.A0, self.A1 = [-1.0, 1.0]
        self.activation = 0
        # self.odor_layers = odor_layers
        # self.num_layers = len(odor_layers)
        self.init_gain(odor_dict)

    def set_gain(self, value, odor_id):
        self.gain[odor_id] = value

    def reset_gain(self, odor_id):
        self.gain[odor_id] = self.base_gain[odor_id]

    def reset_all_gains(self):
        self.gain = self.base_gain

    # def get_gain(self, odor_id):
    #     return self.gain[odor_id]

    def compute_dCon(self, concentrations):
        for id, cur in concentrations.items():
            if id not in self.odor_ids:
                self.add_novel_odor(id, con=cur, gain=0.0)
            else:
                prev = self.Con[id]
                if self.perception == 'log':
                    self.dCon[id] = cur / prev - 1 if prev != 0 else 0
                elif self.perception == 'linear':
                    self.dCon[id] = cur - prev
        self.Con = concentrations

    def get_dCon(self):
        return self.dCon

    def get_gain(self):
        return self.gain

    def step(self, cons):
        # print(cons)
        if len(cons) == 0:
            self.activation = 0
        else:
            self.compute_dCon(cons)
            self.activation *= 1 - self.dt * self.decay_coef
            self.activation += self.dt * np.sum([self.gain[id] * self.dCon[id] for id in self.odor_ids])

            if self.activation > self.A1:
                self.activation = self.A1
            elif self.activation < self.A0:
                self.activation = self.A0

        return self.activation

    def init_gain(self, odor_dict):
        if odor_dict in [None, 'empty_dict']:
            odor_dict = {}
        self.base_gain = {}
        self.Con = {}
        self.dCon = {}
        self.Nodors = len(odor_dict)
        self.odor_ids = list(odor_dict.keys())
        for id, p in odor_dict.items():
            if type(p) == dict:
                m, s = p['mean'], p['std']
                self.base_gain[id] = float(np.random.normal(m, s, 1))
            else:
                self.base_gain[id] = p
            self.Con[id] = 0.0
            self.dCon[id] = 0.0
        self.gain = self.base_gain

    def add_novel_odor(self, id, con=0.0, gain=0.0):
        self.Nodors += 1
        self.odor_ids.append(id)
        self.base_gain[id] = gain
        self.gain[id] = gain
        self.dCon[id] = 0.0
        self.Con[id] = con