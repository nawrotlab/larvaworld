import numpy as np

from lib.model.modules.basic import Effector


class Sensor(Effector):
    def __init__(self, brain, perception='linear', gain_dict={}, decay_coef=1, input_noise=0,
                 brute_force=False, **kwargs):
        super().__init__(**kwargs)
        self.brain = brain
        self.perception = perception
        self.decay_coef = decay_coef
        self.noise = input_noise
        self.brute_force = brute_force
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
            self.compute_dX(input)
            if self.brute_force :
                self.affect_locomotion()
            self.activation *= self.exp_decay_coef
            self.activation += self.dt * np.sum([self.gain[id] * self.dX[id] for id in self.gain_ids])
            if self.activation > self.A1:
                self.activation = self.A1
            elif self.activation < self.A0:
                self.activation = self.A0
        return self.activation

    def affect_locomotion(self):
        pass

    def init_gain(self, gain_dict):
        if gain_dict in [None, 'empty_dict']:
            gain_dict = {}
        self.base_gain = {}
        self.X = {}
        self.dX = {}
        self.Ngains = len(gain_dict)
        self.gain_ids = list(gain_dict.keys())
        for id, p in gain_dict.items():
            if isinstance(p, dict):
                m, s = p['mean'], p['std']
                self.base_gain[id] = float(np.random.normal(m, s, 1))
            else:
                self.base_gain[id] = p
            self.X[id] = 0.0
            self.dX[id] = 0.0
        self.gain = self.base_gain

    def get_dX(self):
        return self.dX

    def get_X_values(self, t, N):
        return list(self.X.values())

    def get_gain(self):
        return self.gain

    def get_activation(self, t):
        return self.activation

    def set_gain(self, value, gain_id):
        self.gain[gain_id] = value

    def reset_gain(self, gain_id):
        self.gain[gain_id] = self.base_gain[gain_id]

    def reset_all_gains(self):
        self.gain = self.base_gain

    def compute_dX(self, input):
        for id, cur in input.items():
            if id not in self.gain_ids:
                self.add_novel_gain(id, con=cur, gain=0.0)
            else:
                prev = self.X[id]
                if self.perception == 'log':
                    self.dX[id] = cur / prev - 1 if prev != 0 else 0
                elif self.perception == 'linear':
                    self.dX[id] = cur - prev
                elif self.perception == 'null':
                    self.dX[id] = cur
        self.X = input

    def add_novel_gain(self, id, con=0.0, gain=0.0):
        self.Ngains += 1
        self.gain_ids.append(id)
        self.base_gain[id] = gain
        self.gain[id] = gain
        self.dX[id] = 0.0
        self.X[id] = con


class Olfactor(Sensor):
    def __init__(self, odor_dict={}, **kwargs):
        super().__init__(gain_dict=odor_dict, **kwargs)
        for id in self.brain.agent.model.odor_ids:
            if id not in self.gain_ids:
                self.add_novel_gain(id)
            # except:
            #     pass

    def affect_locomotion(self):
        if self.activation<0:
            self.brain.intermitter.inhibit_locomotion()
        elif self.activation>0:
            self.brain.intermitter.trigger_locomotion()

    @property
    def first_odor_concentration(self):
        return list(self.X.values())[0]

    @property
    def second_odor_concentration(self):
        return list(self.X.values())[1]

    @property
    def first_odor_concentration_change(self):
        return list(self.dX.values())[0]

    @property
    def second_odor_concentration_change(self):
        return list(self.dX.values())[1]


# @todo add class Thermosensor(Sensor) here with a double gain dict
class Thermosensor(Sensor):
    def __init__(self, thermo_dict={}, **kwargs): #thermodict={"cool", "warm"}
        super().__init__(gain_dict=thermo_dict, **kwargs)
        # for id in self.brain.agent.model.thermo:
        #     if id not in self.gain_ids:
        #         self.add_novel_gain(id)
            # except:
            #     pass

    def affect_locomotion(self):
        if self.activation<0:
            self.brain.intermitter.inhibit_locomotion()
        elif self.activation>0:
            self.brain.intermitter.trigger_locomotion()

    @property
    def detected_temperature(self):
        return list(self.X.values())[0] #@todo do I need to make self.thermoX.values? same for dX.

    @property
    def detected_temperature_change(self):
        return list(self.dX.values())[0]



class Toucher(Sensor):
    def __init__(self,initial_gain, brain, **kwargs):
        gain_dict = {s: initial_gain for s in brain.agent.get_sensors()}
        super().__init__(brain=brain, gain_dict=gain_dict, **kwargs)

    def affect_locomotion(self):
        for id in self.gain_ids:
            if self.dX[id]==1:
                self.brain.intermitter.trigger_locomotion()
                break
            elif self.dX[id]==-1:
                self.brain.intermitter.interrupt_locomotion()
                break

class WindSensor(Sensor):
    def __init__(self,weights, perception='null', **kwargs):
        super().__init__(perception=perception, **kwargs)
        self.weights=weights

