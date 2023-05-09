import numpy as np
import param

from larvaworld.lib import aux
from larvaworld.lib.model.modules.basic import Effector


class Sensor(Effector):
    perception = param.Selector(objects=['linear', 'log', 'null'], label='sensory transduction mode', doc='The method used to calculate the perceived sensory activation from the current and previous sensory input.')
    decay_coef = aux.PositiveNumber(1.0,softmax=2.0, step=0.01, label='sensory decay coef', doc='The linear decay coefficient of the olfactory sensory activation.')
    brute_force = param.Boolean(False, doc='Whether to apply direct rule-based modulation on locomotion or not.')

    def __init__(self, gain_dict={},output_range=(-1.0,1.0),brain=None, **kwargs):
        super().__init__(output_range=output_range, **kwargs)
        self.brain = brain
        self.interruption_counter = 0

        self.exp_decay_coef = np.exp(- self.dt * self.decay_coef)

        self.init_gain(gain_dict)

    def compute_dif(self, input):
        pass

    def update_gain(self):
        pass

    # def update_output(self,output):
    #     return self.apply_noise(output, self.output_noise, range=(self.A0,self.A1))



    def update(self):
        self.update_gain()
        if len(self.input) == 0:
            self.output = 0
        elif self.brute_force:
            self.affect_locomotion()
            self.output = 0
        else:
            self.compute_dX(self.input)
            self.output *= self.exp_decay_coef
            self.output += self.dt * np.sum([self.gain[id] * self.dX[id] for id in self.gain_ids])



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
                    self.dX[id] = cur - prev if prev != 0 else 0
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

    def update_gain(self):
        if self.brain is not None:
            if self.brain.agent is not None:
                for id in self.brain.agent.model.odor_ids:
                    if id not in self.gain_ids:
                        self.add_novel_gain(id)
                    # print(self.brain.agent.unique_id, id, 'new')

    def affect_locomotion(self):
        if self.brain is None :
            return
        L = self.brain.locomotor
        if self.output < 0 and L.crawler.complete_iteration:
            if np.random.uniform(0, 1, 1) <= np.abs(self.output):
                L.intermitter.inhibit_locomotion(L=L)
                self.interruption_counter+=1


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


class Toucher(Sensor):
    def __init__(self, initial_gain, touch_sensors=None, **kwargs):

        if touch_sensors is not None:
            gain_dict = {s: initial_gain for s in touch_sensors}

            # self.brain.agent.add_touch_sensors(touch_sensors)
        else:
            gain_dict = {}

        super().__init__(gain_dict=gain_dict, **kwargs)
        self.touch_sensors = touch_sensors
        self.init_sensors()

    def init_sensors(self):
        if self.brain is not None:
            if self.brain.agent is not None:
                self.brain.agent.touch_sensors = self.touch_sensors
                if self.touch_sensors is not None:
                    self.brain.agent.add_touch_sensors(self.touch_sensors)

    def affect_locomotion(self):
        if self.brain is None :
            return
        L = self.brain.locomotor
        for id in self.gain_ids:
            if self.dX[id] == 1:
                L.intermitter.trigger_locomotion(L=L)
                break
            elif self.dX[id] == -1:
                L.intermitter.interrupt_locomotion(L=L)
                self.interruption_counter += 1
                break


class WindSensor(Sensor):
    def __init__(self, weights, perception='null', **kwargs):
        super().__init__(perception=perception,gain_dict={'windsensor': 1.0}, **kwargs)
        self.weights = weights


# @todo add class Thermosensor(Sensor) here with a double gain dict
class Thermosensor(Sensor):
    def __init__(self, cool_gain=0.0,warm_gain=0.0, **kwargs): #thermodict={"cool", "warm"}
        thermo_dict = aux.AttrDict({'warm': warm_gain, 'cool': cool_gain})
        super().__init__(gain_dict=thermo_dict, **kwargs)



    @property
    def warm_sensor_input(self):
        return self.X['warm'] #@todo do I need to make self.thermoX.values? same for dX.

    @property
    def warm_sensor_perception(self):
        return self.dX['warm']

    @property
    def cool_sensor_input(self):
        return self.X['cool']  # @todo do I need to make self.thermoX.values? same for dX.

    @property
    def cool_sensor_perception(self):
        return self.dX['cool']
