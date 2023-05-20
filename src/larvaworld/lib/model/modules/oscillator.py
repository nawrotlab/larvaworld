import random

import numpy as np
import param
from scipy import signal

from larvaworld.lib import aux


class Timer(param.Parameterized) :
    dt = aux.PositiveNumber(0.1, softmax=1.0, step=0.01, label='timestep', doc='The timestep of the simulation in seconds.')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ticks = 0
        self.total_ticks = 0

        self.active = True
        self.complete_iteration = False

    def count_time(self):
        self.ticks += 1
        self.total_ticks += 1

    @ property
    def t(self):
        return self.ticks * self.dt

    @property
    def total_t(self):
        return self.total_ticks * self.dt

    def reset(self):
        self.ticks = 0
        self.total_ticks = 0

    def start_effector(self):
        self.active = True

    def stop_effector(self):
        self.active = False
        self.ticks = 0


class Oscillator(Timer):
    freq = aux.PositiveNumber(label='oscillation frequency', doc='The initial frequency of the oscillator.')
    phi = aux.Phase(label='orientation', doc='The absolute orientation in space.')

    def __init__(self, random_phi=True, freq_range=None,initial_freq=None, **kwargs):
        if 'phi' not in kwargs.keys():
            kwargs['phi'] = random.uniform(0, 2 * np.pi) if random_phi else 0.0
        self.param.freq.bounds = freq_range
        super().__init__(freq=initial_freq,**kwargs)
        self.initial_freq = self.freq

        self.iteration_counter = 0
        #self.complete_iteration = False

    def set_freq(self, v):
        self.freq = v

    def get_freq(self, t):
        return self.freq

    def oscillate(self):
        self.complete_iteration = False
        phi = self.phi + 2 * np.pi * self.dt * self.freq
        if phi >= 2 * np.pi:
            phi %= 2 * np.pi
            self.complete_iteration = True
            self.act_on_complete_iteration()
            self.iteration_counter += 1
        self.phi = phi


    def act_on_complete_iteration(self):
        pass


    def reset(self):
        # self.ticks = 0
        # self.total_ticks = 0
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0

    def update(self):
        self.complete_iteration = False

    def phi_in_range(self, phi_range):
        return phi_range[0] < self.phi < phi_range[1]

    @property
    def Act_Phi(self):
        return self.phi



