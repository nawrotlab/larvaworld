import random

import numpy as np
import param

from larvaworld.lib import aux
from larvaworld.lib.model.modules.basic import Effector
from larvaworld.lib.param import PositiveNumber, PositiveInteger


class NeuralOscillator(Effector):
    base_activation = PositiveNumber(20.0,bounds=(10.0, 40.0),step=1.0,precedence=1, label='baseline activation', doc='The baseline activation of the oscillator.')
    activation_range = param.Range((10.0, 40.0),bounds=(0.0, 100.0), precedence=1,label='activation range', doc='The activation range of the oscillator.')
    input_range = param.Range((-1, 1),bounds=(-1, 1), precedence=-2,label='input range', doc='The input range of the oscillator.')
    tau = param.Number(0.1,precedence=2, label='time constant', doc='The time constant of the oscillator.')
    w_ee = param.Number(3.0, label='E->E weigths', doc='The E->E synapse connection weights.')
    w_ce = param.Number(0.1, label='C->E weigths', doc='The C->E synapse connection weights.')
    w_ec = param.Number(4.0, label='E->C weigths', doc='The E->C synapse connection weights.')
    w_cc = param.Number(4.0, label='C->C weigths', doc='The C->C synapse connection weights.')
    m = PositiveInteger(100,softmax=1000, label='maximum spike-rate', doc='The maximum allowed spike rate.')
    n = PositiveNumber(2.0,softmax=10.0,step=0.1, label='spike response steepness', doc='The neuron spike-rate response steepness coefficient.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param.base_activation.bounds=self.activation_range

        # self.start_effector()

        self.r1 = self.activation_range[1] - self.base_activation
        self.r0 = self.base_activation - self.activation_range[0]
        self.activation=self.base_activation
        # self.output = self.base_activation

        # Neural populations
        self.E_r = 0  # 28
        self.H_E_r = 0  # 10

        self.E_l = 0  # 30
        self.H_E_l = 0  # 10

        self.C_r = 0
        self.H_C_r = 0  # 10

        self.C_l = 0
        self.H_C_l = 0  # 10

        self.scaled_tau = self.dt / self.tau
        self.warm_up()


    def warm_up(self):
        for i in range(100):
            if random.uniform(0, 1) < 0.5:
                self.step()

    def update(self):
        if self.input < 0:
            a = self.r0 * self.input
        elif self.input >= 0:
            a = self.r1 * self.input
        self.activation=self.base_activation + a


    # @property
    # def Act_coef(self):
    #     return 1
    def act(self):
        self.oscillate()
        self.output = self.E_r - self.E_l

    def inact(self):
        self.output =0

    def oscillate(self):
        A = self.activation
        # print(A)
        # print()
        t = self.scaled_tau
        tau_h = 3 / (1 + (0.04 * A) ** 2)
        t_h = self.dt / tau_h
        g = 6 + (0.09 * A) ** 2

        self.E_l += t * (
                -self.E_l + self.compute_R(A + self.w_ee * self.E_l - self.w_ec * self.C_r, 64 + g * self.H_E_l))
        self.E_r += t * (
                -self.E_r + self.compute_R(A + self.w_ee * self.E_r - self.w_ec * self.C_l, 64 + g * self.H_E_r))
        self.H_E_l += t_h * (-self.H_E_l + self.E_l)
        self.H_E_r += t_h * (-self.H_E_r + self.E_r)

        self.C_l += t * (
                -self.C_l + self.compute_R(A + self.w_ce * self.E_l - self.w_cc * self.C_r, 64 + g * self.H_C_l))
        self.C_r += t * (
                -self.C_r + self.compute_R(A + self.w_ce * self.E_r - self.w_cc * self.C_l, 64 + g * self.H_C_r))
        self.H_C_l += t_h * (-self.H_C_l + self.E_l)
        self.H_C_r += t_h * (-self.H_C_r + self.E_r)

    # @property
    # def Act_Phi(self):
    #
    #     return self.E_r - self.E_l

    def compute_R(self, x, h):
        if x > 0:
            r = self.m * x ** self.n / (x ** self.n + h ** self.n)
            return r
        else:
            return 0.0

    def get_state(self):
        state = [self.E_l, self.H_E_l, self.E_r, self.H_E_r, self.C_l, self.H_C_l, self.C_r, self.H_C_r]
        return state


