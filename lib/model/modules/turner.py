import random

import numpy as np

from lib.model.modules.basic import Effector, Oscillator, ConEffector, StepOscillator, StepEffector
from lib.registry.pars import preg


class NeuralOscillator(StepEffector):
    def __init__(self, base_activation=20, activation_range=(10.0, 40.0), tau=0.1, w_ee=3.0, w_ce=0.1, w_ec=4.0,
                 w_cc=4.0, m=100.0, n=2.0, warm_up=True, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.w_ee = w_ee
        self.w_ce = w_ce
        self.w_ec = w_ec
        self.w_cc = w_cc
        self.m = m
        self.n = n

        self.base_activation = base_activation
        self.r1 = activation_range[1] - base_activation
        self.r0 = base_activation - activation_range[0]
        # self.activation = self.base_activation

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

        if warm_up:
            for i in range(100):
                if random.uniform(0, 1) < 0.5:
                    self.step()

    def update_input(self,A_in=0):
        if A_in == 0:
            a = 0
        elif A_in < 0:
            a = self.r0 * A_in
        elif A_in > 0:
            a = self.r1 * A_in
        return self.base_activation + a

    @property
    def Act_Phi(self):
        A=self.input
        # print(A)
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
        a = self.E_r - self.E_l
        return a

    def compute_R(self, x, h):
        if x > 0:
            r = self.m * x ** self.n / (x ** self.n + h ** self.n)
            return r
        else:
            return 0.0

    def get_state(self):
        state = [self.E_l, self.H_E_l, self.E_r, self.H_E_r, self.C_l, self.H_C_l, self.C_r, self.H_C_r]
        return state




class Turner:
    def __init__(self, mode='neural', continuous=True, rebound=False, dt=0.1,
                 **kwargs):
        D = preg.larva_conf_dict
        self.continuous = continuous
        self.rebound = rebound
        self.buildup = 0
        self.ef0 = D.mdicts2['turner'].mode[mode].class_func(**kwargs, dt=dt)
        # self.ef0.start_effector()

    def step(self, A_in=0.0):
        self.activation = A_in
        self.activity = self.ef0.step(A_in)
        return self.activity


if __name__ == '__main__':
    O = NeuralOscillator(dt=0.1)
    N = 10000
    a = np.zeros([N, 8]) * np.nan
    for i in range(N):
        a[i, :] = O.get_state()
        O.step(A=20)
    print(a.shape)
    Nbins = 1000
    import pyinform

    a0, nbins, l = pyinform.utils.bin_series(a, b=Nbins)
    print(a0.shape)
    d0 = [pyinform.Dist(a0[:, i]) for i in range(a0.shape[1])]

    h0 = [pyinform.shannon.entropy(d0[i], b=2.0) for i in range(len(d0))]
    print(h0)
