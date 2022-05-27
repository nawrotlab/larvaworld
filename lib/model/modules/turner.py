import random

import numpy as np

from lib.model.modules.basic import Effector, Oscillator


class Turner(Oscillator, Effector):
    def __init__(self, mode='neural', activation_noise=0.0, continuous=True, rebound=False, dt=0.1,
                 **kwargs):
        self.mode = mode
        self.activation_noise = activation_noise
        self.continuous = continuous
        self.rebound = rebound
        self.buildup = 0
        self.activation = 0
        self.activity = 0
        if mode == 'neural':
            self.init_neural(dt=dt, **kwargs)

        elif mode == 'sinusoidal':
            self.init_sinusoidal(dt=dt, **kwargs)

        elif mode == 'constant':
            self.init_constant(dt=dt, **kwargs)

        self.start_effector()

    def compute_activity(self):
        if self.mode == 'neural':
            self.neural_oscillator.step(self.activation)
            return self.neural_oscillator.activity
        elif self.mode == 'sinusoidal':
            self.complete_iteration = False
            super().oscillate()
            return self.amp * np.sin(self.phi)
        elif self.mode == 'constant':
            return self.amp

    def update_activation(self, A_in):
        if self.mode == 'neural':
            # A_olf=0
            # Map valence modulation to sigmoid accounting for the non middle location of base_activation
            # b = self.base_activation
            # rd, ru = self.A0, self.A1
            # d, u = self.activation_range
            if A_in == 0:
                a = 0
            elif A_in < 0:
                a = self.r0 * A_in
            elif A_in > 0:
                a = self.r1 * A_in
            # Added the relevance of noise to olfactory valence so that noise is attenuated  when valence is rising
            # noise = np.random.normal(scale=self.base_noise) * (1 - np.abs(v))
            I_T = self.base_activation + a
        #     return  np.random.normal(scale=self.base_noise) * (1 - np.abs(A_olf))
        # else:
        #     return A_olf + np.random.normal(scale=self.activation_noise)
        # return A_olf
        else :
            I_T=A_in
        return I_T
        # return A_olf * (1 + np.random.normal(scale=self.activation_noise))

    def step(self, A_in=0.0):
        self.activation = self.update_activation(A_in)
        if self.effector :
            a=self.compute_activity()
            if self.rebound:
                a+=self.buildup
                self.buildup =0
        else :
            if self.continuous:
                aa=self.compute_activity()
                if self.rebound :
                    self.buildup+= aa
            a= 0.0
        self.activity=a
        return a

    def init_neural(self, dt, base_activation=20, activation_range=None,noise=0.0, **kwargs):
        Effector.__init__(self, dt=dt)
        if activation_range is None:
            activation_range = [10, 40]
        self.base_activation = base_activation
        # self.base_noise = np.abs(self.base_activation * self.activation_noise)
        self.r1 = activation_range[1] - self.base_activation
        self.r0 = self.base_activation - activation_range[0]
        self.activation = self.base_activation
        self.neural_oscillator = NeuralOscillator(dt=self.dt, base_activation=base_activation, **kwargs)
        # for i in range(100):
        #     if random.uniform(0, 1) < 0.5:
        #         self.neural_oscillator.step(base_activation)
        #     self.neural_oscillator.step(base_activation)
        # Multiplicative noise
        # activity += np.random.normal(scale=np.abs(activity * self.noise))
        # Additive noise based on mean activity=14.245 the mean output of the oscillator at baseline activation=20
        self.noise = np.abs(14.245 * noise)

    def init_sinusoidal(self, dt, amp_range=[0.5, 2.0], initial_amp=1.0, initial_freq=0.3, freq_range=[0.1, 1.0],noise=0.0,
                        **kwargs):
        Oscillator.__init__(self, initial_freq=initial_freq, freq_range=freq_range, dt=dt)
        self.initial_amp = initial_amp
        self.amp = initial_amp
        self.amp_range = amp_range
        self.noise = np.abs(self.initial_amp * noise)

    def init_constant(self, dt, amp_range=[0.5, 2.0], initial_amp=1.0,noise=0.0, **kwargs):
        Effector.__init__(self, dt=dt)
        self.initial_amp = initial_amp
        self.amp = initial_amp
        self.amp_range = amp_range
        self.noise = np.abs(self.initial_amp * noise)


class NeuralOscillator:
    def __init__(self, dt,base_activation=20, tau=0.1, w_ee=3.0, w_ce=0.1, w_ec=4.0, w_cc=4.0, m=100.0, n=2.0, warm_up=True,**kwargs):
        self.dt = dt
        self.tau = tau
        self.w_ee = w_ee
        self.w_ce = w_ce
        self.w_ec = w_ec
        self.w_cc = w_cc
        self.m = m
        self.n = n

        # Variable parameters
        # self.g = None
        # self.tau_h = None
        self.activity = 0.0
        self.base_activation = base_activation

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

        if warm_up :
            for i in range(100) :
                if random.uniform(0, 1) < 0.5:
                    self.step(self.base_activation)

    def step(self, A=0):
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
        self.activity = self.E_r - self.E_l
        return self.activity

    def compute_R(self, x, h):
        if x > 0:
            r = self.m * x ** self.n / (x ** self.n + h ** self.n)
            return r
        else:
            return 0.0

    def get_state(self):
        state = [self.E_l, self.H_E_l, self.E_r, self.H_E_r, self.C_l, self.H_C_l, self.C_r, self.H_C_r]
        return state


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
