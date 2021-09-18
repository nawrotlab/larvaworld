import random

import numpy as np

from lib.model.modules.basic import Effector, Oscillator


class Turner(Oscillator, Effector):
    def __init__(self, mode='neural', activation_noise=0.0, noise=0.0, continuous=True, rebound=False, dt=0.1,
                 **kwargs):
        self.mode = mode
        self.noise = noise
        self.activation_noise = activation_noise
        self.continuous = continuous
        self.rebound = rebound
        self.buildup = 0
        self.activation = 0

        if mode == 'neural':
            self.init_neural(dt=dt, **kwargs)

        elif mode == 'sinusoidal':
            self.init_sinusoidal(dt=dt, **kwargs)

        self.start_effector()

    def compute_angular_activity(self):
        return self.compute_activity() if self.effector else 0.0

    def compute_activity(self):
        if self.mode == 'neural':
            self.neural_oscillator.step(self.activation)
            return self.neural_oscillator.activity
        elif self.mode == 'sinusoidal':
            self.complete_iteration = False
            super().oscillate()
            return self.amp * np.sin(self.phi)

    def update_activation(self, A_olf):
        if self.mode == 'neural':
            # Map valence modulation to sigmoid accounting for the non middle location of base_activation
            # b = self.base_activation
            # rd, ru = self.A0, self.A1
            # d, u = self.activation_range
            if A_olf == 0:
                a = 0
            elif A_olf < 0:
                a = self.r0 * A_olf
            elif A_olf > 0:
                a = self.r1 * A_olf
            # Added the relevance of noise to olfactory valence so that noise is attenuated  when valence is rising
            # noise = np.random.normal(scale=self.base_noise) * (1 - np.abs(v))
            return self.base_activation + a + np.random.normal(scale=self.base_noise) * (1 - np.abs(A_olf))
        else:
            return A_olf + np.random.normal(scale=self.activation_noise)

    def step(self, inhibited=False, attenuation=1.0, A_olf=0.0):
        self.activation = self.update_activation(A_olf)
        if not inhibited:
            A = self.compute_angular_activity() + self.buildup
            self.buildup = 0
        else:
            if self.continuous:
                a = self.compute_angular_activity()
                A = a * attenuation + self.buildup
                if self.rebound:
                    self.buildup += a
            else:
                A = 0.0
        n = np.random.normal(scale=self.noise)
        A += n
        return A

    def init_neural(self, dt, base_activation=20, activation_range=None, **kwargs):
        Effector.__init__(self, dt=dt)
        if activation_range is None:
            activation_range = [10, 40]
        self.base_activation = base_activation
        self.base_noise = np.abs(self.base_activation * self.activation_noise)
        self.r1 = activation_range[1] - self.base_activation
        self.r0 = self.base_activation - activation_range[0]
        self.activation = self.base_activation
        self.neural_oscillator = NeuralOscillator(dt=self.dt)
        for i in range(1000):
            if random.uniform(0, 1) < 0.5:
                self.neural_oscillator.step(base_activation)
        # Multiplicative noise
        # activity += np.random.normal(scale=np.abs(activity * self.noise))
        # Additive noise based on mean activity=14.245 the mean output of the oscillator at baseline activation=20
        self.noise = np.abs(14.245 * self.noise)

    def init_sinusoidal(self, dt, amp_range=[0.5, 2.0], initial_amp=1.0, initial_freq=0.3, freq_range=[0.1, 1.0],
                        **kwargs):
        Oscillator.__init__(self, initial_freq=initial_freq, freq_range=freq_range, dt=dt)
        self.initial_amp = initial_amp
        self.amp = initial_amp
        self.amp_range = amp_range
        self.noise = np.abs(self.initial_amp * self.noise)


class NeuralOscillator:
    def __init__(self, dt):
        self.dt = dt
        self.tau = 0.1
        self.w_ee = 3.0
        self.w_ce = 0.1
        self.w_ec = 4.0
        self.w_cc = 4.0
        self.m = 100.0
        self.n = 2.0

        # Variable parameters
        # self.g = None
        # self.tau_h = None
        self.activity = 0.0

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
        # self.scaled_tau_h=None

    def step(self, A=0):
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

    def compute_R(self, x, h):
        if x > 0:
            r = self.m * x ** self.n / (x ** self.n + h ** self.n)
            return r
        else:
            return 0.0