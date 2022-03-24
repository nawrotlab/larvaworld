import numpy as np


class Effector:
    def __init__(self, dt, **kwargs):
        self.dt = dt
        self.t = 0
        self.total_t = 0
        # self.noise = noise
        self.effector = False
        self.ticks = 0
        self.total_ticks = 0

    def count_time(self):
        self.t += self.dt
        self.total_t += self.dt

    def count_ticks(self):
        self.ticks += 1
        self.total_ticks += 1

    def reset_ticks(self):
        self.ticks = 0


    def start_effector(self):
        self.effector = True

    def stop_effector(self):
        self.effector = False
        self.t = 0

    def active(self):
        return self.effector

    def reset(self):
        self.t = 0
        self.total_t = 0


class Oscillator(Effector):
    def __init__(self, freq_range=None, initial_freq=None, initial_freq_std=0, random_phi=True, **kwargs):
        super().__init__(**kwargs)
        # self.freq = initial_freq
        self.freq = float(np.random.normal(loc=initial_freq, scale=initial_freq_std, size=1))
        self.freq_range = freq_range
        self.complete_iteration = False
        self.iteration_counter = 0
        self.d_phi = 2 * np.pi * self.dt * self.freq
        self.timesteps_per_iteration = int(round((1 / self.freq) / self.dt))
        self.phi = np.random.rand() * 2 * np.pi if random_phi else 0

    def set_frequency(self, freq):
        self.freq = freq
        self.timesteps_per_iteration = int(round((1 / self.freq) / self.dt))

    def oscillate(self):
        super().count_time()
        self.phi += self.d_phi
        if self.phi >= 2 * np.pi:
            self.phi %= 2 * np.pi
            self.t = 0
            self.complete_iteration = True
            self.iteration_counter += 1
        else:
            self.complete_iteration = False

    def reset(self):
        self.t = 0
        self.total_t = 0
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0


