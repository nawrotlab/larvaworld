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
        self.initial_phi = np.random.rand() * 2 * np.pi if random_phi else 0
        self.timesteps_per_iteration = int(round((1 / self.freq) / self.dt))
        self.d_phi = 2 * np.pi / self.timesteps_per_iteration
        self.phi = self.initial_phi

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

    def reset(self):
        self.t = 0
        self.total_t = 0
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0


class Oscillator_coupling():
    def __init__(self, crawler_phi_range=[0.0, 0.0],feeder_phi_range=[0.0, 0.0],attenuation=0.0):
        self.crawler_phi_range = crawler_phi_range
        self.feeder_phi_range = feeder_phi_range
        self.attenuation = attenuation

    def step(self, crawler=None, feeder=None):
        return self.resolve_coupling(crawler, feeder)

    def resolve_coupling(self, crawler, feeder):
        if crawler is not None:
            if crawler.effector:
                phi = crawler.phi / np.pi
                p0, p1 = self.crawler_phi_range
                if crawler.waveform in ['realistic', 'gaussian'] and (phi < p0 or phi > p1):
                    return True
                elif crawler.waveform == 'square' and not phi <= 2 * crawler.square_signal_duty:
                    return True
                elif crawler.waveform == 'constant':
                    return True


        if feeder is not None:
            if feeder.effector:
                phi = feeder.phi / np.pi
                p0, p1 = self.feeder_phi_range
                if p0 < phi < p1:
                    return True
        return False
