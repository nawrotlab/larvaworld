import numpy as np

from lib.anal.fitting import gaussian


class DefaultCoupling:
    def __init__(self,locomotor, attenuation = 0.0,suppression_mode='amplitude', **kwargs):
        self.attenuation = attenuation
        self.cur_attenuation = attenuation
        self.locomotor = locomotor
        self.suppression_mode=suppression_mode

    @ property
    def active_effectors(self):
        c, f = self.locomotor.crawler, self.locomotor.feeder
        c_on=True if c is not None and c.effector else False
        f_on=True if f is not None and f.effector else False
        return c_on, f_on


    def step(self):
        A=1
        c_on, f_on = self.active_effectors
        if c_on or f_on :
            A= self.attenuation
        self.cur_attenuation = A
        return A

class SquareCoupling(DefaultCoupling):
    def __init__(self, crawler_phi_range=[0.0, 0.0], feeder_phi_range=[0.0, 0.0], **kwargs):
        super().__init__(**kwargs)
        self.crawler_phi_range = crawler_phi_range
        self.feeder_phi_range = feeder_phi_range

    def step(self):
        A = 1
        c_on, f_on = self.active_effectors
        if c_on:
            c = self.locomotor.crawler
            if c.waveform in ['realistic', 'gaussian'] and not (
                    self.crawler_phi_range[0] < c.phi / np.pi < self.crawler_phi_range[1]):
                A = self.attenuation
            elif c.waveform == 'square' and not c.phi <= 2 * np.pi * c.square_signal_duty:
                A = self.attenuation
            elif c.waveform == 'constant':
                A = self.attenuation
        elif f_on:
            if not self.feeder_phi_range[0] < self.locomotor.feeder.phi / np.pi < self.feeder_phi_range[1]:
                A = self.attenuation
        self.cur_attenuation = A
        return A


class PhasicCoupling(DefaultCoupling) :
    def __init__(self, attenuation_max=0.31, max_attenuation_phase = 3.4, **kwargs):
        super().__init__(**kwargs)
        # self.attenuation_min = attenuation_min
        self.attenuation_max = attenuation_max
        self.max_attenuation_phase = max_attenuation_phase

    # @property
    def step(self):
        A = 1
        c_on, f_on = self.active_effectors
        if c_on:
            A = gaussian(self.locomotor.crawler.phi, self.max_attenuation_phase, 1) * self.attenuation_max + self.attenuation
            if A >= 1:
                A = 1
            elif A <= 0:
                A = 0
        self.cur_attenuation = A
        return A