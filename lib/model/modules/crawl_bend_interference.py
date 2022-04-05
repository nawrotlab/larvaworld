import numpy as np

from lib.anal.fitting import gaussian


class DefaultCoupling:
    def __init__(self,locomotor, attenuation = 0.0, **kwargs):
        self.attenuation = attenuation
        self.cur_attenuation = attenuation
        self.locomotor = locomotor

    def step(self):
        c, f = self.locomotor.crawler, self.locomotor.feeder
        if c is not None and c.effector :
            self.cur_attenuation = self.attenuation
        elif f is not None and f.effector :
            self.cur_attenuation = self.attenuation
        else :
            self.cur_attenuation = 1

        return self.cur_attenuation

class SquareCoupling(DefaultCoupling):
    def __init__(self, crawler_phi_range=[0.0, 0.0], feeder_phi_range=[0.0, 0.0], **kwargs):
        super().__init__(**kwargs)
        self.crawler_phi_range = crawler_phi_range
        self.feeder_phi_range = feeder_phi_range

    def step(self):
        self.cur_attenuation = 1
        c, f = self.locomotor.crawler, self.locomotor.feeder
        if c is not None and c.effector :
            if c.waveform in ['realistic', 'gaussian'] and not (self.crawler_phi_range[0] < c.phi / np.pi < self.crawler_phi_range[1]):
                self.cur_attenuation = self.attenuation
            elif c.waveform == 'square' and not c.phi <= 2*np.pi * c.square_signal_duty:
                self.cur_attenuation = self.attenuation
            elif c.waveform == 'constant':
                self.cur_attenuation = self.attenuation
        elif f is not None and f.effector :
            if not self.feeder_phi_range[0] < f.phi / np.pi < self.feeder_phi_range[1]:
                self.cur_attenuation = self.attenuation
        return self.cur_attenuation

class PhasicCoupling(DefaultCoupling) :
    def __init__(self, attenuation_min=0.2, attenuation_max=0.31, max_attenuation_phase = 2.4, **kwargs):
        super().__init__(**kwargs)
        self.attenuation_min = attenuation_min
        self.attenuation_max = attenuation_max
        self.max_attenuation_phase = max_attenuation_phase

    # @property
    def step(self):
        c, f = self.locomotor.crawler, self.locomotor.feeder
        self.cur_attenuation = gaussian(c.phi, self.max_attenuation_phase, 1) * self.attenuation_max + self.attenuation_min
        if self.cur_attenuation >= 1:
            self.cur_attenuation = 1
        elif self.cur_attenuation <= 0:
            self.cur_attenuation = 0
        return self.cur_attenuation