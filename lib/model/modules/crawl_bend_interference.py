import numpy as np

from lib.anal.fitting import gaussian


class DefaultCoupling():
    def __init__(self,locomotor, **kwargs):
        self.cur_attenuation = 1
        self.locomotor = locomotor

    def step(self):
        return self.cur_attenuation

class SquareCoupling(DefaultCoupling):
    def __init__(self, crawler_phi_range=[0.0, 0.0], feeder_phi_range=[0.0, 0.0], attenuation=0.0, **kwargs):
        super().__init__(**kwargs)
        self.crawler_phi_range = crawler_phi_range
        self.feeder_phi_range = feeder_phi_range
        self.attenuation = attenuation
        self.crawler = self.locomotor.crawler
        self.feeder = self.locomotor.feeder

    def step(self):
        c,f=self.locomotor.crawler, self.locomotor.feeder
        if c is not None:
            if c.effector:
                phi = c.phi / np.pi
                p0, p1 = self.crawler_phi_range
                if c.waveform in ['realistic', 'gaussian'] and (phi < p0 or phi > p1):
                    self.cur_attenuation = self.attenuation
                elif c.waveform == 'square' and not phi <= 2 * self.crawler.square_signal_duty:
                    self.cur_attenuation = self.attenuation
                elif c.waveform == 'constant':
                    self.cur_attenuation = 1
            else:
                self.cur_attenuation = 1
        else:
            self.cur_attenuation = 1


        if f is not None:
            if f.effector:
                phi = self.feeder.phi / np.pi
                p0, p1 = self.feeder_phi_range
                if p0 < phi < p1:
                    self.cur_attenuation = 1
                else :
                    self.cur_attenuation = self.attenuation
        return self.cur_attenuation

class PhasicCoupling(DefaultCoupling) :
    def __init__(self, attenuation_min=0.2, attenuation_max=0.31, max_attenuation_phase = 2.4, **kwargs):
        super().__init__(**kwargs)
        self.attenuation_min = attenuation_min
        self.attenuation_max = attenuation_max
        self.max_attenuation_phase = max_attenuation_phase

    # @property
    def step(self, phi):
        self.cur_attenuation = gaussian(phi, self.max_attenuation_phase, 1) * self.attenuation_max + self.attenuation_min
        if self.cur_attenuation >= 1:
            self.cur_attenuation = 1
        elif self.cur_attenuation <= 0:
            self.cur_attenuation = 0
        return self.cur_attenuation