import numpy as np




class DefaultCoupling:
    def __init__(self, attenuation=0.0, attenuation_max=1.0, suppression_mode='amplitude', **kwargs):

        self.attenuation = attenuation
        self.attenuation_max = attenuation_max
        self.cur_attenuation = attenuation
        self.suppression_mode = suppression_mode

    # @property
    def active_effectors(self, crawler=None, feeder=None):
        c, f = crawler, feeder
        c_on = True if c is not None and c.effector else False
        f_on = True if f is not None and f.effector else False
        return c_on, f_on

    def step(self, crawler=None, feeder=None):
        A = 1
        c_on, f_on = self.active_effectors(crawler, feeder)
        if c_on or f_on:
            A = self.attenuation
        self.cur_attenuation = A
        # print(self.attenuation,self.cur_attenuation)
        return A


class SquareCoupling(DefaultCoupling):
    def __init__(self, crawler_phi_range=[0.0, 0.0], feeder_phi_range=[0.0, 0.0], **kwargs):
        super().__init__(**kwargs)
        self.crawler_phi_range = crawler_phi_range
        self.feeder_phi_range = feeder_phi_range

    def step(self, crawler=None, feeder=None):
        A = 1
        c_on, f_on = self.active_effectors(crawler, feeder)
        if c_on:
            A = self.attenuation
            if crawler.mode in ['realistic', 'gaussian'] and (
                    self.crawler_phi_range[0] < crawler.phi < self.crawler_phi_range[1]):
                A += self.attenuation_max
            elif crawler.mode == 'square' and crawler.phi <= 2 * np.pi * crawler.duty:
                A += self.attenuation_max
            elif crawler.mode == 'constant':
                pass
        elif f_on:
            A = self.attenuation
            if self.feeder_phi_range[0] < feeder.phi < self.feeder_phi_range[1]:
                A += self.attenuation_max
        self.cur_attenuation = A
        return A


class PhasicCoupling(DefaultCoupling):

    def __init__(self, max_attenuation_phase=3.4, **kwargs):
        super().__init__(**kwargs)
        self.max_attenuation_phase = max_attenuation_phase

    def step(self, crawler=None, feeder=None):
        A = 1
        c_on, f_on = self.active_effectors(crawler, feeder)

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        if c_on:
            if hasattr(crawler, 'phi') :
                x=crawler.phi
            else :
                x=0
            A = gaussian(x, self.max_attenuation_phase, 1) * self.attenuation_max + self.attenuation
            if A >= 1:
                A = 1
            elif A <= 0:
                A = 0
        self.cur_attenuation = A
        return A


