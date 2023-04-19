import numpy as np




class DefaultCoupling:
    def __init__(self, attenuation=0.0, attenuation_max=1.0, suppression_mode='amplitude', **kwargs):
        self.attenuation = attenuation
        self.attenuation_max = attenuation_max
        self.suppression_mode = suppression_mode

    def active_effectors(self, crawler=None, feeder=None):
        c, f = crawler, feeder
        c_on = True if c is not None and c.active else False
        f_on = True if f is not None and f.active else False
        return c_on, f_on

    def step(self, A_in=0.0, **kwargs):
        cT0=self.compute_attenuation(**kwargs)
        return self.apply_attenuation(A_in, cT0)

    def compute_attenuation(self, crawler=None, feeder=None):
        c_on, f_on = self.active_effectors(crawler, feeder)
        if c_on or f_on:
            return self.attenuation
        else :
            return 1

    def apply_attenuation(self,A_in, cT0):
        if self.suppression_mode == 'oscillation':
            A_in -= (1 - cT0)
            cT = 1
        elif self.suppression_mode == 'amplitude':
            cT = cT0
        elif self.suppression_mode == 'both':
            A_in -= (1 - cT0)
            cT = cT0
        else :
            raise
        return A_in,cT




class SquareCoupling(DefaultCoupling):
    def __init__(self, crawler_phi_range=[0.0, 0.0], feeder_phi_range=[0.0, 0.0], **kwargs):
        super().__init__(**kwargs)
        self.crawler_phi_range = crawler_phi_range
        self.feeder_phi_range = feeder_phi_range


    def compute_attenuation(self, crawler=None, feeder=None):
        c_on, f_on = self.active_effectors(crawler, feeder)
        if c_on:
            m=crawler.mode
            A = self.attenuation
            if hasattr(crawler, 'phi'):
                if m in ['realistic', 'gaussian'] and crawler.phi_in_range(self.crawler_phi_range):
                    A += self.attenuation_max
                elif m == 'square' and crawler.phi <= 2 * np.pi * crawler.duty:
                    A += self.attenuation_max
                elif m == 'constant':
                    pass
            return A
        elif f_on:
            A = self.attenuation
            if hasattr(feeder, 'phi'):
                if feeder.phi_in_range(self.feeder_phi_range):
                    A += self.attenuation_max
            return A
        else :
            return 1




class PhasicCoupling(DefaultCoupling):

    def __init__(self, max_attenuation_phase=3.4, **kwargs):
        super().__init__(**kwargs)
        self.max_attenuation_phase = max_attenuation_phase

    def compute_attenuation(self, crawler=None, feeder=None):
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
            return A
        else :
            return 1


