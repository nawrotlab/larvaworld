import numpy as np
import param


class DefaultCoupling(param.Parameterized):
    attenuation = param.Magnitude(default=0.0, label='crawl-induced angular attenuation', doc='The attenuation coefficient for the crawl-interference to the angular motion.')
    attenuation_max = param.Magnitude(default=1.0, label='crawl-induced maximum angular attenuation', doc='The suppression relief coefficient for the crawl-interference to the angular motion.')
    suppression_mode = param.Selector(default='amplitude',objects=['amplitude', 'oscillation', 'both'], label='crawl-induced suppression mode', doc='The suppression mode for the crawl-interference to the angular motion.')


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
    crawler_phi_range = param.List(default=[0.0, 0.0], label='crawler suppression relief phase interval', doc='CRAWLER phase range for TURNER suppression lift.')
    feeder_phi_range = param.List(default=[0.0, 0.0], label='feeder suppression relief phase interval', doc='FEEDER phase range for TURNER suppression lift.')



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
    max_attenuation_phase = param.Number(default=3.4, bounds=(0.0, 2 * np.pi), label='max relief phase',
                                 doc='CRAWLER phase of minimum TURNER suppression.')

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


