import numpy as np
import param
from larvaworld.lib import aux



class DefaultCoupling(param.Parameterized):
    attenuation = param.Magnitude(0.0, label='crawl-induced angular attenuation', doc='The attenuation coefficient for the crawl-interference to the angular motion.')
    attenuation_max = param.Magnitude(1.0, label='crawl-induced maximum angular attenuation', doc='The suppression relief coefficient for the crawl-interference to the angular motion.')
    suppression_mode = param.Selector(objects=['amplitude', 'oscillation', 'both'], label='crawl-induced suppression mode', doc='The suppression mode for the crawl-interference to the angular motion.')


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cur_attenuation=1

    def active_effectors(self, crawler=None, feeder=None):
        c, f = crawler, feeder
        c_on = True if c is not None and c.active else False
        f_on = True if f is not None and f.active else False
        return c_on, f_on

    def step(self, **kwargs):
        self.cur_attenuation=self.compute_attenuation(**kwargs)
        return self.apply_attenuation(self.cur_attenuation)

    def compute_attenuation(self, crawler=None, feeder=None):
        c_on, f_on = self.active_effectors(crawler, feeder)
        if c_on or f_on:
            return self.attenuation
        else :
            return 1

    def apply_attenuation(self,cur_att):
        if self.suppression_mode == 'oscillation':
            return cur_att,1
        elif self.suppression_mode == 'amplitude':
            return 1,cur_att
        elif self.suppression_mode == 'both':
            return cur_att,cur_att
        else :
            raise





class SquareCoupling(DefaultCoupling):
    crawler_phi_range = aux.PhaseRange(label='crawler suppression relief phase interval', doc='CRAWLER phase range for TURNER suppression lift.')
    feeder_phi_range = aux.PhaseRange(label='feeder suppression relief phase interval', doc='FEEDER phase range for TURNER suppression lift.')



    def compute_attenuation(self, crawler=None, feeder=None):
        c_on, f_on = self.active_effectors(crawler, feeder)
        if c_on:
            # m=crawler.mode
            A = self.attenuation
            if hasattr(crawler, 'phi'):
                from larvaworld.lib.model import GaussOscillator, PhaseOscillator, SquareOscillator
                if (isinstance(crawler, GaussOscillator) or isinstance(crawler, PhaseOscillator)) and crawler.phi_in_range(self.crawler_phi_range):
                    A += self.attenuation_max
                elif isinstance(crawler, SquareOscillator) and crawler.phi <= 2 * np.pi * crawler.duty:
                    A += self.attenuation_max
                # elif m == 'constant':
                #     pass
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
    max_attenuation_phase = aux.Phase(3.4, label='max relief phase',doc='CRAWLER phase of minimum TURNER suppression.')

    def compute_attenuation(self, crawler=None, feeder=None):
        c_on, f_on = self.active_effectors(crawler, feeder)

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        if not c_on and not f_on :
            return 1

        if c_on:
            x=crawler.phi if hasattr(crawler, 'phi') else 0
        elif f_on :
            x=feeder.phi

        A = gaussian(x, self.max_attenuation_phase, 1) * self.attenuation_max + self.attenuation
        if A >= 1:
            A = 1
        elif A <= 0:
            A = 0
        return A



