import numpy as np
import param
from larvaworld.lib import aux
from larvaworld.lib.param import PhaseRange, Phase


class DefaultCoupling(param.Parameterized):
    attenuation = param.Magnitude(0.0, label='crawl-induced angular attenuation', doc='The attenuation coefficient for the crawl-interference to the angular motion.')
    attenuation_max = param.Magnitude(1.0, label='crawl-induced maximum angular attenuation', doc='The suppression relief coefficient for the crawl-interference to the angular motion.')
    suppression_mode = param.Selector(objects=['amplitude', 'oscillation', 'both'], label='crawl-induced suppression mode', doc='The suppression mode for the crawl-interference to the angular motion.')


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cur_attenuation=1

    # def active_effectors(self, crawler=None, feeder=None):
    #     c, f = crawler, feeder
    #     c_on = True if c is not None and c.active else False
    #     f_on = True if f is not None and f.active else False
    #     return c_on, f_on

    # def step(self, **kwargs):
    #     self.compute_attenuation(**kwargs)
    #     return self.apply_attenuation(self.cur_attenuation)

    # def compute_attenuation(self, crawler=None, feeder=None):
    #     c_on, f_on = self.active_effectors(crawler, feeder)
    #     if c_on:
    #         self.check_crawler(crawler)
    #     elif f_on:
    #         self.check_feeder(feeder)
    #     else:
    #         self.cur_attenuation= 1

    def apply_attenuation(self,cur_att):
        if self.suppression_mode == 'oscillation':
            return cur_att,1
        elif self.suppression_mode == 'amplitude':
            return 1,cur_att
        elif self.suppression_mode == 'both':
            return cur_att,cur_att
        else :
            raise

    # def check(self, crawler=None, feeder=None):
    #     self.cur_attenuation =self.attenuation if (crawler or feeder) else 1

    def check_crawler(self, crawler):
        self.cur_attenuation= self.attenuation

    def check_feeder(self, feeder):
        self.cur_attenuation= self.attenuation



class SquareCoupling(DefaultCoupling):
    crawler_phi_range = PhaseRange(label='crawler suppression relief phase interval', doc='CRAWLER phase range for TURNER suppression lift.')
    feeder_phi_range = PhaseRange(label='feeder suppression relief phase interval', doc='FEEDER phase range for TURNER suppression lift.')

    def check_crawler(self, crawler):
        A = self.attenuation
        if hasattr(crawler, 'phi') and crawler.suppresion_relief(self.crawler_phi_range):
            A += self.attenuation_max
        self.cur_attenuation= A

    def check_feeder(self, feeder):
        A = self.attenuation
        if hasattr(feeder, 'phi') and feeder.suppresion_relief(self.feeder_phi_range):
            A += self.attenuation_max
        self.cur_attenuation= A







class PhasicCoupling(DefaultCoupling):
    max_attenuation_phase = Phase(3.4, label='max relief phase',doc='CRAWLER phase of minimum TURNER suppression.')


    def get(self,x):
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        A = gaussian(x, self.max_attenuation_phase, 1) * self.attenuation_max + self.attenuation
        if A >= 1:
            A = 1
        elif A <= 0:
            A = 0
        return A

    def check_crawler(self, crawler):
        x = crawler.phi if hasattr(crawler, 'phi') else 0
        self.cur_attenuation= self.get(x)

    def check_feeder(self, feeder):
        x = feeder.phi
        self.cur_attenuation= self.get(x)

