import numpy as np
import param
from scipy import signal

from larvaworld.lib import aux
from larvaworld.lib.model.modules.basic import StepOscillator
from larvaworld.lib.param import PositiveNumber, Phase


class StrideOscillator(StepOscillator) :
    stride_dst_mean = PositiveNumber(0.23,softmax=1.0, step=0.01, label='stride distance mean', doc='The mean displacement achieved in a single peristaltic stride as a fraction of the body length.')
    stride_dst_std = PositiveNumber(0.04,softmax=1.0, step=0.01, label='stride distance std', doc='The standard deviation of the displacement achieved in a single peristaltic stride as a fraction of the body length.')


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_to_length = self.new_stride

    @property
    def new_stride(self):
        return np.random.normal(loc=self.stride_dst_mean, scale=self.stride_dst_std)

    @property
    def Act(self):
        return self.freq * self.step_to_length * (1 + self.Act_coef*self.Act_Phi)


    def act_on_complete_iteration(self):
        self.step_to_length = self.new_stride


    def suppresion_relief(self, phi_range):
        return self.phi_in_range(phi_range)


class GaussOscillator(StrideOscillator):
    std = PositiveNumber(0.6, softmax=1.0, step=0.01, label='gaussian stride cycle std', doc='The std of the gaussian window for the velocity oscillation during a stride cycle.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gauss_w=signal.gaussian(360, std=self.std * 360, sym=False)

    @property
    def Act_Phi(self):
        return self.gauss_w[int(np.rad2deg(self.phi))]





class SquareOscillator(StrideOscillator):
    duty = param.Magnitude(0.6, label='square signal duty',doc='The duty parameter(%time at the upper end) of the square signal.')


    @ property
    def Act_Phi(self):
        return float(signal.square(self.phi, duty=self.duty))

    def suppresion_relief(self, phi_range):
        return self.phi <= 2 * np.pi * self.duty

class PhaseOscillator(StrideOscillator):
    max_vel_phase = Phase(3.49, label='max velocity phase',doc='The phase of the crawling oscillation cycle where forward velocity is maximum.')
    max_scaled_vel = PositiveNumber(0.51, softmax=1.5, step=0.01, label='maximum scaled velocity',doc='The maximum scaled forward velocity.')


    @property
    def Act_Phi(self):
        return np.cos(self.phi - self.max_vel_phase)

    @property
    def Act_coef(self):
        return self.max_scaled_vel




