import numpy as np
import param
from scipy import signal

from larvaworld.lib.model.modules.basic import StepOscillator


class StrideOscillator(StepOscillator) :
    stride_dst_mean = param.Magnitude(default=0.23, label='stride distance mean', doc='The mean displacement achieved in a single peristaltic stride as a fraction of the body length.')
    stride_dst_std = param.Magnitude(default=0.04, label='stride distance std', doc='The standard deviation of the displacement achieved in a single peristaltic stride as a fraction of the body length.')


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    #     self.stride_dst_mean, self.stride_dst_std = [np.max([0.0, ii]) for ii in [stride_dst_mean, stride_dst_std]]
        self.step_to_length = self.new_stride
        # self.start_effector()

    @property
    def new_stride(self):
        return np.random.normal(loc=self.stride_dst_mean, scale=self.stride_dst_std)

    @property
    def Act(self):
        if self.complete_iteration:
            self.step_to_length = self.new_stride
        return self.freq * self.step_to_length * (1 + self.Act_coef*self.Act_Phi)



class GaussOscillator(StrideOscillator):
    std = param.Number(default=0.6, softbounds=(0.0, 1.0), label='gaussian stride cycle std', doc='The std of the gaussian window for the velocity oscillation during a stride cycle.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gauss_w=signal.gaussian(360, std=self.std * 360, sym=False)

    @property
    def Act_Phi(self):
        idx = [int(np.rad2deg(self.phi))]
        return self.gauss_w[idx]


class SquareOscillator(StrideOscillator):
    duty = param.Magnitude(default=0.6, label='square signal duty',
                       doc='The duty parameter(%time at the upper end) of the square signal.')


    @ property
    def Act_Phi(self):
        return signal.square(self.phi, duty=self.duty)

class PhaseOscillator(StrideOscillator):
    max_vel_phase = param.Number(default=3.49, bounds=(0.0, 2 * np.pi), label='max velocity phase',
                        doc='The phase of the crawling oscillation cycle where forward velocity is maximum.')
    max_scaled_vel = param.Number(default=0.51, bounds=(0.0, 1.5), label='maximum scaled velocity',
                                 doc='The maximum scaled forward velocity.')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.initial_amp = self.max_scaled_vel
        self.amp = self.initial_amp


    @property
    def Act_Phi(self):
        return np.cos(self.phi - self.max_vel_phase)




