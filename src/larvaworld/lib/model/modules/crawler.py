import numpy as np
from scipy import signal

from larvaworld.lib.model.modules.basic import StepOscillator


class StrideOscillator(StepOscillator) :
    def __init__(self, stride_dst_mean=None, stride_dst_std=0.0, **kwargs):
        super().__init__(**kwargs)
        self.stride_dst_mean, self.stride_dst_std = [np.max([0.0, ii]) for ii in [stride_dst_mean, stride_dst_std]]
        self.step_to_length = self.new_stride

    @property
    def new_stride(self):
        return np.random.normal(loc=self.stride_dst_mean, scale=self.stride_dst_std)

    @property
    def Act(self):
        if self.complete_iteration:
            self.step_to_length = self.new_stride
        return self.freq * self.step_to_length * (1 + self.Act_coef*self.Act_Phi)



class GaussOscillator(StrideOscillator):
    def __init__(self, std,**kwargs):
        super().__init__(**kwargs)
        self.gauss_w=signal.gaussian(360, std=std * 360, sym=False)

    @property
    def Act_Phi(self):
        idx = [int(np.rad2deg(self.phi))]
        return self.gauss_w[idx]


class SquareOscillator(StrideOscillator):
    def __init__(self, duty, **kwargs):
        super().__init__(**kwargs)
        self.duty = duty

    @ property
    def Act_Phi(self):
        return signal.square(self.phi, duty=self.duty)

class PhaseOscillator(StrideOscillator):
    def __init__(self, max_vel_phase,max_scaled_vel,initial_amp=None,  **kwargs):
        self.max_scaled_vel = max_scaled_vel
        super().__init__(initial_amp=max_scaled_vel, **kwargs)
        self.max_vel_phase = max_vel_phase


    @property
    def Act_Phi(self):
        return np.cos(self.phi - self.max_vel_phase)




