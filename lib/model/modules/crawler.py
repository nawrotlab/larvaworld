import numpy as np
from scipy import signal

from lib.model.modules.basic import Oscillator, Effector

class Crawler(Oscillator):
    def __init__(self, waveform, initial_amp=None, square_signal_duty=None, stride_dst_mean=None,
                 stride_dst_std=0.0, initial_freq=1.3,max_scaled_vel=0.6,
                 gaussian_window_std=None, max_vel_phase=3.6, crawler_noise=0, **kwargs):
        # initial_freq = np.random.normal(initial_freq, freq_std)
        super().__init__(initial_freq=initial_freq, **kwargs)
        self.waveform = waveform
        self.activity = 0
        self.amp = initial_amp
        self.noise = crawler_noise

        if waveform == 'square':
            # the percentage of the crawler iteration for which linear force/velocity is applied to the body.
            # It is passed to the duty arg of the square signal of the oscillator
            step_mu, step_std = [np.max([0.0, ii]) for ii in [stride_dst_mean, stride_dst_std]]
            self.square_signal_duty = square_signal_duty
            self.stride_dst_mean = step_mu
            self.stride_dst_std = step_std
            self.step_to_length = self.new_stride
        elif waveform == 'gaussian':
            self.gaussian_window_std = gaussian_window_std
        elif waveform == 'realistic':
            step_mu, step_std = [np.max([0.0, ii]) for ii in [stride_dst_mean, stride_dst_std]]
            self.stride_dst_mean = step_mu
            self.stride_dst_std = step_std
            self.step_to_length = self.new_stride
            self.max_vel_phase = max_vel_phase
            self.max_scaled_vel = max_scaled_vel
        # elif waveform == 'constant':

        waveform_func_dict = {
            'square': self.square_oscillator,
            'gaussian': self.gaussian_oscillator,
            'realistic': self.realistic_oscillator,
            'constant': self.constant_crawler,
        }
        self.waveform_func = waveform_func_dict[waveform]
        self.start_effector()

    @property
    def new_stride(self):
        return np.random.normal(loc=self.stride_dst_mean, scale=self.stride_dst_std)

    def step(self):
        if self.effector:
            super().oscillate()
            self.activity=self.waveform_func()
        else:

            self.activity= 0
        return self.activity

    def gaussian_oscillator(self):
        window = signal.gaussian(self.timesteps_per_iteration,
                                 std=self.gaussian_window_std * self.timesteps_per_iteration,
                                 sym=True) * self.amp
        return window[int(self.t / self.dt)]

    def square_oscillator(self):
        return self.amp * signal.square(self.phi, duty=self.square_signal_duty) + self.amp

    def constant_crawler(self):
        return self.amp

    def realistic_oscillator(self):
        if self.complete_iteration:
            self.step_to_length = self.new_stride
        return self.freq * self.step_to_length * (1 + self.max_scaled_vel * np.cos(self.phi - self.max_vel_phase))
