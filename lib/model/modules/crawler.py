import numpy as np
from scipy import signal

from lib.model.modules.basic import Oscillator, Effector


class Crawler(Oscillator):
    def __init__(self, waveform, initial_amp=None, square_signal_duty=None, stride_dst_mean=None,
                 stride_dst_std=0.0, initial_freq=1.3, max_scaled_vel=0.6,
                 gaussian_window_std=None, max_vel_phase=3.6, crawler_noise=0, **kwargs):
        # initial_freq = np.random.normal(initial_freq, freq_std)
        super().__init__(initial_freq=initial_freq, **kwargs)
        self.waveform = waveform
        self.activity = 0
        self.amp = initial_amp
        self.noise = crawler_noise
        self.max_scaled_vel = max_scaled_vel
        self.max_vel_phase = max_vel_phase
        self.stride_dst_mean, self.stride_dst_std = [np.max([0.0, ii]) for ii in [stride_dst_mean, stride_dst_std]]
        self.step_to_length = self.new_stride


        # if waveform == 'square':
        #     self.square_signal_duty = square_signal_duty
        # elif waveform == 'gaussian':
        #     self.gaussian_window_std = gaussian_window_std
        # elif waveform == 'realistic':
        #     self.max_vel_phase = max_vel_phase

        # waveform_func_dict = {
        #     'square': lambda phi,c: c*(1+self.amp*signal.square(phi, duty=square_signal_duty)),
        #     'gaussian': lambda phi,c: c*(1+self.amp*signal.gaussian(360, gaussian_window_std * 360, sym=False)[int(np.rad2deg(phi))]),
        #     'realistic': lambda phi,c: c*(1+max_scaled_vel*np.cos(phi - self.max_vel_phase)),
        #     'constant': lambda phi,c: self.amp,
        # }
        #
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
            if self.complete_iteration:
                self.step_to_length = self.new_stride

            # self.activity = self.waveform_func(phi=self.phi, c=self.freq * self.step_to_length)
            self.activity = self.freq * self.step_to_length * (1 + self.max_scaled_vel * self.waveform_func())
        else:

            self.activity = 0
        return self.activity

    def gaussian_oscillator(self):
        # A=self.timesteps_per_iteration
        # window = signal.gaussian(360,std=self.gaussian_window_std*360,sym=False)
        # t=int(np.rad2deg(self.phi))
        return signal.gaussian(360, std=self.gaussian_window_std * 360, sym=False)[int(np.rad2deg(self.phi))]

    def square_oscillator(self):

        return signal.square(self.phi, duty=self.square_signal_duty)

    def constant_crawler(self):
        return self.amp

    def realistic_oscillator(self):
        return np.cos(self.phi - self.max_vel_phase)
