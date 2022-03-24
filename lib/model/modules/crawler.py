import numpy as np
from scipy import signal

from lib.model.modules.basic import Oscillator


class Crawler(Oscillator):
    def __init__(self, waveform, initial_amp=None, square_signal_duty=None, step_to_length_mu=None,
                 step_to_length_std=0.0, initial_freq=1.3, freq_std=0.0,
                 gaussian_window_std=None, max_vel_phase=1.0, crawler_noise=0, **kwargs):
        initial_freq = np.random.normal(initial_freq, freq_std)
        super().__init__(initial_freq=initial_freq, **kwargs)
        self.waveform = waveform
        self.activity = 0
        self.amp = initial_amp
        self.noise = crawler_noise
        step_mu, step_std = [np.max([0.0, ii]) for ii in [step_to_length_mu, step_to_length_std]]
        if waveform == 'square':
            # the percentage of the crawler iteration for which linear force/velocity is applied to the body.
            # It is passed to the duty arg of the square signal of the oscillator
            self.square_signal_duty = square_signal_duty
            self.step_to_length_mu = step_mu
            self.step_to_length_std = step_std
            self.step_to_length = self.new_stride
        elif waveform == 'gaussian':
            self.gaussian_window_std = gaussian_window_std
        elif waveform == 'realistic':
            self.step_to_length_mu = step_mu
            self.step_to_length_std = step_std
            self.step_to_length = self.new_stride
            self.max_vel_phase = max_vel_phase * np.pi
        waveform_func_dict = {
            'square': self.square_oscillator,
            'gaussian': self.gaussian_oscillator,
            'realistic': self.realistic_oscillator,
        }
        self.waveform_func = waveform_func_dict[waveform]
        self.start_effector()

    @property
    def new_stride(self):
        return np.random.normal(loc=self.step_to_length_mu, scale=self.step_to_length_std)

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

    def realistic_oscillator(self):
        if self.complete_iteration:
            self.step_to_length = self.new_stride
        return self.freq * self.step_to_length * (1 + 0.6 * np.cos(self.phi - self.max_vel_phase))
