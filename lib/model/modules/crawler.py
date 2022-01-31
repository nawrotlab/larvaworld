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
        # self.noise = self.scaled_noise * self.
        step_mu, step_std = [np.max([0.0, ii]) for ii in [step_to_length_mu, step_to_length_std]]
        if self.waveform == 'square':
            # the percentage of the crawler iteration for which linear force/velocity is applied to the body.
            # It is passed to the duty arg of the square signal of the oscillator
            self.square_signal_duty = square_signal_duty
            self.step_to_length_mu = step_mu
            self.step_to_length_std = step_std
            self.step_to_length = self.generate_step_to_length()
            # self.amp = self.square_oscillator_amp()
        elif self.waveform == 'gaussian':
            self.gaussian_window_std = gaussian_window_std
        elif self.waveform == 'realistic':
            self.step_to_length_mu = step_mu
            self.step_to_length_std = step_std
            self.step_to_length = self.generate_step_to_length()
            self.max_vel_phase = max_vel_phase * np.pi

        self.start_effector()

    # NOTE Computation of linear speed in a squared signal, so that a whole iteration moves the body forward by a
    # proportion of its real_length
    # TODO This is not working as expected probably because of the body drifting even
    #  during the silent half of the circle. For 100 sec with 1 Hz, with sim_length 0.1 and step_to_length we should
    #  get distance traveled=4 but we get 5.45
    def generate_step_to_length(self):
        return np.random.normal(loc=self.step_to_length_mu, scale=self.step_to_length_std)

    # def square_oscillator_amp(self):
    #     return 0.5 * self.step_to_length*self.dt / (self.timesteps_per_iteration * self.square_signal_duty)

    def step(self):
        self.complete_iteration = False
        if self.effector:
            if self.waveform == 'realistic':
                activity = self.realistic_oscillator(phi=self.phi, freq=self.freq,
                                                     sd=self.step_to_length, max_vel_phase=self.max_vel_phase)
            elif self.waveform == 'square':
                activity = self.amp * signal.square(self.phi, duty=self.square_signal_duty) + self.amp
            elif self.waveform == 'gaussian':
                activity = self.gaussian_oscillator()
            elif self.waveform == 'constant':
                activity = self.amp
            super().oscillate()
            if self.complete_iteration and self.waveform == 'realistic':
                self.step_to_length = self.generate_step_to_length()
        else:
            activity = 0

        return activity

    def gaussian_oscillator(self):
        window = signal.gaussian(self.timesteps_per_iteration,
                                 std=self.gaussian_window_std * self.timesteps_per_iteration,
                                 sym=True) * self.amp
        current_t = int(self.t / self.dt)
        value = window[current_t]
        # print(self.t/self.dt, self.timesteps_per_iteration, current_t)
        return value
        # FIXME This is just the x pos on the window. But right now only phi iterates around so I use phi.
        # return window[round(self.phi*self.timesteps_per_iteration/(2*np.pi))]

    def square_oscillator(self):
        r = self.amp * signal.square(self.phi, duty=self.square_signal_duty) + self.amp
        # print(r)
        return r

    # Attention. This equation generates the SCALED velocity per stride
    # See vel_curve.ipynb in notebooks/calibration/crawler
    def realistic_oscillator(self, phi, freq, sd=0.24, k=+1, l=0.6, max_vel_phase=np.pi):
        a = freq * sd * (k + l * np.cos(phi - max_vel_phase))
        # a = (np.cos(-phi) * l + k) * sd * freq
        return a