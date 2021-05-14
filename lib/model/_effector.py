import itertools
import random
import time

import numpy as np
from scipy import signal
from scipy.stats import lognorm, rv_discrete

import lib.aux.sampling as sampling
from lib.aux.functions import flatten_tuple


class Effector:
    def __init__(self, dt, **kwargs):
        self.dt = dt
        self.t = 0
        self.total_t = 0
        # self.noise = noise
        self.effector = False

    def count_time(self):
        self.t += self.dt
        self.total_t += self.dt

    def start_effector(self):
        self.effector = True

    def stop_effector(self):
        self.effector = False
        self.t = 0

    def active(self):
        return self.effector

    def reset(self):
        self.t = 0
        self.total_t = 0


class Oscillator(Effector):
    def __init__(self, freq_range=None, initial_freq=None, initial_freq_std=0, random_phi=True, **kwargs):
        super().__init__(**kwargs)
        # self.freq = initial_freq
        self.freq = float(np.random.normal(loc=initial_freq, scale=initial_freq_std, size=1))
        self.freq_range = freq_range
        self.complete_iteration = False
        self.iteration_counter = 0
        self.initial_phi = np.random.rand() * 2 * np.pi if random_phi else 0
        self.timesteps_per_iteration = int(round((1 / self.freq) / self.dt))
        self.d_phi = 2 * np.pi / self.timesteps_per_iteration
        self.phi = self.initial_phi

    def set_frequency(self, freq):
        self.freq = freq
        self.timesteps_per_iteration = int(round((1 / self.freq) / self.dt))

    def oscillate(self):
        super().count_time()
        self.phi += self.d_phi
        if self.phi >= 2 * np.pi:
            self.phi %= 2 * np.pi
            self.t = 0
            self.complete_iteration = True
            self.iteration_counter += 1

    def reset(self):
        self.t = 0
        self.total_t = 0
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0


class Crawler(Oscillator):
    def __init__(self, waveform, initial_amp=None, square_signal_duty=None, step_to_length_mu=None,
                 step_to_length_std=0,
                 gaussian_window_std=None, max_vel_phase=1.0, crawler_noise=0, **kwargs):
        super().__init__(**kwargs)
        self.waveform = waveform
        self.activity = 0
        self.amp = initial_amp
        self.scaled_noise = crawler_noise
        # self.noise = self.scaled_noise * self.
        if self.waveform == 'square':
            # the percentage of the crawler iteration for which linear force/velocity is applied to the body.
            # It is passed to the duty arg of the square signal of the oscillator
            self.square_signal_duty = square_signal_duty
            self.step_to_length_mu = step_to_length_mu
            self.step_to_length_std = step_to_length_std
            self.step_to_length = self.generate_step_to_length()
        elif self.waveform == 'gaussian':
            self.gaussian_window_std = gaussian_window_std
        elif self.waveform == 'realistic':
            self.step_to_length_mu = step_to_length_mu
            self.step_to_length_std = step_to_length_std
            self.step_to_length = self.generate_step_to_length()
            self.max_vel_phase = max_vel_phase * np.pi

    # NOTE Computation of linear speed in a squared signal, so that a whole iteration moves the body forward by a
    # proportion of its real_length
    # TODO This is not working as expected probably because of the body drifting even
    #  during the silent half of the circle. For 100 sec with 1 Hz, with sim_length 0.1 and step_to_length we should
    #  get distance traveled=4 but we get 5.45
    def generate_step_to_length(self):
        return np.random.normal(loc=self.step_to_length_mu, scale=self.step_to_length_std)

    def adapt_square_oscillator_amp(self, length):
        distance = length * self.step_to_length
        timesteps_active_per_iteration = self.timesteps_per_iteration * self.square_signal_duty
        distance_per_active_timestep = distance / timesteps_active_per_iteration
        lin_vel_when_active = distance_per_active_timestep / self.dt
        self.amp = lin_vel_when_active / 2
        # print(self.amp, self.timesteps_per_iteration * self.square_signal_duty)

    def step(self, length):
        self.complete_iteration = False
        if self.effector:
            if self.waveform == 'realistic':
                activity = self.realistic_oscillator(phi=self.phi, freq=self.freq,
                                                     sd=self.step_to_length, max_vel_phase=self.max_vel_phase) * length
            elif self.waveform == 'square':
                self.adapt_square_oscillator_amp(length)
                activity = self.square_oscillator()
            elif self.waveform == 'gaussian':
                activity = self.gaussian_oscillator()
            elif self.waveform == 'constant':
                activity = self.amp

                # b=0.05-activity
                # if b>0:
                #     add=np.random.uniform(low=0.0, high=b)
                #     activity += add

            # if self.noise:
            #     activity += np.random.normal(scale=np.abs(activity * self.noise))
            super().oscillate()
            if self.complete_iteration and self.waveform == 'realistic':
                self.step_to_length = self.generate_step_to_length()
            activity += np.random.normal(scale=self.scaled_noise * length)
        else:
            activity = 0
        return activity
        # return np.clip(activity, a_min=0, a_max=np.inf)

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


class Turner(Oscillator, Effector):
    def __init__(self, mode='neural', activation_noise=0.0, noise=0.0, continuous=True, rebound=False, dt=0.1,
                 **kwargs):
        self.mode = mode
        self.noise = noise
        self.activation_noise = activation_noise
        self.continuous = continuous
        self.rebound = rebound
        self.buildup = 0
        self.activation = 0

        if mode == 'neural':
            self.init_neural(dt=dt, **kwargs)

        elif mode == 'sinusoidal':
            self.init_sinusoidal(dt=dt, **kwargs)

    def compute_angular_activity(self):
        return self.compute_activity() if self.effector else 0.0

    def compute_activity(self):
        if self.mode == 'neural':
            self.neural_oscillator.step(self.activation)
            return self.neural_oscillator.activity
        elif self.mode == 'sinusoidal':
            self.complete_iteration = False
            super().oscillate()
            return self.amp * np.sin(self.phi)

    def update_activation(self, A_olf):
        if self.mode == 'neural':
            # Map valence modulation to sigmoid accounting for the non middle location of base_activation
            # b = self.base_activation
            # rd, ru = self.A0, self.A1
            # d, u = self.activation_range
            if A_olf == 0:
                a = 0
            elif A_olf < 0:
                a = self.r0 * A_olf
            elif A_olf > 0:
                a = self.r1 * A_olf
            # Added the relevance of noise to olfactory valence so that noise is attenuated  when valence is rising
            # noise = np.random.normal(scale=self.base_noise) * (1 - np.abs(v))
            return self.base_activation + a + np.random.normal(scale=self.base_noise) * (1 - np.abs(A_olf))
        else:
            return A_olf + np.random.normal(scale=self.activation_noise)

    def step(self, inhibited=False, attenuation=1.0, A_olf=0.0):
        self.activation = self.update_activation(A_olf)
        if not inhibited:
            A = self.compute_angular_activity() + self.buildup
            self.buildup = 0
        else:
            if self.continuous:
                a = self.compute_angular_activity()
                A = a * attenuation + self.buildup
                if self.rebound:
                    self.buildup += a
            else:
                A = 0.0
        A += np.random.normal(scale=self.noise)
        return A

    def init_neural(self, dt, base_activation=20, activation_range=None, **kwargs):
        Effector.__init__(self, dt=dt)
        if activation_range is None:
            activation_range = [10, 40]
        self.base_activation = base_activation
        self.base_noise = np.abs(self.base_activation * self.activation_noise)
        self.r1 = activation_range[1] - self.base_activation
        self.r0 = self.base_activation - activation_range[0]
        self.activation = self.base_activation
        self.neural_oscillator = NeuralOscillator(dt=self.dt)
        for i in range(1000):
            if random.uniform(0, 1) < 0.5:
                self.neural_oscillator.step(base_activation)
        # Multiplicative noise
        # activity += np.random.normal(scale=np.abs(activity * self.noise))
        # Additive noise based on mean activity=14.245 the mean output of the oscillator at baseline activation=20
        self.noise = np.abs(14.245 * self.noise)

    def init_sinusoidal(self, dt, amp_range=[0.5, 2.0], initial_amp=1.0, initial_freq=0.3, freq_range=[0.1, 1.0],
                        **kwargs):
        Oscillator.__init__(self, initial_freq=initial_freq, freq_range=freq_range, dt=dt)
        self.initial_amp = initial_amp
        self.amp = initial_amp
        self.amp_range = amp_range
        self.noise = np.abs(self.initial_amp * self.noise)


# class NeuralTurner(Turner):
#     def __init__(self, base_activation=20, activation_range=None, **kwargs):
#         super().__init__(**kwargs)
#         if activation_range is None:
#             activation_range = [10, 40]
#         self.activation_range = activation_range
#         self.base_activation = base_activation
#         self.base_noise = np.abs(self.base_activation * self.activation_noise)
#         self.range_upwards = self.activation_range[1] - self.base_activation
#         self.range_downwards = self.base_activation - self.activation_range[0]
#         self.activation = self.base_activation
#         self.neural_oscillator = NeuralOscillator(dt=self.dt)
#         for i in range(1000):
#             if random.uniform(0, 1) < 0.5:
#                 self.neural_oscillator.step(base_activation)
#         # Multiplicative noise
#         # activity += np.random.normal(scale=np.abs(activity * self.noise))
#         # Additive noise based on mean activity=14.245 the mean output of the oscillator at baseline activation=20
#         self.noise = np.abs(14.245 * self.noise)
#
#     def compute_activity(self):
#         self.neural_oscillator.step(self.activation)
#         return self.neural_oscillator.activity
#
#     def update_activation(self, olfactory_activation):
#         # Map valence modulation to sigmoid accounting for the non middle location of base_activation
#         b = self.base_activation
#         rd, ru = self.range_downwards, self.range_upwards
#         # d, u = self.activation_range
#         v = olfactory_activation
#         if v == 0:
#             a = 0
#         elif v < 0:
#             a = rd * v
#         elif v > 0:
#             a = ru * v
#         # Added the relevance of noise to olfactory valence so that noise is attenuated  when valence is rising
#         noise = np.random.normal(scale=self.base_noise) * (1 - np.abs(v))
#         return b + a + noise


class NeuralOscillator:
    def __init__(self, dt):
        self.dt = dt
        self.tau = 0.1
        self.w_ee = 3.0
        self.w_ce = 0.1
        self.w_ec = 4.0
        self.w_cc = 4.0
        self.m = 100.0
        self.n = 2.0

        # Variable parameters
        # self.g = None
        # self.tau_h = None
        self.activity = 0.0

        # Neural populations
        self.E_r = 0  # 28
        self.H_E_r = 0  # 10

        self.E_l = 0  # 30
        self.H_E_l = 0  # 10

        self.C_r = 0
        self.H_C_r = 0  # 10

        self.C_l = 0
        self.H_C_l = 0  # 10

        self.scaled_tau = self.dt / self.tau
        # self.scaled_tau_h=None

    def step(self, A=0):
        t = self.scaled_tau
        tau_h = 3 / (1 + (0.04 * A) ** 2)
        t_h = self.dt / tau_h
        g = 6 + (0.09 * A) ** 2

        self.E_l += t * (
                -self.E_l + self.compute_R(A + self.w_ee * self.E_l - self.w_ec * self.C_r, 64 + g * self.H_E_l))
        self.E_r += t * (
                -self.E_r + self.compute_R(A + self.w_ee * self.E_r - self.w_ec * self.C_l, 64 + g * self.H_E_r))
        self.H_E_l += t_h * (-self.H_E_l + self.E_l)
        self.H_E_r += t_h * (-self.H_E_r + self.E_r)

        self.C_l += t * (
                -self.C_l + self.compute_R(A + self.w_ce * self.E_l - self.w_cc * self.C_r, 64 + g * self.H_C_l))
        self.C_r += t * (
                -self.C_r + self.compute_R(A + self.w_ce * self.E_r - self.w_cc * self.C_l, 64 + g * self.H_C_r))
        self.H_C_l += t_h * (-self.H_C_l + self.E_l)
        self.H_C_r += t_h * (-self.H_C_r + self.E_r)
        self.activity = self.E_r - self.E_l

    def compute_R(self, x, h):
        if x > 0:
            r = self.m * x ** self.n / (x ** self.n + h ** self.n)
            return r
        else:
            return 0.0


# class SinusoidalTurner(Turner, Oscillator):
#     def __init__(self, amp_range=[0.5, 2.0], initial_amp=1.0, initial_freq=0.3, freq_range=[0.1, 1.0], dt=0.1, **kwargs):
#         Turner.__init__(self, dt=dt, **kwargs)
#
#         # FIXME Will be obsolete when we fix oscillator interference
#         Oscillator.__init__(self, initial_freq=initial_freq, freq_range=freq_range, dt=dt)
#         self.initial_amp = initial_amp
#         self.amp = initial_amp
#         self.amp_range = amp_range
#         self.noise = np.abs(self.initial_amp * self.noise)
#
#     def compute_activity(self):
#         self.complete_iteration = False
#         super().oscillate()
#         return self.amp * np.sin(self.phi)


class Feeder(Oscillator):
    def __init__(self, model, feed_radius, feed_capacity,
                 feeder_initial_freq=2, feeder_freq_range=[1, 3], **kwargs):
        super().__init__(initial_freq=feeder_initial_freq, freq_range=feeder_freq_range, **kwargs)
        self.model = model
        self.feed_radius = feed_radius
        self.feed_capacity = feed_capacity
        # self.feed_success = None

    def step(self):
        self.complete_iteration = False
        if self.effector:
            super().oscillate()
        return self.complete_iteration


class Oscillator_coupling():
    def __init__(self, crawler_phi_range=[0.0, 0.0],
                 feeder_phi_range=[0.0, 0.0],
                 attenuation=0.0):
        self.crawler_phi_range = crawler_phi_range
        self.feeder_phi_range = feeder_phi_range
        self.attenuation = attenuation

    def step(self, crawler=None, feeder=None):
        return self.resolve_coupling(crawler, feeder)

    def resolve_coupling(self, crawler, feeder):
        if crawler is not None:
            if crawler.effector:
                phi = crawler.phi / np.pi
                p0, p1 = self.crawler_phi_range
                if crawler.waveform == 'realistic' and (phi < p0 or phi > p1):
                    return True
                elif crawler.waveform == 'square' and not phi <= 2 * crawler.square_signal_duty:
                    return True
                elif crawler.waveform == 'gaussian' and (phi < p0 or phi > p1):
                    return True

        if feeder is not None:
            if feeder.effector:
                phi = feeder.phi / np.pi
                p0, p1 = self.feeder_phi_range
                if p0 < phi < p1:
                    return True
        return False
        # if self.crawler_inhibits_bend or self.feeder_inhibits_bend :
        #     self.turner_inhibition=True


class Intermitter(Effector):
    def __init__(self, nengo_manager=None,
                 crawler=None, crawl_bouts=False,
                 feeder=None, feed_bouts=False,
                 turner=None, intermittent_turner=False, turner_prepost_lag=[0, 0],
                 pause_dist=None, stridechain_dist=None,
                 EEB_decay=1,
                 EEB=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.nengo_manager = nengo_manager

        self.crawler = crawler
        self.turner = turner
        self.feeder = feeder
        self.EEB = EEB
        self.base_EEB = EEB
        if crawler is None:
            self.crawl_bouts = False
        else:
            self.crawl_bouts = crawl_bouts
        if turner is None:
            self.intermittent_turner = False
        else:
            self.intermittent_turner = intermittent_turner
        if feeder is None:
            self.feed_bouts = False
        else:
            self.feed_bouts = feed_bouts

        if self.nengo_manager is None:
            # self.feeder_reoccurence_rate_on_success = feeder_reoccurence_rate_on_success
            self.EEB_decay = EEB_decay
            # self.feeder_reoccurence_rate = 1 - self.EEB
            # self.feeder_reoccurence_rate = self.feeder_reoccurence_rate_on_success
            self.EEB_exp_coef = np.exp(-self.EEB_decay * self.dt)

        self.turner_pre_lag_ticks = int(turner_prepost_lag[0] / self.dt)
        self.turner_post_lag_ticks = int(turner_prepost_lag[1] / self.dt)

        self.reset()
        # Rest-bout duration distribution
        # Trying to fit curve in fig 3 Ueno(2012)
        # For a=1.5 as mentioned we don't get visual fit. We try other values
        # self.rest_duration_range = (1, 10001)  # in sec
        # self.rest_duration_range = rest_duration_range  # in sec
        # self.rest_power_coef = float(np.random.normal(loc=rest_power_coef, scale=rest_power_coef_std, size=1))
        # self.pause_dist = PowerLawDist(range=self.rest_duration_range, coef=self.rest_power_coef,
        #                               a=self.rest_duration_range[0], b=self.rest_duration_range[1],
        #                               name='power_law_dist')
        if pause_dist['name'] == 'powerlaw':
            self.pause_min, self.pause_max = np.round(np.array(pause_dist['range']) / self.dt).astype(int)
            self.pause_dist = sampling.truncated_power_law(a=pause_dist['alpha'], xmin=self.pause_min,
                                                           xmax=self.pause_max)
        elif pause_dist['name'] == 'lognormal':
            self.pause_dist = None
            self.pause_min, self.pause_max = pause_dist['range']
            self.pause_mean, self.pause_std = pause_dist['mu'], pause_dist['sigma']
            # self.pause_dist = self.lognormal_discrete(mu=int(self.pause_mean / self.dt),
            #                                           sigma=int(self.pause_std / self.dt),
            #                                           min=int(self.pause_min / self.dt),
            #                                           max=int(self.pause_max / self.dt))

        self.stridechain_min, self.stridechain_max = np.array(stridechain_dist['range']).astype(int)
        if stridechain_dist['name'] == 'powerlaw':
            self.stridechain_dist = sampling.truncated_power_law(a=stridechain_dist['alpha'], xmin=self.stridechain_min,
                                                                 xmax=self.stridechain_max)
        elif stridechain_dist['name'] == 'lognormal':
            self.stridechain_mean, self.stridechain_std = stridechain_dist['mu'], stridechain_dist['sigma']
            self.stridechain_dist = self.lognormal_discrete(mu=self.stridechain_mean, sigma=self.stridechain_std,
                                                            min=self.stridechain_min, max=self.stridechain_max)

    def lognormal_discrete(self, mu, sigma, min, max):
        Dd = lognorm(s=sigma, loc=0.0, scale=np.exp(mu))
        pk2 = Dd.cdf(np.arange(min + 1, max + 2)) - Dd.cdf(np.arange(min, max + 1))
        pk2 = pk2 / np.sum(pk2)
        xrng = np.arange(min, max + 1, 1)
        return rv_discrete(a=min, b=max, values=(xrng, pk2))

    def generate_stridechain_length(self):
        if self.stridechain_dist is None:
            return sampling.sample_lognormal_int(mean=self.stridechain_mean, sigma=self.stridechain_std,
                                                 xmin=self.stridechain_min, xmax=self.stridechain_max)
        else:
            return self.stridechain_dist.rvs(size=1)[0]

    def generate_pause_duration(self):
        if self.pause_dist is None:
            return sampling.sample_lognormal(mean=self.pause_mean, sigma=self.pause_std,
                                             xmin=self.pause_min, xmax=self.pause_max)
        else:
            return self.pause_dist.rvs(size=1)[0] * self.dt

    def initialize(self):
        self.pause_dur = np.nan
        self.pause_start = False
        self.pause_stop = False
        self.pause_id = np.nan

        self.stridechain_dur = np.nan
        self.stridechain_start = False
        self.stridechain_stop = False
        self.stridechain_id = np.nan
        self.stridechain_length = np.nan

        # self.feed_bout_dur = np.nan

    def reset(self):
        # Initialize internal variables
        self.initialize()
        self.t = 0
        self.total_t = 0

        self.turner_pre_lag = 0
        self.turner_post_lag = 0

        self.pause_counter = 0
        self.current_pause_duration = None
        self.cum_pause_dur = 0

        self.stridechain_counter = 0
        self.current_stridechain_length = None
        self.cum_stridechain_dur = 0
        self.current_numstrides = 0

    def step(self):
        self.initialize()
        # Check if intermitter is turned on (it could have been turned on last in the previous timestep)
        self.update_state()
        self.update_turner()
        # If the intermitter is on ...
        if self.effector:
            self.pause_id = self.pause_counter
            # ...start an inactivity bout if there is none already running ...
            if self.current_pause_duration is None:
                self.current_pause_duration = self.generate_pause_duration()
                self.pause_start = True
                #  ... and turn off the underlying components
                self.inhibit_locomotion()
            # ... advance the timer of the current inactivity bout ...
            super().count_time()
            if self.intermittent_turner:
                if self.t > self.current_pause_duration - self.turner_pre_lag_ticks * self.dt:
                    if self.nengo_manager is None:
                        self.turner.start_effector()
                    else:
                        self.turner.set_freq(self.turner.default_freq)
            # ... if end of current inactivity bout is reached turn intermitter off ...
            if self.t > self.current_pause_duration:
                self.register_pause()
                # ... and turn on locomotion
                self.current_stridechain_length = self.generate_stridechain_length()
                self.stridechain_start = True
                self.stop_effector()
                self.disinhibit_locomotion()
        else:
            self.stridechain_id = self.stridechain_counter

    def disinhibit_locomotion(self):
        if self.nengo_manager is None:
            if np.random.uniform(0, 1, 1) >= self.EEB:
                if self.crawl_bouts:
                    self.crawler.start_effector()
            else:
                if self.feed_bouts:
                    self.feeder.start_effector()
        else:
            if np.random.uniform(0, 1, 1) >= self.EEB:
                self.crawler.set_freq(self.crawler.default_freq)
            else:
                self.feeder.set_freq(self.feeder.default_freq)

    def inhibit_locomotion(self):
        if self.nengo_manager is None:
            if self.crawl_bouts:
                self.crawler.stop_effector()
            if self.feed_bouts:
                self.feeder.stop_effector()
            if self.intermittent_turner:
                self.turner_post_lag = self.turner_post_lag_ticks
        else:
            self.crawler.set_freq(0)
            self.feeder.set_freq(0)
            self.turner_post_lag = self.turner_post_lag_ticks

    def update_turner(self):
        if self.intermittent_turner:
            self.turner_post_lag -= 1
            if self.turner_post_lag <= 0:
                self.turner_post_lag = 0
                if self.nengo_manager is None:
                    self.turner.stop_effector()
                else:
                    self.turner.set_freq(0)

    def update_state(self):
        if self.nengo_manager is None:
            if self.crawler:
                if self.crawler.complete_iteration:
                    self.current_numstrides += 1
                    if self.current_numstrides >= self.current_stridechain_length:
                        self.start_effector()
                        self.register_stridechain()
            if self.feeder:
                if self.feeder.complete_iteration:
                    if np.random.uniform(0, 1, 1) >= self.EEB:
                        self.start_effector()
        else:
            if not self.effector:
                if np.random.uniform(0, 1, 1) > 0.97:
                    self.start_effector()

    def register_stridechain(self):
        self.stridechain_counter += 1
        self.stridechain_dur = self.t - self.dt
        self.cum_stridechain_dur += self.stridechain_dur
        self.stridechain_length = self.current_numstrides
        self.t = 0
        self.stridechain_stop = True
        self.current_numstrides = 0
        self.current_stridechain_length = None

    def register_pause(self):
        self.pause_counter += 1
        self.pause_dur = self.t - self.dt
        self.cum_pause_dur += self.pause_dur
        self.current_pause_duration = None
        self.t = 0
        self.pause_stop = True
        pass


class BranchIntermitter(Effector):
    def __init__(self, rest_duration_range=(None, None), dt=0.1, sigma=1.0, m=0.01, N=1000):
        self.dt = dt
        self.N = N
        self.m = m
        self.xmin, self.xmax = rest_duration_range
        if self.xmin is None:
            self.xmin = self.dt
        if self.xmax is None:
            self.xmax = 2 ** 9

        # Starting in activity state
        self.S = 0
        self.c_act = 0
        self.c_rest = 0

        self.rest_start = False
        self.rest_stop = False
        self.non_rest_start = True
        self.non_rest_stop = False
        self.rest_dur = np.nan
        self.non_rest_dur = np.nan

        def step():
            self.rest_start = False
            self.rest_stop = False
            self.non_rest_start = False
            self.non_rest_stop = False
            self.rest_dur = np.nan
            self.non_rest_dur = np.nan
            # TODO Right now low threshold has no effect and equals dt
            p = np.clip(sigma * self.S / self.N + self.m / self.N, a_min=0, a_max=1)
            self.S = np.random.binomial(self.N, p)
            if (self.S <= 0):
                if self.c_rest > 0:
                    self.rest_dur = self.c_rest
                    self.rest_stop = True
                    self.non_rest_start = True
                    # D_rest.append(c_rest)
                    self.disinhibit_locomotion()

                    self.c_rest = 0
                self.c_act += self.dt
                if self.c_act >= self.xmax:
                    self.non_rest_dur = self.c_act
                    self.non_rest_stop = True
                    self.rest_start = True
                    self.inhibit_locomotion()
                    self.c_act = 0
                    self.S = 1
                    return
            elif (self.S > 0):
                if self.c_act > 0:
                    # D_act.append(c_act)

                    self.non_rest_dur = self.c_act
                    self.non_rest_stop = True
                    self.rest_start = True
                    self.inhibit_locomotion()
                    self.c_act = 0
                self.c_rest += dt
                if self.c_rest >= self.xmax:
                    self.rest_dur = self.c_rest
                    self.rest_stop = True
                    self.non_rest_start = True
                    self.disinhibit_locomotion()
                    self.c_rest = 0
                    self.S = 0
                    return


class Olfactor(Effector):
    def __init__(self,
                 odor_dict={}, perception='log', decay_coef=1, olfactor_noise=0, **kwargs):
        super().__init__(**kwargs)

        self.perception = perception
        self.decay_coef = decay_coef
        self.noise = olfactor_noise
        self.A0, self.A1 = [-1.0, 1.0]
        self.activation = 0
        # self.odor_layers = odor_layers
        # self.num_layers = len(odor_layers)
        self.init_gain(odor_dict)

    def set_gain(self, value, odor_id):
        self.gain[odor_id] = value

    def reset_gain(self, odor_id):
        self.gain[odor_id] = self.base_gain[odor_id]

    def reset_all_gains(self):
        self.gain = self.base_gain

    # def get_gain(self, odor_id):
    #     return self.gain[odor_id]

    def compute_dCon(self, concentrations):
        for id, cur in concentrations.items():
            if id not in self.odor_ids:
                self.add_novel_odor(id, con=cur, gain=0.0)
            else:
                prev = self.Con[id]
                if self.perception == 'log':
                    self.dCon[id] = cur / prev - 1 if prev != 0 else 0
                elif self.perception == 'linear':
                    self.dCon[id] = cur - prev
        self.Con = concentrations

    def get_dCon(self):
        return self.dCon

    def get_gain(self):
        return self.gain

    def step(self, cons):
        if len(cons) == 0:
            self.activation = 0
        else:
            self.compute_dCon(cons)
            self.activation *= 1 - self.dt * self.decay_coef
            self.activation += self.dt * np.sum([self.gain[id] * self.dCon[id] for id in self.odor_ids])

            if self.activation > self.A1:
                self.activation = self.A1
            elif self.activation < self.A0:
                self.activation = self.A0

        return self.activation

    def init_gain(self, odor_dict):
        if odor_dict in [None, 'empty_dict']:
            odor_dict = {}
        self.base_gain = {}
        self.Con = {}
        self.dCon = {}
        self.Nodors = len(odor_dict)
        self.odor_ids = list(odor_dict.keys())
        for id, p in odor_dict.items():
            if type(p) == dict:
                m, s = p['mean'], p['std']
                self.base_gain[id] = float(np.random.normal(m, s, 1))
            else:
                self.base_gain[id] = p
            self.Con[id] = 0.0
            self.dCon[id] = 0.0
        self.gain = self.base_gain

    def add_novel_odor(self, id, con=0.0, gain=0.0):
        self.Nodors += 1
        self.odor_ids.append(id)
        self.base_gain[id] = gain
        self.gain[id] = gain
        self.dCon[id] = 0.0
        self.Con[id] = con


class RLmemory(Effector):
    def __init__(self,brain, gain, decay_coef, DeltadCon=0.02, state_spacePerOdorSide=3, gain_space=[-500, -50, 50, 500],
                 decay_coef_space = None, update_dt=2, train_dur=30, alpha=0.05, gamma=0.6, epsilon=0.15, **kwargs):
        super().__init__(**kwargs)
        self.brain = brain
        self.effector = True
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.DeltadCon = DeltadCon
        self.gain_space = gain_space
        if decay_coef_space is None :
            decay_coef_space = [decay_coef]
        self.decay_coef_space = decay_coef_space
        # self.gain = gain
        self.odor_ids = list(gain.keys())
        self.Nodors = len(self.odor_ids)
        self.fit_pars = self.odor_ids + ['decay_coef']
        # self.Nfit_pars =self.Nodors + 1
        self.actions = [ii for ii in itertools.product(gain_space, repeat=self.Nodors)]
        self.actions = [flatten_tuple(ii) for ii in itertools.product(self.actions, list(self.decay_coef_space))]
        self.state_spacePerOdorSide = state_spacePerOdorSide
        self.state_space = np.array(
            [ii for ii in itertools.product(range(2 * self.state_spacePerOdorSide + 1), repeat=self.Nodors)])
        # self.q_table = [np.zeros(len(self.state_space), len(self.actions)) for ii in odor_ids]
        self.q_table = np.zeros((self.state_space.shape[0], len(self.actions)))
        self.lastAction = 0
        self.lastState = 0
        self.Niters = int(update_dt * 60 / self.dt)
        self.iterator = self.Niters
        self.train_dur = train_dur
        self.rewardSum = 0
        self.best_gain = gain
        # self.decay_coef = decay_coef
        self.best_decay_coef = decay_coef

        self.table=False

    def state_collapse(self, dCon):

        if len(dCon) > 0:
            # if len(dCon) == 1:
            dCon = [dCon]
        stateV = []
        for index in range(len(dCon)):
            # print(dCon, dCon[index], index, type(dCon))
            # raise
            for i in dCon[index]:
                dConI = dCon[index][i]
                stateIntermitt = np.zeros(self.state_spacePerOdorSide)
                for ii in range(self.state_spacePerOdorSide):
                    stateIntermitt[ii] = np.abs(dConI) > (ii + 1) * self.DeltadCon

            stateV.append(int(np.sign(dConI) * (np.sum(stateIntermitt)) + self.state_spacePerOdorSide))
        state = np.where((self.state_space == stateV).all(axis=1))[0][0]
        return state

    def step(self, gain, dCon, reward, decay_coef):
        if self.table==False :
            temp = self.brain.agent.model.table_collector
            if temp is not  None :
                self.table = temp.tables['best_gains'] if 'best_gains' in list(temp.tables.keys()) else None

        self.count_time()
        if self.effector and self.total_t > self.train_dur * 60:
            self.effector = False
            print(f'Training stopped after {self.train_dur} minutes')
            print(f'Best gain : {self.best_gain}')
        if self.effector:
            self.rewardSum += int(reward) - 0.001
            if self.iterator >= self.Niters:
                self.iterator = 0
                state = self.state_collapse(dCon)

                if random.uniform(0, 1) < self.epsilon:
                    actionID = random.randrange(len(self.actions))
                else:
                    actionID = np.argmax(self.q_table[state])  # Exploit learned values

                old_value = self.q_table[self.lastState, self.lastAction]
                next_max = np.max(self.q_table[state])

                new_value = (1 - self.alpha) * old_value + self.alpha * (self.rewardSum + self.gamma * next_max)

                self.q_table[self.lastState, self.lastAction] = new_value
                self.lastAction = actionID
                self.lastState = state

                action = self.actions[actionID]
                for ii, id in enumerate(self.odor_ids):
                    gain[id] = action[ii]
                decay_coef = action[-1]
                best_combo=self.get_best_combo()
                self.best_gain = {id : best_combo[id] for id in self.odor_ids}
                self.best_decay_coef = best_combo['decay_coef']
                if self.table :
                    for col in list(self.table.keys()) :
                        try :
                            self.table[col].append(getattr(self.brain.agent, col))
                        except :
                            self.table[col].append(np.nan)
                self.rewardSum = 0
            self.iterator += 1

            return gain, decay_coef
        else:
            return self.best_gain, self.best_decay_coef

    # def get_best_gain(self):
    #     return dict(zip(self.odor_ids, self.actions[np.argmax(np.mean(self.q_table, axis=0))]))

    def get_best_combo(self):
        return dict(zip(self.fit_pars, self.actions[np.argmax(np.mean(self.q_table, axis=0))]))


# class TurnerModulator:
#     def __init__(self, base_activation, activation_range, **kwargs):
#         self.base_activation = base_activation
#         self.activation_range = activation_range
#         self.range_upwards = self.activation_range[1] - self.base_activation
#         self.range_downwards = self.base_activation - self.activation_range[0]
#         self.activity = self.base_activation
#
#     # The olfactory input will lie in the range (-1,1) in the sense of positive (going up the gradient)  will cause
#     # less turns and negative (going down the gradient) will cause more turns. If this works in chemotaxis,
#     # this range (-1,1) could be regarded as the cumulative valence of olfactory input.
#     def update_activation(self, olfactory_activation):
#         # Map valence modulation to sigmoid accounting for the non middle location of base_activation
#         if olfactory_activation < 0:
#             self.activity = self.base_activation + self.range_downwards * olfactory_activation
#         elif olfactory_activation > 0:
#             self.activity = self.base_activation + self.range_upwards * olfactory_activation
#         else:
#             self.activity = self.base_activation
#
#     def step(self, olfactory_activation):
#         self.update_activation(olfactory_activation)
#         return self.activity
class Brain():
    def __init__(self, agent, modules, conf):
        self.agent = agent
        self.modules = modules
        self.conf = conf
        self.olfactory_activation = 0
        # self.crawler, self.turner, self.feeder, self.olfactor, self.intermitter = None, None, None, None, None

    def sense_odors(self, pos):
        cons = {}
        for id, layer in self.agent.model.odor_layers.items():
            v = layer.get_value(pos)
            cons[id] = v + np.random.normal(scale=v * self.olfactor.noise)
        return cons


class DefaultBrain(Brain):
    def __init__(self, **kwargs):
        Brain.__init__(self, **kwargs)
        dt = self.agent.model.dt
        if self.modules['interference']:
            self.osc_coupling = Oscillator_coupling(**self.conf['interference_params'])
        else:
            self.osc_coupling = Oscillator_coupling()
        if self.modules['crawler']:
            self.crawler = Crawler(dt=dt, **self.conf['crawler_params'])
            self.crawler.start_effector()
        else:
            self.crawler = None

        if self.modules['turner']:
            self.turner = Turner(dt=dt, **self.conf['turner_params'])
            self.turner.start_effector()
        else:
            self.turner = None
        if self.modules['feeder']:
            self.feeder = Feeder(dt=dt, model=self.agent.model, **self.conf['feeder_params'])
            self.feeder.stop_effector()
        else:
            self.feeder = None
        if self.modules['intermitter']:
            self.intermitter = Intermitter(dt=dt,
                                           crawler=self.crawler, turner=self.turner, feeder=self.feeder,
                                           **self.conf['intermitter_params'])
            self.intermitter.start_effector()
        else:
            self.intermitter = None
        # Initialize sensors
        if self.modules['olfactor']:
            self.olfactor = Olfactor(dt=dt, **self.conf['olfactor_params'])
        else:
            self.olfactor = None

        if self.modules['memory']:
            self.memory = RLmemory(brain=self, dt=dt, decay_coef=self.olfactor.decay_coef,
                                   gain=self.olfactor.gain, **self.conf['memory_params'])
        else:
            self.memory = None

    def run(self, pos):
        if self.intermitter:
            self.intermitter.step()

        # Step the feeder
        feed_motion = self.feeder.step() if self.feeder else False

        if self.memory:
            self.olfactor.gain, self.olfactor.decay_coef = self.memory.step(self.olfactor.get_gain(), self.olfactor.get_dCon(),
                                                  self.agent.food_detected is not None, self.olfactor.decay_coef)
        lin = self.crawler.step(self.agent.get_sim_length()) if self.crawler else 0
        self.olfactory_activation = self.olfactor.step(self.sense_odors(pos)) if self.olfactor else 0
        # ... and finally step the turner...
        ang = self.turner.step(inhibited=self.osc_coupling.step(crawler=self.crawler, feeder=self.feeder),
                               attenuation=self.osc_coupling.attenuation,
                               A_olf=self.olfactory_activation) if self.turner else 0

        return lin, ang, feed_motion
