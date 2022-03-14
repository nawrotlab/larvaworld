import numpy as np

from lib.anal.fitting import BoutGenerator
from lib.aux.ang_aux import restore_bend_2seg
from lib.model.modules.crawl_bend_interference import Oscillator_coupling
from lib.model.modules.crawler import Crawler
from lib.model.modules.feeder import Feeder
from lib.model.modules.intermitter import Intermitter, BranchIntermitter, NengoIntermitter
from lib.model.modules.turner import Turner, NeuralOscillator


class Locomotor:
    def __init__(self, dt, ang_mode='torque', crawler_noise=0, turner_input_noise=0, turner_output_noise=0,
                 torque_coef=1, ang_damp_coef=1, body_spring_k=1, bend_correction_coef = 1.4):
        self.dt = dt
        self.crawler, self.turner, self.feeder, self.intermitter = [None] * 4

        self.bend = 0.0
        self.ang_vel = 0.0
        self.lin_vel = 0.0
        self.feed_motion = False
        self.last_dist=0
        self.bend_correction_coef = bend_correction_coef

        self.crawler_noise = crawler_noise
        self.turner_input_noise = turner_input_noise
        self.turner_output_noise = turner_output_noise

        self.torque_coef = torque_coef
        self.ang_damp_coef = ang_damp_coef
        self.body_spring_k = body_spring_k

        self.cur_state = 'run'
        self.cur_run_dur=0
        self.cur_pause_dur=None

    def update(self):
        if self.cur_state == 'run':
            self.cur_run_dur += self.dt
        elif self.cur_state == 'pause':
            self.cur_pause_dur += self.dt

    def add_noise(self):
        self.lin_vel += np.random.normal(scale=self.crawler_noise)
        self.ang_vel = self.ang_vel * (1 + np.random.normal(scale=self.turner_output_noise))

    def scale2length(self, length):
        self.lin_vel *= length

    def bend_body(self, output):
        # print(output, self.bend, self.ang_vel)
        self.ang_vel += (
                                    output * self.torque_coef - self.ang_damp_coef * self.ang_vel - self.body_spring_k * self.bend) * self.dt
        self.bend += self.ang_vel * self.dt


class DefaultLocomotor(Locomotor):
    def __init__(self, conf, **kwargs):
        super().__init__(**kwargs)
        m, c = conf.modules, conf
        if m['crawler']:
            self.crawler = Crawler(dt=self.dt, **c['crawler_params'])
        if m['turner']:
            self.turner = Turner(dt=self.dt, **c['turner_params'])
        if m['feeder']:
            self.feeder = Feeder(dt=self.dt, **c['feeder_params'])
        self.coupling = Oscillator_coupling(locomotor=self, **c['interference_params']) if m[
            'interference'] else Oscillator_coupling(locomotor=self)
        if m['intermitter']:
            mode = c['intermitter_params']['mode'] if 'mode' in c['intermitter_params'].keys() else 'default'
            if mode == 'default':
                self.intermitter = Intermitter(locomotor=self, dt=self.dt, **c['intermitter_params'])
            elif mode == 'branch':
                self.intermitter = BranchIntermitter(locomotor=self, dt=self.dt, **c['intermitter_params'])

    def step(self, A_in=0, length=1):
        self.lin_vel, ang, feed_motion = 0, 0, False
        if self.intermitter:
            self.intermitter.step()
        if self.feeder:
            self.feed_motion = self.feeder.step()
        if self.crawler:
            self.lin_vel = self.crawler.step()
        if self.turner:
            self.ang_vel = self.turner.step(inhibited=self.coupling.step(),
                                            attenuation=self.coupling.attenuation,
                                            A_in=A_in)

        self.add_noise()
        self.scale2length(length)

        return self.lin_vel, self.ang_vel, self.feed_motion


class Levy_locomotor(Locomotor):
    def __init__(self, dt, conf=None,
                 pause_dist={'range': [0.4, 20.0], 'name': 'uniform'},
                 run_dist={'range': [1.0, 100.0], 'name': 'powerlaw', 'alpha': 1.44791},
                 lin_constant=0.001, ang_constant=60, **kwargs):
        super().__init__(dt=dt)
        self.lin_constant = lin_constant
        self.ang_constant = ang_constant

        self.run_dist = BoutGenerator(**run_dist, dt=self.dt)
        self.pause_dist = BoutGenerator(**pause_dist, dt=self.dt)
        self.cur_run_dur_max = self.run_dist.sample()
        self.cur_pause_dur_max = None

    def intermit(self):
        if self.cur_state == 'run' and self.cur_run_dur >= self.cur_run_dur_max:
            self.cur_run_dur = None
            self.cur_run_dur_max = None
            self.cur_pause_dur_max = self.pause_dist.sample()
            self.cur_pause_dur = 0
            self.cur_state = 'pause'
            self.lin_vel = 0
            self.ang_vel = self.ang_constant * np.random.choice([-1, 1])
        elif self.cur_state == 'pause' and self.cur_pause_dur >= self.cur_pause_dur_max:
            self.cur_pause_dur = None
            self.cur_pause_dur_max = None
            self.cur_run_dur_max = self.run_dist.sample()
            self.cur_run_dur = 0
            self.cur_state = 'run'
            self.lin_vel = self.lin_constant
            self.ang_vel = 0

    def step(self, A_in=0, length=1):
        self.intermit()
        self.update()
        self.add_noise()
        self.scale2length(length)
        return self.lin_vel, self.ang_vel, self.feed_motion


class Wystrach2016(Locomotor):
    def __init__(self, dt, conf=None, lin_constant=0.001,
                 turner_input_constant=19, w_ee=3.0, w_ce=0.1, w_ec=4.0, w_cc=4.0, m=100.0, n=2.0, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.lin_constant = lin_constant

        self.turner_input_constant = turner_input_constant
        self.neural_oscillator = NeuralOscillator(dt=self.dt, w_ee=w_ee, w_ce=w_ce, w_ec=w_ec, w_cc=w_cc, m=m, n=n)

    def step(self, A_in=0, length=1):
        self.bend = restore_bend_2seg(self.bend, self.last_dist, length, correction_coef=self.bend_correction_coef)
        self.lin_vel = self.lin_constant

        input = (self.turner_input_constant + A_in) * (1 + np.random.normal(scale=self.turner_input_noise))

        self.neural_oscillator.step(input)
        self.add_noise()
        self.bend_body(self.neural_oscillator.activity)


        self.scale2length(length)
        # print(self.lin_vel, length, self.lin_constant)
        self.last_dist=self.lin_vel * self.dt

        return self.lin_vel, self.ang_vel, self.feed_motion

    def step_TiPI(self, S, A_in=0):
        ang_vel, bend, lin = S

        lin = self.lin_constant
        lin += np.random.normal(scale=self.crawler_noise)

        input = (self.turner_input_constant + A_in) * (1 + np.random.normal(scale=self.turner_input_noise))

        self.neural_oscillator.step(input)
        output = self.neural_oscillator.activity * (1 + np.random.normal(scale=self.turner_output_noise))

        return np.array([output, lin])


class Davies2015(Locomotor):
    def __init__(self, dt, conf=None, lin_constant=0.001, min_run_dur=1.0,
                 theta_min_headcast=37, theta_max_headcast=120,
                 theta_max_weathervane=20, ang_vel_weathervane=60,
                 r_run2headcast=0.148, r_headcast2run=2.0,
                 r_weathervane_stop=2.0, r_weathervane_resume=1.0, **kwargs):
        super().__init__(dt=dt)
        self.lin_constant = lin_constant
        self.min_run_dur = min_run_dur
        self.theta_min_headcast, self.theta_max_headcast = theta_min_headcast, theta_max_headcast
        self.ang_vel_headcast = 2 * theta_max_headcast
        self.theta_max_weathervane, self.ang_vel_weathervane = theta_max_weathervane, ang_vel_weathervane
        self.r_run2headcast, self.r_headcast2run = r_run2headcast, r_headcast2run
        self.r_weathervane_stop, self.r_weathervane_resume = r_weathervane_stop, r_weathervane_resume

        self.cur_headcast = 0
        self.cur_weathervane = 0

    def step(self, A_in=0, length=1):
        if self.cur_state == 'run' and self.cur_run_dur >= self.min_run_dur:
            if np.random.uniform(0, 1, 1) <= self.r_run2headcast * self.dt:
                self.cur_state = 'headcast'
                sign = np.sign(self.bend)[0] if self.bend != 0 else np.random.choice([-1, 1], 1)[0]
                self.ang_vel = sign * self.ang_vel_headcast
        elif self.cur_state == 'headcast' and np.abs(
                self.cur_headcast) >= self.theta_min_headcast and np.sign(self.cur_headcast * self.ang_vel) == 1:
            if np.random.uniform(0, 1, 1) <= self.r_headcast2run * self.dt:
                self.cur_state = 'run'
                self.ang_vel = np.random.choice([-1, 1], 1)[0] * self.ang_vel_weathervane
        if self.cur_state == 'run':
            self.cur_run_dur += self.dt
            self.lin_vel = self.lin_constant
            self.cur_headcast = 0

            if self.ang_vel == 0.0:
                if np.random.uniform(0, 1, 1) <= self.r_weathervane_resume * self.dt:
                    self.ang_vel = np.random.choice([-1, 1], 1)[0] * self.ang_vel_weathervane
            else:
                if np.random.uniform(0, 1, 1) <= self.r_weathervane_stop * self.dt:
                    self.ang_vel = 0.0
                    self.cur_weathervane = 0

            if np.abs(self.cur_weathervane) >= self.theta_max_weathervane:
                self.ang_vel *= -1
            self.cur_weathervane += self.ang_vel * self.dt
        elif self.cur_state == 'headcast':
            self.lin_vel = 0
            self.cur_run_dur = 0
            self.cur_weathervane = 0
            if np.abs(self.cur_headcast) >= self.theta_max_headcast:
                self.ang_vel *= -1
            self.cur_headcast += self.ang_vel * self.dt
        self.add_noise()
        self.scale2length(length)
        return self.lin_vel, self.ang_vel, self.feed_motion


class Sakagiannis2022(Locomotor):
    def __init__(self, dt, conf=None,
                 pause_dist={'range': [0.4, 20.0], 'name': 'uniform'},
                 run_dist={'range': [1.0, 100.0], 'name': 'powerlaw', 'alpha': 1.44791},
                 initial_freq=1.418, step_mu=0.224, step_std=0.033, max_vel_phase=1.0,
                 attenuation=0.1, crawler_phi_range=(0.5, 1.5),
                 turner_input_constant=19,w_ee=3.0, w_ce=0.1, w_ec=4.0, w_cc=4.0, m=100.0, n=2.0, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.turner_input_constant = turner_input_constant
        self.neural_oscillator = NeuralOscillator(dt=self.dt, w_ee=w_ee, w_ce=w_ce, w_ec=w_ec, w_cc=w_cc, m=m, n=n)

        self.freq = initial_freq
        self.step_to_length_mu = step_mu
        self.step_to_length_std = step_std
        self.step_to_length = np.random.normal(loc=self.step_to_length_mu, scale=self.step_to_length_std)
        self.max_vel_phase = max_vel_phase * np.pi

        self.run_dist = BoutGenerator(**run_dist, dt=self.dt)
        self.pause_dist = BoutGenerator(**pause_dist, dt=self.dt)
        self.cur_run_dur_max = self.run_dist.sample()
        self.cur_pause_dur_max = None

        self.timesteps_per_iteration = int(round((1 / self.freq) / self.dt))
        self.d_phi = 2 * np.pi / self.timesteps_per_iteration
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0
        self.attenuation = attenuation
        self.crawler_phi_range = crawler_phi_range

    def oscillate(self):
        self.phi += self.d_phi
        if self.phi >= 2 * np.pi:
            self.phi %= 2 * np.pi
            self.complete_iteration = True
            self.iteration_counter += 1
            self.step_to_length = np.random.normal(loc=self.step_to_length_mu, scale=self.step_to_length_std)
        else:
            self.complete_iteration = False

    def intermit(self):
        if self.cur_state == 'run' and self.cur_run_dur >= self.cur_run_dur_max:
            self.cur_run_dur = None
            self.cur_run_dur_max = None
            self.cur_pause_dur_max = self.pause_dist.sample()
            self.cur_pause_dur = 0
            self.cur_state = 'pause'
        elif self.cur_state == 'pause' and self.cur_pause_dur >= self.cur_pause_dur_max:
            self.cur_pause_dur = None
            self.cur_pause_dur_max = None
            self.cur_run_dur_max = self.run_dist.sample()
            self.cur_run_dur = 0
            self.cur_state = 'run'

    def step(self, A_in=0, length=1):
        self.bend = restore_bend_2seg(self.bend, self.last_dist, length, correction_coef=self.bend_correction_coef)
        self.intermit()
        if self.cur_state == 'run':
            self.cur_run_dur += self.dt
            self.oscillate()
            self.lin_vel = self.freq * self.step_to_length * (1 + 0.6 * np.cos(self.phi - self.max_vel_phase))
            attenuation_coef = 1 if self.crawler_phi_range[0] <= self.phi/np.pi <= self.crawler_phi_range[
                1] else self.attenuation
        elif self.cur_state == 'pause':
            self.cur_pause_dur += self.dt
            self.phi = 0
            self.lin_vel = 0
            attenuation_coef = 1

        input = (self.turner_input_constant + A_in) * (1 + np.random.normal(scale=self.turner_input_noise))

        self.neural_oscillator.step(input)

        self.add_noise()
        self.bend_body(self.neural_oscillator.activity * attenuation_coef)
        self.scale2length(length)

        self.last_dist = self.lin_vel * self.dt
        return self.lin_vel, self.ang_vel, self.feed_motion

    def step_TiPI(self, S, A_in=0, ):
        ang_vel, bend, lin = S
        self.intermit()
        if self.cur_state == 'run':
            self.oscillate()
            lin = self.freq * self.step_to_length * (1 + 0.6 * np.cos(self.phi - self.max_vel_phase))
            attenuation_coef = 1 if self.crawler_phi_range[0] <= self.phi <= self.crawler_phi_range[
                1] else self.attenuation
        else:
            self.phi = 0
            lin = 0
            attenuation_coef = 1

        lin += np.random.normal(scale=self.crawler_noise)
        # lin *= length

        input = (self.turner_input_constant + A_in) * (1 + np.random.normal(scale=self.turner_input_noise))

        self.neural_oscillator.step(input)

        output = self.neural_oscillator.activity * (
                1 + np.random.normal(scale=self.turner_output_noise)) * attenuation_coef
        return np.array([output, lin])


Levy_locomotor_conf = {
    'lin_constant': 0.001,
    'ang_constant': 60,
    'crawler_noise': 0.0,
    'turner_input_noise': 0.0,
    'turner_output_noise': 0.0,
    'run_dist': {'range': [1.0, 100.0], 'name': 'powerlaw', 'alpha': 1.44791},
    'pause_dist': {'range': [0.4, 20.0], 'name': 'uniform'},
}

Wystrach2016_conf = {
    'lin_constant': 0.001,  # in m/s
    'torque_coef': 1.0,
    'ang_damp_coef': 1.0,
    'body_spring_k': 1.0,
    'turner_input_constant': 19.0,
    'crawler_noise': 0.0,
    'turner_input_noise': 0.0,
    'turner_output_noise': 0.0,
    'w_ee': 3.0,
    'w_ce': 0.1,
    'w_ec': 4.0,
    'w_cc': 4.0,
    'm': 100.0,
    'n': 2.0,
}

Davies2015_conf = {
    'lin_constant': 0.001,
    'min_run_dur': 1,
    'theta_min_headcast': 37,
    'theta_max_headcast': 120,
    'theta_max_weathervane': 20,
    'ang_vel_weathervane': 60.0,
    'r_run2headcast': 0.148,
    'r_headcast2run': 2.0,
    'r_weathervane_stop': 2.0,
    'r_weathervane_resume': 1.0,
}

Sakagiannis2022_conf = {
    'step_mu': 0.224,
    'step_std': 0.033,
    'max_vel_phase': 1.0,
    'initial_freq': 1.418,
    'torque_coef': 1.0,
    'ang_damp_coef': 1.0,
    'body_spring_k': 1.0,
    'turner_input_constant': 19.0,
    'crawler_noise': 0.0,
    'turner_input_noise': 0.0,
    'turner_output_noise': 0.0,
    'attenuation': 0.0,
    'crawler_phi_range': (0.5, 1.5),
    'run_dist': {'range': [1.0, 100.0], 'name': 'powerlaw', 'alpha': 1.44791},
    'pause_dist': {'range': [0.4, 20.0], 'name': 'uniform'},
}

if __name__ == '__main__':
    L = Davies2015(dt=0.1, **Davies2015_conf)
    for i in range(1000):
        L.step()
