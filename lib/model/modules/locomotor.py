import numpy as np

from lib.anal.fitting import BoutGenerator, gaussian
from lib.aux.ang_aux import restore_bend_2seg
from lib.model.modules.crawl_bend_interference import DefaultCoupling, SquareCoupling, PhasicCoupling
from lib.model.modules.crawler import Crawler
from lib.model.modules.feeder import Feeder
from lib.model.modules.intermitter import Intermitter, BranchIntermitter, NengoIntermitter
from lib.model.modules.turner import Turner, NeuralOscillator


class Locomotor:
    def __init__(self, dt, ang_mode='torque', crawler_noise=0, turner_input_noise=0, turner_output_noise=0,
                 torque_coef=1, ang_damp_coef=1, body_spring_k=1, bend_correction_coef=1.4):
        self.dt = dt
        self.crawler, self.turner, self.feeder, self.intermitter = [None] * 4

        self.bend = 0.0
        self.ang_vel = 0.0
        self.lin_vel = 0.0
        self.feed_motion = False
        self.last_dist = 0
        self.bend_correction_coef = bend_correction_coef

        self.crawler_noise = crawler_noise
        self.turner_input_noise = turner_input_noise
        self.turner_output_noise = turner_output_noise

        self.torque_coef = torque_coef
        self.ang_damp_coef = ang_damp_coef
        self.body_spring_k = body_spring_k

        self.cur_state = 'run'
        self.cur_run_dur = 0
        self.cur_pause_dur = None

    def update(self):
        if self.cur_state == 'run':
            self.cur_run_dur += self.dt
        elif self.cur_state == 'pause':
            self.cur_pause_dur += self.dt

    def add_noise(self):
        self.lin_vel = self.lin_vel *(1+np.random.normal(scale=self.crawler_noise))
        self.ang_vel = self.ang_vel * (1 + np.random.normal(scale=self.turner_output_noise))

    def scale2length(self, length):
        self.lin_vel *= length

    def bend_body(self, output):
        dv=output * self.torque_coef - self.ang_damp_coef * self.ang_vel - self.body_spring_k * self.bend
        self.ang_vel += dv * self.dt
        self.bend += self.ang_vel * self.dt


class DefaultLocomotor(Locomotor):
    def __init__(self, conf, **kwargs):
        super().__init__(**kwargs)
        m, c = conf.modules, conf
        if m['crawler']:
            self.crawler = Crawler(dt=self.dt, **c.crawler_params)
        if m['turner']:
            self.turner = Turner(dt=self.dt, **c.turner_params)
        if m['feeder']:
            self.feeder = Feeder(dt=self.dt, **c.feeder_params)
        mode = c.interference_params.mode if 'mode' in c.interference_params.keys() else 'default'
        if mode == 'default':
            self.coupling = DefaultCoupling(locomotor=self, **c.interference_params)
        elif mode == 'square':
            self.coupling = SquareCoupling(locomotor=self, **c.interference_params)
        elif mode == 'phasic':
            self.coupling = PhasicCoupling(locomotor=self, **c.interference_params)

        if m['intermitter']:
            mode = c.intermitter_params.mode if 'mode' in c.intermitter_params.keys() else 'default'
            if mode == 'default':
                self.intermitter = Intermitter(locomotor=self, dt=self.dt, **c.intermitter_params)
            elif mode == 'branch':
                self.intermitter = BranchIntermitter(locomotor=self, dt=self.dt, **c.intermitter_params)
            # elif mode == 'simple':
            #     self.intermitter = SimpleIntermitter(locomotor=self, dt=self.dt, **c.intermitter_params)

    def step(self, A_in=0, length=1):
        self.lin_vel, Aang, feed_motion = 0, 0, False
        if self.intermitter:
            self.intermitter.step()
        if self.feeder:
            self.feed_motion = self.feeder.step()
        if self.crawler:
            self.lin_vel = (self.crawler.step() + np.random.normal(scale=self.crawler_noise)) * length
        if self.turner:
            Aang0=self.turner.step(A_in=A_in)
            Aang = self.coupling.step(self.crawler.phi) * Aang0 * (1 + np.random.normal(scale=self.turner_output_noise))

        return self.lin_vel, Aang, self.feed_motion


class Levy_locomotor(Locomotor):
    def __init__(self, dt, conf=None,
                 pause_dist={'range': [0.4, 20.0], 'name': 'uniform'},
                 run_dist={'range': [1.0, 100.0], 'name': 'powerlaw', 'alpha': 1.44791},
                 run_scaled_velocity_mean=None,run_velocity_mean=None, ang_vel_headcast=0.3, **kwargs):
        super().__init__(dt=dt)
        self.run_scaled_velocity_mean = run_scaled_velocity_mean
        self.run_velocity_mean = run_velocity_mean
        self.ang_vel_headcast = ang_vel_headcast

        self.run_dist = BoutGenerator(**run_dist, dt=self.dt)
        self.pause_dist = BoutGenerator(**pause_dist, dt=self.dt)
        self.cur_run_dur_max = self.run_dist.sample()
        self.cur_pause_dur_max = None

    def step(self, A_in=0, length=1):
        if self.cur_state == 'run' and self.cur_run_dur >= self.cur_run_dur_max:
            self.cur_run_dur = None
            self.cur_run_dur_max = None
            self.cur_pause_dur_max = self.pause_dist.sample()
            self.cur_pause_dur = 0
            self.cur_state = 'pause'
            self.lin_vel = 0
            self.ang_vel = self.ang_vel_headcast * np.random.choice([-1, 1])
        elif self.cur_state == 'pause' and self.cur_pause_dur >= self.cur_pause_dur_max:
            self.cur_pause_dur = None
            self.cur_pause_dur_max = None
            self.cur_run_dur_max = self.run_dist.sample()
            self.cur_run_dur = 0
            self.cur_state = 'run'
            self.lin_vel = self.run_scaled_velocity_mean * length if self.run_scaled_velocity_mean is not None else self.run_velocity_mean
            self.ang_vel = 0
        self.update()
        self.add_noise()
        #self.scale2length(length)
        return self.lin_vel, self.ang_vel, self.feed_motion


class Wystrach2016(Locomotor):
    def __init__(self, dt, conf=None, run_scaled_velocity_mean=None,run_velocity_mean=None,
                 turner_input_constant=19, w_ee=3.0, w_ce=0.1, w_ec=4.0, w_cc=4.0, m=100.0, n=2.0, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.run_scaled_velocity_mean = run_scaled_velocity_mean
        self.run_velocity_mean = run_velocity_mean

        self.turner_input_constant = turner_input_constant
        self.neural_oscillator = NeuralOscillator(dt=self.dt, w_ee=w_ee, w_ce=w_ce, w_ec=w_ec, w_cc=w_cc, m=m, n=n)

    def step(self, A_in=0, length=1):
        self.bend = restore_bend_2seg(self.bend, self.last_dist, length, correction_coef=self.bend_correction_coef)
        self.lin_vel =  self.run_scaled_velocity_mean * length if self.run_scaled_velocity_mean is not None else self.run_velocity_mean

        input = (self.turner_input_constant + A_in) * (1 + np.random.normal(scale=self.turner_input_noise))

        self.neural_oscillator.step(input)
        self.add_noise()
        self.bend_body(self.neural_oscillator.activity)

        #self.scale2length(length)
        # print(self.lin_vel, length, self.lin_constant)
        self.last_dist = self.lin_vel * self.dt

        return self.lin_vel, self.ang_vel, self.feed_motion



class Davies2015(Locomotor):
    def __init__(self, dt, conf=None, run_scaled_velocity_mean=None, run_velocity_mean=None, run_dur_min=1.0,
                 theta_min_headcast=37, theta_max_headcast=120,
                 theta_max_weathervane=20, ang_vel_weathervane=0.1, ang_vel_headcast=0.3,
                 r_run2headcast=0.148, r_headcast2run=2.0,
                 r_weathervane_stop=2.0, r_weathervane_resume=1.0, **kwargs):
        super().__init__(dt=dt)
        self.run_scaled_velocity_mean = run_scaled_velocity_mean
        self.run_velocity_mean = run_velocity_mean
        self.run_dur_min = run_dur_min
        self.theta_min_headcast, self.theta_max_headcast = theta_min_headcast, theta_max_headcast
        self.ang_vel_headcast = ang_vel_headcast
        self.theta_max_weathervane, self.ang_vel_weathervane = theta_max_weathervane, ang_vel_weathervane
        self.r_run2headcast, self.r_headcast2run = r_run2headcast, r_headcast2run
        self.r_weathervane_stop, self.r_weathervane_resume = r_weathervane_stop, r_weathervane_resume

        self.cur_headcast = 0
        self.cur_weathervane = 0

    def step(self, A_in=0, length=1):
        if self.cur_state == 'run' and self.cur_run_dur >= self.run_dur_min:
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
            self.lin_vel =  self.run_scaled_velocity_mean * length if self.run_scaled_velocity_mean is not None else self.run_velocity_mean
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
        return self.lin_vel, self.ang_vel, self.feed_motion

class Sakagiannis2022(Locomotor):
    def __init__(self, dt, conf=None,
                 pause_dist={'range': [0.4, 20.0], 'name': 'uniform'},
                 run_dist={'range': [1.0, 100.0], 'name': 'powerlaw', 'alpha': 1.44791},
                 stridechain_dist=None,
                 initial_freq=1.418, step_mu=0.224, step_std=0.033,
                 attenuation_min=0.2, attenuation_max=0.31, max_vel_phase=3.6,
                 turner_input_constant=19, w_ee=3.0, w_ce=0.1, w_ec=4.0, w_cc=4.0, m=100.0, n=2.0, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.turner_input_constant = turner_input_constant
        self.neural_oscillator = NeuralOscillator(dt=self.dt, w_ee=w_ee, w_ce=w_ce, w_ec=w_ec, w_cc=w_cc, m=m, n=n)

        self.freq = initial_freq
        self.step_to_length_mu = step_mu
        self.step_to_length_std = step_std
        self.step_to_length = self.new_stride

        if stridechain_dist is None :
            self.run_dist = BoutGenerator(**run_dist, dt=self.dt)
            self.cur_run_dur_max = self.run_dist.sample()
            self.current_numstrides = None
            self.stridechain_dist = None
        else :
            self.stridechain_dist = BoutGenerator(**stridechain_dist, dt=1)
            self.cur_stridechain_length = self.stridechain_dist.sample()
            self.current_numstrides = 0
            self.run_dist = None

        self.pause_dist = BoutGenerator(**pause_dist, dt=self.dt)
        self.cur_pause_dur_max = None

        self.d_phi = 2 * np.pi*self.dt*self.freq
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0
        self.attenuation_min = attenuation_min
        self.attenuation_max = attenuation_max
        self.max_vel_phase = max_vel_phase
        self.max_attenuation_phase = max_vel_phase-1.2

    @ property
    def attenuation_func(self):
        a=gaussian(self.phi, self.max_attenuation_phase, 1) * self.attenuation_max + self.attenuation_min
        if a>=1 :
            a=1
        elif a<=0 :
            a=0
        return a

    @ property
    def new_stride(self):
        return np.random.normal(loc=self.step_to_length_mu, scale=self.step_to_length_std)
    def oscillate(self):
        self.phi += self.d_phi
        if self.phi >= 2 * np.pi:
            self.phi %= 2 * np.pi
            self.complete_iteration = True
            self.iteration_counter += 1

        else:
            self.complete_iteration = False

    @ property
    def run_termination(self):
        if self.stridechain_dist is not None and self.current_numstrides>=self.cur_stridechain_length :
            self.cur_stridechain_length = None
            self.current_numstrides = None
            return True
        elif self.run_dist is not None and self.cur_run_dur >= self.cur_run_dur_max:
            self.cur_run_dur = None
            self.cur_run_dur_max = None
            return True
        else :
            return False

    def run_initiation(self):
        if self.stridechain_dist is not None:
            self.cur_stridechain_length = self.stridechain_dist.sample()
            self.current_numstrides = 0
        elif self.run_dist is not None :
            self.cur_run_dur_max = self.run_dist.sample()
            self.cur_run_dur = 0
        self.cur_state = 'run'


    def intermit(self):
        if self.cur_state == 'run' and self.run_termination:
            self.cur_pause_dur_max = self.pause_dist.sample()
            self.cur_pause_dur = 0
            self.cur_state = 'pause'
        elif self.cur_state == 'pause' and self.cur_pause_dur >= self.cur_pause_dur_max:
            self.cur_pause_dur = None
            self.cur_pause_dur_max = None
            self.run_initiation()

    def step(self, A_in=0, length=1):
        self.bend = restore_bend_2seg(self.bend, self.last_dist, length, correction_coef=self.bend_correction_coef)
        self.intermit()
        if self.cur_state == 'run':
            if self.cur_run_dur is not None :
                self.cur_run_dur += self.dt
            self.oscillate()
            if self.complete_iteration and self.current_numstrides is not None :
                self.current_numstrides+=1
                self.step_to_length = self.new_stride
            self.lin_vel = self.freq * self.step_to_length * (1 + 0.6 * np.cos(self.phi - self.max_vel_phase))*length
            attenuation_coef = self.attenuation_func
        elif self.cur_state == 'pause':
            self.cur_pause_dur += self.dt
            self.phi = 0
            self.lin_vel = 0
            attenuation_coef = 1

        input = (self.turner_input_constant + A_in) * (1 + np.random.normal(scale=self.turner_input_noise))

        self.neural_oscillator.step(input)

        self.add_noise()
        self.bend_body(self.neural_oscillator.activity * attenuation_coef)
        self.last_dist = self.lin_vel * self.dt
        return self.lin_vel, self.ang_vel, self.feed_motion



Levy_locomotor_conf = {
    'run_scaled_velocity': 0.1,
    'ang_vel_headcast': np.deg2rad(60),
    'crawler_noise': 0.0,
    'turner_input_noise': 0.0,
    'turner_output_noise': 0.0,
    'run_dist': {'range': [1.0, 100.0], 'name': 'powerlaw', 'alpha': 1.44791},
    'pause_dist': {'range': [0.4, 20.0], 'name': 'uniform'},
}

Wystrach2016_conf = {
    'run_scaled_velocity': 0.1,  # in m/s
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
    'run_scaled_velocity': 0.1,
    'min_run_dur': 1,
    'theta_min_headcast': 37,
    'theta_max_headcast': 120,
    'theta_max_weathervane': 20,
    'ang_vel_weathervane': np.deg2rad(60),
    'ang_vel_headcast': np.deg2rad(60) * 2,
    'r_run2headcast': 0.148,
    'r_headcast2run': 2.0,
    'r_weathervane_stop': 2.0,
    'r_weathervane_resume': 1.0,
}

Sakagiannis2022_conf = {
    'step_mu': 0.224,
    'step_std': 0.033,
    'max_vel_phase': 3.6,
    'initial_freq': 1.418,
    'torque_coef': 1.0,
    'ang_damp_coef': 1.0,
    'body_spring_k': 1.0,
    'turner_input_constant': 19.0,
    'crawler_noise': 0.0,
    'turner_input_noise': 0.0,
    'turner_output_noise': 0.0,
    'attenuation_min': 0.2,
    'attenuation_max': 0.31,
    #'crawler_phi_range': (0.5, 1.5),
    'run_dist': {'range': [1.0, 100.0], 'name': 'powerlaw', 'alpha': 1.44791},
    'pause_dist': {'range': [0.4, 20.0], 'name': 'uniform'},
}

if __name__ == '__main__':
    dt=0.065
    l = 1
    labs = ["Levy", "Wystrach2016", "Davies2015", "Sakagiannis2022"]
    loco_pairs = [[Levy_locomotor, Levy_locomotor_conf], [Wystrach2016, Wystrach2016_conf],
                  [Davies2015, Davies2015_conf], [Sakagiannis2022, Sakagiannis2022_conf]]
    for lab,(func,conf) in zip(labs, loco_pairs) :
        L = func(dt=dt, **conf)
        vs = []
        for i in range(5000):
            v, fov, feed = L.step(length=l)
            vs.append(v)
        print(lab, np.mean(vs))
        # break
