import numpy as np

from lib.anal.fitting import BoutGenerator, gaussian
from lib.aux.ang_aux import restore_bend_2seg
from lib.aux.dictsNlists import AttrDict
from lib.model.modules.crawl_bend_interference import DefaultCoupling, SquareCoupling, PhasicCoupling
from lib.model.modules.crawler import Crawler
from lib.model.modules.feeder import Feeder
from lib.model.modules.intermitter import Intermitter, BranchIntermitter, NengoIntermitter
from lib.model.modules.turner import Turner, NeuralOscillator


class Locomotor:
    def __init__(self, dt, ang_mode='torque', lin_mode='velocity', offline=False, crawler_noise=0, turner_input_noise=0,
                 turner_output_noise=0,
                 torque_coef=1.78, ang_damp_coef=2.6, body_spring_k=50, bend_correction_coef=1.6, ang_vel_coef=1):
        self.offline = offline
        self.ang_mode = ang_mode
        self.lin_mode = lin_mode
        self.dt = dt
        self.crawler, self.turner, self.feeder, self.intermitter = [None] * 4
        self.ang_activity = 0.0
        self.lin_activity = 0.0

        self.bend = 0.0
        self.ang_vel = 0.0
        self.lin_vel = 0.0
        self.feed_motion = False
        self.last_dist = 0
        self.bend_correction_coef = bend_correction_coef
        self.cur_ang_suppression=1

        self.crawler_noise = crawler_noise
        self.turner_input_noise = turner_input_noise
        self.turner_output_noise = turner_output_noise

        self.torque_coef = torque_coef
        self.ang_damp_coef = ang_damp_coef
        self.body_spring_k = body_spring_k
        self.ang_vel_coef = ang_vel_coef

        self.cur_state = 'run'
        self.cur_run_dur = 0
        self.cur_pause_dur = None

        # self.new_run, self.new_pause=False, False

    def update(self):
        if self.cur_state == 'run':
            self.cur_run_dur += self.dt
        elif self.cur_state == 'pause':
            self.cur_pause_dur += self.dt

    def add_noise(self):
        self.lin_activity *= (1 + np.random.normal(scale=self.crawler_noise))
        self.ang_activity *= (1 + np.random.normal(scale=self.turner_output_noise))

    # def scale2length(self, length):
    #     self.lin_activity *= length

    def update_body(self, length):
        if self.lin_mode == 'velocity':
            self.lin_vel = self.lin_activity
        self.last_dist = self.lin_vel * self.dt
        self.bend = restore_bend_2seg(self.bend, self.last_dist, length, correction_coef=self.bend_correction_coef)
        self.bend_body()

    def bend_body(self):
        if self.ang_mode == 'torque':
            dv = self.ang_activity * self.torque_coef - self.ang_damp_coef * self.ang_vel - self.body_spring_k * self.bend
            # print(self.bend,self.ang_vel, dv)
            # raise
            self.ang_vel += dv * self.dt
        elif self.ang_mode == 'velocity':
            self.ang_vel = self.ang_activity * self.ang_vel_coef
        # self.ang_vel *=self.cur_ang_suppression
        self.bend += self.ang_vel * self.dt
        if self.bend > np.pi:
            self.bend = np.pi
        elif self.bend < -np.pi:
            self.bend = -np.pi

    def on_new_pause(self):
        pass

    def on_new_run(self):
        pass


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
        if c.interference_params is None or not m['interference']:
            self.coupling = DefaultCoupling(locomotor=self, attenuation=1)
        else:
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

    def step(self, A_in=0, length=1):
        self.lin_activity, self.ang_activity, self.feed_motion = 0, 0, False
        self.cur_ang_suppression = 1
        if self.intermitter:
            pre_state = self.intermitter.cur_state
            self.intermitter.step()
            if pre_state == 'run' and self.intermitter.cur_state == 'pause':
                self.on_new_pause()
            elif pre_state == 'pause' and self.intermitter.cur_state == 'run':
                self.on_new_run()
        if self.feeder:
            self.feed_motion = self.feeder.step()
        if self.crawler:
            self.lin_activity = self.crawler.step() * length
        if self.turner:
            A_in *= (1 + np.random.normal(scale=self.turner_input_noise))
            if self.coupling.suppression_mode == 'amplitude':
                cT = self.coupling.step()
            elif self.coupling.suppression_mode == 'oscillation':
                A_in -= (1 - self.coupling.step())
                cT = 1
            elif self.coupling.suppression_mode == 'both':
                A_in -= (1 - self.coupling.step())
                cT = self.coupling.step()

            self.cur_ang_suppression=cT
            ang = self.turner.step(A_in=A_in)
            self.ang_activity = ang
            # self.ang_activity = ang * self.cur_ang_suppression
            if self.turner.rebound:
                self.turner.buildup += ang * (1-self.cur_ang_suppression)

        # if self.intermitter.cur_state=='pause' :
        #     print(self.cur_ang_suppression)
        self.add_noise()
        # self.scale2length(length)
        if not self.offline:
            return self.lin_activity, self.ang_activity, self.feed_motion
        else:
            self.update_body(length)
            return self.lin_vel, self.ang_vel, self.feed_motion


class Levy_locomotor(DefaultLocomotor):
    def __init__(self, dt, conf=None,
                 pause_dist={'range': [0.4, 20.0], 'name': 'uniform'},
                 run_dist={'range': [1.0, 100.0], 'name': 'powerlaw', 'alpha': 1.44791},
                 run_scaled_velocity_mean=None, run_velocity_mean=None, ang_vel_headcast=0.3, **kwargs):
        from lib.conf.base.dtypes import null_dict
        if conf is None:
            conf = null_dict('locomotor',
                             crawler_params=null_dict('crawler', waveform='constant',
                                                      initial_amp=run_scaled_velocity_mean),
                             turner_params=null_dict('turner', mode='constant', initial_amp=ang_vel_headcast),
                             interference_params=null_dict('interference', mode='default', attenuation=0),
                             intermitter_params=null_dict('intermitter', run_dist=run_dist, pause_dist=pause_dist,
                                                          run_mode='run')
                             )
        super().__init__(dt=dt, ang_mode='velocity', offline=True, conf=conf)
        # self.run_scaled_velocity_mean = run_scaled_velocity_mean
        # self.run_velocity_mean = run_velocity_mean
        # self.ang_vel_headcast = ang_vel_headcast
        # self.crawler = Crawler(dt=self.dt, waveform='constant', initial_amp=run_scaled_velocity_mean)
        # self.turner = Turner(dt=self.dt, mode='constant', initial_amp=ang_vel_headcast)
        # intermitter_params = AttrDict.from_nested_dicts({
        #     'pause_dist': pause_dist,
        #     'run_dist': run_dist,
        #     'run_mode': 'run',
        #     'crawl_bouts': True,
        #     'EEB': 0,
        # })
        # self.intermitter = Intermitter(locomotor=self, dt=self.dt, **intermitter_params)
        # self.coupling = DefaultCoupling(locomotor=self, cur_attenuation=0)

    def on_new_pause(self):
        self.turner.amp *= np.random.choice([-1, 1])
        # print(np.sign(self.ang_activity))

    # def step(self, A_in=0, length=1):
    #     pre_state = self.intermitter.cur_state
    #     self.intermitter.step()
    #     cur_state = self.intermitter.cur_state
    #     # self.lin_activity = self.crawler.step()
    #     # A_in *= (1 + np.random.normal(scale=self.turner_input_noise))
    #     # self.ang_activity = self.coupling.step() * self.turner.step(A_in=A_in)
    #     if cur_state != pre_state:
    #         if cur_state == 'run':
    #             self.lin_activity = self.run_scaled_velocity_mean * length if self.run_scaled_velocity_mean is not None else self.run_velocity_mean
    #             self.ang_activity = 0
    #         elif cur_state == 'pause':
    #             self.lin_activity = 0
    #             self.ang_activity = self.ang_vel_headcast * np.random.choice([-1, 1])
    #
    #     # if self.cur_state == 'run' and self.cur_run_dur >= self.cur_run_dur_max:
    #     #     self.cur_run_dur = None
    #     #     self.cur_run_dur_max = None
    #     #     self.cur_pause_dur_max = self.pause_dist.sample()
    #     #     self.cur_pause_dur = 0
    #     #     self.cur_state = 'pause'
    #     #     self.lin_vel = 0
    #     #     self.ang_vel = self.ang_vel_headcast * np.random.choice([-1, 1])
    #     # elif self.cur_state == 'pause' and self.cur_pause_dur >= self.cur_pause_dur_max:
    #     #     self.cur_pause_dur = None
    #     #     self.cur_pause_dur_max = None
    #     #     self.cur_run_dur_max = self.run_dist.sample()
    #     #     self.cur_run_dur = 0
    #     #     self.cur_state = 'run'
    #     #     self.lin_vel = self.run_scaled_velocity_mean * length if self.run_scaled_velocity_mean is not None else self.run_velocity_mean
    #     #     self.ang_vel = 0
    #     # self.update()
    #     self.add_noise()
    #     self.update_body(length)
    #     return self.lin_vel, self.ang_vel, self.feed_motion


class Wystrach2016(Locomotor):
    def __init__(self, dt, conf=None, run_scaled_velocity_mean=None, run_velocity_mean=None,
                 turner_input_constant=19, w_ee=3.0, w_ce=0.1, w_ec=4.0, w_cc=4.0, m=100.0, n=2.0, **kwargs):
        super().__init__(dt=dt, ang_mode='torque', offline=True, **kwargs)
        self.run_scaled_velocity_mean = run_scaled_velocity_mean
        self.run_velocity_mean = run_velocity_mean

        self.turner_input_constant = turner_input_constant
        self.neural_oscillator = NeuralOscillator(dt=self.dt, w_ee=w_ee, w_ce=w_ce, w_ec=w_ec, w_cc=w_cc, m=m, n=n)

    def step(self, A_in=0, length=1):
        self.lin_activity = self.run_scaled_velocity_mean * length if self.run_scaled_velocity_mean is not None else self.run_velocity_mean

        input = (self.turner_input_constant + A_in) * (1 + np.random.normal(scale=self.turner_input_noise))

        self.neural_oscillator.step(input)
        self.ang_activity = self.neural_oscillator.activity
        self.add_noise()
        self.update_body(length)
        return self.lin_vel, self.ang_vel, self.feed_motion


class Davies2015(Locomotor):
    def __init__(self, dt, conf=None, run_scaled_velocity_mean=None, run_velocity_mean=None, run_dur_min=1.0,
                 run_dur_max=100.0,
                 theta_min_headcast=37, theta_max_headcast=120, pause_dur_max=100.0,
                 theta_max_weathervane=20, ang_vel_weathervane=0.1, ang_vel_headcast=0.3,
                 r_run2headcast=0.148, r_headcast2run=2.0,
                 r_weathervane_stop=2.0, r_weathervane_resume=1.0, **kwargs):
        super().__init__(dt=dt, ang_mode='velocity', offline=True)
        pause_dur_min = theta_min_headcast / ang_vel_headcast
        intermitter_params = AttrDict.from_nested_dicts({
            'pause_dist': {'range': (pause_dur_min, pause_dur_max), 'name': 'exponential', 'beta': r_headcast2run},
            'run_dist': {'range': (run_dur_min, run_dur_max), 'name': 'exponential', 'beta': r_run2headcast},
            'run_mode': 'run',
            'crawl_bouts': True,
            'EEB': 0,
        })
        self.intermitter = Intermitter(locomotor=self, dt=self.dt, **intermitter_params)
        self.run_scaled_velocity_mean = run_scaled_velocity_mean
        self.run_velocity_mean = run_velocity_mean
        self.theta_min_headcast, self.theta_max_headcast = theta_min_headcast, theta_max_headcast
        self.ang_vel_headcast = ang_vel_headcast
        self.theta_max_weathervane, self.ang_vel_weathervane = theta_max_weathervane, ang_vel_weathervane
        self.r_weathervane_stop, self.r_weathervane_resume = r_weathervane_stop, r_weathervane_resume
        self.cur_turn_amp = 0
        self.state_weathervane = 'on'

    @property
    def headcast_termination_allowed(self):
        return np.abs(self.cur_turn_amp) >= self.theta_min_headcast and np.sign(self.cur_turn_amp * self.ang_vel) == 1

    def step(self, A_in=0, length=1):
        pre_state = self.intermitter.cur_state
        self.intermitter.pause_termination_allowed = self.headcast_termination_allowed
        self.intermitter.step()
        cur_state = self.intermitter.cur_state
        if cur_state != pre_state:
            if cur_state == 'run':
                self.lin_activity = self.run_scaled_velocity_mean * length if self.run_scaled_velocity_mean is not None else self.run_velocity_mean
                self.ang_activity = np.random.choice([-1, 1], 1)[0] * self.ang_vel_weathervane
            elif cur_state == 'pause':
                self.lin_activity = 0
                sign = np.sign(self.bend) if self.bend != 0 else np.random.choice([-1, 1], 1)[0]
                self.ang_activity = sign * self.ang_vel_headcast
            self.cur_turn_amp = 0
        if cur_state == 'run':
            if self.state_weathervane == 'on' and np.random.uniform(0, 1, 1) <= self.r_weathervane_stop * self.dt:
                self.state_weathervane = 'off'
                self.ang_activity = 0.0
                self.cur_turn_amp = 0
            elif self.state_weathervane == 'off' and np.random.uniform(0, 1, 1) <= self.r_weathervane_resume * self.dt:
                self.state_weathervane = 'on'
                self.ang_activity = np.random.choice([-1, 1], 1)[0] * self.ang_vel_weathervane
            if np.abs(self.cur_turn_amp) >= self.theta_max_weathervane:
                self.ang_activity *= -1
        elif cur_state == 'pause':
            if np.abs(self.cur_turn_amp) >= self.theta_max_headcast:
                self.ang_activity *= -1
        self.add_noise()
        self.update_body(length)
        self.cur_turn_amp += self.ang_vel * self.dt
        return self.lin_vel, self.ang_vel, self.feed_motion


class Sakagiannis2022(Locomotor):
    def __init__(self, dt, conf=None,
                 pause_dist={'range': [0.4, 20.0], 'name': 'uniform'},
                 run_dist={'range': [1.0, 100.0], 'name': 'powerlaw', 'alpha': 1.44791},
                 stridechain_dist=None, freq_std=0.18, suppression_mode='amplitude',
                 initial_freq=1.418, step_mu=0.224, step_std=0.033,
                 attenuation=0.2, attenuation_max=0.31, max_vel_phase=3.6,
                 turner_input_constant=19, w_ee=3.0, w_ce=0.1, w_ec=4.0, w_cc=4.0, m=100.0, n=2.0, **kwargs):
        super().__init__(dt=dt, ang_mode='torque', offline=True, **kwargs)
        self.turner_input_constant = turner_input_constant
        self.neural_oscillator = NeuralOscillator(dt=self.dt, w_ee=w_ee, w_ce=w_ce, w_ec=w_ec, w_cc=w_cc, m=m, n=n)

        self.suppression_mode = suppression_mode
        self.freq = initial_freq
        self.stride_dst_mean = step_mu
        self.stride_dst_std = step_std
        self.step_to_length = self.new_stride

        if stridechain_dist is None:
            self.run_dist = BoutGenerator(**run_dist, dt=self.dt)
            self.cur_run_dur_max = self.run_dist.sample()
            self.current_numstrides = None
            self.stridechain_dist = None
        else:
            self.stridechain_dist = BoutGenerator(**stridechain_dist, dt=1)
            self.cur_stridechain_length = self.stridechain_dist.sample()
            self.current_numstrides = 0
            self.run_dist = None

        self.pause_dist = BoutGenerator(**pause_dist, dt=self.dt)
        self.cur_pause_dur_max = None

        self.d_phi = 2 * np.pi * self.dt * self.freq
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0
        self.attenuation = attenuation
        self.attenuation_max = attenuation_max
        self.max_vel_phase = max_vel_phase
        self.max_attenuation_phase = max_vel_phase - 1.2

    @property
    def attenuation_func(self):
        a = gaussian(self.phi, self.max_attenuation_phase, 1) * self.attenuation_max + self.attenuation
        if a >= 1:
            a = 1
        elif a <= 0:
            a = 0
        return a

    @property
    def new_stride(self):
        return np.random.normal(loc=self.stride_dst_mean, scale=self.stride_dst_std)

    def oscillate(self):
        self.phi += self.d_phi
        if self.phi >= 2 * np.pi:
            self.phi %= 2 * np.pi
            self.complete_iteration = True
            self.iteration_counter += 1

        else:
            self.complete_iteration = False

    @property
    def run_termination(self):
        if self.stridechain_dist is not None and self.current_numstrides >= self.cur_stridechain_length:
            self.cur_stridechain_length = None
            self.current_numstrides = None
            return True
        elif self.run_dist is not None and self.cur_run_dur >= self.cur_run_dur_max:
            self.cur_run_dur = None
            self.cur_run_dur_max = None
            return True
        else:
            return False

    def run_initiation(self):
        if self.stridechain_dist is not None:
            self.cur_stridechain_length = self.stridechain_dist.sample()
            self.current_numstrides = 0
        elif self.run_dist is not None:
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
        # self.bend = restore_bend_2seg(self.bend, self.last_dist, length, correction_coef=self.bend_correction_coef)
        self.intermit()
        if self.cur_state == 'run':
            if self.cur_run_dur is not None:
                self.cur_run_dur += self.dt
            self.oscillate()
            if self.complete_iteration and self.current_numstrides is not None:
                self.current_numstrides += 1
                self.step_to_length = self.new_stride
            self.lin_activity = self.freq * self.step_to_length * (
                    1 + 0.6 * np.cos(self.phi - self.max_vel_phase)) * length
            attenuation_coef = self.attenuation_func
        elif self.cur_state == 'pause':
            self.cur_pause_dur += self.dt
            self.phi = 0
            self.lin_activity = 0
            attenuation_coef = 1

        input = (self.turner_input_constant + A_in) * (1 + np.random.normal(scale=self.turner_input_noise))

        if self.suppression_mode == 'amplitude':
            self.neural_oscillator.step(input)
            self.ang_activity = self.neural_oscillator.activity * attenuation_coef
        elif self.suppression_mode == 'oscillation':
            self.neural_oscillator.step(input + 1 - attenuation_coef)
            self.ang_activity = self.neural_oscillator.activity
        elif self.suppression_mode == 'both':
            self.neural_oscillator.step(input + 1 - attenuation_coef)
            self.ang_activity = self.neural_oscillator.activity * attenuation_coef
        self.add_noise()
        self.update_body(length)
        # self.lin_vel = self.lin_activity
        # self.bend_body(self.ang_activity)
        # self.last_dist = self.lin_vel * self.dt
        return self.lin_vel, self.ang_vel, self.feed_motion
