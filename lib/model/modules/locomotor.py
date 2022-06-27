import numpy as np

from lib.aux.dictsNlists import NestDict
from lib.model.modules.crawl_bend_interference import Coupling
from lib.model.modules.crawler import Crawler
from lib.model.modules.feeder import Feeder

from lib.model.modules.turner import Turner


class Locomotor:
    def __init__(self, dt=0.1, ang_mode='torque', lin_mode='velocity', offline=False, crawler_noise=0,
                 turner_input_noise=0,
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
        self.cur_ang_suppression = 1

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
        from lib.aux.ang_aux import restore_bend_2seg

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

    @property
    def active_effectors(self):
        c, f = self.crawler, self.feeder
        c_on = True if c is not None and c.effector else False
        f_on = True if f is not None and f.effector else False
        return c_on, f_on


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
            c.interference_params = NestDict({'mode': 'default', 'attenuation': 1})
        self.coupling = Coupling(**c.interference_params)

        #     self.coupling = DefaultCoupling(locomotor=self, attenuation=1)
        # else:
        #     mode = c.interference_params.mode if 'mode' in c.interference_params.keys() else 'default'
        #     if mode == 'default':
        #         self.coupling = DefaultCoupling(locomotor=self, **c.interference_params)
        #     elif mode == 'square':
        #         self.coupling = SquareCoupling(locomotor=self, **c.interference_params)
        #     elif mode == 'phasic':
        #         self.coupling = PhasicCoupling(locomotor=self, **c.interference_params)

        if m['intermitter']:
            if 'mode' not in c.intermitter_params.keys():
                c.intermitter_params.mode = 'default'
            from lib.model.modules.intermitter import ChoiceIntermitter
            self.intermitter = ChoiceIntermitter(dt=self.dt, **c.intermitter_params)

            # mode = c.intermitter_params.mode if 'mode' in c.intermitter_params.keys() else 'default'
            # if mode == 'default':
            #     self.intermitter = Intermitter(locomotor=self, dt=self.dt, **c.intermitter_params)
            # elif mode == 'branch':
            #     self.intermitter = BranchIntermitter(locomotor=self, dt=self.dt, **c.intermitter_params)

    def step(self, A_in=0, length=1):
        self.lin_activity, self.ang_activity, self.feed_motion = 0, 0, False
        self.cur_ang_suppression = 1
        if self.intermitter:
            pre_state = self.intermitter.cur_state
            self.intermitter.step(locomotor=self)
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
            cT0 = self.coupling.step(self.crawler, self.feeder)
            if self.coupling.suppression_mode == 'amplitude':
                cT = cT0
            elif self.coupling.suppression_mode == 'oscillation':
                A_in -= (1 - cT0)
                cT = 1
            elif self.coupling.suppression_mode == 'both':
                A_in -= (1 - cT0)
                cT = cT0
            self.cur_ang_suppression = cT
            ang = self.turner.step(A_in=A_in)
            self.ang_activity = ang
            if self.turner.rebound:
                self.turner.buildup += ang * (1 - self.cur_ang_suppression)
        self.add_noise()
        if not self.offline:
            return self.lin_activity, self.ang_activity, self.feed_motion
        else:
            self.update_body(length)
            return self.lin_vel, self.ang_vel, self.feed_motion