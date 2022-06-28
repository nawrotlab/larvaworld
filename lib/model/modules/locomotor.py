import numpy as np

from lib.aux.dictsNlists import NestDict
from lib.model.body.controller import PhysicsController
from lib.model.modules.crawl_bend_interference import Coupling
from lib.model.modules.crawler import Crawler
from lib.model.modules.feeder import Feeder

from lib.model.modules.turner import Turner


class Locomotor:
    def __init__(self, dt=0.1):
        self.crawler, self.turner, self.feeder, self.intermitter, self.interference = [None] * 5
        self.dt = dt
        self.cur_state = 'run'
        self.cur_run_dur = 0
        self.cur_pause_dur = None
        self.cur_ang_suppression = 1

        self.ang_activity = 0.0
        self.lin_activity = 0.0
        self.feed_motion = False

    def update(self):
        if self.cur_state == 'run':
            self.cur_run_dur += self.dt
        elif self.cur_state == 'pause':
            self.cur_pause_dur += self.dt

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

    def output(self, length=None):
        return self.lin_activity, self.ang_activity, self.feed_motion


class OfflineLocomotor(Locomotor):
    def __init__(self, dt=0.1, **kwargs):
        super().__init__(dt)
        self.controller = PhysicsController(**kwargs)
        self.bend = 0.0
        self.ang_vel = 0.0
        self.lin_vel = 0.0
        self.last_dist = 0

    def update_body(self, length):
        self.lin_vel, self.ang_vel = self.controller.get_vels(lin=self.lin_activity, ang=self.ang_activity,
                                                              prev_ang_vel=self.ang_vel,
                                                              prev_lin_vel=self.lin_vel,
                                                              bend=self.bend, dt=self.dt,
                                                              ang_suppression=self.cur_ang_suppression)

        from lib.aux.ang_aux import restore_bend_2seg
        self.last_dist = self.lin_vel * self.dt
        self.bend = restore_bend_2seg(self.bend, self.last_dist, length,
                                      correction_coef=self.controller.bend_correction_coef)
        self.bend_body()

    def bend_body(self):
        self.bend += self.ang_vel * self.dt
        if self.bend > np.pi:
            self.bend = np.pi
        elif self.bend < -np.pi:
            self.bend = -np.pi

    # @property
    def output(self, length):
        self.update_body(length)
        return self.lin_vel, self.ang_vel, self.feed_motion

class DefaultLocomotor(OfflineLocomotor, Locomotor):
    def __init__(self, conf, offline=False, **kwargs):
        self.offline = offline
        if offline:
            OfflineLocomotor.__init__(self, **kwargs)
        else:
            Locomotor.__init__(self, **kwargs)
        from lib.registry.pars import preg
        preg.larva_conf_dict.init_loco(conf, self)

    # def output(self, length):
    #     if self.offline :
    #         self.update_body(length)
    #         return self.lin_vel, self.ang_vel, self.feed_motion
    #     else :
    #         return self.lin_activity, self.ang_activity, self.feed_motion

    def step(self, A_in=0, length=1):
        if self.intermitter:
            pre_state = self.intermitter.cur_state
            self.intermitter.step(locomotor=self)
            if pre_state == 'run' and self.intermitter.cur_state == 'pause':
                self.on_new_pause()
            elif pre_state == 'pause' and self.intermitter.cur_state == 'run':
                self.on_new_run()
        self.feed_motion = self.feeder.step() if self.feeder else False
        self.lin_activity = self.crawler.step() * length if self.crawler else 0
        if self.interference:
            cT0 = self.interference.step(self.crawler, self.feeder)

            mm = self.interference.suppression_mode
            if mm == 'oscillation':
                A_in -= (1 - cT0)
                cT = 1
            elif mm == 'amplitude':
                cT = cT0
            elif mm == 'both':
                A_in -= (1 - cT0)
                cT = cT0
        else :
            cT = 1
        self.cur_ang_suppression=cT
        self.ang_activity = self.turner.step(A_in=A_in) if self.turner else 0
        if self.offline:
            self.update_body(length)
            return self.lin_vel, self.ang_vel, self.feed_motion
        else:
            return self.lin_activity, self.ang_activity, self.feed_motion
