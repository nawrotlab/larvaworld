import numpy as np


from larvaworld.lib import reg

class Locomotor:
    def __init__(self, dt=0.1):
        self.crawler, self.turner, self.feeder, self.intermitter, self.interference = [None] * 5
        self.dt = dt
        self.cur_state = 'exec'
        self.cur_run_dur = 0
        self.cur_pause_dur = None
        self.cur_ang_suppression = 1

        self.ang_activity = 0.0
        self.lin_activity = 0.0
        self.feed_motion = False

    def update(self):
        if self.cur_state == 'exec':
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



class DefaultLocomotor(Locomotor):
    def __init__(self, conf, **kwargs):
        super().__init__()
        D = reg.model.dict.model.m
        for k in ['crawler', 'turner', 'interference', 'feeder', 'intermitter']:

            if conf.modules[k]:

                m = conf[f'{k}_params']
                if k == 'feeder':
                    mode = 'default'
                else:
                    mode = m.mode
                kws = {kw: getattr(self, kw) for kw in D[k].kwargs.keys()}
                func = D[k].mode[mode].class_func
                M = func(**m, **kws)
                if k == 'intermitter':
                    M.disinhibit_locomotion(self)
                if k == 'crawler':
                    M.mode = m.mode
            else:
                M = None
            setattr(self, k, M)
        # return L


    def step(self, A_in=0, length=1):

        if self.intermitter:
            pre_state = self.intermitter.cur_state
            self.intermitter.step(locomotor=self)
            if pre_state == 'exec' and self.intermitter.cur_state == 'pause':
                self.on_new_pause()
            elif pre_state == 'pause' and self.intermitter.cur_state == 'exec':
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
        return self.lin_activity, self.ang_activity, self.feed_motion
