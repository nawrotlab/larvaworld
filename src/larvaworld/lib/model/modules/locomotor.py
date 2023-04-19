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



    @property
    def active_effectors(self):
        c, f = self.crawler, self.feeder
        c_on = True if c is not None and c.active else False
        f_on = True if f is not None and f.active else False
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
                # if k == 'intermitter':
                #     M.run_initiation(self)
                if k == 'crawler':
                    M.mode = m.mode
            else:
                M = None
            setattr(self, k, M)

    def on_new_pause(self):
        if self.crawler:
            self.crawler.stop_effector()
        if self.feeder:
            self.feeder.stop_effector()

    def on_new_run(self):
        if self.crawler:
            self.crawler.start_effector()
        if self.feeder:
            self.feeder.stop_effector()

    def on_new_feed(self):
        if self.crawler:
            self.crawler.stop_effector()
        if self.feeder:
            self.feeder.start_effector()

    def step(self, A_in=0, length=1, on_food=False):


        self.feed_motion = self.feeder.step() if self.feeder else False
        if self.crawler :
            self.lin_activity = self.crawler.step() * length
            stride_completed=self.crawler.complete_iteration
        else:
            self.lin_activity =  0
            stride_completed = False
        if self.intermitter:
            pre_state = self.intermitter.cur_state
            cur_state =self.intermitter.step(stride_completed=stride_completed,feed_motion=self.feed_motion, on_food=on_food)
            if pre_state != 'pause' and cur_state == 'pause':
                self.on_new_pause()
            elif pre_state != 'exec' and cur_state == 'exec':
                self.on_new_run()
            elif pre_state != 'feed' and cur_state == 'feed':
                self.on_new_feed()
        if self.interference:
            A_in, cT = self.interference.step(A_in, self.crawler, self.feeder)
        else :
            cT = 1
        self.cur_ang_suppression=cT
        self.ang_activity = self.turner.step(A_in=A_in) if self.turner else 0
        return self.lin_activity, self.ang_activity, self.feed_motion
