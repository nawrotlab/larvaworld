from larvaworld.lib import reg

class Locomotor:
    def __init__(self, dt=0.1):
        self.crawler, self.turner, self.feeder, self.intermitter, self.interference = [None] * 5
        self.dt = dt
        # self.cur_state = 'exec'
        # self.cur_run_dur = 0
        # self.cur_pause_dur = None


        # self.ang_activity = 0.0
        # self.lin_activity = 0.0
        # self.feed_motion = False

    # def update(self):
    #     if self.cur_state == 'exec':
    #         self.cur_run_dur += self.dt
    #     elif self.cur_state == 'pause':
    #         self.cur_pause_dur += self.dt




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

    def step_intermitter(self, **kwargs):
        if self.intermitter:
            pre_state = self.intermitter.cur_state
            cur_state =self.intermitter.step(**kwargs)
            if pre_state != 'pause' and cur_state == 'pause':
                self.on_new_pause()
            elif pre_state != 'exec' and cur_state == 'exec':
                self.on_new_run()
            elif pre_state != 'feed' and cur_state == 'feed':
                self.on_new_feed()
            # print(cur_state)

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
                mm={k:m[k] for k in m.keys() if k!='mode'}
                M = func(**mm, **kws)
            else:
                M = None
            setattr(self, k, M)



    def step(self, A_in=0, length=1, on_food=False):
        C,F,T,If=self.crawler,self.feeder,self.turner,self.interference


        feed_motion = F.step() if F else False
        if C :
            lin = C.step() * length
            stride_completed=C.complete_iteration
        else:
            lin =  0
            stride_completed = False
        self.step_intermitter(stride_completed=stride_completed,feed_motion=feed_motion, on_food=on_food)

        if T :
            if If:
                cur_att_in, cur_att_out = If.step(crawler=C, feeder=F)
            else:
                cur_att_in, cur_att_out = 1, 1
            ang = T.step(A_in=A_in * cur_att_in) * cur_att_out
        else:
            ang = 0

        return lin, ang, feed_motion
