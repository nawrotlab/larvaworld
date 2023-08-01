from larvaworld.lib import reg

class Locomotor:
    def __init__(self, dt=0.1):
        self.crawler, self.turner, self.feeder, self.intermitter, self.interference = [None] * 5
        self.dt = dt


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

    @property
    def stride_completed(self):
        if self.crawler :
            return self.crawler.complete_iteration
        else:
            return False

    @property
    def feed_motion(self):
        if self.feeder:
            return self.feeder.complete_iteration
        else:
            return False


    def step(self, A_in=0, length=1, on_food=False):
        C,F,T,If=self.crawler,self.feeder,self.turner,self.interference
        if If:
            If.cur_attenuation=1
        if F :
            F.step()
            if F.active and If:
                If.check_feeder(F)
        if C :
            lin = C.step() * length
            if C.active and If:
                If.check_crawler(C)
        else:
            lin =  0
        self.step_intermitter(stride_completed=self.stride_completed,feed_motion=self.feed_motion, on_food=on_food)

        if T :
            if If:
                cur_att_in, cur_att_out = If.apply_attenuation(If.cur_attenuation)
                # cur_att_in, cur_att_out = If.step(crawler=C, feeder=F)
            else:
                cur_att_in, cur_att_out = 1, 1
            ang = T.step(A_in=A_in * cur_att_in) * cur_att_out
        else:
            ang = 0

        return lin, ang, self.feed_motion
