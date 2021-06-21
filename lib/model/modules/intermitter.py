import numpy as np

from lib.aux import sampling as sampling
from lib.conf import dtype_dicts as dtypes
from lib.conf.conf import loadConf
from lib.model.modules.basic import Effector


class Intermitter(Effector):
    def __init__(self, crawler=None, crawl_bouts=False,
                 feeder=None, feed_bouts=False,
                 pause_dist=None, stridechain_dist=None, crawl_freq=10 / 7, feed_freq=2.0,
                 EEB_decay=1,
                 EEB=0.5, feeder_reoccurence_rate=None, feeder_reocurrence_as_EEB=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.crawler = crawler
        self.feeder = feeder
        self.EEB = EEB
        self.base_EEB = EEB
        self.crawl_freq = crawl_freq
        self.feed_freq = feed_freq
        self.crawl_ticks = np.round(1 / (crawl_freq * self.dt)).astype(int)
        self.feed_ticks = np.round(1 / (feed_freq * self.dt)).astype(int)
        self.feeder_reoccurence_rate = feeder_reoccurence_rate if feeder_reoccurence_rate is not None else self.EEB
        self.feeder_reocurrence_as_EEB = feeder_reocurrence_as_EEB
        self.crawl_bouts = crawl_bouts
        self.feed_bouts = feed_bouts

        self.EEB_decay = EEB_decay
        self.EEB_exp_coef = np.exp(-self.EEB_decay * self.dt)

        self.reset()

        self.stridechain_dist = sampling.BoutGenerator(**stridechain_dist, dt=1)
        self.pause_dist = sampling.BoutGenerator(**pause_dist, dt=self.dt)

        self.disinhibit_locomotion()

    def initialize(self):
        self.pause_dur = None
        self.pause_start = False
        self.pause_stop = False
        self.pause_id = None

        self.stridechain_dur = None
        self.stride_start = False
        self.stridechain_start = False
        self.stride_stop = False
        self.stridechain_stop = False
        self.stridechain_id = None
        self.stridechain_length = None

        self.feedchain_dur = None
        self.feed_start = False
        self.feedchain_start = False
        self.feed_stop = False
        self.feedchain_stop = False
        self.feedchain_id = None
        self.feedchain_length = None

    def reset(self):
        # print('ddd')
        # Initialize internal variables
        self.initialize()
        self.t = 0
        self.total_t = 0
        self.ticks = 0
        self.total_ticks = 0

        self.pause_counter = 0
        self.current_pause_duration = None
        self.current_pause_ticks = None
        self.cum_pause_dur = 0

        self.stridechain_counter = 0
        self.current_stridechain_length = None
        self.cum_stridechain_dur = 0
        self.current_numstrides = 0
        self.stride_counter = 0

        self.feedchain_counter = 0
        self.current_feedchain_length = None
        self.cum_feedchain_dur = 0
        self.current_numfeeds = 0
        self.feed_counter = 0

    def step(self):
        self.initialize()
        super().count_time()
        # super().count_ticks()
        self.update_state()

    def disinhibit_locomotion(self):
        if np.random.uniform(0, 1, 1) >= self.EEB:
            if self.crawl_bouts:
                self.current_stridechain_length = self.stridechain_dist.sample()
                self.stridechain_start = True
                if self.crawler is not None:
                    self.crawler.start_effector()
                if self.feeder is not None:
                    self.feeder.stop_effector()
        else:
            if self.feed_bouts:
                self.current_feedchain_length = 1
                self.feedchain_start = True
                self.feed_start = True
                if self.feeder is not None:
                    self.feeder.start_effector()
                if self.crawler is not None:
                    self.crawler.stop_effector()

    def inhibit_locomotion(self):
        self.current_pause_duration = self.pause_dist.sample()
        self.pause_start = True
        if self.crawl_bouts and self.crawler is not None:
            self.crawler.stop_effector()
        if self.feed_bouts and self.feeder is not None:
            self.feeder.stop_effector()

    def update_state(self):
        if self.current_stridechain_length is not None:
            if self.crawler.complete_iteration:
                self.current_numstrides += 1
                self.stride_stop = True
                self.stride_counter += 1
                if self.current_numstrides >= self.current_stridechain_length:
                    self.register_stridechain()
                    self.inhibit_locomotion()
                else:
                    self.stride_start = True
            else:
                self.stridechain_id = self.stridechain_counter

        elif self.current_feedchain_length is not None:
            if self.feeder.complete_iteration:
                self.current_numfeeds += 1
                self.feed_stop = True
                self.feed_counter += 1
                if np.random.uniform(0, 1, 1) >= self.feeder_reoccurence_rate:
                    self.register_feedchain()
                    self.inhibit_locomotion()
                else:
                    self.current_feedchain_length += 1
                    self.feed_start = True
            else:
                self.feedchain_id = self.feedchain_counter

        elif self.current_pause_duration is not None:
            if self.t > self.current_pause_duration:
                self.register_pause()
                self.disinhibit_locomotion()
            else:
                self.pause_id = self.pause_counter

    def register_stridechain(self):
        self.stridechain_counter += 1
        self.stridechain_dur = self.t
        self.cum_stridechain_dur += self.stridechain_dur
        self.stridechain_length = self.current_stridechain_length
        self.t = 0
        self.stridechain_stop = True
        self.current_numstrides = 0
        self.current_stridechain_length = None

    def register_feedchain(self):
        self.feedchain_counter += 1
        self.feedchain_dur = self.t
        self.cum_feedchain_dur += self.feedchain_dur
        self.feedchain_length = self.current_feedchain_length
        self.t = 0
        self.feedchain_stop = True
        self.current_feedchain_length = None

    def register_pause(self):
        self.pause_counter += 1
        self.pause_dur = self.t
        self.cum_pause_dur += self.pause_dur
        self.current_pause_duration = None
        self.t = 0
        self.pause_stop = True

    def get_mean_feed_freq(self):
        try :
            f= self.feed_counter / (self.total_ticks*self.dt)
        except :
            f= self.feed_counter / self.total_t
        return f


class NengoIntermitter(Intermitter):
    def __init__(self,nengo_manager, **kwargs):
        super().__init__(**kwargs)
        self.nengo_manager = nengo_manager
        self.stridechain_lengths = []
        self.feedchain_lengths = []
        self.pause_durs = []

    def disinhibit_locomotion(self):
        if np.random.uniform(0, 1, 1) >= self.EEB:
            self.crawler.set_freq(self.crawler.default_freq)
        else:
            self.feeder.set_freq(self.feeder.default_freq)

    def inhibit_locomotion(self):
        self.current_pause_duration = self.pause_dist.sample()
        self.pause_start = True
        self.crawler.set_freq(0)
        self.feeder.set_freq(0)

    def update_state(self):
        if not self.effector:
            if np.random.uniform(0, 1, 1) > 0.97:
                self.start_effector()


class OfflineIntermitter(Intermitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stridechain_lengths = []
        self.feedchain_lengths = []
        self.pause_durs = []

    def step(self):
        super().count_ticks()
        if self.current_stridechain_length is not None:
            if self.ticks >= (self.current_numstrides + 1) * self.crawl_ticks:
                self.current_numstrides += 1
                self.stride_counter += 1
                if self.current_numstrides >= self.current_stridechain_length:
                    self.stridechain_counter += 1
                    self.cum_stridechain_dur += self.ticks * self.dt
                    self.stridechain_lengths.append(self.current_stridechain_length)
                    self.ticks = 0
                    self.current_numstrides = 0
                    self.current_stridechain_length = None
                    self.current_pause_ticks = np.round(self.pause_dist.sample()/self.dt).astype(int)
        elif self.current_feedchain_length is not None:
            if self.ticks >= self.current_feedchain_length * self.feed_ticks:
                self.feed_counter += 1
                if np.random.uniform(0, 1, 1) >= self.feeder_reoccurence_rate:
                    self.feedchain_counter += 1
                    self.cum_feedchain_dur += self.ticks * self.dt
                    self.feedchain_lengths.append(self.current_feedchain_length)
                    self.ticks = 0
                    self.current_feedchain_length = None
                    self.current_pause_ticks = np.round(self.pause_dist.sample()/self.dt).astype(int)
                else:
                    self.current_feedchain_length += 1
        elif self.current_pause_ticks is not None:
            if self.ticks > self.current_pause_ticks:
                self.pause_counter += 1
                dur=self.ticks*self.dt
                self.pause_durs.append(dur)
                self.cum_pause_dur += dur
                self.current_pause_ticks = None
                self.ticks = 0
                if np.random.uniform(0, 1, 1) >= self.EEB:
                    if self.crawl_bouts:
                        self.current_stridechain_length = self.stridechain_dist.sample()
                else:
                    if self.feed_bouts:
                        self.current_feedchain_length = 1
        # print(t)
        # print(self.current_stridechain_length, self.current_feedchain_length, self.current_pause_duration)


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


def get_EEB_poly1d(sample_dataset=None, dt=None, **kwargs):
    if sample_dataset is not None:
        sample = loadConf(sample_dataset, 'Ref')
        kws = {
            'crawl_freq': sample['crawl_freq'],
            'feed_freq': sample['feed_freq'],
            'dt': dt if dt is not None else sample['dt'],
            'crawl_bouts': True,
            'feed_bouts': True,
            'stridechain_dist': sample['stride']['best'],
            'pause_dist': sample['pause']['best'],
            'feeder_reoccurence_rate': sample['feeder_reoccurence_rate'],
        }
    else:
        kws = {**kwargs, 'dt': dt}

    EEBs = np.arange(0, 1, 0.05)
    ms = []
    for EEB in EEBs:
        inter = OfflineIntermitter(**dtypes.get_dict('intermitter', EEB=EEB, **kws))
        max_ticks=int(60 * 60/inter.dt)
        while inter.total_ticks < max_ticks:
            inter.step()
        ms.append(inter.get_mean_feed_freq())
    ms = np.array(ms)
    z = np.poly1d(np.polyfit(ms, EEBs, 5))
    return z


def get_best_EEB(deb, sample_dataset=None, dt=None, **kwargs):
    if sample_dataset is not None:
        sample = loadConf(sample_dataset, 'Ref')
        z = np.poly1d(sample['EEB_poly1d']) if dt in [None, sample['dt']] else get_EEB_poly1d(sample_dataset, dt,
                                                                                              **kwargs)
    else:
        z = get_EEB_poly1d(sample_dataset, dt, **kwargs)

    if type(deb) == dict:
        s = deb['feed_freq_estimate']
    else:
        s = deb.feed_freq_estimate

    return np.clip(z(s), a_min=0, a_max=1)