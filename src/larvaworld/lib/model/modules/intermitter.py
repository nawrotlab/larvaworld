import os
import numpy as np
import pandas as pd



from larvaworld.lib.model.modules.basic import Timer
from larvaworld.lib import reg, aux, util
from larvaworld.lib.aux import nam


default_bout_distros=aux.AttrDict({'turn_dur': {'range': [0.25, 3.25], 'name': 'exponential', 'beta': 4.70099},
 'turn_amp': {'range': [0.00042, 305.7906],
  'name': 'lognormal',
  'mu': 2.08133,
  'sigma': 1.26487},
 'turn_vel_max': {'range': [0.00031, 2752.80403],
  'name': 'levy',
  'mu': -1.22434,
  'sigma': 11.93284},
 'run_dur': {'range': [0.44, 114.25],
  'name': 'lognormal',
  'mu': 1.148,
  'sigma': 1.11329},
 'run_dst': {'range': [0.00022, 0.14457],
  'name': 'levy',
  'mu': 0.00017,
  'sigma': 0.00125},
 'pause_dur': {'range': [0.12, 15.94], 'name': 'exponential', 'beta': 1.01503},
 'run_count': {'range': [1, 142],
  'name': 'lognormal',
  'mu': 1.39115,
  'sigma': 1.14667}})

class Intermitter(Timer):
    def __init__(self, pause_dist=None, stridechain_dist=None, run_dist=None, run_mode='stridechain',
                 feeder_reoccurence_rate=None, EEB=0.5,feed_bouts=False,EEB_decay=1, **kwargs):
        super().__init__(**kwargs)
        self.reset()

        self.cur_state = None
        self.feed_bouts = feed_bouts

        if run_mode=='stridechain' :
            if stridechain_dist is None or stridechain_dist.range is None:
                stridechain_dist = default_bout_distros.run_count
            self.stridechain_min, self.stridechain_max = stridechain_dist.range
            self.stridechain_dist = util.BoutGenerator(**stridechain_dist, dt=1)
            self.run_dist = None

        elif run_mode=='exec' :
            if run_dist is not None or run_dist.range is None:
                run_dist = default_bout_distros.run_dur
            self.stridechain_min, self.stridechain_max = run_dist.range
            self.run_dist = util.BoutGenerator(**run_dist, dt=self.dt)
            self.stridechain_dist = None
        else :
            raise ValueError ('None of stidechain or exec distribution exist')

        if pause_dist is None or pause_dist.range is None :
            pause_dist = default_bout_distros.pause_dur
        self.pau_min, self.pau_max = (np.array(pause_dist.range) / self.dt).astype(int)
        self.pause_dist = util.BoutGenerator(**pause_dist, dt=self.dt)


        self.stride_counter = 0
        self.stridechain_counter = 0
        self.run_counter = 0
        self.pause_counter = 0
        self.feed_counter = 0
        self.feedchain_counter = 0
        self.feed_success_counter = 0
        self.feed_fail_counter = 0
        self.EEB = EEB
        self.base_EEB = EEB
        self.EEB_decay = EEB_decay
        # self.cur_state = None


        self.expected_stridechain_length = None
        self.current_numstrides = 0
        self.expected_run_duration = None
        # self.current_run_duration =0
        self.expected_pause_duration = None
        self.current_feedchain_length = None

        self.cum_feedchain_dur = 0
        self.cum_stridechain_dur = 0
        self.cum_run_dur = 0
        self.cum_pause_dur = 0

        self.stridechain_lengths = []
        self.stridechain_durs = []
        self.feedchain_lengths = []
        self.feedchain_durs = []
        self.pause_durs = []
        self.run_durs = []
        self.feed_durs = []
        self.stride_durs = []

        self.feeder_reoccurence_rate = feeder_reoccurence_rate if feeder_reoccurence_rate is not None else self.EEB


    def alternate_crawlNpause(self,stride_completed=False):
        if self.cur_state is None :
            self.run_initiation()
        elif self.expected_stridechain_length is not None and stride_completed:
            self.current_numstrides += 1
            if self.current_numstrides >= self.expected_stridechain_length:
                self.register('stridechain')
                self.expected_pause_duration = self.generate_pause()
                self.cur_state = 'pause'


        elif self.expected_pause_duration is not None:
            if self.t > self.expected_pause_duration:
                self.register('pause')
                self.run_initiation()


        elif self.expected_run_duration is not None:
            if self.t > self.expected_run_duration:
                self.register('run')
                self.expected_pause_duration = self.generate_pause()
                self.cur_state = 'pause'


    def alternate_exploreNexploit(self,feed_motion=False,on_food=False):
        if feed_motion :
            if self.current_feedchain_length is None:
                raise
            else :
                self.feed_counter += 1
                if not on_food:
                    self.feed_fail_counter +=1
                    self.register('feedchain')
                    self.run_initiation()
                else:
                    self.feed_success_counter += 1
                    if np.random.uniform(0, 1, 1) >= self.feeder_reoccurence_rate:
                        self.register('feedchain')
                        self.run_initiation()
                    else:
                        self.current_feedchain_length += 1
        elif on_food and self.current_feedchain_length is None and np.random.uniform(0, 1, 1) <= self.EEB:
            self.current_feedchain_length = 1
            self.register(self.cur_state)
            self.cur_state = 'feed'
            # self.reset()

    def register(self, bout):
        dur = self.t
        self.ticks = 0
        if bout=='exec' :
            if self.stridechain_dist is None and self.run_dist is not None:
                bout='run'
            elif self.stridechain_dist is not None and self.run_dist is None:
                bout = 'stridechain'

        if bout == 'feedchain':
            self.feedchain_counter += 1
            self.cum_feedchain_dur += dur
            self.feedchain_lengths.append(self.current_feedchain_length)
            self.feedchain_durs.append(dur)
            self.current_feedchain_length = None

        elif bout == 'stridechain':
            self.stridechain_counter += 1
            self.cum_stridechain_dur += dur
            self.stridechain_lengths.append(self.current_numstrides)
            self.stridechain_durs.append(dur)
            self.expected_stridechain_length = None
            self.current_numstrides = 0

        elif bout == 'run':
            self.run_counter += 1
            self.cum_run_dur += dur
            self.run_durs.append(dur)
            self.expected_run_duration = None

        elif bout == 'pause':
            self.pause_counter += 1
            self.cum_pause_dur += dur
            self.pause_durs.append(dur)
            self.expected_pause_duration = None

    def update_state(self, stride_completed=False, feed_motion=False,on_food=False):
        if self.feed_bouts :
            self.alternate_exploreNexploit(feed_motion, on_food)
        self.alternate_crawlNpause(stride_completed)

        # self.update_state(locomotor,on_food=on_food)
        return self.cur_state


    def step(self, **kwargs):
        self.count_time()
        return self.update_state(**kwargs)

    def generate_stridechain(self):
        return self.stridechain_dist.sample()

    def generate_run(self):
        return self.run_dist.sample()

    def run_initiation(self):
        if self.stridechain_dist is not None:
            self.expected_stridechain_length = self.stridechain_dist.sample()
            self.current_numstrides = 0
        elif self.run_dist is not None:
            self.expected_run_duration = self.run_dist.sample()
        self.cur_state = 'exec'
        self.ticks = 0


    def generate_pause(self):
        return self.pause_dist.sample()


    def build_dict(self):
        cum_t=nam.cum('t')
        d = {}
        if self.total_t != 0:
            d[cum_t] = self.total_t
        else:
            d[cum_t] = self.total_ticks * self.dt
        d[nam.num('tick')] = int(self.total_ticks)
        for c0 in ['feed', 'stride']:
            c = nam.chain(c0)
            Nc0, Nc = nam.num([c0, c])
            l = nam.length(c)
            d[l] = [int(ll) for ll in getattr(self, f'{l}s')]
            d[Nc0] = int(sum(d[l]))
            d[nam.mean(nam.freq(c0))] = d[Nc0] / d[cum_t]

        for c in ['feedchain', 'stridechain', 'pause']:
            t = nam.dur(c)
            d[t] = getattr(self, f'{t}s')
            d[nam.num(c)] = int(getattr(self, f'{c}_counter'))
            cum_chunk_t = nam.cum(t)
            d[cum_chunk_t] = getattr(self, cum_chunk_t)
            d[nam.dur_ratio(c)] = d[cum_chunk_t] / d[cum_t]
        return d

    def save_dict(self, path=None, dic=None):
        if dic is None:
            dic = self.build_dict()
        if path is not None:
            file = f'{path}/intermitter_dict.txt'
            if not os.path.exists(path):
                os.makedirs(path)
            aux.save_dict(dic, file)

    @property
    def active_bouts(self):
        return self.expected_stridechain_length, self.current_feedchain_length, self.expected_pause_duration, self.expected_run_duration

    def get_mean_feed_freq(self):
        try:
            f = self.feed_counter / (self.total_ticks * self.dt)
        except:
            f = self.feed_counter / self.total_t
        return f



class OfflineIntermitter(Intermitter):
    def __init__(self, crawl_freq=10 / 7, feed_freq=2.0, **kwargs):
        super().__init__(**kwargs)
        self.crawl_freq = crawl_freq
        self.feed_freq = feed_freq
        self.crawl_ticks = np.round(1 / (crawl_freq * self.dt)).astype(int)
        self.feed_ticks = np.round(1 / (feed_freq * self.dt)).astype(int)

    def step(self, stride_completed=None, feed_motion=None,on_food=False):
        self.count_time()
        t = int(self.t/self.dt)
        if feed_motion is None:

            if self.cur_state == 'feed' and t%self.feed_ticks==0:
                feed_motion = True
            else:
                feed_motion = False
        if stride_completed is None:
            if self.cur_state == 'exec' and t%self.crawl_ticks==0:
                stride_completed = True
            else:
                stride_completed = False

        # self.update_state(locomotor,on_food=on_food)
        return self.update_state(stride_completed, feed_motion, on_food)
    #     if on_food and self.current_feedchain_length is None and self.feed_bouts:
    #         if np.random.uniform(0, 1, 1) <= self.base_EEB:
    #             self.current_feedchain_length = 1
    #             self.cur_state = 'feed'
    #             if self.current_stridechain_length is not None:
    #                 self.register('stride')
    #                 self.current_numstrides = 0
    #                 self.current_stridechain_length = None
    #             if self.current_pause_duration is not None:
    #                 self.register('pause')
    #                 self.current_pause_ticks = None
    #             return
    #     self.stride_stop = False
    #     # print(self.current_stridechain_length, self.current_feedchain_length, self.current_pause_ticks)
    #     if self.current_stridechain_length and t >= self.current_crawl_ticks:
    #         self.current_numstrides += 1
    #         self.stride_stop = True
    #         if self.current_numstrides >= self.current_stridechain_length:
    #             self.register('stride')
    #             self.current_numstrides = 0
    #             self.current_stridechain_length = None
    #             self.current_pause_ticks = int(self.generate_pause() / self.dt)
    #             self.inhibit_locomotion(L=locomotor)
    #     elif self.current_feedchain_length and t >= self.current_feed_ticks:
    #         self.feed_counter += 1
    #         if np.random.uniform(0, 1, 1) >= self.feeder_reoccurence_rate:
    #             self.register('feed')
    #             self.current_feedchain_length = None
    #             self.current_pause_ticks = int(self.generate_pause() / self.dt)
    #             self.inhibit_locomotion(L=locomotor)
    #         else:
    #             self.current_feedchain_length += 1
    #     elif self.current_pause_ticks and t > self.current_pause_ticks:
    #         self.register('pause')
    #         self.current_pause_ticks = None
    #         self.disinhibit_locomotion(L=locomotor)
    #
    # def disinhibit_locomotion(self, L=None):
    #     if np.random.uniform(0, 1, 1) >= self.EEB:
    #         if self.crawl_bouts:
    #             self.current_stridechain_length = self.generate_stridechain()
    #     else:
    #         if self.feed_bouts:
    #             self.current_feedchain_length = 1
    #
    # def inhibit_locomotion(self, L=None):
    #     pass
    #
    # def register(self, bout):
    #     if self.register_bouts:
    #         t = self.ticks
    #         dur = t * self.dt
    #         if bout == 'feed':
    #             self.feedchain_counter += 1
    #             self.cum_feedchain_dur += dur
    #             self.feedchain_lengths.append(self.current_feedchain_length)
    #
    #         elif bout == 'stride':
    #             self.stridechain_counter += 1
    #             self.cum_stridechain_dur += dur
    #             self.stridechain_lengths.append(self.current_stridechain_length)
    #
    #         elif bout == 'pause':
    #             self.pause_counter += 1
    #             self.cum_pause_dur += dur
    #             self.pause_durs.append(dur)
    #
    #     self.reset_ticks()
    #
    # @property
    # def current_crawl_ticks(self):
    #     return (self.current_numstrides + 1) * self.crawl_ticks
    #
    # @property
    # def current_feed_ticks(self):
    #     return self.current_feedchain_length * self.feed_ticks


class NengoIntermitter(Intermitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.current_stridechain_length = self.generate_stridechain()

    # def disinhibit_locomotion(self, L=None):
    #     if np.random.uniform(0, 1, 1) >= self.EEB:
    #         L.crawler.set_freq(L.crawler.initial_freq)
    #         if L.feeder is not None:
    #             L.feeder.set_freq(0)
    #         self.current_stridechain_length = self.generate_stridechain()
    #     else:
    #         if L.feeder is not None:
    #             L.feeder.set_freq(L.feeder.initial_freq)
    #         L.crawler.set_freq(0)
    #         self.current_feedchain_length = 1
    #
    # def inhibit_locomotion(self, L=None):
    #     L.crawler.set_freq(0)
    #     if L.feeder is not None:
    #         L.feeder.set_freq(0)


class BranchIntermitter(Intermitter):
    def __init__(self,beta=None,c=0.7,sigma=1,**kwargs):
        super().__init__(feed_bouts=False,**kwargs)
        self.c = c
        self.beta = beta
        self.sigma = sigma
        # pause_dist, stridechain_dist = self.check_distros(pause_dist=pause_dist,stridechain_dist=stridechain_dist)
        #
        #
        # if run_mode == 'stridechain':
        #     if stridechain_dist is not None:
        #         self.stridechain_min, self.stridechain_max = stridechain_dist.range
        #         self.stridechain_dist = util.BoutGenerator(**stridechain_dist, dt=1)
        #         self.run_dist = None
        #     else:
        #         run_mode = 'exec'
        # if run_mode == 'exec':
        #     if run_dist is not None:
        #         self.run_dist = util.BoutGenerator(**run_dist, dt=self.dt)
        #         self.stridechain_min, self.stridechain_max = run_dist.range
        #         self.stridechain_dist = None
        #     else:
        #         raise ValueError('None of stidechain or exec distribution exist')
        # self.pau_min, self.pau_max = (np.array(pause_dist.range)/self.dt).astype(int)
        # self.pause_dist = util.BoutGenerator(**pause_dist, dt=self.dt)

    def generate_stridechain(self):
        return util.exp_bout(beta=self.beta, tmax=self.stridechain_max, tmin=self.stridechain_min)

    def generate_pause(self):
        return util.critical_bout(c=self.c, sigma=self.sigma, N=1000, tmax=self.pau_max, tmin=self.pau_min)*self.dt


class FittedIntermitter(OfflineIntermitter):
    def __init__(self, refID, **kwargs):
        cRef = reg.retrieveRef(refID)
        stored_conf = {
            'crawl_freq': cRef['crawl_freq'],
            'feed_freq': cRef['feed_freq'],
            'dt': cRef['dt'],
            'stridechain_dist': cRef['stride']['best'],
            'pause_dist': cRef['pause']['best'],
            'feeder_reoccurence_rate': cRef['feeder_reoccurence_rate'],
        }
        stored_conf.update(kwargs)
        stored_conf['feed_bouts'] = True if stored_conf['feed_freq'] is not None else False
        super().__init__(**stored_conf)

def get_EEB_poly1d(**kws):
    max_dur = 60 * 60
    EEBs = np.arange(0, 1.05, 0.05)
    ms = []
    for EEB in EEBs:
        inter = OfflineIntermitter(EEB=EEB, **kws)
        # inter.disinhibit_locomotion()
        while inter.total_t < max_dur:
            inter.step()
        ms.append(inter.get_mean_feed_freq())
    z = np.poly1d(np.polyfit(np.array(ms), EEBs, 5))
    return z

def get_EEB_time_fractions(refID=None, dt=None, **kwargs):
    if refID is not None:
        kws = reg.retrieveRef(refID)['intermitter']
    else:
        kws = kwargs
    if dt is not None:
        kws['dt'] = dt
    max_dur = 60 * 60
    rts = {f'{q} ratio': nam.dur_ratio(p) for p, q in
           zip(['stridechain', 'pause', 'feedchain'], ['crawl', 'pause', 'feed'])}
    EEBs = np.round(np.arange(0, 1, 0.02), 2)
    data = []
    for EEB in EEBs:
        inter = OfflineIntermitter(EEB=EEB, **kws)
        while inter.total_t < max_dur:
            inter.step()
        dic = inter.build_dict()
        ffr = inter.get_mean_feed_freq()
        entry = {'EEB': EEB, **{k: np.round(dic[v], 2) for k, v in rts.items()}, nam.mean(nam.freq('feed')): ffr}
        data.append(entry)
    df = pd.DataFrame.from_records(data=data)
    return df

