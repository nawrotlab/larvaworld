import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lib.aux.dictsNlists
from lib.anal.fitting import BoutGenerator
from lib.aux import naming as nam
from lib.conf.stored.conf import loadConf
from lib.conf.base.dtypes import null_dict
from lib.model.modules.basic import Effector


class Intermitter(Effector):
    def __init__(self, brain=None, crawl_bouts=False, feed_bouts=False,
                 pause_dist=None, stridechain_dist=None, crawl_freq=10 / 7, feed_freq=2.0,
                 EEB_decay=1, save_to=None,
                 EEB=0.5, feeder_reoccurence_rate=None, feeder_reocurrence_as_EEB=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.brain = brain
        self.save_to = save_to

        self.crawler = brain.crawler if brain is not None else None
        self.feeder = brain.feeder if brain is not None else None
        self.turner = brain.turner if brain is not None else None
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

        self.stridechain_dist = BoutGenerator(**stridechain_dist, dt=1)
        self.pause_dist = BoutGenerator(**pause_dist, dt=self.dt)

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

        self.stridechain_lengths = []
        self.stridechain_durs = []
        self.feedchain_lengths = []
        self.feedchain_durs = []
        self.pause_durs = []
        self.feed_durs = []
        self.stride_durs = []

    def step(self):
        self.initialize()
        super().count_time()
        # super().count_ticks()
        self.update_state()
        # print(self.brain.agent.unique_id, self.EEB, self.feeder_reoccurence_rate)
        # print(self.current_stridechain_length, self.current_feedchain_length, self.current_pause_duration)

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
        self.stridechain_lengths.append(self.current_stridechain_length)
        self.stridechain_durs.append(self.stridechain_dur)
        self.t = 0
        self.stridechain_stop = True
        self.current_numstrides = 0
        self.current_stridechain_length = None

    def register_feedchain(self):
        self.feedchain_counter += 1
        self.feedchain_dur = self.t
        self.cum_feedchain_dur += self.feedchain_dur
        self.feedchain_length = self.current_feedchain_length
        self.feedchain_lengths.append(self.current_feedchain_length)
        self.feedchain_durs.append(self.feedchain_dur)
        self.t = 0
        self.feedchain_stop = True
        self.current_feedchain_length = None

    def register_pause(self):
        self.pause_counter += 1
        self.pause_dur = self.t
        self.cum_pause_dur += self.pause_dur
        self.pause_durs.append(self.pause_dur)
        self.current_pause_duration = None
        self.t = 0
        self.pause_stop = True

    def get_mean_feed_freq(self):
        try:
            f = self.feed_counter / (self.total_ticks * self.dt)
        except:
            f = self.feed_counter / self.total_t
        return f

    def build_dict(self):
        d = {}
        if self.total_t != 0:
            d[nam.cum('t')] = self.total_t
        else:
            d[nam.cum('t')] = self.total_ticks * self.dt
        d[nam.num('tick')] = self.total_ticks
        for c in ['feedchain', 'stridechain', 'pause']:
            t = nam.dur(c)
            l = nam.length(c)
            N = nam.num(c)
            cum_t = nam.cum(t)
            rt = nam.dur_ratio(c)
            d[t] = getattr(self, f'{t}s')
            d[N] = getattr(self, f'{c}_counter')
            d[cum_t] = getattr(self, cum_t)
            d[rt] = d[cum_t] / d[nam.cum('t')]
            if c in ['feedchain', 'stridechain']:
                d[l] = getattr(self, f'{l}s')
        d[nam.num('feed')] = sum(d[nam.length('feedchain')])
        d[nam.num('stride')] = sum(d[nam.length('stridechain')])
        d[nam.mean(nam.freq('feed'))] = d[nam.num('feed')] / d[nam.cum('t')]
        d[nam.mean(nam.freq('stride'))] = d[nam.num('stride')] / d[nam.cum('t')]

        return d

    def save_dict(self, path=None):
        dic = self.build_dict()
        if path is None:
            if self.save_to is not None:
                path = self.save_to
            else:
                raise ValueError('No path to save intermittency dict')
        if dic is not None:
            file = f'{path}/{self.brain.agent.unique_id}.txt'
            if not os.path.exists(path):
                os.makedirs(path)
            lib.aux.dictsNlists.save_dict(dic, file)

    def update(self, max_refeed_rate=0.9, refeed_rate_coef=0, food_present=None, feed_success=None):
        if food_present is None:
            self.EEB *= self.EEB_exp_coef
        else:
            self.EEB = self.base_EEB
        if feed_success == True:
            if self.feeder_reocurrence_as_EEB:
                self.feeder_reoccurence_rate = self.EEB
            else:
                self.feeder_reoccurence_rate = max_refeed_rate
        elif feed_success == False:
            self.feeder_reoccurence_rate *= refeed_rate_coef

    @property
    def current_crawl_ticks(self):
        return (self.current_numstrides + 1) * self.crawl_ticks

    @property
    def current_feed_ticks(self):
        return self.current_feedchain_length * self.feed_ticks

    @property
    def active_bouts(self):
        return self.current_stridechain_length, self.current_feedchain_length, self.current_pause_ticks


class OfflineIntermitter(Intermitter):
    def __init__(self,register_bouts=True, **kwargs):
        super().__init__(**kwargs)
        self.register_bouts = register_bouts

    def step(self):
        super().count_ticks()
        t = self.ticks
        self.stride_stop = False
        if self.current_stridechain_length and t >= self.current_crawl_ticks:
            self.current_numstrides += 1
            self.stride_counter += 1
            self.stride_stop = True
            if self.current_numstrides >= self.current_stridechain_length:
                self.register('stride')
                self.current_numstrides = 0
                self.current_stridechain_length = None
                self.current_pause_ticks = int(self.pause_dist.sample() / self.dt)
                self.inhibit_locomotion()
        elif self.current_feedchain_length and t >= self.current_feed_ticks:
            self.feed_counter += 1
            if np.random.uniform(0, 1, 1) >= self.feeder_reoccurence_rate:
                self.register('feed')
                self.current_feedchain_length = None
                self.current_pause_ticks = int(self.pause_dist.sample() / self.dt)
                self.inhibit_locomotion()
            else:
                self.current_feedchain_length += 1
        elif self.current_pause_ticks and t > self.current_pause_ticks:
            self.register('pause')
            self.current_pause_ticks = None
            self.disinhibit_locomotion()

    def disinhibit_locomotion(self):
        if np.random.uniform(0, 1, 1) >= self.EEB:
            if self.crawl_bouts:
                self.current_stridechain_length = self.stridechain_dist.sample()
        else:
            if self.feed_bouts:
                self.current_feedchain_length = 1

    def inhibit_locomotion(self):
        pass




    def register(self, bout):
        if self.register_bouts :
            t = self.ticks
            dur = t * self.dt
            if bout=='feed' :
                self.feedchain_counter += 1
                self.cum_feedchain_dur += dur
                self.feedchain_lengths.append(self.current_feedchain_length)

            elif bout=='stride' :
                self.stridechain_counter += 1
                self.cum_stridechain_dur += dur
                self.stridechain_lengths.append(self.current_stridechain_length)

            elif bout=='pause' :
                self.pause_counter += 1
                self.cum_pause_dur += dur
                self.pause_durs.append(dur)

        self.reset_ticks()



class NengoIntermitter(OfflineIntermitter):
    def __init__(self, nengo_manager, **kwargs):
        super().__init__(**kwargs)
        self.nengo_manager = nengo_manager
        self.current_stridechain_length = self.stridechain_dist.sample()

    def disinhibit_locomotion(self):
        if np.random.uniform(0, 1, 1) >= self.EEB:
            self.crawler.set_freq(self.crawler.default_freq)
            self.feeder.set_freq(0)
            self.current_stridechain_length = self.stridechain_dist.sample()
        else:
            self.feeder.set_freq(self.feeder.default_freq)
            self.crawler.set_freq(0)
            self.current_feedchain_length = 1

    def inhibit_locomotion(self):
        self.crawler.set_freq(0)
        self.feeder.set_freq(0)

    # def update_state(self):
    #     if not self.effector:
    #         if np.random.uniform(0, 1, 1) > 0.97:
    #             self.start_effector()


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


class FittedIntermitter(OfflineIntermitter):
    def __init__(self, sample_dataset, **kwargs):
        sample = loadConf(sample_dataset, 'Ref')
        stored_conf = {
            'crawl_freq': sample['crawl_freq'],
            'feed_freq': sample['feed_freq'],
            'dt': sample['dt'],
            'stridechain_dist': sample['stride']['best'],
            'pause_dist': sample['pause']['best'],
            'feeder_reoccurence_rate': sample['feeder_reoccurence_rate'],
        }
        stored_conf.update(kwargs)
        stored_conf['crawl_bouts'] = True if stored_conf['crawl_freq'] is not None else False
        stored_conf['feed_bouts'] = True if stored_conf['feed_freq'] is not None else False
        # print(kwargs)
        super().__init__(**stored_conf)


def get_EEB_poly1d(sample=None, dt=None, **kwargs):
    if sample is not None:
        if type(sample) == str:
            sample = loadConf(sample, 'Ref')
        kws = sample['intermitter']
    else:
        kws = kwargs
    if dt is not None:
        kws['dt'] = dt

    EEBs = np.arange(0, 1.05, 0.05)
    ms = []
    for EEB in EEBs:
        inter = OfflineIntermitter(EEB=EEB,register_bouts=False, **kws)
        max_ticks = int(60 * 60 / inter.dt)
        while inter.total_ticks < max_ticks:
            inter.step()
        ms.append(inter.get_mean_feed_freq())
    z = np.poly1d(np.polyfit(np.array(ms), EEBs, 5))
    return z


def get_best_EEB(deb, sample):
    z = np.poly1d(sample['EEB_poly1d'])
    if type(deb) == dict:
        s = deb['feed_freq_estimate']
    else:
        s = deb.fr_feed
    return np.clip(z(s), a_min=0, a_max=1)

def get_EEB_time_fractions(sample=None, dt=None,**kwargs):
    if sample is not None:
        if type(sample) == str:
            sample = loadConf(sample, 'Ref')
        kws = sample['intermitter']
    else:
        kws = kwargs
    if dt is not None:
        kws['dt'] = dt

    rts= {f'{q} ratio' : nam.dur_ratio(p) for p,q in zip(['stridechain', 'pause','feedchain'], ['crawl', 'pause','feed'])}
    EEBs = np.round(np.arange(0, 1, 0.02),2)
    data=[]
    for EEB in EEBs:
        inter = OfflineIntermitter(EEB=EEB, **kws)
        max_ticks = int(1*60 * 60 / inter.dt)
        while inter.total_ticks < max_ticks:
            inter.step()
        dic=inter.build_dict()
        ffr=inter.get_mean_feed_freq()
        entry = {'EEB': EEB,  **{k : np.round(dic[v],2) for k,v in rts.items()}, nam.mean(nam.freq('feed')) : ffr}
        data.append(entry)
    df=pd.DataFrame.from_records(data=data)
    return df


if __name__ == "__main__":
    from lib.stor.larva_dataset import LarvaDataset
    sample = 'None.200_controls'
    sample = loadConf(sample, 'Ref')
    d=LarvaDataset(sample['dir'])
    d.config['EEB_poly1d'] = get_EEB_poly1d(**d.config['intermitter']).c.tolist()
    d.save_config()

    raise
    ffrs=np.arange(0,2,0.1)
    fig,ax=plt.subplots(1,1)
    sample='None.200_controls'
    for dt in [0.1, 0.08,0.12, 0.0625]:
        z=get_EEB_poly1d(sample,dt)
        plt.plot(ffrs,z(ffrs), label=dt)
        # dt=0.1
        # df=get_EEB_time_fractions(sample, dt)
        # print(dt, )
        # print(nam.mean(nam.freq('feed')))
    plt.show()
    raise
    inter = FittedIntermitter(sample_dataset='reference', dt=0.001, EEB=0.8, EEB_decay=1.0)
    # print(inter.EEB_decay)
    sample = loadConf('reference', 'Ref')
    kws = {
        'crawl_freq': sample['crawl_freq'],
        'feed_freq': sample['feed_freq'],
        'dt': 0.001,
        'crawl_bouts': True,
        'feed_bouts': True,
        'stridechain_dist': sample['stride']['best'],
        'pause_dist': sample['pause']['best'],
        'feeder_reoccurence_rate': sample['feeder_reoccurence_rate'],
    }
    inter2 = OfflineIntermitter(**null_dict('intermitter', EEB=0.8, **kws))
