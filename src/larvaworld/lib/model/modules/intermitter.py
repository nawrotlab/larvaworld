import os
import numpy as np
import pandas as pd
import param

from larvaworld.lib.model.modules.oscillator import Timer
from larvaworld.lib import reg, aux, util
from larvaworld.lib.aux import nam
from larvaworld.lib.param import PositiveNumber, OptionalPositiveNumber

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
    EEB = param.Magnitude(1.0, label='exploitation-exploration balance', doc='The baseline exploitation-exploration balance. 0 means only exploitation, 1 only exploration.')
    EEB_decay = PositiveNumber(1.0, softmax=2.0, doc='The exponential decay coefficient of the exploitation-exploration balance when no food is detected.')
    run_mode = param.Selector(objects=['stridechain','exec'], doc='The generation mode of the crawling epochs.')
    feeder_reoccurence_rate = OptionalPositiveNumber(softmax=1.0, label='feed reoccurence', doc='The default reoccurence rate of the feeding motion.')
    feed_bouts = param.Boolean(False, doc='Whether feeding epochs are generated.')

    def __init__(self, pause_dist=None, stridechain_dist=None, run_dist=None, crawl_freq=10 / 7, feed_freq=2.0, **kwargs):
        super().__init__(**kwargs)
        self.crawl_freq = crawl_freq
        self.feed_freq = feed_freq
        self.reset()

        self.cur_state = None


        if self.run_mode=='stridechain' :
            if stridechain_dist is None or stridechain_dist.range is None:
                stridechain_dist = default_bout_distros.run_count
            self.stridechain_min, self.stridechain_max = stridechain_dist.range
            self.stridechain_dist = util.BoutGenerator(**stridechain_dist, dt=1)
            self.run_dist = None

        elif self.run_mode=='exec' :
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


        self.Nstrides = 0
        self.Nstridechains = 0
        self.Nruns = 0
        self.Npauses = 0
        self.Nfeeds = 0
        self.Nfeedchains = 0
        self.Nfeeds_success = 0
        self.Nfeeds_fail = 0
        self.base_EEB = self.EEB


        self.exp_Nstrides = None
        self.cur_Nstrides = 0
        self.exp_Trun = None
        # self.current_run_duration =0
        self.exp_Tpause = None
        self.cur_Nfeeds = None

        # self.cum_feedchain_dur = 0
        # self.cum_stridechain_dur = 0
        # self.cum_run_dur = 0
        # self.cum_pause_dur = 0

        self.stridechain_lengths = []
        self.stridechain_durs = []
        self.feedchain_lengths = []
        self.feedchain_durs = []
        self.pause_durs = []
        self.run_durs = []
        self.feed_durs = []
        self.stride_durs = []




    @ property
    def pause_completed(self):
        t=self.exp_Tpause
        return t is not None and self.t > t

    @property
    def run_completed(self):
        t = self.exp_Trun
        return t is not None and self.t > t

    @property
    def stridechain_completed(self):
        n = self.exp_Nstrides
        return n is not None and self.cur_Nstrides > n

    def alternate_crawlNpause(self,stride_completed=False):
        if stride_completed :
            self.cur_Nstrides += 1


        if self.stridechain_completed :
            self.register('stridechain')
            self.exp_Tpause = self.generate_pause()
            self.cur_state = 'pause'

        elif self.pause_completed :
            self.register('pause')
            self.run_initiation()

        elif self.run_completed:
            self.register('run')
            self.exp_Tpause = self.generate_pause()
            self.cur_state = 'pause'

    @ property
    def feed_repeated(self):
        r = self.feeder_reoccurence_rate if self.feeder_reoccurence_rate is not None else self.EEB
        return np.random.uniform(0, 1, 1) < r


    def alternate_exploreNexploit(self,feed_motion=False,on_food=False):

        if feed_motion :
            if self.cur_Nfeeds is None:
                raise
            else :
                self.Nfeeds += 1
                if not on_food:
                    self.Nfeeds_fail +=1
                    self.register('feedchain')
                    self.run_initiation()
                else:
                    self.Nfeeds_success += 1
                    if self.feed_repeated :
                        self.cur_Nfeeds += 1
                    else:
                        self.register('feedchain')
                        self.run_initiation()
        elif on_food and self.cur_Nfeeds is None and np.random.uniform(0, 1, 1) <= self.EEB:
            self.cur_Nfeeds = 1
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
            self.Nfeedchains += 1
            self.feedchain_lengths.append(self.cur_Nfeeds)
            self.feedchain_durs.append(dur)
            self.cur_Nfeeds = None

        elif bout == 'stridechain':
            self.Nstridechains += 1
            self.stridechain_lengths.append(self.cur_Nstrides)
            self.stridechain_durs.append(dur)
            self.exp_Nstrides = None
            self.cur_Nstrides = 0

        elif bout == 'run':
            self.Nruns += 1
            self.run_durs.append(dur)
            self.exp_Trun = None

        elif bout == 'pause':
            self.Npauses += 1
            self.pause_durs.append(dur)
            self.exp_Tpause = None

    def update_state(self, stride_completed=False, feed_motion=False,on_food=False):
        if self.feed_bouts :
            self.alternate_exploreNexploit(feed_motion, on_food)
        self.alternate_crawlNpause(stride_completed)
        # print(self.EEB, self.feeder_reoccurence_rate, self.base_EEB)
        return self.cur_state


    def step(self, **kwargs):
        if self.cur_state is None :
            self.run_initiation()
        self.count_time()
        return self.update_state(**kwargs)

    def generate_stridechain(self):
        return self.stridechain_dist.sample()

    def generate_run(self):
        return self.run_dist.sample()

    def run_initiation(self):
        if self.stridechain_dist is not None:
            self.exp_Nstrides = self.stridechain_dist.sample()
        elif self.run_dist is not None:
            self.exp_Trun = self.run_dist.sample()
        self.cur_Nstrides = 0
        self.cur_state = 'exec'
        self.ticks = 0


    def generate_pause(self):
        return self.pause_dist.sample()


    def build_dict(self):
        cum_t=nam.cum('t')
        d = {}
        d[cum_t] = self.total_t
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
            d[nam.num(c)] = len(d[t])
            cum_chunk_t = nam.cum(t)
            d[cum_chunk_t] = np.sum(d[t])
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
        return self.exp_Nstrides, self.cur_Nfeeds, self.exp_Tpause, self.exp_Trun

    @property
    def mean_feed_freq(self):
        return self.Nfeeds / self.total_t



class OfflineIntermitter(Intermitter):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        self.crawl_ticks = np.round(1 / (self.crawl_freq * self.dt)).astype(int)
        self.feed_ticks = np.round(1 / (self.feed_freq * self.dt)).astype(int)

    def step(self, stride_completed=None, feed_motion=None,on_food=False):
        self.count_time()
        if feed_motion is None:
            feed_motion = self.cur_state == 'feed' and self.ticks%self.feed_ticks==0
        if stride_completed is None:
            stride_completed = self.cur_state == 'exec' and self.ticks%self.crawl_ticks==0
        return self.update_state(stride_completed, feed_motion, on_food)



class NengoIntermitter(Intermitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class BranchIntermitter(Intermitter):
    def __init__(self,beta=None,c=0.7,sigma=1,feed_bouts=False,**kwargs):
        super().__init__(feed_bouts=feed_bouts,**kwargs)
        self.c = c
        self.beta = beta
        self.sigma = sigma

    def generate_stridechain(self):
        return util.exp_bout(beta=self.beta, tmax=self.stridechain_max, tmin=self.stridechain_min)

    def generate_pause(self):
        return util.critical_bout(c=self.c, sigma=self.sigma, N=1000, tmax=self.pau_max, tmin=self.pau_min)*self.dt


class FittedIntermitter(OfflineIntermitter):
    def __init__(self, refID, **kwargs):
        c = reg.getRef(refID)['intermitter']
        stored_conf = {
            'crawl_freq': c['crawl_freq'],
            'feed_freq': c['feed_freq'],
            'dt': c['dt'],
            'stridechain_dist': c['stride']['best'],
            'pause_dist': c['pause']['best'],
            'feeder_reoccurence_rate': c['feeder_reoccurence_rate'],
        }
        stored_conf.update(kwargs)
        stored_conf['feed_bouts'] = True if stored_conf['feed_freq'] is not None else False
        super().__init__(**stored_conf)

def get_EEB_poly1d(**kws):
    max_dur = 60 * 60
    EEBs = np.arange(0, 1.05, 0.05)
    ms = []
    for EEB in EEBs:
        M = OfflineIntermitter(EEB=EEB, **kws)
        while M.total_t < max_dur:
            M.step()
        ms.append(M.mean_feed_freq)
    z = np.poly1d(np.polyfit(np.array(ms), EEBs, 5))
    return z

def get_EEB_time_fractions(refID=None, dt=None, **kwargs):
    if refID is not None:
        kws = reg.getRef(refID)['intermitter']
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
        M = OfflineIntermitter(EEB=EEB, **kws)
        while M.total_t < max_dur:
            M.step()
        dic = M.build_dict()
        entry = {'EEB': EEB, **{k: np.round(dic[v], 2) for k, v in rts.items()}, nam.mean(nam.freq('feed')): M.mean_feed_freq}
        data.append(entry)
    df = pd.DataFrame.from_records(data=data)
    return df

