import copy
import os

import numpy as np
import pandas as pd

from lib.aux import dictsNlists as dNl
from lib.registry.dtypes import null_dict
from lib.registry.ga_dict import ga_dict, interference_ga_dict
from lib.conf.stored.conf import loadConf, loadRef, expandConf, saveConf
from lib.registry.pars import preg
from lib.plot.base import BasePlot


class Calibration:
    def __init__(self, refID, turner_mode='neural', physics_keys=None, absolute=True, shorts=None):
        if shorts is None:
            shorts = ['b', 'fov', 'foa']
        if physics_keys is None:
            physics_keys = []
        if turner_mode == 'neural':
            turner_keys =['base_activation','n','tau']
        elif turner_mode == 'sinusoidal':
            turner_keys = ['initial_amp','initial_freq']
        PH = ga_dict(name='physics', only=physics_keys)
        TUR = ga_dict(name='turner', only=turner_keys)
        space_dict={**PH, **TUR}

        self.turner_mode = turner_mode
        self.base_turner= {**null_dict('base_turner', mode=self.turner_mode),  **null_dict(f'{self.turner_mode}_turner')}
        self.space_dict = space_dict
        self.turner_keys = turner_keys
        self.physics_keys = physics_keys
        self.refID=refID
        self.refDataset =d= loadRef(refID)
        d.load(contour=False)
        s, e, c = d.step_data, d.endpoint_data, d.config
        self.absolute = absolute
        self.shorts = shorts
        self.target = {sh : d.get_chunk_par(chunk='pause', short=sh, min_dur=3, mode='distro') for sh in self.shorts}
        self.N = self.target[self.shorts[0]].shape[0]
        self.dt = c.dt
        self.best = None
        self.KS_dic = None

    def build_modelConf(self,new_id=None, **kwargs) :
        if new_id is None :
            new_id =f'fitted_{self.turner_mode}_turner'
        m= self.refDataset.average_modelConf(new_id=new_id, **self.best, **kwargs)
        return {new_id : m}




    def plot_turner_distros(self, sim, fig=None, axs=None,in_deg=False,**kwargs):
        Nps = len(self.shorts)
        P=BasePlot(name='turner_distros',**kwargs)
        P.build(Ncols=Nps, fig=fig, axs=axs,figsize=(5 * Nps, 5), sharey=True)
        for i, sh in enumerate(self.shorts):
            p, lab = preg.getPar(sh, to_return=['d', 'lab'])
            vs = self.target[sh]
            if in_deg :
                vs=np.rad2deg(vs)
            lim = np.nanquantile(vs, 0.99)
            bins = np.linspace(-lim, lim, 100)

            ws = np.ones_like(vs) / float(len(vs))
            P.axs[i].hist(vs, weights=ws, label='experiment', bins=bins, color='red', alpha=0.5)

            vs0 = sim[sh]
            if in_deg :
                vs0=np.rad2deg(vs0)
            ws0 = np.ones_like(vs0) / float(len(vs0))
            P.axs[i].hist(vs0, weights=ws0, label='model', bins=bins, color='blue', alpha=0.5)
            P.conf_ax(i, ylab='probability' if i == 0 else None, xlab=lab,
                      xMaxN=4, yMaxN=4, leg_loc='upper left'if i == 0 else None)
        P.adjust(LR=(0.1,0.95), BT=(0.15,0.95), W=0.01, H=0.1)
        return P.get()

    def sim_turner(self, turner, physics, N=2000):
        from lib.model.modules.turner import Turner
        from lib.aux.ang_aux import wrap_angle_to_0

        L = Turner(dt=self.dt, **turner)

        def compute_ang_vel(b, torque, fov):
            dv = -physics.ang_damping * fov - physics.body_spring_k * b + torque
            return fov + dv * self.dt

        simFOV = np.zeros(N) * np.nan
        simB = np.zeros(N) * np.nan
        b = 0
        fov = 0
        for i in range(N):
            ang=L.step(0)
            fov = compute_ang_vel(b, physics.torque_coef * ang, fov)
            b = wrap_angle_to_0(b + fov * self.dt)
            simFOV[i] = fov
            simB[i] = b

        simB=np.rad2deg(simB)
        simFOV=np.rad2deg(simFOV)
        simFOA = np.diff(simFOV, prepend=[0]) / self.dt

        if 'tur_t' in self.shorts or 'tur_fou' in self.shorts:
            from lib.process.aux import detect_turns, process_epochs

            Lturns, Rturns = detect_turns(pd.Series(simFOV), self.dt)
            Ldurs, Lamps, Lmaxs = process_epochs(simFOV, Lturns, self.dt, return_idx=False)
            Rdurs, Ramps, Rmaxs = process_epochs(simFOV, Rturns, self.dt, return_idx=False)
            Tamps = np.concatenate([Lamps, Ramps])
            Tdurs = np.concatenate([Ldurs, Rdurs])


            sim = {'b': simB, 'fov': simFOV, 'foa': simFOA, 'tur_t' : Tdurs, 'tur_fou' : Tamps}

        else :
            sim = {'b': simB, 'fov': simFOV, 'foa': simFOA}
        return sim


    def eval_turner(self, sim):
        from scipy.stats import ks_2samp

        if not self.absolute:
            Ks_dic = {sh: np.round(ks_2samp(self.target[sh], sim[sh])[0], 3) for sh in self.shorts}
        else:
            Ks_dic = {sh: np.round(ks_2samp(np.abs(self.target[sh]), np.abs(sim[sh]))[0], 3) for sh in self.shorts}
        err = np.sum(list(Ks_dic.values()))
        return err, Ks_dic

    def retrieve_modules(self,q, Ndec=None):
        dic = dNl.NestDict({k: q0 for (k, dic), q0 in zip(self.space_dict.items(), q)})
        turner = dNl.NestDict(copy.deepcopy(self.base_turner))

        if Ndec is not None :
            physics = null_dict('physics', **{k: np.round(dic[k], Ndec) for k in self.physics_keys})
            turner.update({k: np.round(dic[k], Ndec) for k in self.turner_keys})
        else :
            physics = null_dict('physics', **{k: dic[k] for k in self.physics_keys})
            turner.update({k: dic[k] for k in self.turner_keys})
        # print(q, physics)
        return physics, turner

    def optimize_turner(self, q=None, return_sim=False, N=4000):

        physics, turner = self.retrieve_modules(q)
        sim = self.sim_turner(turner, physics, N=N)
        if return_sim :
            return sim
        else :
            err, Ks_dic = self.eval_turner(sim)
            return err

    def run(self, method='Nelder-Mead',**kwargs):
        from scipy.optimize import minimize

        print(f'Calibrating parameters {list(self.space_dict.keys())}')
        bnds = [(dic['min'], dic['max']) for k,dic in self.space_dict.items()]
        init = np.array([dic['initial_value'] for k,dic in self.space_dict.items()])

        #print(bnds)
        res = minimize(self.optimize_turner, init, method=method, bounds=bnds, **kwargs)
        self.best, self.KS_dic = self.plot_turner(q=res.x)

    def plot_turner(self, q):
        from lib.aux.sim_aux import fft_max

        physics, turner = self.retrieve_modules(q, Ndec=2)
        sim = self.sim_turner(turner, physics, N=self.N)
        err, Ks_dic = self.eval_turner(sim)

        for key, val in turner.items():
            print(key, ' : ', val)
        for key, val in physics.items():
            print(key, ' : ', val)

        print(Ks_dic)
        ffov = fft_max(sim['fov'], self.dt, fr_range=(0.1, 0.8))
        print('ffov : ', np.round(ffov, 2), 'dt : ', self.dt)
        _=self.plot_turner_distros(sim)
        best = dNl.NestDict({'turner': turner, 'physics': physics})

        return best, Ks_dic
        # pass

def calibrate_interference(mID,refID, dur=None, N=10, Nel=2, Ngen=20,**kwargs):
    from lib.ga.robot.larva_offline import LarvaOffline
    from lib.conf.stored.ga_conf import distro_KS_interference_evaluation
    from lib.ga.util.ga_launcher import GAlauncher

    d = loadRef(refID)
    c=d.config
    if dur is None :
        dur=c.Nticks*c.dt/60

    build_kws = {
        'fitness_target_refID': refID,
        'fitness_target_kws': {'eval_shorts': ['fov', 'foa','b'], 'pooled_cycle_curves': ['fov', 'foa','b']},
        'base_model': mID,
        'bestConfID': mID,
        'exclude_func': None,
        'init_mode': 'model',
        'robot_class': LarvaOffline,
        'space_dict': interference_ga_dict(mID),
        'fitness_func': distro_KS_interference_evaluation,
        'plot_func': None,
    }

    kws = {'sim_params': null_dict('sim_params', duration=dur, timestep=c.dt),
           'scene': 'no_boxes',
           'experiment': 'realism',
           'env_params':expandConf('arena_200mm', 'Env'),
           'offline' : True,
           'show_screen' : False
           }

    kws['ga_select_kws'] = null_dict('ga_select_kws', Nagents=N, Nelits=Nel, Ngenerations=Ngen)
    kws['ga_build_kws'] = null_dict('ga_build_kws', **build_kws)
    kws.update(kwargs)

    conf = null_dict('GAconf', **kws)

    GA = GAlauncher(**conf)
    best_genome = GA.run()

    mm=loadConf(mID, 'Model')
    IF =mm.brain.interference_params
    if IF.attenuation + IF.attenuation_max > 1 :
        mm.brain.interference_params.attenuation_max = np.round(1-IF.attenuation,2)
        saveConf(id=mID, conf_type='Model', conf=mm)

    return {mID: mm}

def adapt_crawler(ee, waveform='realistic', average=True):
    if waveform=='realistic':
        if average:
            crawler = null_dict('crawler',waveform='realistic',
                                initial_freq=np.round(ee[preg.getPar('fsv')].median(), 2),
                                stride_dst_mean=np.round(ee[preg.getPar('str_sd_mu')].median(), 2),
                                stride_dst_std=np.round(ee[preg.getPar('str_sd_std')].median(), 2),
                                max_vel_phase=np.round(ee['phi_scaled_velocity_max'].median(), 2),
                                max_scaled_vel=np.round(ee[preg.getPar('str_sv_max')].median(), 2))

        else:
            crawler = null_dict('crawler',waveform='realistic',
                                initial_freq=ee[preg.getPar('fsv')],
                                stride_dst_mean=ee[preg.getPar('str_sd_mu')],
                                stride_dst_std=ee[preg.getPar('str_sd_std')],
                                max_vel_phase=ee['phi_scaled_velocity_max'],
                                max_scaled_vel=ee[preg.getPar('str_sv_max')])
    elif waveform=='constant':
        if average:
            crawler = null_dict('crawler',waveform='constant',
                                initial_amp=np.round(ee[preg.getPar('run_v_mu')].median(), 2))
        else:
            crawler = null_dict('crawler',waveform='constant',
                                initial_amp=ee[preg.getPar('run_v_mu')]
                                )
    return crawler

def adapt_intermitter(c, e, **kwargs) :
    intermitter = null_dict('intermitter')
    intermitter.stridechain_dist = c.bout_distros.run_count
    try:
        ll1, ll2 = intermitter.stridechain_dist.range
        intermitter.stridechain_dist.range = (int(ll1), int(ll2))
    except:
        pass

    intermitter.run_dist = c.bout_distros.run_dur
    try:
        ll1, ll2 = intermitter.run_dist.range
        intermitter.run_dist.range = (np.round(ll1, 2), np.round(ll2, 2))
    except:
        pass
    intermitter.pause_dist = c.bout_distros.pause_dur
    try:
        ll1, ll2 = intermitter.pause_dist.range
        intermitter.pause_dist.range = (np.round(ll1, 2), np.round(ll2, 2))
    except:
        pass
    intermitter.crawl_freq = np.round(e[preg.getPar('fsv')].median(), 2)
    return intermitter

def adapt_interference(c, e, mode='phasic', average=True) :
    if average :
        at_phiM = np.round(e['phi_attenuation_max'].median(), 1)

        pau_fov_mu=e[preg.getPar('pau_fov_mu')]

        att0 = np.clip(np.round((e[preg.getPar('run_fov_mu')] / pau_fov_mu).median(), 2), a_min=0, a_max=1)
        fov_curve = c.pooled_cycle_curves['fov']['abs']
        #print(fov_curve)
        # att0=np.round(np.clip(np.nanmean(att0s),a_min=0, a_max=1),2)
        att1 = np.min(fov_curve) / pau_fov_mu.median()
        att2 = np.max(fov_curve) / pau_fov_mu.median() - att1
        att1 = np.round(np.clip(att1, a_min=0, a_max=1), 2)
        att2 = np.round(np.clip(att2, a_min=0, a_max=1 - att1), 2)

        if mode=='phasic':

            interference = {**null_dict('base_interference', mode='phasic',suppression_mode='amplitude', attenuation_max=att2, attenuation=att1),
                      **null_dict('phasic_interference', max_attenuation_phase=at_phiM)}

            # interference = null_dict('interference', mode='phasic', suppression_mode='amplitude', max_attenuation_phase=at_phiM,
            #                      attenuation_max=att2, attenuation=att1)
        elif mode == 'square':
            interference = {**null_dict('base_interference', mode='square',suppression_mode='amplitude', attenuation_max=att2, attenuation=att0),
                            **null_dict('square_interference',  crawler_phi_range=(at_phiM - 1, at_phiM + 1))}

            # interference = null_dict('interference', mode='square', suppression_mode='amplitude', attenuation_max=att2,
            #                       max_attenuation_phase=None,
            #                       crawler_phi_range=(at_phiM - 1, at_phiM + 1),
            #                       attenuation=att0)
    else :
        raise ValueError ('Not implemented')
    return interference

def adapt_turner(e, mode = 'neural', average=True) :
    if mode == 'neural':
        if average:
            fr_mu = e[preg.getPar('ffov')].median()
            coef, intercept = 0.024, 5
            A_in_mu = np.round(fr_mu / coef + intercept)

            turner = {**null_dict('base_turner', mode='neural'),
            **null_dict('neural_turner', base_activation=A_in_mu,
                      activation_range=(10.0, 40.0)
                      )}
        else:
            raise ValueError('Not implemented')
    elif mode == 'sinusoidal':
        if average:
            fr_mu = e[preg.getPar('ffov')].median()
            turner = {**null_dict('base_turner', mode='sinusoidal'),
            **null_dict('sinusoidal_turner',
                        initial_freq=np.round(fr_mu, 2),
                               freq_range = (0.1,0.8),
                               initial_amp = np.round(e[preg.getPar('pau_foa_mu')].median(), 2)/10,
                               amp_range = (0.0,100.0)
                      )}

        else:
            raise ValueError('Not implemented')
    elif mode == 'constant':
        if average:
            turner = {**null_dict('base_turner', mode='constant'),
                      **null_dict('constant_turner',
                                  initial_amp=np.round(e[preg.getPar('pau_foa_mu')].median(), 2),
                                  # amp_range=(-1000.0, 1000.0)
                                  )}
        else:
            raise ValueError('Not implemented')
    return turner

def adapt_locomotor(c,e,average=True):
    if average:
        m=null_dict('locomotor')
        m.turner_params = adapt_turner(e, mode='neural',average=True)
        m.crawler_params = adapt_crawler(e, waveform='realistic',average=True)
        m.intermitter_params = adapt_intermitter(c, e)
        m.interference_params = adapt_interference(c, e, mode='phasic', average=True)
        m.feeder_params = None

    else:
        raise ValueError('Not implemented')
    return m

def calibrate_4models(refID='None.150controls') :
    mIDs=[]
    mdict={}
    for Tmod, Tlab in zip(['neural', 'sinusoidal'], ['NEU', 'SIN']) :
        C = Calibration(refID=refID, turner_mode=Tmod)
        C.run()
        for IFmod, IFlab in zip(['phasic', 'square'], ['PHI', 'SQ']) :
            mID=f'{IFlab}on{Tlab}'
            mIDs.append(mID)
            m = C.build_modelConf(new_id=mID, interference_mode=IFmod)
            mm = calibrate_interference(mID=mID, refID=refID)
            mdict.update(mm)

    update_modelConfs(d=loadRef(refID), mIDs=mIDs)
    return mdict

def update_modelConfs(d, mIDs):
    save_to = f'{d.dir_dict.model_tables}/4models'
    os.makedirs(save_to, exist_ok=True)
    for mID in mIDs:
        d.config.modelConfs.average[mID] = loadConf(mID, 'Model')
    d.save_config(add_reference=True)


if __name__ == '__main__':

    refID = 'None.150controls'
    # mIDs = ['PHIonNEU', 'SQonNEU', 'PHIonSIN', 'SQonSIN']
    # for mID in mIDs:
    #     m=loadConf(mID, 'Model')
    #     print(m)
    calibrate_4models(refID)
