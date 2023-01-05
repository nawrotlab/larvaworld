import numpy as np
import pandas as pd

from lib.model.agents._larva import PhysicsController
from lib import reg, aux


class Calibration:
    def __init__(self, refID, turner_mode='neural', physics_keys=None, absolute=True, shorts=None):
        self.refID = refID
        self.refDataset = d = reg.loadRef(refID)
        d.load(contour=False)
        s, e, c = d.step_data, d.endpoint_data, d.config
        if shorts is None:
            shorts = ['b', 'fov', 'foa', 'tur_t', 'tur_fou']
        self.shorts = shorts
        self.target = {sh: d.get_chunk_par(chunk='pause', short=sh, min_dur=3, mode='distro') for sh in self.shorts}
        self.N = self.target[self.shorts[0]].shape[0]
        self.dt = c.dt

        if physics_keys is None:
            physics_keys = []
        if turner_mode == 'neural':
            turner_keys = ['base_activation', 'n', 'tau']
        elif turner_mode == 'sinusoidal':
            turner_keys = ['initial_amp', 'initial_freq']
        elif turner_mode == 'constant':
            turner_keys = ['initial_amp']

        self.mkeys = aux.NestDict({
            'turner': turner_keys,
            'physics': physics_keys,
            'all': physics_keys + turner_keys
        })

        self.D = reg.model
        self.mdicts = aux.NestDict({
            'turner': self.D.dict.model.m['turner'].mode[turner_mode].args,
            'physics': self.D.dict.model.m['physics'].args
        })
        self.mconfs = aux.NestDict({
            'turner': None,
            'physics': None
        })

        init, bounds = [], []
        for k in self.mkeys.turner:
            p = self.mdicts.turner[k]
            init.append(p.v)
            bounds.append(p.lim)
        for k in self.mkeys.physics:
            p = self.mdicts.physics[k]
            init.append(p.v)
            bounds.append(p.lim)
        self.init = np.array(init)
        self.bounds = bounds

        self.Tfunc = self.D.dict.model.m['turner'].mode[turner_mode].class_func
        self.turner_mode = turner_mode
        self.absolute = absolute

        self.best = None
        self.KS_dic = None

    def sim_turner(self, N=2000):
        T = self.Tfunc(**self.mconfs.turner, dt=self.dt)
        PH = PhysicsController(**self.mconfs.physics)
        simFOV = np.zeros(N) * np.nan
        simB = np.zeros(N) * np.nan
        b = 0
        fov = 0
        for i in range(N):
            ang = T.step(0)
            fov = PH.compute_ang_vel(torque=PH.torque_coef * ang, v=fov, b=b,
                                     c=PH.ang_damping, k=PH.body_spring_k, dt=T.dt)
            b = aux.wrap_angle_to_0(b + fov * self.dt)
            simFOV[i] = fov
            simB[i] = b

        simB = np.rad2deg(simB)
        simFOV = np.rad2deg(simFOV)
        simFOA = np.diff(simFOV, prepend=[0]) / self.dt

        if 'tur_t' in self.shorts or 'tur_fou' in self.shorts:
            from lib.process.annotation import detect_turns
            from lib.process.annotation import process_epochs

            Lturns, Rturns = detect_turns(pd.Series(simFOV), self.dt)
            Ldurs, Lamps, Lmaxs = process_epochs(simFOV, Lturns, self.dt, return_idx=False)
            Rdurs, Ramps, Rmaxs = process_epochs(simFOV, Rturns, self.dt, return_idx=False)
            Tamps = np.concatenate([Lamps, Ramps])
            Tdurs = np.concatenate([Ldurs, Rdurs])

            sim = {'b': simB, 'fov': simFOV, 'foa': simFOA, 'tur_t': Tdurs, 'tur_fou': Tamps}

        else:
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

    def retrieve_modules(self, q, Ndec=None):
        if Ndec is not None:
            q = [np.round(q0, Ndec) for q0 in q]
        dic = aux.NestDict({k: q0 for k, q0 in zip(self.mkeys.all, q)})

        for k, mdict in self.mdicts.items():
            kwargs = {kk: dic[kk] for kk in self.mkeys[k]}
            self.mconfs[k] = self.D.conf(mdict=mdict, **kwargs)
        # return mconfs

    def optimize_turner(self, q=None, return_sim=False, N=2000):
        self.retrieve_modules(q)
        sim = self.sim_turner(N=N)
        if return_sim:
            return sim
        else:
            err, Ks_dic = self.eval_turner(sim)
            # print(err)
            return err

    def run(self, method='Nelder-Mead', **kwargs):
        from scipy.optimize import minimize

        # print(f'Calibrating parameters {list(self.space_dict.keys())}')
        # bnds = [(dic['min'], dic['max']) for k, dic in self.space_dict.items()]
        # init = np.array([dic['initial_value'] for k, dic in self.space_dict.items()])

        # print(bnds)
        # print(self.init)
        # print(self.bounds)
        # raise
        res = minimize(self.optimize_turner, self.init, method=method, bounds=self.bounds, **kwargs)
        self.best, self.KS_dic = self.eval_best(q=res.x)
        # self.best, self.KS_dic = self.plot_turner(q=res.x)

    #
    def eval_best(self, q):
        self.retrieve_modules(q, Ndec=2)
        sim = self.sim_turner(N=self.N)
        err, Ks_dic = self.eval_turner(sim)
        best = self.mconfs
        return best, Ks_dic

