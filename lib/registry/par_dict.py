from types import FunctionType
from typing import Tuple, List

import numpy as np
import param

from lib.aux import naming as nam, dictsNlists as dNl
from lib.aux.par_aux import bar, wave, sub, subsup, th, Delta, dot, circledast, omega, ddot, mathring
from lib.registry.units import ureg


def get_vfunc(dtype, lim, vs):
    func_dic = {
        float: param.Number,
        int: param.Integer,
        str: param.String,
        bool: param.Boolean,
        dict: param.Dict,
        list: param.List,
        type: param.ClassSelector,
        List[int]: param.List,
        List[str]: param.List,
        List[float]: param.List,
        List[Tuple[float]]: param.List,
        FunctionType: param.Callable,
        Tuple[float]: param.Range,
        Tuple[int]: param.NumericTuple,
    }
    if dtype == float and lim == (0.0, 1.0):
        return param.Magnitude
    if type(vs) == list and dtype in [str, int]:
        return param.Selector
    else:
        return func_dic[dtype]


def vpar(vfunc, v0, h, lab, lim, dv, vs):
    f_kws = {
        'default': v0,
        'doc': h,
        'label': lab,
        'allow_None': True
    }
    if vfunc in [param.List, param.Number, param.Range]:
        if lim is not None:
            f_kws['bounds'] = lim
    if vfunc in [param.Range, param.Number]:
        if dv is not None:
            f_kws['step'] = dv
    if vfunc in [param.Selector]:
        f_kws['objects'] = vs
    func = vfunc(**f_kws, instantiate=True)
    return func


def preparePar(p, k=None, dtype=float, d=None, disp=None, sym=None, codename=None, lab=None, h=None, u_name=None,
               required_ks=[], u=ureg.dimensionless, v0=None, lim=None, dv=None, vs=None,
               vfunc=None, vparfunc=None, func=None, **kwargs):
    codename = p if codename is None else codename
    d = p if d is None else d
    disp = d if disp is None else disp
    k = k if k is not None else d
    sym = k if sym is None else sym
    if lab is None:
        if u == ureg.dimensionless:
            lab = f'{disp}'
        else:
            lab = fr'{disp} ({u})'

    h = lab if h is None else h
    if vparfunc is None:
        if vfunc is None:
            vfunc = get_vfunc(dtype=dtype, lim=lim, vs=vs)
        vparfunc = vpar(vfunc, v0, h, lab, lim, dv, vs)
    else:
        vparfunc = vparfunc()

    kws = {
        'name': p,
        'p': p,
        'd': d,
        'k': k,
        'disp': disp,
        'sym': sym,
        'codename': codename,
        'dtype': dtype,
        'func': func,
        'u': u,
        'required_ks': required_ks,
        'vparfunc': vparfunc,
        'dv': dv,
        'v0': v0,

    }
    return dNl.NestDict(kws)


class BaseParDict:
    def __init__(self, func_dict, in_rad=True, in_m=True):
        self.func_dict = func_dict
        self.dict = dNl.NestDict()
        self.dict_entries = []
        self.build_initial()
        self.build_angular(in_rad)
        self.build_spatial(in_m)
        self.build_chunks()
        self.build_sim_pars()

    def build_initial(self):
        kws = {'u': ureg.s}
        self.add(
            **{'p': 'model.dt', 'k': 'dt', 'd': 'timestep', 'sym': '$dt$', 'lim': (0.01, 0.5), 'dv': 0.01, 'v0': 0.1,
               **kws})
        self.add(
            **{'p': 'cum_dur', 'k': nam.cum('t'), 'sym': sub('t', 'cum'), 'lim': (0.0, None), 'dv': 0.1, 'v0': 0.0,
               **kws})
        self.add(
            **{'p': 'num_ticks', 'k': 'N_ticks', 'sym': sub('N', 'ticks'),'dtype':int, 'lim': (0, None), 'dv': 1})

    def add(self, **kwargs):
        prepar = preparePar(**kwargs)
        self.dict[prepar.k] = prepar
        self.dict_entries.append(prepar)

    def add_rate(self, k0=None, k_time='t', p=None, k=None, d=None, sym=None, k_num=None, k_den=None, **kwargs):
        if k0 is not None:
            b = self.dict[k0]
            if p is None:
                p = f'd_{k0}'
            if k is None:
                k = f'd_{k0}'
            if d is None:
                d = f'{b.d} rate'
            if sym is None:
                sym = dot(b.sym)
            if k_num is None:
                k_num = f'D_{k0}'
        if k_den is None:
            k_den = f'D_{k_time}'

        b_num = self.dict[k_num]
        b_den = self.dict[k_den]
        # u = b_num.u / b_den.u

        kws = {
            'p': p,
            'k': k,
            'd': d,
            'sym': sym,
            'u': b_num.u / b_den.u,
            'required_ks': [k_num, k_den],
            # **kwargs
        }
        kws.update(kwargs)
        self.add(**kws)

    def add_operators(self, k0):
        b = self.dict[k0]
        kws0 = {'u': b.u, 'required_ks': [k0]}

        funcs = self.func_dict.ops

        mu_kws = {'d': nam.mean(b.d), 'p': nam.mean(b.p), 'sym': bar(b.sym), 'disp': f'mean {b.disp}',
                  'func': funcs.mean(b.d), 'k': f'{b.k}_mu'}

        std_kws = {'d': nam.std(b.d), 'p': nam.std(b.p), 'sym': wave(b.sym), 'disp': f'std {b.disp}',
                   'func': funcs.std(b.d),
                   'k': f'{b.k}_std'}

        min_kws = {'d': nam.min(b.d), 'p': nam.min(b.p), 'sym': sub(b.sym, 'min'), 'disp': f'minimum {b.disp}',
                   'func': funcs.min(b.d), 'k': f'{b.k}_min'}

        max_kws = {'d': nam.max(b.d), 'p': nam.max(b.p), 'sym': sub(b.sym, 'max'), 'disp': f'maximum {b.disp}',
                   'func': funcs.max(b.d), 'k': f'{b.k}_max'}

        fin_kws = {'d': nam.final(b.d), 'p': nam.final(b.p), 'sym': sub(b.sym, 'fin'), 'disp': f'final {b.disp}',
                   'func': funcs.final(b.d), 'k': f'{b.k}_fin'}

        init_kws = {'d': nam.initial(b.d), 'p': nam.initial(b.p), 'sym': sub(b.sym, '0'), 'disp': f'initial {b.disp}',
                    'func': funcs.initial(b.d), 'k': f'{b.k}0'}

        cum_kws = {'d': nam.cum(b.d), 'p': nam.cum(b.p), 'sym': sub(b.sym, 'cum'), 'disp': f'total {b.disp}',
                   'func': funcs.cum(b.d), 'k': nam.cum(b.k)}

        for kws in [mu_kws, std_kws, min_kws, max_kws, fin_kws, init_kws, cum_kws]:
            self.add(**kws, **kws0)

    def add_chunk(self, pc, kc, func=None, required_ks=[]):
        f_kws = {'func': func, 'required_ks': required_ks}

        ptr = nam.dur_ratio(pc)
        pl = nam.length(pc)
        pN = nam.num(pc)
        pN_mu = nam.mean(pN)
        ktr = f'{kc}_tr'
        kl = f'{kc}_l'
        kN = f'{kc}_N'
        kN_mu = f'{kN}_mu'
        kt = f'{kc}_t'

        kwlist = [
            {
                'p': pc,
                'k': kc,
                'sym': f'${kc}$',
                'disp': pc
            },
            {
                'p': nam.start(pc),
                'k': f'{kc}0',
                'u': ureg.s,
                'sym': subsup('t', kc, 0),
                'disp': f'{pc} start',
                **f_kws
            },
            {'p': nam.stop(pc),
             'k': f'{kc}1',
             'u': ureg.s,
             'sym': subsup('t', kc, 1),
             'disp': f'{pc} end',
             **f_kws},
            {
                'p': nam.id(pc),
                'k': f'{kc}_id',
                'sym': sub('idx', kc),
                'disp': f'{pc} idx',
                'dtype': str
            },
            {'p': ptr,
             'k': ktr,
             'sym': sub('r', kc),
             'disp': f'% time in {pc}s',
             'lim': (0.0, 1.0),
             'required_ks': [nam.cum(nam.dur(pc)), nam.cum(nam.dur(''))],
             'func': self.func_dict.tr(pc)},
            {
                'p': pN,
                'k': kN,
                'sym': sub('N', f'{pc}s'),
                'disp': f'# {pc}s',
                'dtype': int,
                **f_kws
            },
            {
                'p': nam.dur(pc),
                'k': kt,
                'sym': sub(Delta('t'), kc),
                'disp': f'{pc} duration',
                'u': ureg.s,
                **f_kws
            }]

        for kws in kwlist:
            self.add(**kws)

        for ii in ['on', 'off']:
            self.add(**{'p': f'{pN_mu}_{ii}_food', 'k': f'{kN_mu}_{ii}_food'})
            self.add(**{'p': f'{ptr}_{ii}_food', 'k': f'{ktr}_{ii}_food', 'lim': (0.0, 1.0)})

        self.add_rate(k_num=kN, k_den=nam.cum('t'), k=kN_mu, p=pN_mu, sym=bar(kN), disp=f' mean # {pc}s/sec', func=func)
        self.add_operators(k0=kt)

        if str.endswith(pc, 'chain'):
            self.add(**{'p': pl, 'k': kl, 'sym': sub('l', kc), 'dtype': int, **f_kws})
            self.add_operators(k0=kl)

    def add_chunk_track(self, kc, k):
        bc = self.dict[kc]
        b = self.dict[k]
        b0, b1 = self.dict[f'{kc}0'], self.dict[f'{kc}1']
        kws = {
            'func': self.func_dict.track_par(bc.p, b.p),
            'u': b.u
        }
        k01 = f'{kc}_{k}'
        kws0 = {
            'p': nam.at(b.p, b0.p),
            'k': f'{kc}_{k}0',
            'disp': f'{b.disp} at {bc.p} start',
            'sym': subsup(b.sym, kc, 0),
            **kws
        }
        kws1 = {
            'p': nam.at(b.p, b1.p),
            'k': f'{kc}_{k}1',
            'disp': f'{b.disp} at {bc.p} stop',
            'sym': subsup(b.sym, kc, 1),
            **kws
        }

        kws01 = {
            'p': nam.chunk_track(bc.p, b.p),
            'k': k01,
            'disp': f'{b.disp} during {bc.p}s',
            'sym': sub(Delta(b.sym), kc),
            **kws
        }
        self.add(**kws0)
        self.add(**kws1)
        self.add(**kws01)
        self.add_operators(k0=k01)

    def add_velNacc(self, k0, p_v=None, k_v=None, d_v=None, sym_v=None, disp_v=None, p_a=None, k_a=None, d_a=None,
                    sym_a=None, disp_a=None, func_v=None):
        b = self.dict[k0]
        b_dt = self.dict['dt']
        if p_v is None:
            p_v = nam.vel(b.p)
        if p_a is None:
            p_a = nam.acc(b.p)
        if d_v is None:
            d_v = nam.vel(b.d)
        if d_a is None:
            d_a = nam.acc(b.d)
        if k_v is None:
            k_v = f'{b.k}v'
        if k_a is None:
            k_a = f'{b.k}a'
        if sym_v is None:
            sym_v = dot(b.sym)
        if sym_a is None:
            sym_a = ddot(b.sym)

        if func_v is None:
            def func_v(d):
                from lib.aux.vel_aux import comp_rate
                s, e, c = d.step_data, d.endpoint_data, d.config
                comp_rate(s, c, p=b.d, pv=d_v)

        self.add(
            **{'p': p_v, 'k': k_v, 'd': d_v, 'u': b.u / b_dt.u, 'sym': sym_v, 'disp': disp_v, 'required_ks': [k0],
               'func': func_v})

        def func_a(d):
            from lib.aux.vel_aux import comp_rate
            s, e, c = d.step_data, d.endpoint_data, d.config
            comp_rate(s, c, p=d_v, pv=d_a)

        self.add(
            **{'p': p_a, 'k': k_a, 'd': d_a, 'u': b.u / b_dt.u ** 2, 'sym': sym_a, 'disp': disp_a, 'required_ks': [k_v],
               'func': func_a})

    def add_scaled(self, k0, **kwargs):
        b = self.dict[k0]
        b_l = self.dict['l']

        def func(d):
            from lib.process.spatial import scale_to_length
            s, e, c = d.step_data, d.endpoint_data, d.config
            scale_to_length(s, e, c, pars=[b.d], keys=None)

        kws = {
            'p': nam.scal(b.p),
            'k': f's{k0}',
            'd': nam.scal(b.d),
            'u': b.u / b_l.u,
            'sym': mathring(b.sym),
            'disp': f'scaled {b.disp}',
            'required_ks': [k0],
            'func': func
        }

        kws.update(kwargs)
        self.add(**kws)

    def add_unwrap(self, k0, **kwargs):
        b = self.dict[k0]
        if b.u == ureg.deg:
            in_deg = True
        elif b.u == ureg.rad:
            in_deg = False

        kws = {
            'p': nam.unwrap(b.p),
            'd': nam.unwrap(b.d),
            'k': f'{b.k}u',
            'u': b.u,
            'sym': b.sym,
            'disp': b.disp,
            'lim': None,
            'required_ks': [k0],
            'dv': b.dv,
            'v0': b.v0,
            'func': self.func_dict.unwrap(b.d, in_deg)
        }
        kws.update(kwargs)

        self.add(**kws)

    def add_dst(self, point='', **kwargs):
        xd, yd = nam.xy(point)
        xk, bx = [(k, p) for k, p in self.dict.items() if p.d == xd][0]
        yk, by = [(k, p) for k, p in self.dict.items() if p.d == yd][0]

        if bx.u == by.u:
            u = bx.u
        else:
            raise
        if bx.dv == by.dv:
            dv = bx.dv
        else:
            raise

        kws = {
            'p': nam.dst(point),
            'd': nam.dst(point),
            'k': f'{point}d',
            'u': u,
            'sym': sub('d', point),
            'disp': f'{point} distance',
            'lim': (0.0, None),
            'required_ks': [xk, yk],
            'dv': dv,
            'v0': 0.0,
            'func': self.func_dict.dst(point=point)
        }
        kws.update(kwargs)

        self.add(**kws)

    def add_freq(self, k0, **kwargs):
        b = self.dict[k0]
        kws = {
            'p': nam.freq(b.p),
            'd': nam.freq(b.d),
            'k': f'f{b.k}',
            'u': ureg.Hz,
            'sym': sub(b.sym, 'freq'),
            'disp': f'{b.disp} dominant frequency',
            'required_ks': [k0],
            'func': self.func_dict.freq(b.d)
        }
        kws.update(kwargs)
        self.add(**kws)

    def add_dsp(self, range=(0, 40), u=ureg.m):
        a = 'dispersion'
        k0 = 'dsp'
        s0 = circledast('d')
        r0, r1 = range
        dur = int(r1 - r0)
        p = f'{a}_{r0}_{r1}'
        k = f'{k0}_{r0}_{r1}'

        self.add(**{'p': p, 'k': k, 'u': u, 'sym': subsup(s0, f'{r0}', f'{r1}'),
                    'func': self.func_dict.dsp(range), 'required_ks': ['x', 'y'],
                    'lab': f"dispersal in {dur}''"})
        self.add_scaled(k0=k)
        self.add_operators(k0=k)
        self.add_operators(k0=f's{k}')

    def add_tor(self, dur):
        p0 = 'tortuosity'
        k0 = 'tor'
        k = f'{k0}{dur}'
        self.add(
            **{'p': f'{p0}_{dur}', 'k': k, 'lim': (0.0, 1.0), 'sym': sub(k0, dur), 'disp': f"{p0} over {dur}''",
               'func': self.func_dict.tor(dur)})
        self.add_operators(k0=k)

    def build_angular(self, in_rad=True):
        if in_rad:
            u = ureg.rad
            amax = np.pi
        else:
            u = ureg.deg
            amax = 180
        kws = {'dv': np.round(amax / 180, 2), 'u': u, 'v0': 0.0}
        self.add(
            **{'p': 'bend', 'k': 'b', 'sym': th('b'), 'disp': 'bending angle', 'lim': (-amax, amax), **kws})
        self.add_velNacc(k0='b', sym_v=omega('b'), sym_a=dot(omega('b')), disp_v='bending angular speed',
                         disp_a='bending angular acceleration')

        angs = [
            ['f', 'front', '', ''],
            ['r', 'rear', 'r', 'rear '],
            ['h', 'head', 'h', 'head '],
            ['t', 'tail', 't', 'tail '],
        ]

        for suf, psuf, ksuf, lsuf in angs:
            p0 = nam.orient(psuf)
            p_v, p_a = nam.vel(p0), nam.acc(p0)
            ko = f'{suf}o'
            kou = f'{ko}u'
            self.add(**{'p': p0, 'k': ko, 'sym': th(ksuf), 'disp': f'{lsuf}orientation',
                        'lim': (0, 2 * amax), **kws})

            self.add_unwrap(k0=ko)

            self.add_velNacc(k0=kou, k_v=f'{suf}ov', k_a=f'{suf}oa', p_v=p_v, d_v=p_v, p_a=p_a, d_a=p_a,
                             sym_v=omega(ksuf),sym_a=dot(omega(ksuf)), disp_v=f'{lsuf}angular speed',
                             disp_a=f'{lsuf}angular acceleration')
        for k0 in ['b', 'bv', 'ba', 'fov', 'foa', 'rov', 'roa', 'fo', 'ro']:
            self.add_freq(k0=k0)
            self.add_operators(k0=k0)

    def build_spatial(self, in_m=True):
        tor_durs = [1, 2, 5, 10, 20, 60, 120, 240, 300, 600]
        dsp_ranges = [(0, 40), (0, 60), (20, 80), (0, 120), (0, 240), (0, 300), (0, 600), (60, 120), (60, 300)]
        if in_m:
            u = ureg.m
            s = 1
        else:
            u = ureg.mm
            s = 1000

        kws = {'u': u}
        self.add(**{'p': 'x', 'disp': 'X position', 'sym': 'X', **kws})
        self.add(**{'p': 'y', 'disp': 'Y position', 'sym': 'Y', **kws})
        self.add(
            **{'p': 'real_length', 'k': 'l', 'd': 'length', 'disp': 'body length',
               'sym': '$l$','v0': 0.004 * s, 'lim': (0.0005 * s, 0.01 * s), 'dv': 0.0005 * s, **kws})

        self.add(
            **{'p': 'dispersion', 'k': 'dsp', 'sym': circledast('d'), 'disp': 'dispersal', **kws})

        d_d, d_v, d_a = nam.dst(''), nam.vel(''), nam.acc('')
        d_sd, d_sv, d_sa = nam.scal([d_d, d_v, d_a])
        self.add_dst(point='')
        self.add_velNacc(k0='d', k_v='v', k_a='a', p_v=d_v, d_v=d_v, p_a=d_a, d_a=d_a,
                         sym_v='v', sym_a=dot('v'), disp_v='speed', disp_a='acceleration',
                         func_v=self.func_dict.vel(d_d, d_v))
        self.add_scaled(k0='d')
        self.add_velNacc(k0='sd', k_v='sv', k_a='sa', p_v=d_sv, d_v=d_sv, p_a=d_sa, d_a=d_sa, sym_v=mathring('v'),
                         sym_a=dot(mathring('v')), disp_v='scaled speed', disp_a='scaled acceleration',
                         func_v=self.func_dict.vel(d_sd, d_sv))
        for k0 in ['l', 'd', 'sd', 'v', 'sv', 'a', 'sa', 'x', 'y']:
            self.add_freq(k0=k0)
            self.add_operators(k0=k0)

        for i in dsp_ranges:
            self.add_dsp(range=i, u=u)
        self.add(**{'p': 'tortuosity', 'k': 'tor', 'lim': (0.0, 1.0), 'sym': 'tor'})
        for dur in tor_durs:
            self.add_tor(dur=dur)
        self.add(**{'p': 'anemotaxis', 'sym': 'anemotaxis'})

    def build_chunks(self):
        d0 = {
            'str': 'stride',
            'pau': 'pause',
            'fee': 'feed',
            'tur': 'turn',
            'Ltur': 'Lturn',
            'Rtur': 'Rturn',
            'run': 'run',
            'str_c': nam.chain('stride'),
            'fee_c': nam.chain('feed')
        }
        for kc, pc in d0.items():
            temp = self.func_dict.chunk(kc)
            func = temp.func
            required_ks = temp.required_ks

            self.add_chunk(pc=pc, kc=kc, func=func, required_ks=required_ks)
            for k in ['fov', 'rov', 'foa', 'roa', 'x', 'y', 'fo', 'fou', 'ro', 'rou', 'b', 'bv', 'ba', 'v', 'sv', 'a',
                      'sa', 'd', 'sd']:
                self.add_chunk_track(kc=kc, k=k)
        self.add_rate(k_num='Ltur_N', k_den='tur_N', k='tur_H', p='handedness_score',
                      sym=sub('H', 'tur'), lim=(0.0, 1.0), disp='Handedness score')
        for ii in ['on', 'off']:
            self.add(**{'p': f'handedness_score_{ii}_food', 'k': f'tur_H_{ii}_food'})

    def build_sim_pars(self):
        for ii, jj in zip(['C', 'T'], ['crawler', 'turner']):
            self.add(**{'p': f'brain.locomotor.{jj}.output', 'k': f'A_{ii}', 'd': f'{jj} output', 'sym': sub('A', ii)})
            self.add(**{'p': f'brain.locomotor.{jj}.input', 'k': f'I_{ii}', 'd': f'{jj} input', 'sym': sub('I', ii)})

        self.add(**{'p': 'brain.locomotor.cur_ang_suppression', 'k': 'c_CT', 'd': 'ang_suppression',
                    'disp': 'angular suppression output', 'sym': sub('c', 'CT'), 'lim': (0.0, 1.0)})

        self.add(**{'p': 'brain.intermitter.EEB', 'k': 'EEB', 'd': 'exploitVSexplore_balance', 'lim': (0.0, 1.0),
                    'disp': 'exploitVSexplore_balance', 'sym': 'EEB'})

        for ii, jj in zip(['1', '2'], ['first', 'second']):
            k = f'c_odor{ii}'
            dk = f'd{k}'
            self.add(**{'p': f'brain.olfactor.{jj}_odor_concentration', 'k': k, 'd': k,
                        'disp': f'Odor {ii} concentration', 'sym': subsup('C', 'od', ii)})
            self.add(**{'p': f'brain.olfactor.{jj}_odor_concentration_change', 'k': dk, 'd': dk,
                        'disp': f'Odor {ii} concentration change', 'sym': subsup(Delta('C'), 'od', ii)})

        for ii, jj in zip(['W', 'C'], ['warm', 'cool']):
            k = f'temp_{ii}'
            dk = f'd{k}'

            self.add(**{'p': f'brain.thermosensor.{jj}_sensor_input', 'k': k, 'd': k,
                        'disp': f'{jj} sensor input', 'sym': sub('Temp', ii)})
            self.add(**{'p': f'brain.thermosensor.{jj}_sensor_perception', 'k': dk, 'd': dk, 'lim': (-0.1, 0.1),
                        'disp': f'{jj} sensor perception', 'sym': sub(Delta('Temp'), ii)})

        for ii, jj in zip(['olf', 'tou', 'wind', 'therm'], ['olfactor', 'toucher', 'windsensor', 'thermosensor']):
            self.add(
                **{'p': f'brain.{jj}.output', 'k': f'A_{ii}', 'd': f'{jj} output',
                   'disp': f'{jj} output', 'lim': (0.0, 1.0),
                   'sym': sub('A', ii)})
