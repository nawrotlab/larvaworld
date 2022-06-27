from typing import Tuple

import numpy as np
import param


from lib.aux import naming as nam, dictsNlists as dNl
from lib.aux.par_aux import bar, wave, sub, subsup, th, Delta, dot, circledast, omega, ddot, mathring
from lib.registry.units import ureg

func_dic = {
    # SupportsFloat: param.Number,
    float: param.Number,
    int: param.Integer,
    str: param.String,
    bool: param.Boolean,
    dict: param.Dict,
    list: param.List,
    Tuple[float]: param.Range,
    Tuple[int]: param.NumericTuple,
}


def vpar(vfunc, v0, h, lab, lim, dv,vs):
    f_kws = {
        'default': v0,
        'doc': h,
        'label': lab,
        'allow_None': True
    }
    if vfunc in [param.List,param.Number, param.Range]:
        if lim is not None:
            f_kws['bounds'] = lim
    if vfunc in [param.Range,param.Number]:
        if dv is not None:
            f_kws['step'] = dv
    if vfunc in [param.Selector]:
        f_kws['objects'] = vs
    func = vfunc(**f_kws, instantiate=True)
    return func


def preparePar(p, k, dtype=float, d=None, disp=None, sym=None, codename=None, lab=None, h=None, u_name=None,
                 required_ks=[], u=ureg.dimensionless, v0=None, lim=None, dv=None,vs=None,
                 vfunc=None, vparfunc=None, func=None, **kwargs):
    codename = p if codename is None else codename
    d = p if d is None else d
    disp = d if disp is None else disp
    sym = k if sym is None else sym
    if lab is None :
        if u==ureg.dimensionless :
            lab = f'{disp}'
        else :
            lab = fr'{disp} ({u})'

    h = lab if h is None else h
    if vparfunc is None:
        if vfunc is None:
            vfunc = func_dic[dtype]
        vparfunc = vpar(vfunc, v0, h, lab, lim, dv,vs)
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
        self.dict_entries=[]
        self.build_initial()
        self.build_angular(in_rad)
        self.build_spatial(in_m)
        self.build_chunks()
        self.build_sim_pars()


    def build_initial(self):
        kws = {'u': ureg.s, 'vfunc': param.Number}
        self.addPar(**{'p': 'model.dt', 'k': 'dt', 'd': 'timestep', 'sym': '$dt$', 'codename': 'model.dt',
                       'lim': (0.01, 0.5), 'dv': 0.01, 'v0': 0.1, **kws})
        self.addPar(
            **{'p': 'cum_dur', 'k': nam.cum('t'), 'd': nam.cum('dur'), 'sym': sub('t', 'cum'), 'lim': (0.0, None),
               'dv': 0.1, 'v0': 0.0, **kws})

    def addPar(self, **kwargs):
        prepar = preparePar(**kwargs)
        self.dict[prepar.k]=prepar
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
        u = b_num.u / b_den.u
        self.addPar(
            **{'p': p, 'k': k, 'd': d, 'sym': sym, 'u': u, 'required_ks': [k_num, k_den], 'vfunc': param.Number},
            **kwargs)

    def add_operators(self, k0):
        b = self.dict[k0]
        kws0 = {'vfunc': param.Number, 'u': b.u, 'required_ks': [k0]}

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
            self.addPar(**kws, **kws0)

    def add_chunk(self, pc, kc, func=None, required_ks=[]):
        f_kws = {'func': func, 'required_ks': required_ks}

        p0, p1, pt, pid, ptr, pN, pl = nam.start(pc), nam.stop(pc), nam.dur(pc), nam.id(pc), nam.dur_ratio(pc), nam.num(
            pc), nam.length(pc)
        pN_mu = nam.mean(pN)

        k0, k1, kid, ktr, kN, kt, kl = f'{kc}0', f'{kc}1', f'{kc}_id', f'{kc}_tr', f'{kc}_N', f'{kc}_t', f'{kc}_l'
        kN_mu = f'{kN}_mu'

        self.addPar(**{'p': pc, 'k': kc, 'sym': f'${kc}$', 'disp': pc})
        self.addPar(
            **{'p': p0, 'k': k0, 'u': ureg.s, 'sym': subsup('t', kc, 0), 'disp': f'{pc} start', 'vfunc': param.Number,
               **f_kws})
        self.addPar(
            **{'p': p1, 'k': k1, 'u': ureg.s, 'sym': subsup('t', kc, 1), 'disp': f'{pc} end', 'vfunc': param.Number,
               **f_kws})
        self.addPar(
            **{'p': pid, 'k': kid, 'sym': sub('idx', kc), 'dtype': str, 'disp': f'{pc} idx', 'vfunc': param.String})

        self.addPar(
            **{'p': ptr, 'k': ktr, 'sym': sub('r', kc), 'disp': f'% time in {pc}s', 'vfunc': param.Magnitude,
               'required_ks': [nam.cum(pt), nam.cum(nam.dur(''))], 'func': self.func_dict.tr(pc)})
        self.addPar(
            **{'p': pN, 'k': kN, 'sym': sub('N', f'{pc}s'), 'dtype': int, 'disp': f'# {pc}s', 'vfunc': param.Integer,
               **f_kws})

        for ii in ['on', 'off']:
            self.addPar(**{'p': f'{pN_mu}_{ii}_food', 'k': f'{kN_mu}_{ii}_food', 'vfunc': param.Number})
            self.addPar(**{'p': f'{ptr}_{ii}_food', 'k': f'{ktr}_{ii}_food', 'vfunc': param.Magnitude})

        self.add_rate(k_num=kN, k_den=nam.cum('t'), k=kN_mu, p=pN_mu, d=pN_mu, sym=bar(kN), disp=f' mean # {pc}s/sec',
                      func=func)
        self.addPar(**{'p': pt, 'k': kt, 'u': ureg.s, 'sym': sub(Delta('t'), kc), 'disp': f'{pc} duration',
                       'vfunc': param.Number, **f_kws})
        self.add_operators(k0=kt)

        if str.endswith(pc, 'chain'):
            self.addPar(**{'p': pl, 'k': kl, 'sym': sub('l', kc), 'dtype': int, 'vfunc': param.Integer, **f_kws})
            self.add_operators(k0=kl)

    def add_chunk_track(self, kc, k):
        bc = self.dict[kc]
        b = self.dict[k]
        b0, b1 = self.dict[f'{kc}0'], self.dict[f'{kc}1']
        p0, p1 = nam.at(b.p, b0.p), nam.at(b.p, b1.p)
        p01 = nam.chunk_track(bc.p, b.p)
        # print(p01,p0,p1)
        k01 = f'{kc}_{k}'
        k0,k1=f'{k01}0',f'{k01}1'
        disp01 = f'{b.disp} during {bc.p}s'
        disp0,disp1 =f'{b.disp} at {bc.p} start',f'{b.disp} at {bc.p} stop'

        sym01=sub(Delta(b.sym), kc)
        sym0,sym1=subsup(b.sym, kc, 0),subsup(b.sym, kc, 1)
        # func=self.func_dict.track_par(bc.p, b.p)
        kws={
            'vfunc': param.Number,
            'func': self.func_dict.track_par(bc.p, b.p),
            'u': b.u
        }
        self.addPar(**{'p': p01, 'k': k01, 'sym': sym01, 'disp': disp01, **kws})
        self.addPar(**{'p': p0, 'k': k0, 'sym': sym0, 'disp': disp0, **kws})
        self.addPar(**{'p': p1, 'k': k1, 'sym': sym1, 'disp': disp1, **kws})
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

        self.addPar(
            **{'p': p_v, 'k': k_v, 'd': d_v, 'u': b.u / b_dt.u, 'sym': sym_v, 'disp': disp_v, 'vfunc': param.Number,
               'required_ks': [k0],
               'func': func_v})

        def func_a(d):
            from lib.aux.vel_aux import comp_rate
            s, e, c = d.step_data, d.endpoint_data, d.config
            comp_rate(s, c, p=d_v, pv=d_a)

        self.addPar(
            **{'p': p_a, 'k': k_a, 'd': d_a, 'u': b.u / b_dt.u ** 2, 'sym': sym_a, 'disp': disp_a, 'required_ks': [k_v],
               'vfunc': param.Number, 'func': func_a})

    def add_scaled(self, k0, sym=None, disp=None, **kwargs):
        b = self.dict[k0]
        b_l = self.dict['l']
        u = b.u / b_l.u

        d = nam.scal(b.d)
        p = nam.scal(b.p)
        k = f's{k0}'

        if sym is None:
            sym = mathring(b.sym)
        if disp is None:
            disp = f'scaled {b.disp}'

        def func(d):
            from lib.process.spatial import scale_to_length
            s, e, c = d.step_data, d.endpoint_data, d.config
            scale_to_length(s, e, c, pars=[b.d], keys=None)

        self.addPar(
            **{'p': p, 'k': k, 'd': d, 'u': u, 'sym': sym, 'disp': disp, 'required_ks': [k0], 'vfunc': param.Number,
               'func': func},
            **kwargs)

    def add_unwrap(self, k0, k=None, d=None, p=None, disp=None, sym=None):
        b = self.dict[k0]
        if k is None:
            k = f'{b.k}u'
        if sym is None:
            sym = b.sym
        if d is None:
            d = nam.unwrap(b.d)
        if p is None:
            p = nam.unwrap(b.p)
        if disp is None:
            disp = b.disp
        if b.u == ureg.deg:
            in_deg = True
        elif b.u == ureg.rad:
            in_deg = False

        self.addPar(**{'p': p, 'd': d, 'k': k, 'u': b.u, 'sym': sym, 'disp': disp, 'lim': None, 'required_ks': [k0],
                       'vfunc': param.Number, 'dv': b.dv, 'v0': b.v0,
                       'func': self.func_dict.unwrap(b.d, in_deg)})

    def add_dst(self, k=None, d=None, p=None, disp=None, sym=None, point=''):
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

        if k is None:
            k = f'{point}d'
        if sym is None:
            sym = sub('d', point)
        if d is None:
            d = nam.dst(point)
        if p is None:
            p = nam.dst(point)
        if disp is None:
            disp = f'{point} distance'

        self.addPar(
            **{'p': p, 'd': d, 'k': k, 'u': u, 'sym': sym, 'disp': disp, 'lim': (0.0, None), 'required_ks': [xk, yk],
               'vfunc': param.Number, 'dv': dv, 'v0': 0.0, 'func': self.func_dict.dst(point=point)})

    def add_freq(self, k0, k=None, d=None, p=None, disp=None, sym=None, **kwargs):
        b = self.dict[k0]
        if k is None:
            k = f'f{b.k}'
        if sym is None:
            sym = sub(b.sym, 'freq')
        if d is None:
            d = nam.freq(b.d)
        if p is None:
            p = nam.freq(b.p)
        if disp is None:
            disp = f'{b.disp} dominant frequency'
        self.addPar(
            **{'p': p, 'k': k, 'd': d, 'u': ureg.Hz, 'sym': sym, 'disp': disp, 'required_ks': [k0],
               'vfunc': param.Number, 'func': self.func_dict.freq(b.d)},
            **kwargs)

    def add_dsp(self, range=(0, 40), u=ureg.m):
        a = 'dispersion'
        k0 = 'dsp'
        s0 = circledast('d')
        r0, r1 = range
        dur = int(r1 - r0)
        p = f'{a}_{r0}_{r1}'
        k = f'{k0}_{r0}_{r1}'

        self.addPar(**{'p': p, 'k': k, 'u': u, 'sym': subsup(s0, f'{r0}', f'{r1}'), 'vfunc': param.Number,
                       'func': self.func_dict.dsp(range), 'required_ks': ['x', 'y'],
                       'lab': f"dispersal in {dur}''"})
        self.add_scaled(k0=k)
        self.add_operators(k0=k)
        self.add_operators(k0=f's{k}')

    def add_tor(self, dur):
        p0 = 'tortuosity'
        p = f'{p0}_{dur}'
        k0 = 'tor'
        k = f'{k0}{dur}'
        disp = f"{p0} over {dur}''"
        self.addPar(
            **{'p': p, 'k': k, 'd': p, 'lim': (0.0, 1.0), 'sym': sub(k0, dur), 'disp': disp, 'vfunc': param.Magnitude,
               'func': self.func_dict.tor(dur)})
        self.add_operators(k0=k)

    def build_angular(self, in_rad=True):
        if in_rad:
            u = ureg.rad
            amax = np.pi
        else:
            u = ureg.deg
            amax = 180
        kws = {'dv': np.round(amax / 180, 2), 'u': u, 'v0': 0.0, 'vfunc': param.Number}
        self.addPar(
            **{'p': 'bend', 'codename': 'bend', 'k': 'b', 'sym': th('b'), 'disp': 'bending angle', 'lim': (-amax, amax),
               **kws})
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
            pou = nam.unwrap(p0)
            p_v, p_a = nam.vel(p0), nam.acc(p0)
            ko = f'{suf}o'
            kou = f'{ko}u'
            self.addPar(**{'p': p0, 'k': ko, 'sym': th(ksuf), 'disp': f'{lsuf}orientation',
                           'lim': (0, 2 * amax), **kws})

            self.add_unwrap(k0=ko)

            self.add_velNacc(k0=kou, k_v=f'{suf}ov', k_a=f'{suf}oa', p_v=p_v, d_v=p_v, p_a=p_a, d_a=p_a,
                             sym_v=omega(ksuf),
                             sym_a=dot(omega(ksuf)), disp_v=f'{lsuf}angular speed',
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

        kws = {'u': u, 'vfunc': param.Number}
        self.addPar(**{'p': 'x', 'k': 'x', 'd': 'x', 'disp': 'X position', 'sym': 'X', **kws})
        self.addPar(**{'p': 'y', 'k': 'y', 'd': 'y', 'disp': 'Y position', 'sym': 'Y', **kws})
        self.addPar(
            **{'p': 'real_length', 'k': 'l', 'd': 'length', 'codename': 'real_length', 'disp': 'body length',
               'sym': '$l$',
               'v0': 0.004 * s,
               'lim': (0.0005 * s, 0.01 * s), 'dv': 0.0005 * s, **kws})

        self.addPar(
            **{'p': 'dispersion', 'k': 'dsp', 'd': 'dispersion', 'sym': circledast('d'), 'disp': 'dispersal', **kws})

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
        self.addPar(**{'p': 'tortuosity', 'k': 'tor', 'd': 'tortuosity', 'lim': (0.0, 1.0), 'sym': 'tor',
                       'vfunc': param.Magnitude})
        for dur in tor_durs:
            self.add_tor(dur=dur)
        self.addPar(**{'p': 'anemotaxis', 'k': 'anemotaxis', 'd': 'anemotaxis', 'sym': 'anemotaxis'})

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
        self.add_rate(k_num='Ltur_N', k_den='tur_N', k='tur_H', p='handedness_score', d='handedness_score',
                      sym=sub('H', 'tur'), lim=(0.0, 1.0), disp='Handedness score')
        self.addPar(**{'p': f'handedness_score_on_food', 'k': 'tur_H_on_food'})
        self.addPar(**{'p': f'handedness_score_off_food', 'k': 'tur_H_off_food'})

    def build_sim_pars(self):
        kws = {}
        self.addPar(
            **{'p': 'brain.locomotor.crawler.activity', 'k': 'Act_cr', 'd': 'crawler output', 'disp': 'crawler output',
               'sym': sub('A', 'C'), 'vfunc': param.Number, **kws})
        self.addPar(
            **{'p': 'brain.locomotor.turner.activity', 'k': 'Act_tur', 'd': 'turner output', 'disp': 'turner output',
               'sym': sub('A', 'T'), 'vfunc': param.Number, **kws})
        self.addPar(
            **{'p': 'brain.locomotor.turner.activation', 'k': 'A_tur', 'd': 'crawler input', 'disp': 'crawler input',
               'sym': sub('I', 'T'), 'vfunc': param.Number, **kws})
        self.addPar(**{'p': 'brain.locomotor.cur_ang_suppression', 'k': 'c_CT', 'd': 'ang_suppression',
                       'disp': 'angular suppression output', 'sym': sub('c', 'CT'), 'vfunc': param.Magnitude, **kws})
        self.addPar(**{'p': 'brain.olfactory_activation', 'k': 'A_olf', 'd': 'olfactory activation',
                       'disp': 'olfactory activation', 'sym': sub('A', 'olf'), 'vfunc': param.Number, **kws})
        self.addPar(
            **{'p': 'brain.touch_activation', 'k': 'A_touch', 'd': 'tactile activation', 'disp': 'tactile activation',
               'sym': sub('A', 'touch'), 'vfunc': param.Number, **kws})
        self.addPar(**{'p': 'brain.wind_activation', 'k': 'A_wind', 'd': 'wind activation', 'disp': 'wind activation',
                       'sym': sub('A', 'wind'), 'vfunc': param.Number, **kws})
        self.addPar(**{'p': 'brain.intermitter.EEB', 'k': 'EEB', 'd': 'exploitVSexplore_balance',
                       'disp': 'exploitVSexplore_balance', 'sym': 'EEB', 'vfunc': param.Magnitude, **kws})

        for ii,jj in (['1', '2'], ['first', 'second']) :
            k=f'c_odor{ii}'
            dk=f'd{k}'
            self.addPar(**{'p': f'brain.olfactor.{jj}_odor_concentration', 'k': k, 'd': k,
                           'disp': f'Odor {ii} concentration', 'sym': subsup('C', 'od',ii), 'vfunc': param.Number, **kws})
            self.addPar(**{'p': f'brain.olfactor.{jj}_odor_concentration_change', 'k': dk, 'd': dk,
                           'disp': f'Odor {ii} concentration change', 'sym': subsup(Delta('C'), 'od',ii), 'vfunc': param.Number, **kws})
