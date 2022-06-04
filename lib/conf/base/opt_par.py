import siunits as siu
import param
import numpy as np
from param import Parameterized, XYCoordinates, Selector, String, Boolean, Number, Path, Integer, List, Dict
from typing import TypedDict, List, Tuple
import lib.aux.naming as nam
from lib.aux.par_aux import bar, wave, sub, subsup, th, Delta, dot, circledcirc, circledast, odot, paren, brack, dot_th, \
    ddot_th, omega, ddot, mathring

from lib.aux.dictsNlists import AttrDict, load_dicts
from lib.conf.base.init_pars import init_pars, unit_dic

func_dic = {
    float: param.Number,
    int: param.Integer,
    str: param.String,
    bool: param.Boolean,
    dict: param.Dict,
    list: param.List,
    Tuple[float]: param.Range,
    Tuple[int]: param.NumericTuple,
}



def vpar(vfunc, v0, h, lab, lim, dv):
    # f = func_dic[dtype]
    f_kws = {
        'default': v0,
        'doc': h,
        'label': lab,
    }
    if vfunc in [param.List, param.Range] :
        if lim is not None:
            f_kws['bounds'] = lim
        if dv is not None:
            f_kws['step'] = dv
    func = vfunc(**f_kws, instantiate=True)
    return func




def buildPar(p, k, dtype=float, d=None, disp=None, sym=None,codename=None, lab=None, h=None, u_name=None,u=None, v0=None, lim=None, dv=None,
             vfunc=None,vparfunc=None,func=None,**kwargs):


    if u is None :
        u = unit_dic[u_name]
    if u_name is None :
        u_name = u.unit.abbrev
    codename = p if codename is None else codename
    d = p if d is None else d
    disp = d if disp is None else disp
    k = p if k is None else k
    sym = k if sym is None else sym
    lab = f'{disp} ({u_name})' if lab is None else lab
    h = lab if h is None else h
    if vparfunc is None :
        if vfunc is None :
            vfunc=func_dic[dtype]
        vparfunc = vpar(vfunc, v0, h, lab, lim, dv)
    #print(k, lab)
    class LarvaworldParNew(param.Parameterized):
        p = param.String('p')
        d = param.String('')
        disp = param.String('')
        k = param.String('k')
        sym = param.String('')
        codename = param.String('')
        u = param.Parameter('-')
        dtype = param.Parameter(float)
        v = vparfunc

        @property
        def unit(self):
            if self.u is None :
                return None
            else :
                return self.u.unit.abbrev

        @property
        def short(self):
            return self.k

        @ property
        def initial_value(self):
            return self.param.v.default

        @property
        def symbol(self):
            return self.sym

        @property
        def label(self):
            return self.param.v.label

        @property
        def tooltip(self):
            return self.param.v.doc

        @property
        def help(self):
            return self.param.v.doc

        @property
        def min(self):
            try:
                vmin,vmax = self.param.v.bounds
                return vmin
            except:
                return None

        @property
        def max(self):
            try:
                vmin, vmax = self.param.v.bounds
                return vmax
            except:
                return None

        @property
        def lim(self):
            try:
                lim = self.param.v.bounds
                return lim
            except:
                return None


        @property
        def step(self):
            try :
                step=self.param.v.step
                return step
            except:
                return None

        @property
        def get_ParsArg(self):
            from lib.anal.argparsers import build_ParsArg
            return build_ParsArg(name=self.name, k=self.k, h=self.help, t=self.dtype, v=self.initial_value, vs=None)

        # @property
        def exists(self, dataset) :
            s, e = dataset.step_data, dataset.endpoint_data
            dic={'step' : self.d in s.columns, 'end' : self.d in e.columns}
            return dic

        def get(self,dataset, key='step'):
            return dataset.get_par(key=key,par=self.d)


        def compute(self,dataset):
            res=self.exists(dataset)
            if not res['step'] and not res['end']:
                if self.func is not None :
                    self.func(dataset)
                    print(f'Parameter {self.disp} computed successfully')



    par = LarvaworldParNew(name=p, p=p, k=k, d=d,dtype=dtype, disp=disp, sym=sym, u=u, codename=codename)
    par.param.add_parameter('func' , param.Callable(default=func,doc='Function to get the parameter from a dataset', constant=True, allow_None=True))
    return par

class NewParDict :
    def __init__(self, mode='build', object=None, save=False):
        if mode=='load' :
            self.load()
        elif mode=='reconstruct' :
            self.reconstruct()
        elif mode=='build' :
            self.build(save=save, object=object)

    def addPar(self, **kwargs):
        p = buildPar(**kwargs)
        self.dict[p.k] = p

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

        b_num = self.dict[k_num] if k_num is not None else None
        b_den = self.dict[k_den] if k_den is not None else None
        if b_num is not None and b_den is not None:
            u = b_num.u / b_den.u
        else:
            u = 1 * siu.I
        self.addPar(**{'p': p, 'k': k, 'd': d, 'sym': sym, 'u': u, 'vfunc': param.Number}, **kwargs)

    def add_operators(self, k0):
        b = self.dict[k0]
        mu_kws = {'d': nam.mean(b.d), 'p': nam.mean(b.p), 'sym': bar(b.sym), 'disp': f'mean {b.disp}', 'u': b.u,
                  'k': f'{b.k}_mu'}
        std_kws = {'d': nam.std(b.d), 'p': nam.std(b.p), 'sym': wave(b.sym), 'disp': f'std {b.disp}', 'u': b.u,
                   'k': f'{b.k}_std'}
        min_kws = {'d': nam.min(b.d), 'p': nam.min(b.p), 'sym': sub(b.sym, 'min'), 'disp': f'minimum {b.disp}',
                   'u': b.u, 'k': f'{b.k}_min'}
        max_kws = {'d': nam.max(b.d), 'p': nam.max(b.p), 'sym': sub(b.sym, 'max'), 'disp': f'maximum {b.disp}',
                   'u': b.u, 'k': f'{b.k}_max'}
        fin_kws = {'d': nam.final(b.d), 'p': nam.final(b.p), 'sym': sub(b.sym, 'fin'), 'disp': f'final {b.disp}',
                   'u': b.u, 'k': f'{b.k}_fin'}
        cum_kws = {'d': nam.cum(b.d), 'p': nam.cum(b.p), 'sym': sub(b.sym, 'cum'), 'disp': f'total {b.disp}', 'u': b.u,
                   'k': nam.cum(b.k)}

        for kws in [mu_kws, std_kws, min_kws, max_kws, fin_kws, cum_kws]:
            kws['vfunc']=param.Number
            self.addPar(**kws)

    def add_chunk(self, pc, kc):
        p0, p1, pt, pid, ptr, pN, pl = nam.start(pc), nam.stop(pc), nam.dur(pc), nam.id(pc), nam.dur_ratio(pc), nam.num(pc),nam.length(pc)
        pN_mu=nam.mean(pN)

        k0,k1,kid,ktr,kN,kt, kl=f'{kc}0',f'{kc}1',f'{kc}_id',f'{kc}_tr',f'{kc}_N',f'{kc}_t',f'{kc}_l'
        kN_mu=f'{kN}_mu'

        self.addPar(**{'p': pc, 'k': kc, 'sym': f'${kc}$', 'disp': pc})
        self.addPar(
            **{'p': p0, 'k': k0, 'u_name': 'sec', 'sym': subsup('t', kc, 0), 'disp': f'{pc} start', 'vfunc': param.Number})
        self.addPar(
            **{'p': p1, 'k': k1,'u_name': 'sec', 'sym': subsup('t', kc, 1), 'disp': f'{pc} end', 'vfunc': param.Number})
        self.addPar(**{'p': pid, 'k': kid, 'u_name': None, 'sym': sub('idx', kc),'dtype': str, 'disp': f'{pc} idx', 'vfunc': param.String})
        self.addPar(
            **{'p': ptr, 'k': ktr, 'u_name': None, 'sym': sub('r', kc), 'disp': f'% time in {pc}s', 'vfunc': param.Magnitude})
        self.addPar(**{'p': pN, 'k': kN, 'u_name': None, 'sym': sub('N', f'{pc}s'),'dtype': int, 'disp': f'# {pc}s', 'vfunc': param.Integer})


        for ii in ['on', 'off'] :
            self.addPar(**{'p': f'{pN_mu}_{ii}_food', 'k': f'{kN_mu}_{ii}_food', 'vfunc': param.Number})
            self.addPar(**{'p': f'{ptr}_{ii}_food', 'k': f'{ktr}_{ii}_food', 'vfunc': param.Magnitude})

        self.add_rate(k_num=kN, k_den=nam.cum('t'), k=kN_mu, p=pN_mu, d=pN_mu,sym=bar(kN), disp=f' mean # {pc}s/sec')
        s00 = Delta('t')
        self.addPar(**{'p': pt, 'k': kt, 'u_name': 'sec', 'sym': sub(s00, kc), 'disp': f'{pc} duration', 'vfunc': param.Number})
        self.add_operators(k0=kt)

        if str.endswith(pc, 'chain'):
            self.addPar(**{'p': pl, 'k': kl, 'sym': sub('l', kc),'dtype': int, 'vfunc': param.Integer})
            self.add_operators(k0=kl)

    def add_chunk_track(self, kc, k, extrema=True):
        bc = self.dict[kc]
        b = self.dict[k]
        b0, b1 = self.dict[f'{kc}0'], self.dict[f'{kc}1']
        p0, p1 = nam.at(b.p, b0.p), nam.at(b.p, b1.p)

        k00 = f'{kc}_{k}'
        disp = f'{b.disp} during {bc.p}s'
        self.addPar(
            **{'p': nam.chunk_track(bc.p, b.p), 'k': k00, 'u': b.u, 'sym': sub(Delta(b.sym), kc),
               'disp': disp, 'vfunc': param.Number})
        self.add_operators(k0=k00)

        if extrema:
            self.addPar(**{'p': p0, 'k': f'{kc}_{k}0', 'u': b.u, 'sym': subsup(b.sym, kc, 0), 'vfunc': param.Number})
            self.addPar(**{'p': p1, 'k': f'{kc}_{k}1', 'u': b.u, 'sym': subsup(b.sym, kc, 1), 'vfunc': param.Number})

    def add_diff(self, k0, **kwargs):
        b = self.dict[k0]
        k = f'D_{k0}'
        self.addPar(
            **{'p': f'D_{b.p}', 'k': k, 'd': f'{b.d} change', 'u': b.u, 'sym': Delta(b.sym), 'lim': b.param.v.bounds,
               'dv': b.param.v.step, 'dtype': b.dtype, 'vfunc': param.Number}, **kwargs)

    def add_velNacc(self, k0, p_v=None, k_v=None, d_v=None, sym_v=None, disp_v=None, p_a=None, k_a=None, d_a=None,
                    sym_a=None, disp_a=None):
        # self.add_diff(k0)
        b = self.dict[k0]
        b_dt = self.dict['dt']
        u_v = b.u / b_dt.u
        u_a = u_v / b_dt.u
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
        self.addPar(**{'p': p_v, 'k': k_v, 'd': d_v, 'u': u_v, 'sym': sym_v, 'disp': disp_v,'vfunc': param.Number})
        # self.add_diff(k_v)
        self.addPar(**{'p': p_a, 'k': k_a, 'd': d_a, 'u': u_a, 'sym': sym_a, 'disp': disp_a,'vfunc': param.Number})

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

        def func(d) :
            from lib.process.spatial import scale_to_length
            s, e, c = d.step_data, d.endpoint_data, d.config
            scale_to_length(s, e, c, pars=[b.d], keys=None)

        self.addPar(**{'p': p, 'k': k, 'd': d, 'u': u, 'sym': sym, 'disp': disp,'vfunc': param.Number, 'func':func}, **kwargs)

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

        from lib.process.aux import get_freq
        def func(d) :
            get_freq(d, par=b.d, fr_range=(0.0, +np.inf))

        # func= {'func':get_freq, 'kws' : {'par' : b.d, 'fr_range' : (0.0, +np.inf)}}
        self.addPar(**{'p': p, 'k': k, 'd': d, 'u': 1 * siu.hz, 'sym': sym, 'disp': disp,'vfunc': param.Number, 'func': func}, **kwargs)

    def add_dsp(self, range=(0, 40)):
        a = 'dispersion'
        k0 = 'dsp'
        s0 = circledast('d')
        r0, r1 = range
        dur = int(r1 - r0)
        p = f'{a}_{r0}_{r1}'
        k = f'{k0}_{r0}_{r1}'
        self.addPar(**{'p': p, 'k': k, 'u_name': 'm', 'sym': subsup(s0, f'{r0}', f'{r1}'),'vfunc': param.Number,
                       'lab': f"dispersal in {dur}''"})
        self.add_scaled(k0=k)
        self.add_operators(k0=k)
        self.add_operators(k0=f's{k}')

    def build_initial(self):
        kws = {'u_name': 'sec', 'vfunc': param.Number}
        self.addPar(**{'p': 'model.dt', 'k': 'dt', 'd': 'timestep', 'sym': '$dt$', 'codename': 'model.dt',
                       'lim': (0.01, 0.5), 'dv': 0.01, 'v0': 0.1,**kws})
        self.addPar(**{'p': 'cum_dur', 'k': nam.cum('t'), 'd': nam.cum('dur'), 'sym': sub('t', 'cum'), 'lim': (0.0, None),
                 'dv': 0.1, 'v0': 0.0, **kws})

        kws = {'u_name': 'm', 'vfunc': param.Number}
        self.addPar(**{'p': 'x0', 'k': 'x0','d': 'x0', 'disp': 'initial X position', 'sym': sub('X', 0), 'v0': 0.0, **kws})
        self.addPar(**{'p': 'y0', 'k': 'y0','d': 'y0', 'disp': 'initial Y position', 'sym': sub('Y', 0), 'v0': 0.0, **kws})
        self.addPar(**{'p': 'x', 'k': 'x', 'd': 'x','disp': 'X position', 'sym': 'X', **kws})
        self.addPar(** {'p': 'y', 'k': 'y', 'd': 'y','disp': 'Y position', 'sym': 'Y', **kws})
        self.addPar(**{'p': 'real_length', 'k': 'l', 'd': 'length','codename': 'length', 'disp': 'body length', 'sym': '$l$', 'v0': 0.004,
                 'lim': (0.0005, 0.01), 'dv': 0.0005, **kws})
        self.addPar(**{'p': nam.dst(''), 'k': 'd', 'sym': 'd', 'disp': 'distance', **kws})

        kws = {'dv': np.round(np.pi / 180, 2), 'u_name': 'rad', 'v0': 0.0, 'vfunc': param.Number}
        for ii1, ii2 in zip(['front', 'rear', 'head', 'tail'], ['f', 'r', 'h', 't']) :
            self.addPar(**{'p': nam.orient(ii1), 'k': f'{ii2}o', 'sym': th(ii2), 'disp': f'{ii1} orientation',
             'lim': (0, 2 * np.pi), **kws})
            self.addPar(**{'p': nam.unwrap(nam.orient(ii1)), 'k': f'{ii2}ou', 'sym': th(ii2), 'disp': f'{ii1} orientation',
             'lim': None, **kws})

        self.addPar(**{'p': 'bend', 'k': 'b', 'sym': th('b'), 'disp': 'bending angle', 'lim': (-np.pi, np.pi), **kws})

    def build(self, save=True, object=None):
        siu.day = siu.s * 24 * 60 * 60
        siu.cm = siu.m * 10 ** -2
        siu.mm = siu.m * 10 ** -3
        siu.g = siu.kg * 10 ** -3
        siu.deg = siu.I.rename("deg", "deg", "plain angle")
        siu.microM = siu.mol * 10 ** -6
        self.dict=AttrDict.from_nested_dicts({})
        self.build_initial()
        self.build_basic()
        self.build_chunks()

        if save:
            self.save()

    def save(self, save_pdf=False):
        pass


    def load(self):
        pass
        # self.dict = load_dicts([paths.path('ParDict')])[0]

    def reconstruct(self):
        # frame = load_dicts([paths.path('ParDict')])[0]
        # for k, args in frame.items():
        #     self.dict[k] = Parameter(**args, par_dict=self.dict)
        # for k, p in self.dict.items():
        #     p.par_dict = self.dict
        pass


    def build_basic(self):
        # for k0 in ['fo', 'ro', 'x', 'y'] :
        #     self.add_diff(k0)

        bv, fov, rov,hov, tov = nam.vel('bend'), nam.vel(nam.orient('front')), nam.vel(nam.orient('rear')),nam.vel(nam.orient('head')), nam.vel(nam.orient('tail'))
        ba, foa, roa,hoa, toa = nam.acc('bend'), nam.acc(nam.orient('front')), nam.acc(nam.orient('rear')),nam.acc(nam.orient('head')), nam.acc(nam.orient('tail'))
        self.add_velNacc(k0='b',sym_v=omega('b'),sym_a=dot(omega('b')), disp_v='bending angular velocity', disp_a='bending angular acceleration')
        self.add_velNacc(k0='fou',k_v='fov',k_a='foa',p_v=fov,d_v=fov,p_a=foa,d_a=foa, sym_v=omega(''),sym_a=dot(omega('')), disp_v='angular velocity', disp_a='angular acceleration')
        self.add_velNacc(k0='rou',k_v='rov',k_a='roa',p_v=rov,d_v=rov,p_a=roa,d_a=roa, sym_v=omega('r'),sym_a=dot(omega('r')), disp_v='rear angular velocity', disp_a='rear angular acceleration')
        self.add_velNacc(k0='hou',k_v='hov',k_a='hoa',p_v=hov,d_v=hov,p_a=hoa,d_a=hoa, sym_v=omega('h'),sym_a=dot(omega('h')), disp_v='head angular velocity', disp_a='head angular acceleration')
        self.add_velNacc(k0='tou',k_v='tov',k_a='toa',p_v=tov,d_v=tov,p_a=toa,d_a=toa, sym_v=omega('t'),sym_a=dot(omega('t')), disp_v='tsil angular velocity', disp_a='tail angular acceleration')
        self.add_velNacc(k0='d',k_v='v',k_a='a',p_v=nam.vel(''),d_v=nam.vel(''),p_a=nam.acc(''),d_a=nam.acc(''), sym_v='v',sym_a=dot('v'), disp_v='velocity', disp_a='acceleration')
        self.add_scaled(k0='d')
        sv,sa=nam.scal(nam.vel('')),nam.scal(nam.acc(''))
        self.add_velNacc(k0='sd',k_v='sv',k_a='sa', p_v=sv,d_v=sv,p_a=sa,d_a=sa,sym_v=mathring('v'),sym_a=dot(mathring('v')),disp_v='scaled velocity', disp_a='scaled acceleration')
        for k0 in ['l','d', 'sd', 'v', 'sv','a', 'sa','b', 'bv', 'ba','fov', 'foa','rov', 'roa']:
            self.add_freq(k0=k0)
            self.add_operators(k0=k0)
        self.addPar(**{'p': 'dispersion', 'k': 'dsp', 'd': 'dispersion', 'u': 1 * siu.m, 'sym': circledast('d'), 'disp': 'dispersal'})
        for i in [(0, 40), (0, 60), (20, 80), (0, 120), (0, 240), (0, 300), (0, 600), (60, 120), (60, 300)]:
            self.add_dsp(range=i)

        for i in ['', 1, 2, 5, 10, 20, 60, 120, 240, 300, 600]:
            p0 = 'tortuosity'
            if i == '':
                p = p0
            else:
                p = f'{p0}_{i}'
            k0 = 'tor'
            k = f'{k0}{i}'
            disp = f"{p0} over {i}''"
            self.addPar(**{'p': p, 'k': k, 'd': p, 'u_name': None,'lim': (0.0,1.0), 'sym': sub(k0, i),'disp': disp, 'vfunc': param.Magnitude})
            self.add_operators(k0=k)
        self.addPar(**{'p': 'anemotaxis', 'k': 'anemotaxis', 'd': 'anemotaxis', 'u_name': 'm', 'sym': 'anemotaxis'})

    def build_chunks(self):
        chunk_dict = {
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
        for kc, pc in chunk_dict.items():
            self.add_chunk(pc=pc, kc=kc)
            for k in ['fov', 'rov', 'foa', 'roa', 'x', 'y', 'fo', 'fou', 'ro', 'rou', 'b', 'bv', 'ba', 'v', 'sv', 'a', 'sa']:
                self.add_chunk_track(kc=kc, k=k)
            if pc in ['stride', 'run']:
                for kk in ['d']:
                    b=self.dict[kk]
                    k=f'{kc}_{kk}'
                    self.addPar(
                        **{'p': nam.chunk_track(pc, b.p), 'k': k, 'u': b.u, 'sym': sub(Delta(b.sym), kc), 'disp' : f'{pc} distance'})

                    self.add_scaled(k0=k)
                    self.add_operators(k0=k)
                    self.add_operators(k0=f's{k}')
        self.add_rate(k_num='Ltur_N', k_den='tur_N', k='tur_H', p='handedness_score', d='handedness_score',
                      sym=sub('H', 'tur'), lim=(0.0, 1.0), disp='Handedness score')
        self.addPar(**{'p':f'handedness_score_on_food', 'k':'tur_H_on_food'})
        self.addPar(**{'p':f'handedness_score_off_food', 'k':'tur_H_off_food'})

newParDict=NewParDict().dict

def base_dtype(dtype):
    if dtype in [float, Tuple[float], List[float], List[Tuple[float]]]:
        base_t = float
    elif dtype in [int, Tuple[int], List[int], List[Tuple[int]]]:
        base_t = int
    else:
        base_t = dtype
    return base_t

def init2opt_dict(name):
    init_dic = init_pars()[name]
    opt_dict = {}
    for k, v in init_dic.items():
        if 't' not in v.keys():
            v['t'] = float
    for k, v in init_dic.items():
        dtype = v['t']
        dtype0 = base_dtype(dtype)
        func = func_dic[dtype]
        kws = {
            'doc': v['h'],
            'default': v['v'] if 'v' in v.keys() else None,
            'label': v['lab'] if 'lab' in v.keys() else k,

        }
        if dtype0 in [float, int]:
            b0 = v['min'] if 'min' in v.keys() else dtype0(0.0)
            b1 = v['max'] if 'max' in v.keys() else None
            bounds = (b0, b1)
            step = v['dv'] if 'dv' in v.keys() else 0.1
            kws.update({
                'step': step,
                'bounds': bounds,
            })

        opt_dict[k] = func(**kws)
    return opt_dict

if __name__ == '__main__':
    # print(newParDict.b)
    #
    #
    # raise


    for k,p in newParDict.items() :
        print(k,p.min, p.lim, p.step)


        pass

    raise

