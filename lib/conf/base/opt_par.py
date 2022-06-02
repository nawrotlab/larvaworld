import siunits as siu
import param
import numpy as np
from param import Parameterized, XYCoordinates, Selector, String, Boolean, Number, Path, Integer, List, Dict
from param.parameterized import get_all_slots, get_occupied_slots
from typing import TypedDict, List, Tuple

from lib.aux.dictsNlists import AttrDict
from lib.aux.par_aux import sub, sup, bar, circle, wave, tilde, subsup
from lib.conf.stored.conf import expandConf, saveConf, copyConf, kConfDict, deleteConf
from lib.conf.base.init_pars import init_pars
from lib.conf.base.par import ParDict, getPar

siu.day = siu.s * 24 * 60 * 60
siu.cm = siu.m * 10 ** -2
siu.mm = siu.m * 10 ** -3
siu.g = siu.kg * 10 ** -3
siu.deg = siu.I.rename("deg", "deg", "plain angle")
siu.microM = siu.mol * 10 ** -6


def base_dtype(t):
    if t in [float, Tuple[float], List[float], List[Tuple[float]]]:
        base_t = float
    elif t in [int, Tuple[int], List[int], List[Tuple[int]]]:
        base_t = int
    else:
        base_t = t
    return base_t


def maxNdigits(array, Min=None):
    N = len(max(array.astype(str), key=len))
    if Min is not None:
        N = max([N, Min])
    return N


def define_dv(dv, cur_dtype):
    if dv is None:
        if cur_dtype == int:
            dv = 1
        elif cur_dtype == float:
            dv = 0.1
    return dv


def define_lim(lim, vs, min, max, u, wrap_mode, cur_dtype):
    if lim is not None:
        return lim
    if u.unit == siu.deg:
        if wrap_mode == 'positive':
            lim = (0.0, 360.0)
        elif wrap_mode == 'zero':
            lim = (-180.0, 180.0)
    elif u.unit == siu.rad:
        if wrap_mode == 'positive':
            lim = (0.0, 2 * np.pi)
        elif wrap_mode == 'zero':
            lim = (-np.pi, np.pi)
    else:
        if cur_dtype in [float, int]:
            if vs is not None:
                lim = (np.min(vs), np.max(vs))
            else:
                if min is None:
                    min = 0
                if max is None:
                    max = 1
                lim = (min, max)
    return lim


def define_vs(vs, dv, lim, cur_dtype):
    if vs is not None:
        return vs
    if dv is not None and lim is not None:
        ar = np.arange(lim[0], lim[1] + dv, dv)
        if cur_dtype == float:
            Ndec = len(str(format(dv, 'f')).split('.')[1])
            ar = np.round(ar, Ndec)
        vs = ar.astype(cur_dtype)

        vs = vs.tolist()
    return vs


def define_range(dtype, lim, vs, dv, min, max, u, wrap_mode):
    cur_dtype = base_dtype(dtype)
    dv = define_dv(dv, cur_dtype)
    lim = define_lim(lim, vs, min, max, u, wrap_mode, cur_dtype)
    vs = define_vs(vs, dv, lim, cur_dtype)
    return dv, lim, vs


func_dic = {
    float: param.Number,
    int: param.Integer,
    str: param.String,
    bool: param.Boolean,
    dict: param.Dict,
    list: param.List,
    Tuple[float]: param.NumericTuple,
    Tuple[int]: param.NumericTuple,
}


class Parameter:
    def __init__(self, p, u=1 * siu.I, t=float, k=None, disp=None, s=None,sym=None, lim=None, v=None, vs=None, min=None, max=None, dv=None,
                 h='',d=None, lab=None, combo=None, wrap_mode=None, codename=None):
        self.p = p
        self.d = self.p if d is None else d
        self.disp = self.d if disp is None else disp
        self.k = self.p if k is None else k
        self.sym = self.k if sym is None else sym

        self.s = s
        self.h = h
        self.dtype = t
        self.v = v
        self.u = u
        self.codename = codename
        self.combo = combo

        self.dv, self.lim, self.vs = define_range(self.dtype, lim, vs, dv, min, max, self.u, wrap_mode)
        # self.symbol = symbol
        self.lab = f'{self.disp} ({self.unit})' if lab is None else lab

    @property
    def unit(self):
        try:
            u = self.u.unit.abbrev
        except:
            u = '-'
        return u

    @property
    def vpar(self):
        f = func_dic[self.dtype]
        f_kws = {
            'default': self.v,
            'doc': self.h,
            'label': self.lab,

        }
        if self.lim is not None:
            f_kws['bounds'] = self.lim
        if self.dv is not None:
            f_kws['step'] = self.dv

        func = f(**f_kws)

        return func


class OptPar(Parameterized):
    def __init__(self, dic, **kwargs):
        par = Parameter(**dic)
        str_keys = ['k', 'p', 'd','sym', 'codename', 's']
        super().__init__(name=par.p)
        for k in str_keys:
            self.param.add_parameter(k, param.String(getattr(par, k)))
        self.param.add_parameter('v', par.vpar)
        self.param['v']._internal_name = par.k
        for k, v in kwargs.items():
            self.__setattr__(k, v)


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


class OptParDict(Parameterized):
    def __init__(self, name, **kwargs):
        super().__init__(name=name)
        opt_dict = {name: init2opt_dict(name)}
        for k, v in opt_dict[name].items():
            self.param.add_parameter(k, v)
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    @property
    def dict(self):
        dic = self.param.values()
        dic.pop('name', None)
        return AttrDict.from_nested_dicts(dic)

    @property
    def entry(self):
        return AttrDict.from_nested_dicts({self.name : self.dict})


class SimParConf(OptParDict):
    def __init__(self, exp=None, conf_type='Exp', sim_ID=None, path=None,duration=None, **kwargs):
        if exp is not None and conf_type is not None:
            from lib.conf.stored.conf import loadConf, next_idx
            if duration is None:
                try :
                    exp_conf = loadConf(exp, conf_type)
                    duration = exp_conf.sim_params.duration
                except :
                    duration = 3.0
            if sim_ID is None:
                sim_ID = f'{exp}_{next_idx(exp, conf_type)}'
            if path is None:
                if conf_type == 'Exp':
                    path = f'single_runs/{exp}'
                elif conf_type == 'Ga':
                    path = f'ga_runs/{exp}'
                elif conf_type == 'Batch':
                    path = f'batch_runs/{exp}'
                elif conf_type == 'Eval':
                    path = f'eval_runs/{exp}'
        super().__init__(name='sim_params', sim_ID=sim_ID, path=path,duration=duration, **kwargs)


if __name__ == '__main__':
    conf = SimParConf(exp='dish').entry

    print(conf.sim_params.path)

    raise
    ddd = {}
    ddd['body'] = {
        'initial_length': {'v': 0.004, 'lim': (0.0, 0.01), 'dv': 0.0001, 'p': 'real_length', 'k': 'l', 'u': 1 * siu.m,
                           'd': 'length', 'sym': '$l$', 'disp': 'body length',
                           'combo': 'length', 'h': 'The initial body length.'},
        'length_std': {'v': 0.0, 'max': 0.001, 'dv': 0.0001, 'aux_vs': ['sample'], 'disp': 'std',
                       'combo': 'length', 'h': 'The standard deviation of the initial body length.'},
        'Nsegs': {'t': int, 'v': 2, 'min': 1, 'max': 12, 'label': 'number of body segments', 'symbol': '-',
                  'u': '# $segments$',
                  'h': 'The number of segments comprising the larva body.'},
        'seg_ratio': {'max': 1.0,
                      'h': 'The length ratio of the body segments. If null, equal-length segments are generated.'},
        # [5 / 11, 6 / 11]
        'touch_sensors': {'t': int, 'min': 0, 'max': 8,
                          'h': 'The number of touch sensors existing on the larva body.'},
        'shape': {'t': str, 'v': 'drosophila_larva', 'vs': ['drosophila_larva', 'zebrafish_larva'],
                  'h': 'The body shape.'},
        }

    ddd_l = ddd['body']['initial_length']

    ppp_l = OptPar(dic=ddd_l)

    print(ppp_l.sym)