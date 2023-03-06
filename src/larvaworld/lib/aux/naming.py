import itertools

import numpy as np
from larvaworld.lib import aux

def join(s, p, loc, c='_'):
    if loc == 'suf':
        return f'{p}{c}{s}'
    elif loc == 'pref':
        return f'{s}{c}{p}'


def name(s, ps, loc='suf', c='_'):
    if type(ps) == str:
        if ps == '':
            return s
        else:
            return join(s, ps, loc, c)
    elif type(ps) == list:
        return [join(s, p, loc, c) if p != '' else s for p in ps]



class NamingRegistry(aux.AttrDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ks=['mean', 'std', 'var', 'min', 'max', 'final', 'initial', 'cum', 'freq', 'lin', 'scal', 'abs', 'non',
                  'chain','dur', 'unwrap', 'dst', 'dst_to', 'bearing_to', 'vel', 'acc', 'orient', 'scal']
        # for k in  :
        #     self[k]=self.get_func(k)


    def get_kws(self, k):
        loc_pref = ['final', 'initial', 'cum', 'lin', 'scal', 'abs','dst_to', 'bearing_to', 'non']

        noseparator = ['chain']

        key_pairs = {
            'vel': 'velocity',
            'acc': 'acceleration',
            'scal': 'scaled',
            'orient': 'orientation',
            'unwrap': 'unwrapped',
            # 'scal': 'scaled',
        }

        kws = {}
        if k in loc_pref:
            kws['loc'] = 'pref'
        if k in key_pairs.keys():
            kws['s'] = key_pairs[k]
        else:
            kws['s'] = k
        if k in noseparator:
            kws['c'] = ''
        return kws


    def get_func(self, k):
        kws=self.get_kws(k)
        def func(ps, **kwargs):
            kws.update(kwargs)
            return name(ps=ps, **kws)
        return func

    def __getattr__(self, item):

        # raise ValueError(item, 'attr')
        return self[item]

    def __getitem__(self, k):
        return self.get_func(k)
        # # raise ValueError(k, 'item')
        # if not k in self.keys() :
        #     print(f'{k} does not exist')
        #     self[k]=self.get_func(k)
        # return self.get_func(k)

    def num(self,chunk):
        s = 'num'
        temp = name(s, chunk, 'pref')
        return name('s', temp, 'suf', c='')

    def xy(self,points, flat=False):
        if type(points) == str:
            if points == '':
                return ['x', 'y']
            else:
                return [f'{points}_x', f'{points}_y']
        elif type(points) == list:
            t = [[f'{p}_x', f'{p}_y'] if p != '' else ['x', 'y'] for p in points]
            return [item for sublist in t for item in sublist] if flat else t

    def chunk_track(self, chunk_name, params):
        return self[chunk_name](params, loc='pref')

    def contour(self, Nc):
        return [f'contour{i}' for i in range(Nc)]

    def midline(self, N, type='point'):
        if N >= 2:
            points = ['head'] + [f'{type}{i}' for i in np.arange(2, N, 1)] + ['tail']
        elif N == 1:
            points = ['body']
        else:
            points = []
        return points

    def contour_xy(self, Nc, flat=False):
        return self.xy(self.contour(Nc), flat=flat)

    def midline_xy(self,N, flat=False):
        return self.xy(self.midline(N), flat=flat)

    def at(self, p, t):
        return self[f'{p}_at'](t, loc='pref')
        # return f'{p}_at_{t}'


nam=NamingRegistry()

def h5_kdic(p, N, Nc):
    def epochs_ps():
        cs = ['turn', 'Lturn', 'Rturn', 'pause', 'exec', 'stride', 'stridechain']
        pars = ['id', 'start', 'stop', 'dur', 'dst', nam.scal('dst'), 'length', nam.max('vel'), 'count']
        pars = aux.flatten_list([nam.chunk_track(c, pars) for c in cs])
        return pars

    def dspNtor_ps():
        tor_ps = [f'tortuosity_{dur}' for dur in [1, 2, 5, 10, 20, 30, 60, 100, 120, 240, 300]]
        dsp_ps = [f'dispersion_{t0}_{t1}' for (t0, t1) in
                  itertools.product([0, 5, 10, 20, 30, 60], [30, 40, 60, 90, 120, 240, 300])]
        pars = tor_ps + dsp_ps + nam.scal(dsp_ps)
        return pars

    def base_spatial_ps(p=''):
        d, v, a = ps = [nam.dst(p), nam.vel(p), nam.acc(p)]
        ld, lv, la = lps = nam.lin(ps)
        ps0 = nam.xy(p) + ps + lps + nam.cum([d, ld])
        return ps0 + nam.scal(ps0)

    def ang_pars(angs):
        avels = nam.vel(angs)
        aaccs = nam.acc(angs)
        uangs = nam.unwrap(angs)
        avels_min, avels_max = nam.min(avels), nam.max(avels)
        return avels + aaccs + uangs + avels_min + avels_max

    def angular(N):
        Nangles = np.clip(N - 2, a_min=0, a_max=None)
        Nsegs = np.clip(N - 1, a_min=0, a_max=None)
        ors = nam.orient(aux.unique_list(['front', 'rear', 'head', 'tail'] + nam.midline(Nsegs, type='seg')))
        ang = ors + [f'angle{i}' for i in range(Nangles)] + ['bend']
        return aux.unique_list(ang + ang_pars(ang))

    dic = aux.AttrDict({
        'contour': nam.contour_xy(Nc, flat=True),
        'midline': nam.midline_xy(N, flat=True),
        'epochs': epochs_ps(),
        'base_spatial': base_spatial_ps(p),
        'angular': angular(N),
        'dspNtor': dspNtor_ps(),
    })
    return dic


