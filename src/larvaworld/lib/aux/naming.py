import itertools

import numpy as np
from larvaworld.lib import aux

def join(s, p, loc, c='_'):
    if loc == 'suf':
        return f'{p}{c}{s}'
    elif loc == 'pref':
        return f'{s}{c}{p}'


def name(s, ps, loc='suf', c='_'):
    if isinstance(ps,str):
        if ps == '':
            return s
        else:
            return join(s, ps, loc, c)
    elif isinstance(ps,list):
        return aux.SuperList([join(s, p, loc, c) if p != '' else s for p in ps])



class NamingRegistry(aux.AttrDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ks=['mean', 'std', 'var', 'min', 'max', 'final', 'initial', 'cum', 'freq', 'lin', 'scal', 'abs', 'non',
                  'chain','dur', 'unwrap', 'dst', 'dst_to', 'bearing_to', 'vel', 'acc', 'orient', 'scal']

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
        return self[item]

    def __getitem__(self, k):
        return self.get_func(k)

    def num(self,chunk):
        s = 'num'
        temp = name(s, chunk, 'pref')
        return name('s', temp, 'suf', c='')

    def xy(self,points, flat=False, xsNys=False):
        if type(points) == str:
            if points == '':
                t= ['x', 'y']
            else:
                t= [f'{points}_x', f'{points}_y']

        elif type(points) == list:
            t = [self.xy(p) for p in points]
            if xsNys:
                t=[np.array(t)[:,i].tolist() for i in [0,1]]
            if flat :
                t=[item for sublist in t for item in sublist]
        return aux.SuperList(t)

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

    def atStartStopChunk(self,p,chunk):
        return [
            self.at(p, self.start(chunk)),
            self.at(p, self.stop(chunk)),
            self.chunk_track(chunk, p)
        ]



nam=NamingRegistry()
