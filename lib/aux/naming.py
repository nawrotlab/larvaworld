import itertools
import numpy as np




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


def xy(points, flat=False):
    if type(points) == str:
        if points == '':
            return ['x', 'y']
        else:
            return [f'{points}_x', f'{points}_y']
    elif type(points) == list:
        t = [[f'{p}_x', f'{p}_y'] if p != '' else ['x', 'y'] for p in points]
        return [item for sublist in t for item in sublist] if flat else t


def dst(points):
    s = 'dst'
    if points == '':
        return s
    else:
        return name(s, points, 'suf')


def dst2(points, **kwargs):
    s = 'dst_to'
    return name(s, points, 'pref', **kwargs)


def bearing2(points, **kwargs):
    s = 'bearing_to'
    return name(s, points, 'pref', **kwargs)


def straight_dst(points):
    s = dst('straight')
    return name(s, points, 'suf')


def vel(params):
    s = 'velocity'
    if params == '':
        return s
    else:
        return name(s, params, 'suf')


def acc(params):
    s = 'acceleration'
    if params == '':
        return s
    else:
        return name(s, params, 'suf')


def lin(params):
    s = 'lin'
    return name(s, params, 'pref')


def scal(params):
    s = 'scaled'
    return name(s, params, 'pref')


def abs(params):
    s = 'abs'
    return name(s, params, 'pref')


def final(params):
    s = 'final'
    return name(s, params, 'pref')


def initial(params):
    s = 'initial'
    return name(s, params, 'pref')


def cum(params):
    s = 'cum'
    return name(s, params, 'pref')


def filt(params):
    s = 'filt'
    return name(s, params, 'suf')


def min(params):
    s = 'min'
    return name(s, params, 'suf')


def orient(segs):
    s = 'orientation'
    return name(s, segs, 'suf')


def unwrap(segs):
    s = 'unwrapped'
    return name(s, segs, 'suf')


def max(params):
    s = 'max'
    return name(s, params, 'suf')


def freq(params):
    s = 'freq'
    return name(s, params, 'suf')


def start(chunk):
    s = 'start'
    return name(s, chunk, 'suf')


def stop(chunk):
    s = 'stop'
    return name(s, chunk, 'suf')


def dur(chunk):
    s = 'dur'
    return name(s, chunk, 'suf')


def dur_ratio(chunk):
    s = 'dur_ratio'
    return name(s, chunk, 'suf')


def num(chunk):
    s = 'num'
    temp = name(s, chunk, 'pref')
    return name('s', temp, 'suf', c='')


def contact(chunk):
    s = 'contact'
    return name(s, chunk, 'suf')


def id(chunk):
    s = 'id'
    return name(s, chunk, 'suf')


def mean(chunk):
    s = 'mean'
    return name(s, chunk, 'suf')


def std(chunk):
    s = 'std'
    return name(s, chunk, 'suf')


def var(chunk):
    s = 'var'
    return name(s, chunk, 'suf')


def non(chunk):
    s = 'non'
    return name(s, chunk, 'pref')


def length(chunk):
    s = 'length'
    return name(s, chunk, 'suf')


def chain(chunk):
    s = 'chain'
    return name(s, chunk, 'suf', c='')


def overlap_ratio(base_chunk, overlapping_chunk):
    return f'{base_chunk}_{overlapping_chunk}_overlap'


def chunk_track(chunk_name, params):
    s = chunk_name
    return name(s, params, 'pref')


def contour(Nc):
    contour = [f'contour{i}' for i in range(Nc)]
    return contour


def midline(N, type='point'):
    if N >= 2:
        points = ['head'] + [f'{type}{i}' for i in np.arange(2, N, 1)] + ['tail']
    elif N == 1:
        points = ['body']
    else:
        points = []
    return points


def at(p, t):
    s = f'{p}_at'
    return name(s, t, 'pref')
    # return f'{p}_at_{t}'


def base_spatial_ps(p=''):
    d, v, a = ps = [dst(p), vel(p), acc(p)]
    ld, lv, la = lps = lin(ps)
    ps0 = xy(p) + ps + lps + cum([d, ld])
    return ps0 + scal(ps0)


def epoch_ps(c):
    pars = ['id', 'start', 'stop', 'dur', 'dst', scal('dst'), 'length', max('vel'), 'count']
    return chunk_track(c, pars)


def epochs_ps(cs=['turn', 'Lturn', 'Rturn', 'pause', 'exec', 'stride', 'stridechain']):
    from lib.aux import dictsNlists as dNl
    cs = ['turn', 'Lturn', 'Rturn', 'pause', 'exec', 'stride', 'stridechain']
    pars = dNl.flatten_list([epoch_ps(c) for c in cs])
    return pars


def dspNtor_ps():
    tor_ps = [f'tortuosity_{dur}' for dur in [1, 2, 5, 10, 20, 30, 60, 100, 120, 240, 300]]
    dsp_ps = [f'dispersion_{t0}_{t1}' for (t0, t1) in
              itertools.product([0, 5, 10, 20, 30, 60], [30, 40, 60, 90, 120, 240, 300])]
    pars = tor_ps + dsp_ps + scal(dsp_ps)
    return pars


def contour_xy(Nc, flat=False):
    return xy(contour(Nc), flat=flat)


def midline_xy(N, flat=False):
    return xy(midline(N, type='point'), flat=flat)


def segs(Nsegs):
    return midline(Nsegs, type='seg')

def angs():
    angs = orient(['front', 'rear', 'head', 'tail']) + ['bend']
    return angs

def ang_pars(angs):
    avels=vel(angs)
    aaccs=acc(angs)
    uangs=unwrap(angs)
    avels_min, avels_max=min(avels), max(avels)
    return avels+aaccs+uangs+avels_min+avels_max

def angular(N) :
    Nangles = np.clip(N - 2, a_min=0, a_max=None)
    angs = angles(Nangles)
    Nsegs = np.clip(N - 1, a_min=0, a_max=None)
    ssegs = segs(Nsegs)
    ors=orient(['front', 'rear', 'head', 'tail'])+ orient(ssegs)
    ang=ors+angs+['bend']
    ang_ps=ang_pars(ang)
    return ang+ang_ps

def angles(Nangles):
    return [f'angle{i}' for i in range(Nangles)]



def retrieve_metrics(obj, c):
    sp_conf = c.metric_definition.spatial
    if sp_conf.fitted is None:
        point_idx = sp_conf.hardcoded.point_idx
        use_component_vel = sp_conf.hardcoded.use_component_vel
    else:
        point_idx = sp_conf.fitted.point_idx
        use_component_vel = sp_conf.fitted.use_component_vel

    try:
        points = midline(c.Npoints, type='point')
        p = points[point_idx - 1]
    except:
        p = 'centroid'
    c.point=p
    obj = define_metrics(obj, N=c.Npoints, Nc=c.Ncontour, p=c.point, use_component_vel=use_component_vel)
    return obj



def define_metrics(obj, N=None, Nc=None, p=None, use_component_vel=False):
    from lib.aux import dictsNlists as dNl
    obj.points = midline(N, type='point')
    obj.Nangles = np.clip(N - 2, a_min=0, a_max=None)
    obj.angles = angles(obj.Nangles)
    obj.Nsegs = np.clip(N - 1, a_min=0, a_max=None)
    obj.segs = segs(obj.Nsegs)
    obj.midline_xy = midline_xy(N, flat=False)
    obj.contour = contour(Nc)
    obj.contour_xy = contour_xy(Nc,flat=False)

    obj.point = p
    obj.distance = dst(obj.point)
    obj.velocity = vel(obj.point)
    obj.acceleration = acc(obj.point)
    if use_component_vel:
        obj.velocity = lin(obj.velocity)
        obj.acceleration = lin(obj.acceleration)

    obj.h5_kdic = h5_kdic(p, N, Nc)
    obj.load_h5_kdic = dNl.NestDict({h5k: "w" for h5k in obj.h5_kdic.keys()})
    return obj

def h5_kdic(p, N, Nc):
    from lib.aux import dictsNlists as dNl
    dic = dNl.NestDict({
        'contour': contour_xy(Nc, flat=True),
        'midline': midline_xy(N, flat=True),
        'epochs': epochs_ps(),
        'base_spatial': base_spatial_ps(p),
        'angular':angular(N),
        'dspNtor': dspNtor_ps(),
    })
    return dic
