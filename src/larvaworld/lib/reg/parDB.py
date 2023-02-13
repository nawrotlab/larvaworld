from types import FunctionType
import warnings
import numpy as np
from typing import List, Tuple, TypedDict
import param
warnings.simplefilter(action='ignore', category=FutureWarning)

from larvaworld.lib.aux import naming as nam
from larvaworld.lib.aux.par_aux import tilde, circle, bar, wave, subsup, sub, sup, th, Delta, dot, circledast, omega, ddot, mathring, delta
from larvaworld.lib import reg, aux, util, decorators

proc_type_keys = ['angular', 'spatial', 'source', 'dispersion', 'tortuosity', 'PI', 'wind']
anot_type_keys = ['bout_detection', 'bout_distribution', 'interference', 'source_attraction', 'patch_residency']


def update_default(name, dic, **kwargs):
    if name not in ['visualization', 'enrichment']:
        # aux.update_nestdict(dic, kwargs)
        dic.update(kwargs)
        return dic
    else:
        for k, v in dic.items():
            if k in list(kwargs.keys()):
                dic[k] = kwargs[k]
            elif isinstance(v, dict):
                for k0, v0 in v.items():
                    if k0 in list(kwargs.keys()):
                        dic[k][k0] = kwargs[k0]
        return aux.AttrDict(dic)

def get_default(d,key='v') :
    if d is None:
        return None
    null = aux.AttrDict()
    for k, v in d.items():
        if not isinstance(v, dict):
            null[k] = v
        elif 'k' in v.keys() or 'h' in v.keys() or 'dtype' in v.keys():
            null[k] = None if key not in v.keys() else v[key]
        else:
            null[k] = get_default(v,key)
    return null




def ConfID_entry(conftype, ids=None, default=None, k=None, symbol=None, single_choice=True):
    def loadConfDic(k):
        return aux.load_dict(reg.Path[k])

    def selector_func(objects, default=None, single_choice=True, **kwargs):
        kws = {
            'objects': objects,
            'default': default,
            'allow_None': True,
        }

        kwargs.update(kws)
        if single_choice:
            func = param.Selector
        else:
            func = param.ListSelector
        try:
            f = func(empty_default=True, **kwargs)
        except:
            f = func(**kwargs)
        return f

    if ids is None:
        ids = list(loadConfDic(conftype).keys())

    def ConfSelector(**kwargs):
        def func():
            return selector_func(objects=ids, **kwargs)

        return func



    if single_choice:
        t = str
        IDstr = 'ID'
    else:
        t = List[str]
        IDstr = 'IDs'

    low = conftype.lower()
    if k is None:
        k = f'{low}{IDstr}'
    if symbol is None:
        symbol = sub(IDstr, low)
    d = {'dtype': t, 'vparfunc': ConfSelector(default=default, single_choice=single_choice),
         'vs': ids, 'v': default,
         'symbol': symbol, 'k': k, 'h': f'The {conftype} configuration {IDstr}',
         'disp': f'{conftype} {IDstr}'}
    return aux.AttrDict(d)

@decorators.timeit
def buildInitDict():
    bF, bT = {'dtype': bool, 'v': False}, {'dtype': bool, 'v': True}

    def pCol(v, obj, **kwargs):
        return {'dtype': str, 'vfunc': param.Color, 'v': v, 'disp': 'color',
                'h': f'The default color of the {obj}.', **kwargs}

    def pPath(conftype=None, h=None, k=None, **kwargs):
        if h is None:
            h = f'The relative path to store the {conftype} datasets.'
        return {'dtype': str, 'h': h, 'k': k,
                'vfunc': param.Foldername, **kwargs}

    def pSaveTo(h='The directory to save data, plots and media', k='save_to', **kwargs):
        return pPath(h=h, k=k, **kwargs)

    def pID(conftype, h=None, k=None, **kwargs):
        if h is None:
            h = f'The unique ID   of the {conftype}.'
        return {'dtype': str, 'h': h, 'k': k, **kwargs}

    def pXYs(conftype, h=None, lim=(-1.0, 1.0), **kwargs):
        if h is None:
            h = f'The XY coordinates of the {conftype}.'
        return {'dtype': List[Tuple[float]], 'lim': lim, 'h': h, **kwargs}

    def substrate():
        from larvaworld.lib.model.deb.substrate import substrate_dict
        d = aux.AttrDict()
        d['substrate_composition'] = {
            n: {'v': 0.0, 'lim': (0.0, 10.0), 'h': f'{n} density in g/cm**3.'} for
            n in
            ['glucose', 'dextrose', 'saccharose', 'yeast', 'agar', 'cornmeal']}

        d['substrate'] = {
            'type': {'dtype': str, 'v': 'standard', 'vs': list(substrate_dict.keys()),
                     'h': 'The type of substrate.'},
            'quality': {'v': 1.0, 'lim': (0.0, 1.0),
                        'h': 'The substrate quality as percentage of nutrients relative to the intact substrate type.'}

        }
        return d

    def food(d):
        d['odor'] = {
            'odor_id': pID('odorant', disp='ID'),
            'odor_intensity': {'lim': (0.0, 10.0), 'disp': 'C peak',
                               'h': 'The peak concentration of the odorant in micromoles.'},
            'odor_spread': {'lim': (0.0, 10.0), 'disp': 'spread',
                            'h': 'The spread of the concentration gradient around the peak.'}
        }
        # })

        d['food_grid'] = {
            'unique_id': pID(' food grid', disp='ID', v='Food_grid'),
            'grid_dims': {'dtype': Tuple[int], 'v': (50, 50), 'lim': (10, 200), 'disp': 'XY dims',
                          'vfunc': param.Tuple,
                          'h': 'The spatial resolution of the food grid.'},
            'initial_value': {'v': 0.1, 'lim': (0.0, 10.0), 'dv': 0.01, 'disp': 'Initial amount',
                              'h': 'The initial amount of food in each cell of the grid.'},
            'distribution': {'dtype': str, 'v': 'uniform', 'vs': ['uniform'],
                             'h': 'The distribution of food in the grid.'},
            'default_color': pCol('green', 'food grid'),
            **d['substrate']
        }

        d['agent'] = {
            'group': pID('agent group', disp=' group ID', k='gID'),
            'odor': d['odor'],
            'pos': d['xy'],
            'default_color': pCol('green', 'agent'),
            'radius': {'v': 0.003, 'lim': (0.0, 0.1), 'dv': 0.001,
                       'h': 'The spatial radius of the source in meters.'},
            'regeneration': {**bF, 'h': 'Whether to regenerate a source when depleted.'},
            'regeneration_pos': {
                'loc': d['xy'],
                'scale': d['xy'],
            }
        }

        d['source'] = {
            **d['agent'],
            'can_be_carried': {**bF, 'disp': 'carriable', 'h': 'Whether the source can be carried around.'},
            'can_be_displaced': {**bF, 'disp': 'displaceable',
                                 'h': 'Whether the source can be displaced by wind/water.'},


        }

        d['nutrient'] = {

            'amount': {'v': 0.0, 'lim': (0.0, 10.0), 'h': 'The food amount in the source.'},
            'radius': {'v': 0.003, 'lim': (0.0, 0.1), 'dv': 0.001,
                       'h': 'The spatial radius of the source in meters.'},
            **d['substrate']
        }

        d['Source'] = {

            'amount': {'v': 0.0, 'lim': (0.0, 10.0), 'h': 'The food amount in the source.'},
            **d['source'],
            **d['substrate']
        }

        d['SourceGroup'] = {
            'distribution': d['spatial_distro'],
            **{k:v for k,v in d['Source'].items() if k not in ['pos', 'group']}

        }

        d['Source_distro'] = d['spatial_distro']

        d['food_params'] = {'source_groups': {'dtype': dict, 'v': {}, 'disp': 'source groups', 'k': 'gSources',
                                              'symbol': sub('source', 'G'), 'entry': 'SourceGroup',
                                              'h': 'The groups of odor or food sources available in the arena',
                                              },
                            'food_grid': {'dtype': dict, 'v': None, 'disp': 'food grid', 'k': 'gFood',
                                          'symbol': sub('food', 'G'),
                                          'h': 'The food grid in the arena',
                                          },
                            'source_units': {'dtype': dict, 'v': {}, 'disp': 'source units', 'k': 'gUnits',
                                             'symbol': sub('source', 'U'), 'entry': 'source',
                                             'h': 'The individual sources  of odor or food in the arena',

                                             }
                            }

        return d

    def life(d):
        d['epoch'] = {
            'start': {'lim': (0.0, 250.0), 'h': 'The beginning of the epoch in hours post-hatch.'},
            'stop': {'lim': (0.0, 250.0), 'h': 'The end of the epoch in hours post-hatch.'},
            'substrate': d['substrate']

        }

        d['Life'] = {
            'age': {'v': 0.0, 'lim': (0.0, 250.0), 'dv': 1.0,
                    'h': 'The larva age in hours post-hatch.'},
            'epochs': {'dtype': TypedDict, 'v': {}, 'entry': 'epoch', 'disp': 'life epochs',
                       'h': 'The feeding epochs comprising life history.'}

        }
        return d

    def xy_distros():
        d = aux.AttrDict({
            'xy': {'dtype': Tuple[float], 'v': (0.0, 0.0), 'k': 'xy', 'lim': (-1.0, 1.0),
                   'vfunc': param.XYCoordinates,
                   'h': 'The xy spatial position coordinates.'},

            'logn_dist': {
                'range': {'dtype': Tuple[float], 'v': (0.0, 2.0), 'lim': (0.0, 10.0), 'dv': 1.0},
                'name': {'dtype': str, 'v': 'lognormal', 'vs': ['lognormal']},
                'mu': {'v': 1.0, 'lim': (0.0, 10.0)},
                'sigma': {'v': 0.0, 'lim': (0.0, 10.0)},
                'fit': bF
            },
        })
        d['spatial_distro'] = {
            'mode': {'dtype': str, 'v': 'normal', 'vs': ['normal', 'periphery', 'uniform', 'grid'],
                     'disp': 'placing',
                     'h': 'The wa to place agents in the distribution shape.'},
            'shape': {'dtype': str, 'v': 'circle', 'vs': ['circle', 'rect', 'oval'],
                      'h': 'The space of the spatial distribution.'},
            'N': {'dtype': int, 'v': 10, 'lim': (0, 1000),
                  'h': 'The number of agents in the group.'},
            'loc': d['xy'],
            'scale': d['xy'],
        }

        d['larva_distro'] = {
            **d['spatial_distro'],
            'orientation_range': {'dtype': Tuple[float], 'v': (0.0, 360.0), 'lim': (0.0, 360.0),
                                  'dv': 1.0,
                                  'disp': 'heading',
                                  'h': 'The range of larva body orientations to sample from, in degrees.'}
        }

        d['source_distro'] =d['spatial_distro']
        return d

    def scapeConfs():

        d = aux.AttrDict({
            'odorscape': {
                'odorscape': {'dtype': str, 'v': 'Gaussian', 'vs': ['Gaussian', 'Diffusion'],
                              'k': 'odorscape_mod',
                              'h': 'The algorithm used for odorscape generation.'},
                'grid_dims': {'dtype': Tuple[int], 'v': (51, 51), 'lim': (10, 100), 'vfunc': param.Tuple,
                              'k': 'grid_dims',
                              'h': 'The odorscape grid resolution.'},
                'evap_const': {'lim': (0.0, 1.0), 'k': 'c_evap',
                               'h': 'The evaporation constant of the diffusion algorithm.'},
                'gaussian_sigma': {'dtype': Tuple[float], 'lim': (0.0, 1.0), 'vfunc': param.NumericTuple,
                                   'k': 'gau_sigma',
                                   'h': 'The sigma of the gaussian difusion algorithm.'}
            },
            'thermoscape': {
                'thermo_sources': {'v': [(0.5, 0.05), (0.05, 0.5), (0.5, 0.95), (0.95, 0.5)],
                                   'dtype': List[Tuple[float]],
                                   'lim': (-100.0, 100.0), 'h': 'The xy coordinates of the thermal sources',
                                   'disp': 'thermal sources',
                                   'k': 'temp_sources'},
                'plate_temp': {'v': 22.0, 'lim': (0.0, 100.0), 'h': 'reference temperature',
                               'disp': 'reference temperature',
                               'k': 'temp_0'},
                'thermo_source_dTemps': {'v': [8.0, -8.0, 8.0, -8.0], 'dtype': List[float],
                                         'lim': (-100.0, 100.0),
                                         'h': 'The relative temperature of the thermal sources',
                                         'disp': 'thermal gradients', 'k': 'dtemp_sources'}
            },
            'air_puff': {
                'duration': {'v': 1.0, 'lim': (0.0, 100.0), 'h': 'The duration of the air-puff in seconds.'},
                'speed': {'v': 10.0, 'lim': (0.0, 1000.0), 'h': 'The wind speed of the air-puff.'},
                'direction': {'v': 0.0, 'lim': (0.0, 100.0), 'h': 'The directions of the air puff in radians.'},
                'start_time': {'v': 0.0, 'lim': (0.0, 10000.0), 'dv': 1.0,
                               'h': 'The starting time of the air-puff in seconds.'},
                'N': {'dtype': int, 'lim': (0, 10000),
                      'h': 'The number of repetitions of the puff. If N>1 an interval must be provided'},
                'interval': {'v': 5.0, 'lim': (0.0, 10000.0),
                             'h': 'Whether the puff will reoccur at constant time intervals in seconds. Ignored if N=1'},
            },
            'windscape': {
                'wind_direction': {'dtype': float, 'v': np.pi, 'lim': (0.0, 2 * np.pi), 'dv': 0.1,
                                   'h': 'The absolute polar direction of the wind/air puff.'},
                'wind_speed': {'dtype': float, 'v': 0.0, 'lim': (0.0, 100.0), 'dv': 1.0,
                               'h': 'The speed of the wind/air puff.'},
                'puffs': {'dtype': TypedDict, 'v': {}, 'entry': 'air_puff', 'disp': 'air-puffs',
                          'h': 'Repetitive or single air-puff stimuli.'}
            },
            'odor_gains': {
                'unique_id': pID('odorant'),
                'mean': {'lim': (0.0, 1000.0), 'dv': 10.0,
                         'h': 'The mean gain/valence for the odorant. Positive/negative for appettitive/aversive valence.'},
                'std': {'lim': (0.0, 10.0), 'dv': 1.0,
                        'h': 'The standard deviation for the odorant gain/valence.'}
            },

            'arena': {
                'dims': {'dtype': Tuple[float], 'v': (0.1, 0.1), 'lim': (0.0, 2.0), 'dv': 0.01,
                               'disp': 'X,Y (m)',
                               'vfunc': param.NumericTuple,
                               'h': 'The arena dimensions in meters.'},
                'shape': {'dtype': str, 'v': 'circular', 'vs': ['circular', 'rectangular'],
                                'disp': 'shape',
                                'h': 'The arena shape.'},
                'torus':{**bF, 'h': 'Whether to allow a toroidal space.'}
            },

        })
        d['Border'] = {
            'default_color': pCol('black', 'border'),
            'width': {'v': 0.001, 'lim': (0.0, 10.0), 'h': 'The width of the border.'},
            'points': pXYs('border segments', lim=(-1.0, 10.10)),
        }

        d['border_list'] = {
            'default_color': pCol('black', 'border'),
            'points': pXYs('border segments', lim=(-1.0, 10.10)),
        }
        return d

    def runConfs():

        d = aux.AttrDict({
            'Essay': {
                # 'essay_ID': pID('essay'),
                # 'path': pPath('essay'),
                'N': {'dtype': int, 'lim': (1, 100), 'disp': '# larvae',
                      'h': 'The number of larvae per larva-group.'}
            },
            'sim_params': {
                # 'sim_ID': pID('simulation', k='id'),
                # 'path': pPath('simulation', k='path'),
                'duration': {'v': 5.0,'lim': (0.0, 100000.0), 'h': 'The duration of the simulation in minutes.',
                             'k': 't'},
                'timestep': {'v': 0.1, 'lim': (0.0, 0.4), 'dv': 0.05,
                             'h': 'The timestep of the simulation in seconds.',
                             'k': 'dt'},
                # 'Box2D': {**bF, 'h': 'Whether to use the Box2D physics engine or not.', 'k': 'Box2D'},

            },
            'build_conf': {
                'min_duration_in_sec': {'v': 170.0, 'lim': (0.0, 3600.0), 'dv': 1.0,
                                        'symbol': sub('T', 'min'), 'k': 'dur_min',
                                        'disp': 'Min track duration (sec)'},
                'min_end_time_in_sec': {'v': 0.0, 'lim': (0.0, 3600.0), 'dv': 1.0,
                                        'symbol': subsup('t', 'min', 1), 'k': 't1_min',
                                        'disp': 'Min track termination time (sec)'},
                'start_time_in_sec': {'v': 0.0, 'lim': (0.0, 3600.0), 'dv': 1.0,
                                      'symbol': sup('t', 0), 'k': 't0',
                                      'disp': 'Track initiation time (sec)'},
                'max_Nagents': {'dtype': int, 'v': 500, 'lim': (0, 5000),
                                'symbol': sub('N', 'max'), 'k': 'N_max',
                                'disp': 'Max number of larva tracks'},
                'save_mode': {'dtype': str, 'v': 'semifull',
                              'symbol': sub('mod', 'build'), 'k': 'mod_build',
                              'vs': ['minimal', 'semifull', 'full', 'points'],
                              'disp': 'Storage mode'
                              },
            },
            'output': {n: bF for n in list(reg.output_dict.keys())}
        })
        return d

    def enrConfs():
        # to_drop_keys = ['midline', 'contour', 'stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn',
        #                 'turn',
        #                 'unused']
        d = aux.AttrDict()

        d['ang_definition'] = {
            'bend': {'dtype': str, 'v': 'from_vectors', 'vs': ['from_angles', 'from_vectors'],
                     'h': 'Whether bending angle is computed as a sum of sequential segmental angles or as the angle between front and rear body vectors.'},
            'front_vector': {'dtype': Tuple[int], 'v': (1, 2), 'lim': (-12, 12), 'vfunc': param.Tuple,
                             'h': 'The initial & final segment of the front body vector.'},
            'rear_vector': {'dtype': Tuple[int], 'v': (-2, -1), 'lim': (-12, 12), 'vfunc': param.Tuple,
                            'h': 'The initial & final segment of the rear body vector.'},
            'front_body_ratio': {'v': 0.5, 'lim': (0.0, 1.0), 'disp': 'front_ratio',
                                 'h': 'The fraction of the body considered front, relevant for bend computation from angles.'}
        }
        d['spatial_definition'] = {
            'point_idx': {'dtype': int, 'lim': (-1, 12),
                          'h': 'The index of the segment used as the larva spatial position (-1 means using the centroid).'},
            'use_component_vel': {**bF, 'disp': 'vel_component',
                                  'h': 'Whether to use the component velocity ralative to the axis of forward motion.'}
        }

        d['metric_definition'] = {
            'angular': d['ang_definition'],
            'spatial': d['spatial_definition'],
            'dispersion': {
                'dsp_starts': {'dtype': List[int], 'v': [0], 'lim': (0, 200), 'dv': 1, 'disp': 'starts',
                               'h': 'The timepoints to start calculating dispersion in seconds.'},
                'dsp_stops': {'dtype': List[int], 'v': [40, 60], 'lim': (0, 200), 'dv': 1, 'disp': 'stops',
                              'h': 'The timepoints to stop calculating dispersion in seconds.'},
            },
            'tortuosity': {
                'tor_durs': {'dtype': List[int], 'v': [5, 20], 'lim': (0, 200), 'dv': 1, 'disp': 't (sec)',
                             'h': 'The time windows to use when calculating tortuosity in seconds.'}
            },
            'stride': {
                'track_point': {'dtype': str,
                                'h': 'The midline point to use when detecting the strides. When none is provided, the default position of the larva is used (see spatial definition).'},
                'use_scaled_vel': {**bT, 'disp': 'vel_scaled',
                                   'h': 'Whether to use the velocity scaled to the body length.'},
                'vel_threshold': {'v': 0.3, 'lim': (0.0, 2.0), 'disp': 'vel_thr',
                                  'h': 'The velocity threshold to be reached in every stride cycle.'},
            },
            # 'pause': {

            'turn': {
                'min_ang': {'v': 30.0, 'lim': (0.0, 180.0), 'dv': 1.0,
                            'h': 'The minimum orientation angle change required to detect a turn.'},
                'min_ang_vel': {'v': 0.0, 'lim': (0.0, 1000.0), 'dv': 1.0,
                                'h': 'The minimum angular velocity maximum required to detect a turn.'},
                'chunk_only': {'dtype': str, 'vs': ['', 'stride', 'pause'],
                               'h': 'Whether to only detect turns whithin a given bout type.'},
            }
        }

        d['preprocessing'] = {
            'rescale_by': {'lim': (0.0, 10.0),
                           'h': 'Whether to rescale spatial coordinates by a scalar in meters.'},
            'drop_collisions': {**bF, 'h': 'Whether to drop timepoints where larva collisions are detected.'},
            'interpolate_nans': {**bF, 'h': 'Whether to interpolate missing values.'},
            'filter_f': {'lim': (0.0, 10.0), 'disp': 'filter frequency',
                         'h': 'Whether to filter spatial coordinates by a grade-1 low-pass filter of the given cut-off frequency.'},
            'transposition': {'dtype': str, 'vs': [None, 'origin', 'arena', 'center'],
                              'h': 'Whether to transpose spatial coordinates.'}
        }
        d['processing'] = {t: {**bT, 'h': f'Whether to apply {t} processing'} for t in proc_type_keys}
        d['annotation'] = {
            # **{b: {**bF, 'h': f'Whether to annotate {b} epochs'} for b in ['stride', 'pause', 'turn']},
            'bout_detection': {**bT, 'h': f'Whether to detect epochs'},
            'bout_distribution': {**bT, 'h': f'Whether to fit distributions to epoch durations'},
            'interference': {**bT, 'h': f'Whether to compute interference'},
            'source_attraction': {**bF, 'h': f'Whether to compute bearing to sources'},
            'patch_residency': {**bF, 'h': f'Whether to compute patch residency'},
            # 'fits': {**bT, 'h': f'Whether to fit epochs'}
        }
        # d['to_drop'] = {kk: {**bF, 'h': f'Whether to drop {kk} parameters'} for kk in to_drop_keys}
        d['enrichment'] = {**{k: d[k] for k in
                              ['metric_definition', 'preprocessing', 'processing', 'annotation']},
                           'recompute': {**bF, 'h': f'Whether to recompute'},
                           'mode': {'dtype': str, 'v': 'minimal', 'vs': ['minimal', 'full'],
                                    'h': f'The processing mode'}
                           }
        return d

    def init_vis():
        d = aux.AttrDict()
        d['render'] = {
            'mode': {'dtype': str, 'v': None, 'vs': [None, 'video', 'image'], 'h': 'The visualization mode',
                     'k': 'm'},
            'image_mode': {'dtype': str, 'vs': [None, 'final', 'snapshots', 'overlap'],
                           'h': 'The image-render mode',
                           'k': 'im'},
            'video_speed': {'dtype': int, 'v': 60, 'lim': (1, 100), 'h': 'The video speed', 'k': 'fps'},
            'media_name': {'dtype': str,
                           'h': 'Filename for the saved video/image. File extension mp4/png sutomatically added.',
                           'k': 'media'},
            'show_display': {'dtype': bool, 'v': True, 'h': 'Hide display', 'k': 'hide'},
            'intro_text': {'dtype': bool, 'v': False, 'h': 'Display an introductory screen before launching the simulation'},
        }
        d['draw'] = {
            'draw_head': {'dtype': bool, 'v': False, 'h': 'Draw the larva head'},
            'draw_centroid': {'dtype': bool, 'v': False, 'h': 'Draw the larva centroid'},
            'draw_midline': {'dtype': bool, 'v': True, 'h': 'Draw the larva midline'},
            'draw_contour': {'dtype': bool, 'v': True, 'h': 'Draw the larva contour'},
            'draw_sensors': {'dtype': bool, 'v': False, 'h': 'Draw the larva sensors'},
            'trails': {'dtype': bool, 'v': False, 'h': 'Draw the larva trajectories'},
            'trajectory_dt': {'lim': (0.0, 100.0), 'h': 'Duration of the drawn trajectories'},
            'odor_aura': {'dtype': bool, 'v': False, 'h': 'Draw the aura around odor sources'},
        }
        d['color'] = {
            'black_background': {'dtype': bool, 'v': False, 'h': 'Set the background color to black'},
            'random_colors': {'dtype': bool, 'v': False, 'h': 'Color each larva with a random color'},
            'color_behavior': {'dtype': bool, 'v': False,
                               'h': 'Color the larvae according to their instantaneous behavior'},
        }
        d['aux'] = {
            'visible_clock': {'dtype': bool, 'v': True, 'h': 'Hide/show the simulation clock'},
            'visible_scale': {'dtype': bool, 'v': True, 'h': 'Hide/show the simulation scale'},
            'visible_state': {'dtype': bool, 'v': False, 'h': 'Hide/show the simulation state'},
            'visible_ids': {'dtype': bool, 'v': False, 'h': 'Hide/show the larva IDs'},
        }
        d['visualization'] = {
            'render': d['render'],
            'aux': d['aux'],
            'draw': d['draw'],
            'color': d['color'],

        }

        return d

    def init_mods():
        d = aux.AttrDict({
            'bout_distro': {
                'fit': {**bT, 'combo': 'distro',
                        'h': 'Whether the distribution is sampled from a reference dataset. Once this is set to "ON" no other parameter is taken into account.'},
                'range': {'dtype': Tuple[float], 'lim': (0.0, 500.0), 'combo': 'distro',
                          'h': 'The distribution range.'},
                'name': {'dtype': str,
                         'vs': ['powerlaw', 'exponential', 'lognormal', 'lognormal-powerlaw', 'levy', 'normal',
                                'uniform'],
                         'combo': 'distro', 'h': 'The distribution name.'},
                'mu': {'lim': (-1000.0, 1000.0), 'disp': 'mean', 'combo': 'distro',
                       'h': 'The "mean" argument for constructing the distribution.'},
                'sigma': {'lim': (-1000.0, 1000.0), 'disp': 'std', 'combo': 'distro',
                          'h': 'The "sigma" argument for constructing the distribution.'},
                'alpha': {'lim': (-1000.0, 1000.0), 'disp': 'alpha', 'combo': 'distro',
                          'h': 'The "alpha" argument for constructing the distribution.'},
                'beta': {'lim': (-1000.0, 1000.0), 'disp': 'beta', 'combo': 'distro',
                         'h': 'The "beta" argument for constructing the distribution.'},
            },
            'physics': {
                'torque_coef': {'v': 0.5, 'lim': (0.1, 1.0), 'dv': 0.01, 'label': 'torque coefficient',
                                'symbol': sub('c', 'T'), 'u_name': sup('sec', -2), 'u': reg.units.s ** -2,
                                'h': 'Conversion coefficient from TURNER output to torque-per-inertia-unit.'},
                'ang_vel_coef': {'v': 1.0, 'lim': (0.0, 5.0), 'dv': 0.01,
                                 'label': 'angular velocity coefficient',
                                 'h': 'Conversion coefficient from TURNER output to angular velocity.'},
                'ang_damping': {'v': 1.0, 'lim': (0.1, 2.0), 'label': 'angular damping', 'symbol': 'z',
                                'u_name': sup('sec', -1), 'u': reg.units.s ** -1,
                                'h': 'Angular damping exerted on angular velocity.'},
                'lin_damping': {'v': 1.0, 'lim': (0.0, 10.0), 'label': 'linear damping', 'symbol': 'zl',
                                'u_name': sup('sec', -1), 'u': reg.units.s ** -1,
                                'h': 'Linear damping exerted on forward velocity.'},
                'body_spring_k': {'v': 1.0, 'lim': (0.0, 10.0), 'dv': 0.1, 'label': 'body spring constant',
                                  'symbol': 'k', 'u_name': sup('sec', -2), 'u': reg.units.s ** -2,
                                  'h': 'Larva-body torsional spring constant reflecting deformation resistance.'},
                'bend_correction_coef': {'v': 1.0, 'lim': (0.8, 1.5), 'label': 'bend correction coefficient',
                                         'symbol': sub('c', 'b'),
                                         'h': 'Correction coefficient of bending angle during forward motion.'},
                'ang_mode': {'dtype': str, 'v': 'torque', 'vs': ['torque', 'velocity'], 'label': 'angular mode',
                             'h': 'Whether the Turner module output is equivalent to torque or angular velocity.'},
            },
            'crawler': {
                'mode': {'dtype': str, 'v': 'realistic', 'k': 'Cr_mod',
                         'vs': ['realistic', 'square', 'gaussian', 'constant'],
                         'symbol': subsup('A', 'C', 'mode'),
                         'label': 'crawler waveform',
                         'h': 'The waveform of the repetitive crawling oscillator (CRAWLER) module.'},
                'initial_freq': {'v': 1.418, 'lim': (0.5, 2.5), 'dv': 0.1, 'aux_vs': ['sample'],
                                 'disp': 'initial',
                                 'k': 'f_C0',
                                 'label': 'crawling frequency', 'symbol': sub('f', 'C'), 'u': reg.units.Hz,
                                 'combo': 'frequency', 'codename': 'scaled_velocity_freq',
                                 'h': 'The initial frequency of the repetitive crawling behavior.'},
                'max_scaled_vel': {'v': 0.6, 'lim': (0.0, 1.5), 'label': 'maximum scaled velocity',
                                   'codename': 'stride_scaled_velocity_max', 'k': 'sstr_v_max', 'dv': 0.1,
                                   'symbol': sub(circle('v'), 'max'), 'u': reg.units.s ** -1,
                                   'u_name': '$body-lengths/sec$',
                                   'h': 'The maximum scaled forward velocity.'},
                'stride_dst_mean': {'v': 0.224, 'lim': (0.0, 1.0), 'dv': 0.01, 'aux_vs': ['sample'],
                                    'disp': 'mean',
                                    'k': 'sstr_d_mu',
                                    'label': r'stride distance mean', 'symbol': sub(bar(circle('d')), 'S'),
                                    'u_name': '$body-lengths$',
                                    'combo': 'scaled distance / stride', 'codename': 'scaled_stride_dst_mean',
                                    'h': 'The mean displacement achieved in a single peristaltic stride as a fraction of the body length.'},
                'stride_dst_std': {'v': 0.033, 'lim': (0.0, 1.0), 'aux_vs': ['sample'], 'disp': 'std',
                                   'k': 'sstr_d_std',
                                   'label': 'stride distance std', 'symbol': sub(tilde(circle('d')), 'S'),
                                   'u_name': '$body-lengths$',
                                   'combo': 'scaled distance / stride', 'codename': 'scaled_stride_dst_std',
                                   'h': 'The standard deviation of the displacement achieved in a single peristaltic stride as a fraction of the body length.'},
                'initial_amp': {'lim': (0.0, 2.0), 'disp': 'initial', 'combo': 'amplitude', 'k': 'A_C0',
                                'label': 'initial crawler amplitude', 'dv': 0.1,
                                'symbol': subsup('A', 'C', '0'),
                                'h': 'The initial amplitude of the CRAWLER-generated forward velocity if this is hardcoded (e.g. constant waveform).'},
                'noise': {'v': 0.0, 'lim': (0.0, 1.0), 'dv': 0.01, 'disp': 'noise', 'combo': 'amplitude',
                          'k': 'A_Cnoise', 'symbol': subsup('A', 'C', 'noise'),
                          'label': 'crawler output noise',
                          'h': 'The intrinsic output noise of the CRAWLER-generated forward velocity.'},
                'max_vel_phase': {'v': 3.6, 'lim': (0.0, 2 * np.pi), 'label': 'max velocity phase',
                                  'k': 'phi_v_max', 'dv': 0.1,
                                  'symbol': subsup('$\phi$', 'C', 'v'), 'u_name': 'rad', 'u': reg.units.rad,
                                  'codename': 'phi_scaled_velocity_max',
                                  'h': 'The phase of the crawling oscillation cycle where forward velocity is maximum.'}
            },
            'olfactor': {
                'perception': {'dtype': str, 'v': 'log', 'vs': ['log', 'linear', 'null'],
                               'label': 'olfaction sensing transduction mode',
                               'k': 'mod_O',
                               'symbol': sub('mod', 'O'), 'u_name': None,
                               'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input.'},
                'input_noise': {'v': 0.0, 'lim': (0.0, 1.0), 'h': 'The intrinsic noise of the sensory input.'},
                'decay_coef': {'v': 0.0, 'lim': (0.0, 2.0), 'label': 'olfactory decay coef',
                               'symbol': sub('c', 'O'), 'k': 'c_O',
                               'h': 'The linear decay coefficient of the olfactory sensory activation.'},
                'brute_force': {**bF,
                                'h': 'Whether to apply direct rule-based modulation on locomotion or not.'}
            },
            'thermosensor': {
                'perception': {'dtype': str, 'v': 'linear', 'vs': ['log', 'linear', 'null'],
                               'label': 'thermosensing transduction mode',
                               'k': 'mod_th',
                               'symbol': sub('mod', 'th'), 'u_name': None,
                               'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input.'},
                'input_noise': {'v': 0.0, 'lim': (0.0, 1.0), 'h': 'The intrinsic noise of the sensory input.'},
                'decay_coef': {'v': 0.0, 'lim': (0.0, 2.0), 'label': 'thermosensation decay coef',
                               'symbol': sub('c', 'th'), 'k': 'c_th',
                               'h': 'The linear decay coefficient of the thermosensory activation.'},
                'brute_force': {**bF,
                                'h': 'Whether to apply direct rule-based modulation on locomotion or not.'},
                'cool_gain': {'v': 200.0, 'lim': (-1000.0, 1000.0),
                              'label': 'tactile cool_gain coef', 'symbol': sub('G', 'cool'), 'k': 'G_cool',
                              'h': 'The initial gain of the tactile sensor.'},
                'warm_gain': {'v': 0.0, 'lim': (-1000.0, 1000.0),
                              'label': 'warm_gain', 'symbol': sub('G', 'warm'), 'k': 'G_warm',
                              'h': 'The initial gain of the tactile sensor.'},
            },
            'windsensor': {
                'weights': {
                    'hunch_lin': {'v': 10.0, 'lim': (-100.0, 100.0), 'label': 'HUNCH->CRAWLER',
                                  'symbol': sub('w', 'HC'), 'k': 'w_HC',
                                  'h': 'The connection weight between the HUNCH neuron ensemble and the CRAWLER module.'},
                    'hunch_ang': {'v': 0.0, 'lim': (-100.0, 100.0), 'label': 'HUNCH->TURNER',
                                  'symbol': sub('w', 'HT'), 'k': 'w_HT',
                                  'h': 'The connection weight between the HUNCH neuron ensemble and the TURNER module.'},
                    'bend_lin': {'v': 0.0, 'lim': (-100.0, 100.0), 'label': 'BEND->CRAWLER',
                                 'symbol': sub('w', 'BC'), 'k': 'w_BC',
                                 'h': 'The connection weight between the BEND neuron ensemble and the CRAWLER module.'},
                    'bend_ang': {'v': -10.0, 'lim': (-100.0, 100.0), 'label': 'BEND->TURNER',
                                 'symbol': sub('w', 'BT'), 'k': 'w_BT',
                                 'h': 'The connection weight between the BEND neuron ensemble and the TURNER module.'},
                }
            },
            'toucher': {
                'perception': {'dtype': str, 'v': 'linear', 'vs': ['log', 'linear'],
                               'symbol': sub('mod', 'T'),
                               'k': 'mod_T', 'label': 'tactile sensing transduction mode',
                               'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input.'},
                'input_noise': {'v': 0.0, 'lim': (0.0, 1.0), 'h': 'The intrinsic noise of the sensory input.'},
                'decay_coef': {'v': 0.1, 'lim': (0.0, 2.0), 'label': 'tactile decay coef',
                               'symbol': sub('c', 'T'), 'k': 'c_T',
                               'h': 'The exponential decay coefficient of the tactile sensory activation.'},
                'state_specific_best': {**bT,
                                        'h': 'Whether to use the state-specific or the global highest evaluated gain after the end of the memory training period.'},
                'brute_force': {**bF,
                                'h': 'Whether to apply direct rule-based modulation on locomotion or not.'},
                'initial_gain': {'v': 40.0, 'lim': (-100.0, 100.0),
                                 'label': 'tactile sensitivity coef', 'symbol': sub('G', 'T'), 'k': 'G_T',
                                 'h': 'The initial gain of the tactile sensor.'},
                'touch_sensors': {'dtype': List[int], 'lim': (0, 8), 'k': 'sens_touch',
                                  'symbol': sub('N', 'T'), 'label': 'tactile sensor contour locations',
                                  'h': 'The number of touch sensors existing on the larva body.'},
            },
            'feeder': {
                'freq_range': {'dtype': Tuple[float], 'v': (1.0, 3.0), 'lim': (0.0, 4.0), 'disp': 'range',
                               'combo': 'frequency',
                               'h': 'The frequency range of the repetitive feeding behavior.'},
                'initial_freq': {'v': 2.0, 'lim': (0.0, 4.0), 'disp': 'initial', 'combo': 'frequency',
                                 'k': 'f_F0',
                                 'label': 'feeding frequency', 'symbol': sub('f', 'F'), 'u': reg.units.Hz,
                                 'h': 'The initial default frequency of the repetitive feeding behavior'},
                'feed_radius': {'v': 0.1, 'lim': (0.1, 10.0), 'symbol': sub('rad', 'F'),
                                'label': 'feeding radius', 'k': 'rad_F',
                                'h': 'The radius around the mouth in which food is consumable as a fraction of the body length.'},
                'V_bite': {'v': 0.0005, 'lim': (0.0001, 0.01), 'dv': 0.0001,
                           'symbol': sub('V', 'F'), 'label': 'feeding volume ratio', 'k': 'V_F',
                           'h': 'The volume of food consumed on a single feeding motion as a fraction of the body volume.'}
            },
            'memory': {
                'modality': {'dtype': str, 'v': 'olfaction', 'vs': ['olfaction', 'touch'],
                             'h': 'The modality for which the memory module is used.'},
                'Delta': {'v': 0.1, 'lim': (0.0, 10.0), 'h': 'The input sensitivity of the memory.'},
                'state_spacePerSide': {'dtype': int, 'v': 0, 'lim': (0, 20), 'disp': 'state space dim',
                                       'h': 'The number of discrete states to parse the state space on either side of 0.'},
                'gain_space': {'dtype': List[float], 'v': [-300.0, -50.0, 50.0, 300.0],
                               'lim': (-1000.0, 1000.0),
                               'dv': 1.0, 'h': 'The possible values for memory gain to choose from.'},
                'update_dt': {'v': 1.0, 'lim': (0.0, 10.0), 'dv': 1.0,
                              'h': 'The interval duration between gain switches.'},
                'alpha': {'v': 0.05, 'lim': (0.0, 2.0), 'dv': 0.01,
                          'h': 'The alpha parameter of reinforcement learning algorithm.'},
                'gamma': {'v': 0.6, 'lim': (0.0, 2.0),
                          'h': 'The probability of sampling a random gain rather than exploiting the currently highest evaluated gain for the current state.'},
                'epsilon': {'v': 0.3, 'lim': (0.0, 2.0),
                            'h': 'The epsilon parameter of reinforcement learning algorithm.'},
                'train_dur': {'v': 20.0, 'lim': (0.0, 100.0),
                              'h': 'The duration of the training period after which no further learning will take place.'}
            },
            'modules': {
                'crawler': {**bF, 'k': 'C'},
                'turner': {**bF, 'k': 'T'},
                'interference': {**bF, 'k': 'If'},
                'intermitter': {**bF, 'k': 'Im'},
                'feeder': {**bF, 'k': 'F'},
                'olfactor': {**bF, 'k': 'O'},
                'windsensor': {**bF, 'k': 'W'},
                'toucher': {**bF, 'k': 'To'},
                'thermosensor': {**bF, 'k': 'Th'},
                'memory': {**bF, 'k': 'O_mem'},
                # 'touch_memory': {**bF, 'k':'To_mem'},
            },
            'square_interference': {
                'crawler_phi_range': {'dtype': Tuple[float], 'v': (0.0, 0.0), 'lim': (0.0, 2 * np.pi),
                                      'label': 'suppression relief phase interval',
                                      'symbol': '$[\phi_{C}^{\omega_{0}},\phi_{C}^{\omega_{1}}]$',
                                      'u': reg.units.rad,
                                      'h': 'CRAWLER phase range for TURNER suppression lift.'},
                'feeder_phi_range': {'dtype': Tuple[float], 'v': (0.0, 0.0), 'lim': (0.0, 2 * np.pi),
                                     'label': 'feeder suppression relief phase interval',
                                     'symbol': '$[\phi_{F}^{\omega_{0}},\phi_{F}^{\omega_{1}}]$',
                                     'u': reg.units.rad,
                                     'h': 'FEEDER phase range for TURNER suppression lift.'}

            },
            'phasic_interference': {
                'max_attenuation_phase': {'v': 3.4, 'lim': (0.0, 2 * np.pi), 'label': 'max relief phase',
                                          'symbol': '$\phi_{C}^{\omega}$', 'u': reg.units.rad, 'k': 'phi_fov_max',
                                          'h': 'CRAWLER phase of minimum TURNER suppression.'}
            },
            'base_interference': {
                'mode': {'dtype': str, 'v': 'square', 'k': 'IF_mod', 'vs': ['default', 'square', 'phasic'],
                         'h': 'CRAWLER:TURNER suppression phase mode.'},
                'suppression_mode': {'dtype': str, 'v': 'amplitude', 'vs': ['amplitude', 'oscillation', 'both'],
                                     'k': 'IF_target',
                                     'label': 'suppression target', 'symbol': '-', 'u_name': None,
                                     'h': 'CRAWLER:TURNER suppression target.'},
                'attenuation': {'v': 1.0, 'lim': (0.0, 1.0), 'label': 'suppression coefficient',
                                'symbol': '$c_{CT}^{0}$', 'k': 'c_CT0',
                                'h': 'CRAWLER:TURNER baseline suppression coefficient'},
                'attenuation_max': {'v': 0.31, 'lim': (0.0, 1.0),
                                    'label': 'suppression relief coefficient',
                                    'symbol': '$c_{CT}^{1}$', 'k': 'c_CT1',
                                    'h': 'CRAWLER:TURNER suppression relief coefficient.'},
            }
        })

        d['interference'] = {
            **d['base_interference'],
            **d['square_interference'],
            **d['phasic_interference']
        }

        d['neural_turner'] = {
            'base_activation': {'v': 20.0, 'lim': (10.0, 40.0), 'dv': 0.1, 'disp': 'mean',
                                'combo': 'activation',
                                'label': 'tonic input', 'symbol': '$I_{T}^{0}$', 'k': 'I_T0',
                                'h': 'The baseline activation/input of the TURNER module.'},
            'activation_range': {'dtype': Tuple[float], 'v': (10.0, 40.0), 'lim': (0.0, 100.0), 'dv': 0.1,
                                 'disp': 'range',
                                 'k': 'I_T_r',
                                 'label': 'input range', 'symbol': r'$[I_{T}^{min},I_{T}^{max}]$',
                                 'combo': 'activation',
                                 'h': 'The activation/input range of the TURNER module.'},

            'tau': {'v': 0.1, 'lim': (0.05, 0.5), 'dv': 0.01, 'label': 'time constant',
                    'symbol': r'$\tau_{T}$',
                    'u': reg.units.s,
                    'h': 'The time constant of the neural oscillator.'},
            'm': {'dtype': int, 'v': 100, 'lim': (50, 200), 'label': 'maximum spike-rate',
                  'symbol': '$SR_{max}$',
                  'h': 'The maximum allowed spike rate.'},
            'n': {'v': 2.0, 'lim': (1.5, 3.0), 'dv': 0.01, 'label': 'spike response steepness',
                  'symbol': '$n_{T}$',
                  'h': 'The neuron spike-rate response steepness coefficient.'}

        }

        d['sinusoidal_turner'] = {
            'initial_amp': {'v': 19.27, 'lim': (0.0, 100.0), 'disp': 'initial',
                            'combo': 'amplitude',
                            'label': 'output amplitude', 'symbol': '$A_{T}^{0}$', 'k': 'A_T0',
                            'h': 'The initial activity amplitude of the TURNER module.'},
            'amp_range': {'dtype': Tuple[float], 'lim': (0.0, 1000.0), 'v': (0.0, 100.0),
                          'disp': 'range', 'combo': 'amplitude',
                          'label': 'output amplitude range', 'symbol': r'$[A_{T}^{min},A_{T}^{max}]$',
                          'k': 'A_T_r',
                          'h': 'The activity amplitude range of the TURNER module.'},
            'initial_freq': {'v': 0.58, 'lim': (0.01, 2.0), 'dv': 0.01, 'disp': 'initial', 'combo': 'frequency',
                             'k': 'f_T0',
                             'label': 'bending frequency', 'symbol': sub('f', 'T'), 'u_name': '$Hz$',
                             'u': reg.units.Hz,
                             'h': 'The initial frequency of the repetitive lateral bending behavior if this is hardcoded (e.g. sinusoidal mode).'},
            'freq_range': {'dtype': Tuple[float], 'lim': (0.01, 2.0), 'dv': 0.01, 'disp': 'range',
                           'combo': 'frequency',
                           'label': 'bending frequency range', 'k': 'f_T_r', 'v': (0.1, 0.8),
                           'symbol': r'$[f_{T}^{min},f_{T}^{max}]$', 'u_name': '$Hz$', 'u': reg.units.Hz,
                           'h': 'The frequency range of the repetitive lateral bending behavior.'}
        }

        d['constant_turner'] = {
            'initial_amp': {'lim': (0.1, 20.0), 'disp': 'initial', 'combo': 'amplitude', 'k': 'A_T0',
                            'label': 'output amplitude', 'symbol': '$A_{T}^{0}$', 'u_name': None,
                            'h': 'The initial activity amplitude of the TURNER module.'},
        }

        d['base_turner'] = {
            'mode': {'dtype': str, 'v': 'neural', 'vs': ['', 'neural', 'sinusoidal', 'constant'],
                     'k': 'Tur_mod',
                     'h': 'The implementation mode of the lateral oscillator (TURNER) module.'},

            'noise': {'v': 0.0, 'disp': 'noise', 'combo': 'amplitude', 'k': 'A_Tnoise', 'lim': (0.0, 1.0),
                      'h': 'The intrinsic output noise of the TURNER activity amplitude.'},
            'activation_noise': {'v': 0.0, 'disp': 'noise', 'combo': 'activation', 'k': 'I_Tnoise',
                                 'lim': (0.0, 1.0),
                                 'h': 'The intrinsic input noise of the TURNER module.'}
        }

        d['turner'] = {
            **d['base_turner'],
            **d['neural_turner'],
            **d['sinusoidal_turner']
        }

        d['Box2D_joint_N'] = {'dtype': int, 'v': 0, 'lim': (0, 2)}

        d['friction_joint'] = {'N': d['Box2D_joint_N'],
                               'args': {'maxForce': {'v': 10 ** 0, 'lim': (0.0, 10 ** 5)},
                                        'maxTorque': {'v': 10 ** 0, 'lim': (0.0, 10 ** 5)}
                                        }}
        d['revolute_joint'] = {'N': d['Box2D_joint_N'], 'args': {
            'enableMotor': bT,  # )
            'maxMotorTorque': {'v': 0.0, 'lim': (0.0, 10 ** 5)},
            'motorSpeed': {'v': 0.0, 'lim': (0.0, 10 ** 5)}
        }}
        d['distance_joint'] = {'N': d['Box2D_joint_N'], 'args': {
            'frequencyHz': {'v': 5.0, 'lim': (0.0, 10 ** 5)},
            'dampingRatio': {'v': 1.0, 'lim': (0.0, 10 ** 5)},
        }}

        d['Box2D_params'] = {
            'joint_types': {
                'friction': d['friction_joint'],
                'revolute': d['revolute_joint'],
                'distance': d['distance_joint']
            }
        }

        d['body'] = {
            'initial_length': {'v': 0.004, 'lim': (0.0, 0.01), 'dv': 0.0001, 'aux_vs': ['sample'],
                               'disp': 'initial',
                               'label': 'length', 'symbol': '$l$', 'u': reg.units.m, 'k': 'l0',
                               'combo': 'length', 'h': 'The initial body length.'},
            'length_std': {'v': 0.0, 'lim': (0.0, 0.001), 'dv': 0.0001, 'u': reg.units.m, 'aux_vs': ['sample'],
                           'disp': 'std', 'k': 'l_std',
                           'combo': 'length', 'h': 'The standard deviation of the initial body length.'},
            'Nsegs': {'dtype': int, 'v': 2, 'lim': (1, 12), 'label': 'number of body segments', 'symbol': '-',
                      'u_name': '# $segments$', 'k': 'Nsegs',
                      'h': 'The number of segments comprising the larva body.'},
            'seg_ratio': {'k': 'seg_r', 'lim': (0.0, 1.0),
                          'h': 'The length ratio of the body segments. If null, equal-length segments are generated.'},

            'shape': {'dtype': str, 'v': 'drosophila_larva', 'vs': ['drosophila_larva', 'zebrafish_larva'],
                      'k': 'body_shape', 'h': 'The body shape.'},
        }

        d['intermitter'] = {
            'mode': {'dtype': str, 'v': 'default', 'vs': ['', 'default', 'branch', 'nengo'],
                     'h': 'The implementation mode of the intermittency (INTERMITTER) module.'},
            'run_mode': {'dtype': str, 'v': 'stridechain', 'vs': ['stridechain', 'exec'],
                         'h': 'The generation mode of exec epochs.'},
            'stridechain_dist': d['bout_distro'],
            'run_dist': d['bout_distro'],
            'pause_dist': d['bout_distro'],
            'EEB': {'v': 0.0, 'lim': (0.0, 1.0), 'symbol': 'EEB', 'k': 'EEB',
                    'h': 'The baseline exploitation-exploration balance. 0 means only exploitation, 1 only exploration.'},
            'EEB_decay': {'v': 1.0, 'lim': (0.0, 2.0), 'symbol': sub('c', 'EEB'),
                          'k': 'c_EEB',
                          'h': 'The exponential decay coefficient of the exploitation-exploration balance when no food is detected.'},
            'crawl_bouts': {**bT, 'disp': 'crawling bouts',
                            'h': 'Whether crawling bouts (runs/stridechains) are generated.'},
            'feed_bouts': {**bF, 'disp': 'feeding bouts',
                           'h': 'Whether feeding bouts (feedchains) are generated.'},
            'crawl_freq': {'v': 1.43, 'lim': (0.5, 2.5), 'k': 'f_C', 'dv': 0.01, 'u': reg.units.Hz,
                           'symbol': sub('f', 'C'),
                           'disp': 'crawling frequency',
                           'h': 'The default frequency of the CRAWLER oscillator when simulating offline.'},
            'feed_freq': {'v': 2.0, 'lim': (0.5, 4.0), 'dv': 0.01, 'k': 'f_F', 'u': reg.units.Hz,
                          'symbol': sub('f', 'F'),
                          'disp': 'feeding frequency',
                          'h': 'The default frequency of the FEEDER oscillator when simulating offline.'},
            'feeder_reoccurence_rate': {'lim': (0.0, 1.0), 'disp': 'feed reoccurence', 'symbol': sub('r', 'F'),
                                        'h': 'The default reoccurence rate of the feeding motion.'}

        }

        d['locomotor'] = {
            'modules': {'turner': bT,
                        'crawler': bT,
                        'interference': bT,
                        'intermitter': bT,
                        'feeder': bF},
            **{f'{m}_params': d[m] for m in ['crawler', 'turner', 'interference', 'intermitter', 'feeder']}
        }

        d['brain'] = {
            'modules': d['modules'],
            **{f'{m}_params': d[m] for m in d['modules'].keys()},
            'nengo': {**bF, 'k': 'nengo'}
        }

        d['gut'] = {
            'M_gm': {'v': 10 ** -2, 'lim': (0.0, 0.1), 'disp': 'gut scaled capacity',
                     'symbol': subsup('M', 'm', 'gut'),
                     'k': 'gut_M_m',
                     'h': 'Gut capacity in C-moles per unit of gut volume.'},
            'y_P_X': {'v': 0.9, 'lim': (0.0, 1.0), 'disp': 'food->product yield',
                      'symbol': sub('y', 'PX'), 'k': 'y_P_X',
                      'h': 'Yield of product per unit of food.'},
            'J_g_per_cm2': {'v': 10 ** -2 / (24 * 60 * 60), 'lim': (0.0, 0.1),
                            'disp': 'digestion secretion rate',
                            'symbol': subsup('J', 'g', 'gut'), 'k': 'gut_J_g',
                            'h': 'Secretion rate of enzyme per unit of gut surface per second.'},
            'k_g': {'v': 1.0, 'lim': (0.0, 1.0), 'disp': 'digestion decay rate',
                    'symbol': subsup('k', 'g', 'gut'),
                    'k': 'gut_k_g',
                    'h': 'Decay rate of digestive enzyme.'},
            'k_dig': {'v': 1.0, 'lim': (0.0, 1.0), 'disp': 'digestion rate',
                      'symbol': subsup('k', 'dig', 'gut'),
                      'k': 'gut_k_dig',
                      'h': 'Rate constant for digestion : k_X * y_Xg.'},
            'f_dig': {'v': 1.0, 'lim': (0.0, 1.0), 'disp': 'digestion response',
                      'symbol': subsup('f', 'dig', 'gut'), 'k': 'gut_f_dig',
                      'h': 'Scaled functional response for digestion : M_X/(M_X+M_K_X)'},
            'M_c_per_cm2': {'v': 5 * 10 ** -8, 'lim': (0.0, 0.1), 'disp': 'carrier density',
                            'symbol': subsup('M', 'c', 'gut'), 'k': 'gut_M_c',
                            'h': 'Area specific amount of carriers in the gut per unit of gut surface.'},
            'constant_M_c': {**bT, 'disp': 'constant carrier density', 'symbol': subsup('M_c', 'con', 'gut'),
                             'k': 'gut_M_c_con',
                             'h': 'Whether to assume a constant amount of carrier enzymes on the gut surface.'},
            'k_c': {'v': 1.0, 'lim': (0.0, 1.0), 'disp': 'carrier release rate',
                    'symbol': subsup('k', 'c', 'gut'),
                    'k': 'gut_k_c',
                    'h': 'Release rate of carrier enzymes.'},
            'k_abs': {'v': 1.0, 'lim': (0.0, 1.0), 'disp': 'absorption rate',
                      'symbol': subsup('k', 'abs', 'gut'),
                      'k': 'gut_k_abs',
                      'h': 'Rate constant for absorption : k_P * y_Pc.'},
            'f_abs': {'v': 1.0, 'lim': (0.0, 1.0), 'disp': 'absorption response',
                      'symbol': subsup('f', 'abs', 'gut'), 'k': 'gut_f_abs',
                      'h': 'Scaled functional response for absorption : M_P/(M_P+M_K_P)'},
        }

        d['DEB'] = {
            'species': {'dtype': str, 'v': 'default', 'vs': ['default', 'rover', 'sitter'], 'disp': 'phenotype',
                        'k': 'species',
                        'h': 'The phenotype/species-specific fitted DEB model to use.'},
            'f_decay': {'v': 0.1, 'lim': (0.0, 2.0), 'dv': 0.1, 'symbol': sub('c', 'DEB'), 'k': 'c_DEB',
                        'label': 'DEB functional response decay coef',
                        'h': 'The exponential decay coefficient of the DEB functional response.'},
            'V_bite': {'v': 0.0005, 'lim': (0.0, 0.01), 'dv': 0.0001,
                       'symbol': sub('V', 'bite'),
                       'k': 'V_bite',
                       'h': 'The volume of food consumed on a single feeding motion as a fraction of the body volume.'},
            'hunger_as_EEB': {**bT,
                              'h': 'Whether the DEB-generated hunger drive informs the exploration-exploitation balance.',
                              'symbol': 'H_as_EEB', 'k': 'H_as_EEB'},
            'hunger_gain': {'v': 0.0, 'lim': (0.0, 1.0), 'symbol': sub('G', 'H'),
                            'k': 'G_H', 'label': 'hunger sensitivity to reserve reduction',
                            'h': 'The sensitivy of the hunger drive in deviations of the DEB reserve density.'},
            'assimilation_mode': {'dtype': str, 'v': 'gut', 'vs': ['sim', 'gut', 'deb'],
                                  'symbol': sub('m', 'ass'), 'k': 'ass_mod',
                                  'h': 'The method used to calculate the DEB assimilation energy flow.'},
            'DEB_dt': {'lim': (0.0, 1000.0), 'disp': 'DEB timestep (sec)',
                       'symbol': sub('dt', 'DEB'),
                       'k': 'DEB_dt',
                       'h': 'The timestep of the DEB energetics module in seconds.'},
        }

        d['energetics'] = {
            'DEB': d['DEB'],
            'gut': d['gut']
        }
        d['obstacle_avoidance'] = {
            'sensor_delta_direction': {'v': 0.4, 'dv': 0.01, 'lim': (0.2, 1.2),
                                       'h': 'Sensor delta_direction'},
            'sensor_saturation_value': {'dtype': int, 'v': 40, 'lim': (0, 200),
                                        'h': 'Sensor saturation value'},
            'obstacle_sensor_error': {'v': 0.35, 'dv': 0.01, 'lim': (0.0, 1.0),
                                      'h': 'Proximity sensor error'},
            'sensor_max_distance': {'v': 0.9, 'dv': 0.01, 'lim': (0.1, 1.5),
                                    'h': 'Sensor max_distance'},
            'motor_ctrl_coefficient': {'dtype': int, 'v': 8770, 'lim': (0, 10000),
                                       'h': 'Motor ctrl_coefficient'},
            'motor_ctrl_min_actuator_value': {'dtype': int, 'v': 35, 'lim': (0, 50),
                                              'h': 'Motor ctrl_min_actuator_value'},
        }
        d['Model'] = {
            'brain': d['brain'],
            'body': d['body'],
            'energetics': d['energetics'],
            'physics': d['physics'],
            'Box2D_params': d['Box2D_params'],
        }


        return d

    def Ga0(d):
        d['ga_select_kws'] = {
            'Nagents': {'dtype': int, 'v': 30, 'lim': (2, 1000),
                        'h': 'Number of agents per generation', 'k': 'N'},
            'Nelits': {'dtype': int, 'v': 3, 'lim': (0, 1000),
                       'h': 'Number of elite agents preserved per generation', 'k': 'Nel'},
            'Ngenerations': {'dtype': int, 'lim': (0, 1000), 'h': 'Number of generations to exec',
                             'k': 'Ngen'},
            'Pmutation': {'v': 0.3, 'lim': (0.0, 1.0), 'h': 'Probability of genome mutation',
                          'k': 'Pmut'},
            'Cmutation': {'v': 0.1, 'lim': (0.0, 1.0), 'h': 'Mutation coefficient', 'k': 'Cmut'},
            'selection_ratio': {'v': 0.3, 'lim': (0.0, 1.0),
                                'h': 'Fraction of agents to be selected for the next generation', 'k': 'Rsel'},
            'verbose': {'dtype': int, 'v': 0, 'vs': [0, 1, 2, 3],
                        'h': 'Verbose argument for GA launcher', 'k': 'verb'}
        }

        d['ga_build_kws0'] = {
            # 'space_mkeys': {'dtype': List[str], 'h': 'The module keys to optimize'},
            # 'robot_class': {'v': 'LarvaRobot', 'dtype': str, 'vs': ['LarvaRobot', 'LarvaOffline'],
            #                 'h': 'The agent class to use in the simulations'},
            'base_model': ConfID_entry('Model', default='RE_NEU_PHI_DEF_nav', k='mID0', symbol=sub('mID', 0)),
            'bestConfID': {'dtype': str,
                           'h': 'The model configuration ID to store the best genome',
                           'k': 'mID1'},
            'init_mode': {'dtype': str, 'v': 'random', 'vs': ['default', 'random', 'model'],
                          'h': 'The initialization mode for the first generation', 'k': 'mGA'},
            # 'exclusion_mode': {**bF, 'h': 'Whether to use exclusion_mode', 'k': 'exclusion_mode'},
            # 'multicore': {**bF, 'h': 'Whether to use multiple cores', 'k': 'multicore'},
            'fitness_target_refID': ConfID_entry('Ref'),
            # 'fitness_target_kws': {'dtype': dict, 'v': {},
            #                        'h': 'The target data to derive from the reference dataset for evaluation'},
            # 'fitness_func': {'dtype': FunctionType, 'h': 'The method for fitness evaluation'},
            # 'exclude_func': {'dtype': FunctionType,
            #                  'h': 'The method for real-time excluding agents'},
        }

        d['ga_build_kws'] = {
            **d['ga_build_kws0'],
            'space_mkeys': {'dtype': List[str], 'h': 'The module keys to optimize'},
            'robot_class': {'v':'LarvaRobot', 'dtype': str,'vs': ['LarvaRobot', 'LarvaOffline'], 'h': 'The agent class to use in the simulations'},
            # 'base_model': ConfID_entry('Model', default='RE_NEU_PHI_DEF_nav', k='mID0', symbol=sub('mID', 0)),
            # 'bestConfID': {'dtype': str,
            #                'h': 'The model configuration ID to store the best genome',
            #                'k': 'mID1'},
            # 'init_mode': {'dtype': str, 'v': 'random', 'vs': ['default', 'random', 'model'],
            #               'h': 'The initialization mode for the first generation', 'k': 'mGA'},
            'exclusion_mode': {**bF, 'h': 'Whether to use exclusion_mode', 'k': 'exclusion_mode'},
            'multicore': {**bF, 'h': 'Whether to use multiple cores', 'k': 'multicore'},
            # 'fitness_target_refID': ConfID_entry('Ref'),
            'fitness_target_kws': {'dtype': dict, 'v': {},
                                   'h': 'The target data to derive from the reference dataset for evaluation'},
            'fitness_func': {'dtype': FunctionType, 'h': 'The method for fitness evaluation'},
            'exclude_func': {'dtype': FunctionType,
                             'h': 'The method for real-time excluding agents'},
        }
        return d

    def Ga1(d):
        d['Ga'] = {
            'scene': {'dtype': str, 'v': 'no_boxes', 'h': 'The name of the scene to load'},
            # 'scene_speed': {'dtype': int, 'v': 0, 'lim': (0, 1000),
            #                 'h': 'The rendering speed of the scene'},
            'env_params': ConfID_entry('Env',default='arena_200mm'),

            # 'env_params': confID_entry('Env'),
            'sim_params': d['sim_params'],
            'experiment': ConfID_entry('Ga',default='exploration'),
            # 'experiment': confID_entry('Ga', default='exploration'),
            # 'caption': {'dtype': str, 'h': 'The screen caption'},
            # 'save_to': pSaveTo(),
            'show_screen': {**bT, 'h': 'Whether to render the screen visualization', 'k': 'hide'},
            'offline': {**bF, 'h': 'Whether to exec a full LarvaworldSim environment', 'k': 'offline'},
            'ga_build_kws': d['ga_build_kws'],
            'ga_select_kws': d['ga_select_kws'],
            # 'ga_kws': {**d['GAengine'], **d['GAselector']},
        }

        return d

    def batch(d):
        d0 = aux.AttrDict({
            'optimization': {
                'fit_par': {'dtype': str, 'disp': 'Utility metric', 'h': 'The utility parameter optimized.'},
                'minimize': {**bT, 'h': 'Whether to minimize or maximize the utility parameter.'},
                'threshold': {'v': 0.001, 'lim': (0.0, 0.01), 'dv': 0.0001,
                              'h': 'The utility threshold to reach before terminating the batch-exec.'},
                'max_Nsims': {'dtype': int, 'v': 7, 'lim': (0, 100),
                              'h': 'The maximum number of single runs before terminating the batch-exec.'},
                'Nbest': {'dtype': int, 'v': 3, 'lim': (0, 20),
                          'h': 'The number of best parameter combinations to use for generating the next generation.'},
                'operations': {
                    'mean': {**bT, 'h': 'Whether to use the mean of the utility across individuals'},
                    'std': {**bF,
                            'h': 'Whether to use the standard deviation of the utility across individuals'},
                    'abs': {**bF, 'h': 'Whether to use the absolute value of the utility'}
                },
            },
            # 'batch_methods': {
            #     'exec': {'dtype': str, 'v': 'default',
            #             'vs': ['null', 'default', 'deb', 'odor_preference', 'exp_fit'],
            #             'h': 'The method to be applied on simulated data derived from every individual exec'},
            #     'post': {'dtype': str, 'v': 'default', 'vs': ['null', 'default'],
            #              'h': 'The method to be applied after a generation of runs is completed to judge whether space-search will continue or batch-exec will be terminated.'},
            #     'final': {'dtype': str, 'v': 'null',
            #               'vs': ['null', 'scatterplots', 'deb', 'odor_preference'],
            #               'h': 'The method to be applied once the batch-exec is complete to plot/save the results.'}
            # },
            'space_search_par': {
                'range': {'dtype': Tuple[float], 'lim': (-100.0, 100.0), 'dv': 1.0,
                          'k': 'ss.range',
                          'h': 'The parameter range to perform the space-search.'},
                'Ngrid': {'dtype': int, 'lim': (0, 100), 'disp': '# steps', 'k': 'ss.Ngrid',
                          'h': 'The number of equally-distanced values to parse the parameter range.'},
                'values': {'dtype': List[float], 'lim': (-100.0, 100.0), 'k': 'ss.vs',
                           'h': 'A list of values of the parameter to space-search. Once this is filled no range/# steps parameters are taken into account.'}
            },
            'space_search': {
                'pars': {'dtype': List[str], 'h': 'The parameters for space search.', 'k': 'ss.pars'},
                'ranges': {'dtype': List[Tuple[float]], 'lim': (-100.0, 100.0), 'dv': 1.0,
                           'h': 'The range of the parameters for space search.', 'k': 'ss.ranges'},
                'Ngrid': {'dtype': int, 'lim': (0, 100), 'h': 'The number of steps for space search.',
                          'k': 'ss.Ngrid'}},

        })
        d.update(d0)
        d['Batch'] = {'exp': {'dtype': str},
                           'space_search': d['space_search'],
                           'optimization': d['optimization'],
                           'exp_kws': {'dtype': dict, 'v': {'enrichment': d['enrichment']},
                                       'h': 'Keywords for the exp exec.'},
                           }
        return d


    def conftypes(d):
        d['Body'] = {
            'symmetry': {'dtype': str, 'v': 'bilateral', 'vs': ['bilateral', 'radial'],
                         'h': 'The body symmetry.'},
            'Nsegs': {'dtype': int, 'v': 2, 'lim': (1, 12),
                      'h': 'The number of segments comprising the larva body.'},
            'seg_ratio': {'lim': (0.0, 1.0),
                          'h': 'The length ratio of the body segments. If null, equal-length segments are generated.'},

            'olfaction_sensors': {'dtype': List[int], 'lim': (0, 16), 'v': [0], 'disp': 'olfaction',
                                  'h': 'The indexes of the contour points bearing olfaction sensors.'},

            'touch_sensors': {'dtype': List[int], 'lim': (0, 16), 'disp': 'touch',
                              'h': 'The indexes of the contour points bearing touch sensors.'},
            'points': pXYs('body contour', lim=(-1.0, 1.0), disp='contour')
        }
        d['Tracker'] = {
            'resolution': {
                'fr': {'v': 10.0, 'lim': (0.0, 100.0), 'disp': 'framerate (Hz)',
                       'h': 'The framerate of the tracker recordings.'},
                'Npoints': {'dtype': int, 'v': 1, 'lim': (0, 20), 'disp': '# midline xy',
                            'h': 'The number of points tracked along the larva midline.'},
                'Ncontour': {'dtype': int, 'v': 0, 'lim': (0, 100), 'disp': '# contour xy',
                             'h': 'The number of points tracked around the larva contour.'}
            },
            'arena': d['arena'],
            'filesystem': {
                'read_sequence': {'dtype': List[str], 'disp': 'columns',
                                  'h': 'The sequence of columns in the tracker-exported files.'},
                'read_metadata': {**bF, 'disp': 'metadata',
                                  'h': 'Whether metadata files are available for the tracker-exported files/folders.'},
                'folder': {
                    'pref': {'dtype': str, 'h': 'A prefix for detecting a raw-data folder.'},
                    'suf': {'dtype': str, 'h': 'A suffix for detecting a raw-data folder.'}},
                'file': {'pref': {'dtype': str, 'h': 'A prefix for detecting a raw-data file.'},
                         'suf': {'dtype': str, 'h': 'A suffix for detecting a raw-data file.'},
                         'sep': {'dtype': str, 'h': 'A separator for detecting a raw-data file.'}}
            },

        }

        d['Group'] = {
            'path': pPath('Group'),
            'tracker': d['Tracker'],
            'enrichment': d['enrichment'],
        }
        d['Env'] = {'arena': d['arena'],
                         'border_list': {'dtype': dict, 'v': {}},
                         'food_params': d['food_params'],
                         'odorscape': {'dtype': dict},
                         'windscape': {'dtype': dict},
                         'thermoscape': {'dtype': dict},
                         }

        d['Exp'] = {
            'env_params': ConfID_entry('Env'),
            'larva_groups': {'dtype': dict, 'v': {}},
            'sim_params': d['sim_params'],
            'trials': ConfID_entry('Trial', default='default'),
            'collections': {'dtype': List[str], 'v': ['pose']},
            'enrichment': d['enrichment'],
            'experiment': ConfID_entry('Exp'),
            'Box2D': {**bF, 'h': 'Whether to use the Box2D physics engine or not.', 'k': 'Box2D'},
        }

        d['Replay'] = {
            'env_params': ConfID_entry('Env'),
            # 'env_params': confID_entry('Env'),
            'transposition': {'dtype': str, 'vs': [None, 'origin', 'arena', 'center'],
                              'symbol': sub('mod', 'trans'), 'k': 'trans',
                              'h': 'Whether to transpose the dataset spatial coordinates.'},
            'agent_ids': {'dtype': List[int], 'symbol': 'ids', 'k': 'ids',
                          'h': 'Whether to only display some larvae of the dataset, defined by their indexes.'},
            'dynamic_color': {'dtype': str, 'vs': [None, 'lin_color', 'ang_color'], 'symbol': sub('color', 'dyn'),
                              'k': 'dyn_col',
                              'h': 'Whether to display larva tracks according to the instantaneous forward or angular velocity.'},
            'time_range': {'dtype': Tuple[float], 'lim': (0.0, 1000.0), 'dv': 1.0, 'symbol': sub('t', 'range'),
                           'k': 't_range',
                           'h': 'Whether to only replay a defined temporal slice of the dataset.'},
            'track_point': {'dtype': int, 'lim': (-1, 12), 'symbol': sub('p', 'track'), 'k': 'track_p',
                            'h': 'The midline point to use for defining the larva position.'},
            'draw_Nsegs': {'dtype': int, 'lim': (1, 12), 'symbol': subsup('N', 'segs', 'draw'), 'k': 'Nsegs',
                           'h': 'Whether to artificially simplify the experimentally tracked larva body to a segmented virtual body of the given number of segments.'},
            'fix_point': {'dtype': int, 'lim': (1, 12), 'symbol': sub('fix', 'p'), 'k': 'fix_p',
                          'h': 'Whether to fixate a specific midline point to the center of the screen. Relevant when replaying a single larva track.'},
            'fix_segment': {'dtype': int, 'vs': [-1, 1], 'symbol': sub('fix', 'seg'), 'k': 'fix_seg',
                            'h': 'Whether to additionally fixate the above or below body segment.'},
            'close_view': {**bF, 'symbol': sub('view', 'close'), 'k': 'vis0',
                           'h': 'Whether to visualize a small arena.'},
            'overlap_mode': {**bF, 'symbol': sub('mod', 'overlap'), 'k': 'overlap',
                             'h': 'Whether to draw overlapped image of the track.'},
            'refID': ConfID_entry('Ref'),
            'dir': {'dtype': str, 'symbol': 'dir',
                              'k': 'dir',
                              'h': 'The path to the stored dataset relative to Root/data. Alternative to providing refID'}
        }

        d['Eval'] = {
            'refID': ConfID_entry('Ref', default='None.150controls'),
            'modelIDs': ConfID_entry('Model', single_choice=False, k='mIDs'),
            'dataset_ids': {'dtype': List[str], 'h': 'The ids for the generated datasets', 'k': 'dIDs'},
            'offline': {**bF, 'h': 'Whether to exec a full LarvaworldSim environment', 'k': 'offline'},
            'N': {'dtype': int, 'v': 5, 'lim': (2, 1000),
                  'h': 'Number of agents per model ID',
                  'k': 'N'},
            # 'id': pID('evaluation exec', k='id'),

        }

        d['ModelGroup'] = {
            'ModelGroupID': ConfID_entry('ModelGroup'),
            'model families': ConfID_entry('Model', single_choice=False)
        }
        d['ExpGroup'] = {
            'ExpGroupID': ConfID_entry('ExpGroup'),
            'simulations': ConfID_entry('Exp', single_choice=False)
        }

        return aux.AttrDict(d)

    def larvaGroup(d):
        d['LarvaGroup'] = {
             'model': ConfID_entry('Model', default='explorer'),
            'sample': {'dtype': str, 'v': 'None.150controls', 'h': 'The reference dataset to sample from.'},
            'default_color': pCol('black', 'larva group'),
            'imitation': {**bF, 'h': 'Whether to imitate the reference dataset.'},
            'distribution': d['larva_distro'],
            'life_history': d['Life'],
            'odor': d['odor']
        }

        return d


    dic = {
        'vis': init_vis,
        'xy': xy_distros,
        'substrate': substrate,
        'scape': scapeConfs,
        'exec': runConfs,
        'enrich': enrConfs,
        'model': init_mods,
    }
    dic0={}
    d = {}
    for k,f in dic.items() :
        dic0[k]=f()
        d.update(dic0[k])

    for f in [food,life,larvaGroup,batch,conftypes,Ga0,Ga1]:
        dic0 = f(d)
        d.update(dic0)
    return aux.AttrDict(d)

def buildDefaultDict(d0):
    dic = {}
    for name, d in d0.items():
        dic[name] = get_default(d, key='v')
    return aux.AttrDict(dic)


@decorators.timeit
class ParamClass:
    def __init__(self,func_dict,in_rad=True, in_m=True):
        self.func_dict = func_dict
        self.dict_entries = self.build(in_rad=in_rad, in_m=in_m)

        self.kdict = self.finalize_dict(self.dict_entries)
        # self.ddict = aux.AttrDict({p.d: p for k, p in self.kdict.items()})
        # self.pdict = aux.AttrDict({p.p: p for k, p in self.kdict.items()})


    @decorators.timeit
    def build(self, in_rad=True, in_m=True):
        self.dict = aux.AttrDict()
        self.dict_entries = []
        self.build_initial()
        self.build_angular(in_rad)
        self.build_spatial(in_m)
        self.build_chunks()
        self.build_sim_pars()
        self.build_deb_pars()
        return self.dict_entries

    def build_initial(self):
        kws = {'u': reg.units.s}
        self.add(
            **{'p': 'model.dt', 'k': 'dt', 'd': 'timestep', 'sym': '$dt$', 'lim': (0.01, 0.5), 'dv': 0.01, 'v0': 0.1,
               **kws})
        self.add(
            **{'p': 'cum_dur', 'k': nam.cum('t'), 'sym': sub('t', 'cum'), 'lim': (0.0, None), 'dv': 0.1, 'v0': 0.0,
               **kws})
        self.add(
            **{'p': 'num_ticks', 'k': 'N_ticks', 'sym': sub('N', 'ticks'), 'dtype': int, 'lim': (0, None), 'dv': 1})

    def add(self, **kwargs):
        prepar = util.preparePar(**kwargs)
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

        kws = {
            'p': p,
            'k': k,
            'd': d,
            'sym': sym,
            'u': b_num.u / b_den.u,
            'required_ks': [k_num, k_den],
        }
        kws.update(kwargs)
        self.add(**kws)

    def add_operators(self, k0):
        b = self.dict[k0]
        kws0 = {'u': b.u, 'required_ks': [k0]}

        funcs = self.func_dict

        mu_kws = {'d': nam.mean(b.d), 'p': nam.mean(b.p), 'sym': bar(b.sym), 'disp': f'mean {b.disp}',
                  'func': funcs.mean(b.d), 'k': f'{b.k}_mu'}

        std_kws = {'d': nam.std(b.d), 'p': nam.std(b.p), 'sym': wave(b.sym), 'disp': f'std {b.disp}',
                   'func': funcs.std(b.d),
                   'k': f'{b.k}_std'}

        var_kws = {'d': nam.var(b.d), 'p': nam.var(b.p), 'sym': wave(b.sym), 'disp': f'var {b.disp}',
                   'func': funcs.var(b.d),
                   'k': f'{b.k}_var'}

        min_kws = {'d': nam.min(b.d), 'p': nam.min(b.p), 'sym': sub(b.sym, 'min'), 'disp': f'minimum {b.disp}',
                   'func': funcs.min(b.d), 'k': f'{b.k}_min'}

        max_kws = {'d': nam.max(b.d), 'p': nam.max(b.p), 'sym': sub(b.sym, 'max'), 'disp': f'maximum {b.disp}',
                   'func': funcs.max(b.d), 'k': f'{b.k}_max'}

        fin_kws = {'d': nam.final(b.d), 'p': nam.final(b.p), 'sym': sub(b.sym, 'fin'), 'disp': f'final {b.disp}',
                   'func': funcs.final(b.d), 'k': f'{b.k}_fin'}

        init_kws = {'d': nam.initial(b.d), 'p': nam.initial(b.p), 'sym': sub(b.sym, '0'), 'disp': f'initial {b.disp}',
                    'func': funcs.initial(b.d), 'k': f'{b.k}0'}

        if k0 == 'd':
            disp = 'pathlength'
        elif k0 == 'sd':
            disp = 'scaled pathlength'
        else:
            disp = f'total {b.disp}'
        cum_kws = {'d': nam.cum(b.d), 'p': nam.cum(b.p), 'sym': sub(b.sym, 'cum'), 'disp': disp,
                   'func': funcs.cum(b.d), 'k': nam.cum(b.k)}

        for kws in [mu_kws, std_kws,var_kws, min_kws, max_kws, fin_kws, init_kws, cum_kws]:
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
                'u': reg.units.s,
                'sym': subsup('t', kc, 0),
                'disp': f'{pc} start',
                **f_kws
            },
            {'p': nam.stop(pc),
             'k': f'{kc}1',
             'u': reg.units.s,
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
             'disp': f'time fraction in {pc}s',
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
                'u': reg.units.s,
                **f_kws
            }]

        for kws in kwlist:
            self.add(**kws)

        for ii in ['on', 'off']:
            self.add(**{'p': f'{pN_mu}_{ii}_food', 'k': f'{kN_mu}_{ii}_food'})
            self.add(**{'p': f'{ptr}_{ii}_food', 'k': f'{ktr}_{ii}_food', 'lim': (0.0, 1.0)})

        self.add_rate(k_num=kN, k_den=nam.cum('t'), k=kN_mu, p=pN_mu, sym=bar(kN), disp=f'avg. # {pc}s per sec',
                      func=func)
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
                s, e, c = d.step_data, d.endpoint_data, d.config
                s[d_v]=aux.apply_per_level(s[b.d], aux.rate, dt=c.dt).flatten()
                # s[d_v]=aux.comp_rate(s[b.d], c.dt)

        self.add(
            **{'p': p_v, 'k': k_v, 'd': d_v, 'u': b.u / b_dt.u, 'sym': sym_v, 'disp': disp_v, 'required_ks': [k0],
               'func': func_v})

        def func_a(d):
            s, e, c = d.step_data, d.endpoint_data, d.config
            s[d_a] = aux.apply_per_level(s[d_v], aux.rate, dt=c.dt).flatten()
            # s[d_a]=aux.comp_rate(s[d_v], c.dt)

        self.add(
            **{'p': p_a, 'k': k_a, 'd': d_a, 'u': b.u / b_dt.u ** 2, 'sym': sym_a, 'disp': disp_a, 'required_ks': [k_v],
               'func': func_a})

    def add_scaled(self, k0, **kwargs):
        b = self.dict[k0]
        b_l = self.dict['l']

        def func(d):
            from larvaworld.lib.process.spatial import scale_to_length
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
        if b.u == reg.units.deg:
            in_deg = True
        elif b.u == reg.units.rad:
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
            'u': reg.units.Hz,
            'sym': sub(b.sym, 'freq'),
            'disp': f'{b.disp} frequency',
            # 'disp': f'{b.disp} dominant frequency',
            'required_ks': [k0],
            'func': self.func_dict.freq(b.d)
        }
        kws.update(kwargs)
        self.add(**kws)

    def add_dsp(self, range=(0, 40), u=reg.units.m):
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
            u = reg.units.rad
            amax = np.pi
        else:
            u = reg.units.deg
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
                             sym_v=omega(ksuf), sym_a=dot(omega(ksuf)), disp_v=f'{lsuf}angular speed',
                             disp_a=f'{lsuf}angular acceleration')
        for k0 in ['b', 'bv', 'ba', 'fov', 'foa', 'rov', 'roa', 'fo', 'ro', 'ho', 'to']:
            self.add_freq(k0=k0)
            self.add_operators(k0=k0)

    def build_spatial(self, in_m=True):
        tor_durs = [1, 2, 5, 10, 20, 60, 120, 240, 300, 600]
        dsp_ranges = [(0, 40), (0, 60), (20, 80), (0, 120), (0, 240), (0, 300), (0, 600), (60, 120), (60, 300)]
        if in_m:
            u = reg.units.m
            s = 1
        else:
            u = reg.units.mm
            s = 1000

        kws = {'u': u}
        self.add(**{'p': 'x', 'disp': 'X position', 'sym': 'X', **kws})
        self.add(**{'p': 'y', 'disp': 'Y position', 'sym': 'Y', **kws})
        self.add(
            **{'p': 'real_length', 'k': 'l', 'd': 'length', 'disp': 'body length',
               'sym': '$l$', 'v0': 0.004 * s, 'lim': (0.0005 * s, 0.01 * s), 'dv': 0.0005 * s, **kws})

        self.add(
            **{'p': 'dispersion', 'k': 'dsp', 'sym': circledast('d'), 'disp': 'dispersal', **kws})

        d_d, d_v, d_a = nam.dst(''), nam.vel(''), nam.acc('')
        d_sd, d_sv, d_sa = nam.scal([d_d, d_v, d_a])
        self.add_dst(point='')
        self.add_velNacc(k0='d', k_v='v', k_a='a', p_v=d_v, d_v=d_v, p_a=d_a, d_a=d_a,
                         sym_v='v', sym_a=dot('v'), disp_v='crawling speed', disp_a='crawling acceleration',
                         func_v=self.func_dict.vel(d_d, d_v))
        for k0 in ['x', 'y', 'd']:
            self.add_scaled(k0=k0)
        self.add_velNacc(k0='sd', k_v='sv', k_a='sa', p_v=d_sv, d_v=d_sv, p_a=d_sa, d_a=d_sa, sym_v=mathring('v'),
                         sym_a=dot(mathring('v')), disp_v='scaled crawling speed',
                         disp_a='scaled crawling acceleration',
                         func_v=self.func_dict.vel(d_sd, d_sv))
        for k0 in ['l', 'd', 'sd', 'v', 'sv', 'a', 'sa', 'x', 'y']:
            self.add_freq(k0=k0)
            self.add_operators(k0=k0)
        for k0 in [nam.cum('d')]:
            self.add_scaled(k0=k0)

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
            'run': 'run',
            'fee': 'feed',
            'tur': 'turn',
            'Ltur': 'Lturn',
            'Rtur': 'Rturn',
            'exec': 'exec',
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
            self.add(**{'p': f'handedness_score_{kc}', 'k': f'tur_H_{kc}'})

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
            sym = subsup('C', 'odor', ii)
            dsym = subsup(delta('C'), 'odor', ii)
            ddisp = f'{sym} sensed (C/{sub("C", 0)} - 1)'
            self.add(**{'p': f'brain.olfactor.{jj}_odor_concentration', 'k': k, 'd': k,
                        'disp': sym, 'sym': sym, 'u': reg.units.micromol})
            self.add(**{'p': f'brain.olfactor.{jj}_odor_concentration_change', 'k': dk, 'd': dk,
                        'disp': ddisp, 'sym': dsym})

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

        self.add_rate(k_num='Ltur_N', k_den='tur_N', k='tur_H', p='handedness_score',
                      disp=f'handedness score ({sub("N", "Lturns")} / {sub("N", "turns")})',
                      sym=sub('H', 'tur'), lim=(0.0, 1.0))
        for ii in ['on', 'off']:
            k = f'{ii}_food'
            self.add(**{'p': k, 'k': k, 'dtype': bool})
            self.add(**{'p': nam.dur(k), 'k': f'{k}_t', 'disp': f'time {ii} food'})
            self.add(**{'p': nam.cum(nam.dur(k)), 'k': nam.cum(f'{k}_t'), 'disp': f'total time {ii} food'})
            self.add(**{'p': nam.dur_ratio(k), 'k': f'{k}_tr', 'lim': (0.0, 1.0), 'disp': f'time fraction {ii} food'})
            self.add(**{'p': f'handedness_score_{k}', 'k': f'tur_H_{k}', 'disp': f'handedness score {ii} food'})
            for kk in ['fov', 'rov', 'foa', 'roa', 'x', 'y', 'fo', 'fou', 'ro', 'rou', 'b', 'bv', 'ba', 'v', 'sv', 'a',
                       'v_mu', 'sv_mu',
                       'sa', 'd', 'sd']:
                b = self.dict[kk]
                k0 = f'{kk}_{k}'
                p0 = f'{b.p}_{k}'
                self.add(**{'p': p0, 'k': k0, 'disp': f'{b.disp} {ii} food'})

    def build_deb_pars(self):
        ks = ['f_am', 'sf_am_Vg', 'f_am_V', 'sf_am_V', 'sf_am_A', 'sf_am_M']
        ps = ['amount_eaten', 'deb.ingested_gut_volume_ratio', 'deb.volume_ingested', 'deb.ingested_body_volume_ratio',
              'deb.ingested_body_area_ratio', 'deb.ingested_body_mass_ratio']
        ds = ['amount_eaten', 'ingested_gut_volume_ratio', 'ingested_volume', 'ingested_body_volume_ratio',
              'ingested_body_area_ratio', 'ingested_body_mass_ratio']
        disps = ['food consumed', 'ingested food as gut volume fraction', 'ingested food volume',
                 'ingested food as body volume fraction', 'ingested food as body area fraction',
                 'ingested food as body mass fraction']
        for k, p, d, disp in zip(ks, ps, ds, disps):
            self.add(**{'p': p, 'k': k, 'd': d, 'disp': disp})

    @decorators.timeit
    def finalize_dict(self, entries):
        dic = aux.AttrDict()
        for prepar in entries:
            p = util.v_descriptor(**prepar)
            dic[p.k] = p
        return dic

@decorators.timeit
class ParamRegistry:
    def __init__(self):
        self.PI = buildInitDict()
        self.DEF = buildDefaultDict(self.PI)

        self.dict = None


    def null(self,name, key='v', **kwargs):
        if key != 'v':
            raise
        d0=self.DEF[name]
        return d0.update_nestdict(kwargs)

    def get_null(self, name, key='v', **kwargs):
        if key != 'v':
            raise
        # return update_default(name, aux.copyDict(self.DEF[name]), **kwargs)
        return update_default(name, self.DEF[name].get_copy(), **kwargs)

    def metric_def(self, ang={}, sp={}, **kwargs):
        def ang_def(fv=(1, 2), rv=(-2, -1), **kwargs):
            return self.get_null('ang_definition',front_vector=fv, rear_vector=rv, **kwargs)

        return self.get_null('metric_definition',
                             angular=ang_def(**ang),
                             spatial=self.get_null('spatial_definition', **sp),
                             **kwargs)



    def enr_dict(self, proc=[], anot=[], pre_kws={},
                 def_kws={}, metric_definition=None, **kwargs):
        kw_dic0={
            'preprocessing' : pre_kws,
            'processing' : {k: True if k in proc else False for k in proc_type_keys},
            'annotation' : {k: True if k in anot else False for k in anot_type_keys}
                }
        kws={k:self.get_null(k,**v) for k,v in kw_dic0.items()}

        if metric_definition is None:
            metric_definition = self.metric_def(**def_kws)
        dic = self.get_null('enrichment',
                                      metric_definition=metric_definition, **kws, **kwargs)
        return dic

    def base_enrich(self, **kwargs):
        return self.enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
                             anot=['bout_detection', 'bout_distribution', 'interference'],
                             **kwargs)

    def get(self, k, d, compute=True):
        p = self.kdict[k]
        res = p.exists(d)

        if res['step']:
            if hasattr(d, 'step_data'):
                return d.step_data[p.d]
            else:
                return d.read(key='step')[p.d]
        elif res['end']:
            if hasattr(d, 'endpoint_data'):
                return d.endpoint_data[p.d]
            else:
                return d.read(key='end')[p.d]
        else:
            for key in res.keys():
                if key not in ['step', 'end'] and res[key]:
                    return d.read(key=f'{key}.{p.d}', file='aux')

        if compute:
            self.compute(k, d)
            return self.get(k, d, compute=False)
        else:
            print(f'Parameter {p.disp} not found')

    def compute(self, k, d):
        p = self.kdict[k]
        res = p.exists(d)
        if not any(list(res.values())):
            k0s = p.required_ks
            for k0 in k0s:
                self.compute(k0, d)
            p.compute(d)

    def getPar(self, k=None, p=None, d=None, to_return='d'):
        if k is not None:
            d0 = self.kdict
            k0 = k
        elif d is not None:
            d0 = aux.AttrDict({p.d: p for k, p in self.kdict.items()})
            k0 = d
        elif p is not None:
            d0 = aux.AttrDict({p.p: p for k, p in self.kdict.items()})
            k0 = p
        else :
            raise

        if type(k0) == str:
            par = d0[k0]
            if type(to_return) == list:
                return [getattr(par, i) for i in to_return]
            elif type(to_return) == str:
                return getattr(par, to_return)
        elif type(k0) == list:
            pars = [d0[i] for i in k0]
            if type(to_return) == list:
                return [[getattr(par, i) for par in pars] for i in to_return]
            elif type(to_return) == str:
                return [getattr(par, to_return) for par in pars]

    def runtime_pars(self):
        return [v.d for k, v in self.kdict.items()]

    def auto_load(self, ks, datasets):
        dic = {}
        for k in ks:
            dic[k] = {}
            for d in datasets:
                vs = self.get(k=k, d=d, compute=True)
                dic[k][d.id] = vs
        return aux.AttrDict(dic)

    @property
    def kdict(self):
        if self.dict is None :
            self.dict = ParamClass(func_dict=reg.funcs.param_computing).kdict
        return self.dict

par = ParamRegistry()
