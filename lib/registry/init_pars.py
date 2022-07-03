import param

from lib.aux.par_aux import subsup

proc_type_keys = ['angular', 'spatial', 'source', 'dispersion', 'tortuosity', 'PI', 'wind']
bout_keys = ['stride', 'pause', 'turn']
to_drop_keys = ['midline', 'contour', 'stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn', 'turn',
                'unused']

# Compound densities (g/cm**3)
substrate_dict = {
    'agar': {
        'glucose': 0,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 0,
        'agar': 16 / 1000,
        'cornmeal': 0
    },
    'standard': {  # w_X = 20.45 g/mol
        'glucose': 100 / 1000,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 50 / 1000,
        'agar': 16 / 1000,
        'cornmeal': 0
        # 'KPO4': 0.1/1000,
        # 'Na_K_tartrate': 8/1000,
        # 'NaCl': 0.5/1000,
        # 'MgCl2': 0.5/1000,
        # 'Fe2(SO4)3': 0.5/1000,
    },
    'cornmeal': {
        'glucose': 517 / 17000,
        'dextrose': 1033 / 17000,
        'saccharose': 0,
        'yeast': 0,
        'agar': 93 / 17000,
        'cornmeal': 1716 / 17000
    },
    'PED_tracker': {
        'glucose': 0,
        'dextrose': 0,
        'saccharose': 2 / 200,
        'yeast': 3 * 0.05 * 0.125 / 0.1,
        'agar': 500 * 2 / 200,
        'cornmeal': 0
    },
    #     [1] M. E. Wosniack, N. Hu, J. Gjorgjieva, and J. Berni, “Adaptation of Drosophila larva foraging in response to changes in food distribution,” bioRxiv, p. 2021.06.21.449222, 2021.
    'cornmeal2': {
        'glucose': 0,
        'dextrose': 450 / 6400,
        'saccharose': 0,
        'yeast': 90 / 6400,
        'agar': 42 / 6400,
        'cornmeal': 420 / 6400
    },
    'sucrose': {
        'glucose': 3.42 / 200,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 0,
        'agar': 0.8 / 200,
        'cornmeal': 0
    }
    # 'apple_juice': {
    #         'glucose': 0.342/200,
    #         'dextrose': 0,
    #         'saccharose': 0,
    #         'yeast': 0,
    #         'agar': 0.8 / 200,
    #         'cornmeal': 0,
    #         'apple_juice': 1.05*5/200,
    #     },

}

def ConfSelector(conf_type, default=None, single_choice=True, **kwargs):
    from lib.conf.stored.conf import kConfDict
    def func():

        kws = {
            'objects': kConfDict(conf_type),
            'default': default,
            'allow_None': True,
            'empty_default': True,
        }
        if single_choice:
            func0 = param.Selector
        else:
            func0 = param.ListSelector
        return func0(**kws, **kwargs)

    return func

def confID_entry(conftype, default=None, k=None, symbol=None, single_choice=True) :
    from typing import List
    from lib.conf.stored.conf import kConfDict
    from lib.aux.par_aux import sub
    from lib.aux import dictsNlists as dNl
    low = conftype.lower()

    if single_choice :
        t=str
        IDstr='ID'
    else :
        t=List[str]
        IDstr = 'IDs'
    if k is None :
        k = f'{low}{IDstr}'
    if symbol is None :
        symbol = sub(IDstr, low)
    d = {'t': t, 'vparfunc': ConfSelector(conftype, default=default, single_choice=single_choice), 'vs': kConfDict(conftype), 'v': default,
           'symbol': symbol, 'k': k, 'h': f'The {conftype} configuration {IDstr}',
                                'disp': f'{conftype} {IDstr}'}
    return dNl.NestDict(d)


class ParInitDict:
    def __init__(self):
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        from typing import List, Tuple, TypedDict
        from types import FunctionType
        import numpy as np
        import param
        from lib.aux import dictsNlists as dNl
        from lib.aux.collecting import output_keys

        bF, bT = {'t': bool, 'v': False}, {'t': bool, 'v': True}

        def pCol(v, obj):
            return {'t': str, 'vfunc': param.Color, 'v': v, 'disp': 'color',
                    'h': f'The default color of the {obj}.'}

        def pPath(v=None, h=None, k=None):
            if h is None: h = f'The relative p ath to store the {v} datasets.'
            return {'t': str, 'h': h, 'k': k,
                    'vfunc': param.Foldername}

        def pID(v=None, h=None, k=None):
            if h is None:
                h = f'The unique ID   of the {v}.'
            return {'t': str, 'h': h, 'k': k}

        def init_vis():
            d = dNl.NestDict()
            d['render'] = {
                'mode': {'t': str, 'v': None, 'vs': [None, 'video', 'image'], 'h': 'The visualization mode', 'k': 'm'},
                'image_mode': {'t': str, 'vs': [None, 'final', 'snapshots', 'overlap'], 'h': 'The image-render mode',
                               'k': 'im'},
                'video_speed': {'t': int, 'v': 60, 'min': 1, 'max': 100, 'h': 'The video speed', 'k': 'fps'},
                'media_name': {'t': str,
                               'h': 'Filename for the saved video/image. File extension mp4/png sutomatically added.',
                               'k': 'media'},
                'show_display': {'t': bool, 'v': True, 'h': 'Hide display', 'k': 'hide'},
            }
            d['draw'] = {
                'draw_head': {'t': bool, 'v': False, 'h': 'Draw the larva head'},
                'draw_centroid': {'t': bool, 'v': False, 'h': 'Draw the larva centroid'},
                'draw_midline': {'t': bool, 'v': True, 'h': 'Draw the larva midline'},
                'draw_contour': {'t': bool, 'v': True, 'h': 'Draw the larva contour'},
                'draw_sensors': {'t': bool, 'v': False, 'h': 'Draw the larva sensors'},
                'trails': {'t': bool, 'v': False, 'h': 'Draw the larva trajectories'},
                'trajectory_dt': {'max': 100.0, 'h': 'Duration of the drawn trajectories'},
            }
            d['color'] = {
                'black_background': {'t': bool, 'v': False, 'h': 'Set the background color to black'},
                'random_colors': {'t': bool, 'v': False, 'h': 'Color each larva with a random color'},
                'color_behavior': {'t': bool, 'v': False,
                                   'h': 'Color the larvae according to their instantaneous behavior'},
            }
            d['aux'] = {
                'visible_clock': {'t': bool, 'v': True, 'h': 'Hide/show the simulation clock'},
                'visible_scale': {'t': bool, 'v': True, 'h': 'Hide/show the simulation scale'},
                'visible_state': {'t': bool, 'v': False, 'h': 'Hide/show the simulation state'},
                'visible_ids': {'t': bool, 'v': False, 'h': 'Hide/show the larva IDs'},
            }
            d['visualization'] = {
                'render': d['render'],
                'aux': d['aux'],
                'draw': d['draw'],
                'color': d['color'],

            }

            return d

        def init_batch_pars():
            d = dNl.NestDict({
                'optimization': {
                    'fit_par': {'t': str, 'disp': 'Utility metric', 'h': 'The utility parameter optimized.'},
                    'minimize': {**bT, 'h': 'Whether to minimize or maximize the utility parameter.'},
                    'threshold': {'v': 0.001, 'lim': (0.0, 0.01), 'dv': 0.0001,
                                  'h': 'The utility threshold to reach before terminating the batch-run.'},
                    'max_Nsims': {'t': int, 'v': 7, 'lim': (0, 100),
                                  'h': 'The maximum number of single runs before terminating the batch-run.'},
                    'Nbest': {'t': int, 'v': 3, 'lim': (0, 20),
                              'h': 'The number of best parameter combinations to use for generating the next generation.'},
                    'operations': {
                        'mean': {**bT, 'h': 'Whether to use the mean of the utility across individuals'},
                        'std': {**bF, 'h': 'Whether to use the standard deviation of the utility across individuals'},
                        'abs': {**bF, 'h': 'Whether to use the absolute value of the utility'}
                    },
                },
                'batch_methods': {
                    'run': {'t': str, 'v': 'default',
                            'vs': ['null', 'default', 'deb', 'odor_preference', 'exp_fit'],
                            'h': 'The method to be applied on simulated data derived from every individual run'},
                    'post': {'t': str, 'v': 'default', 'vs': ['null', 'default'],
                             'h': 'The method to be applied after a generation of runs is completed to judge whether space-search will continue or batch-run will be terminated.'},
                    'final': {'t': str, 'v': 'null',
                              'vs': ['null', 'scatterplots', 'deb', 'odor_preference'],
                              'h': 'The method to be applied once the batch-run is complete to plot/save the results.'}
                },
                'space_search_par': {
                    'range': {'t': Tuple[float], 'lim': (-100.0, 100.0), 'dv': 1.0,
                              'k': 'ss.range',
                              'h': 'The parameter range to perform the space-search.'},
                    'Ngrid': {'t': int, 'lim': (0, 100), 'disp': '# steps', 'k': 'ss.Ngrid',
                              'h': 'The number of equally-distanced values to parse the parameter range.'},
                    'values': {'t': List[float], 'lim': (-100.0, 100.0), 'k': 'ss.vs',
                               'h': 'A list of values of the parameter to space-search. Once this is filled no range/# steps parameters are taken into account.'}
                },
                'space_search': {
                    'pars': {'t': List[str], 'h': 'The parameters for space search.', 'k': 'ss.pars'},
                    'ranges': {'t': List[Tuple[float]], 'lim': (-100.0, 100.0), 'dv': 1.0,
                               'h': 'The range of the parameters for space search.', 'k': 'ss.ranges'},
                    'Ngrid': {'t': int, 'lim': (0, 100), 'h': 'The number of steps for space search.',
                              'k': 'ss.Ngrid'}},
                'batch_setup': {
                    'batch_id': pID('batch-run', k='b_id'),
                    'save_hdf5': {**bF, 'h': 'Whether to store the batch-run data', 'k': 'store_batch'}
                }
            })

            return d

        def init_enr_pars():
            d = dNl.NestDict()

            d['ang_definition'] = {
                'bend': {'t': str, 'v': 'from_vectors', 'vs': ['from_angles', 'from_vectors'],
                         'h': 'Whether bending angle is computed as a sum of sequential segmental angles or as the angle between front and rear body vectors.'},
                'front_vector': {'t': Tuple[int], 'v': (1, 2), 'lim': (-12, 12), 'vfunc': param.Tuple,
                                 'h': 'The initial & final segment of the front body vector.'},
                'rear_vector': {'t': Tuple[int], 'v': (-2, -1), 'lim': (-12, 12), 'vfunc': param.Tuple,
                                'h': 'The initial & final segment of the rear body vector.'},
                'front_body_ratio': {'v': 0.5, 'lim': (0.0, 1.0), 'disp': 'front_ratio',
                                     'h': 'The fraction of the body considered front, relevant for bend computation from angles.'}
            }
            d['spatial_definition'] = {
                'point_idx': {'t': int, 'lim': (-1, 12),
                              'h': 'The index of the segment used as the larva spatial position (-1 means using the centroid).'},
                'use_component_vel': {**bF, 'disp': 'vel_component',
                                      'h': 'Whether to use the component velocity ralative to the axis of forward motion.'}
            }

            d['metric_definition'] = {
                'angular': d['ang_definition'],
                'spatial': d['spatial_definition'],
                'dispersion': {
                    'dsp_starts': {'t': List[int], 'v': [0], 'lim': (0, 200), 'dv': 1, 'disp': 'starts',
                                   'h': 'The timepoints to start calculating dispersion in seconds.'},
                    'dsp_stops': {'t': List[int], 'v': [40,60], 'lim': (0, 200), 'dv': 1, 'disp': 'stops',
                                  'h': 'The timepoints to stop calculating dispersion in seconds.'},
                },
                'tortuosity': {
                    'tor_durs': {'t': List[int], 'v': [5, 20], 'lim': (0, 200), 'dv': 1, 'disp': 't (sec)',
                                 'h': 'The time windows to use when calculating tortuosity in seconds.'}
                },
                'stride': {
                    'track_point': {'t': str,
                                    'h': 'The midline point to use when detecting the strides. When none is provided, the default position of the larva is used (see spatial definition).'},
                    'use_scaled_vel': {**bT, 'disp': 'vel_scaled',
                                       'h': 'Whether to use the velocity scaled to the body length.'},
                    'vel_threshold': {'v': 0.3, 'lim': (0.0, 2.0), 'disp': 'vel_thr',
                                      'h': 'The velocity threshold to be reached in every stride cycle.'},
                },
                # 'pause': {
                #     'stride_non_overlap': {**bT, 'disp': 'excl. strides',
                #                            'h': 'Whether pause bouts are required not to overlap with strides.'},
                #     'min_dur': {'v': 0.4, 'max': 2.0, 'h': 'The minimum duration for detecting a pause, in seconds.'},
                # },
                'turn': {
                    'min_ang': {'v': 30.0, 'max': 180.0, 'dv': 1.0,
                                'h': 'The minimum orientation angle change required to detect a turn.'},
                    'min_ang_vel': {'v': 0.0, 'max': 1000.0, 'dv': 1.0,
                                    'h': 'The minimum angular velocity maximum required to detect a turn.'},
                    'chunk_only': {'t': str, 'vs': ['', 'stride', 'pause'],
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
                'transposition': {'t': str, 'vs': ['', 'origin', 'arena', 'center'],
                                  'h': 'Whether to transpose spatial coordinates.'}
            }
            d['processing'] = {t: bF for t in proc_type_keys}
            d['annotation'] = {**{b: bF for b in bout_keys},
                               'on_food': bF,
                               'fits': bT}
            d['to_drop'] = {kk: bF for kk in to_drop_keys}
            d['enrichment'] = {**{k: d[k] for k in
                                  ['metric_definition', 'preprocessing', 'processing', 'annotation', 'to_drop']},
                               'recompute': bF,
                               'mode': {'t': str, 'v': 'minimal', 'vs': ['minimal', 'full']}
                               }
            return d

        def init_distpars():
            d = dNl.NestDict({
                'xy': {'t': Tuple[float], 'v': (0.0, 0.0), 'k': 'xy', 'lim': (-1.0, 1.0), 'min': -1.0, 'max': 1.0,
                       'vfunc': param.XYCoordinates,
                       'h': 'The xy spatial position coordinates.'},
                'odor': {
                    'odor_id': {'t': str, 'disp': 'ID', 'h': 'The unique ID of the odorant.'},
                    'odor_intensity': {'max': 10.0, 'disp': 'C peak',
                                       'h': 'The peak concentration of the odorant in micromoles.'},
                    'odor_spread': {'max': 10.0, 'disp': 'spread',
                                    'h': 'The spread of the concentration gradient around the peak.'}
                }

            })
            d['substrate_composition'] = {
                n: {'v': 0.0, 'max': 10.0, 'h': f'{n} density in g/cm**3.'} for
                n in
                ['glucose', 'dextrose', 'saccharose', 'yeast', 'agar', 'cornmeal']}

            d['substrate'] = {
                'type': {'t': str, 'v': 'standard', 'vs': list(substrate_dict.keys()),
                         'h': 'The type of substrate.'},
                'quality': {'v': 1.0, 'lim': (0.0, 1.0),
                            'h': 'The substrate quality as percentage of nutrients relative to the intact substrate type.'}

            }

            d['food'] = {
                'radius': {'v': 0.003, 'lim': (0.0, 0.1), 'dv': 0.001,
                           'h': 'The spatial radius of the source in meters.'},
                'amount': {'v': 0.0, 'lim': (0.0, 10.0), 'h': 'The food amount in the source.'},
                'can_be_carried': {**bF, 'disp': 'carriable', 'h': 'Whether the source can be carried around.'},
                'can_be_displaced': {**bF, 'disp': 'displaceable',
                                     'h': 'Whether the source can be displaced by wind/water.'},
                **d['substrate']
            }
            d['food_grid'] = {
                'unique_id': {'t': str, 'v': 'Food_grid', 'disp': 'ID',
                              'h': 'The unique ID of the food grid.'},
                'grid_dims': {'t': Tuple[int], 'v': (50, 50), 'lim': (10, 200), 'disp': 'XY dims',
                              'vfunc': param.Tuple,
                              'h': 'The spatial resolution of the food grid.'},
                'initial_value': {'v': 0.1, 'lim': (0.0, 10.0), 'dv': 0.01, 'disp': 'Initial amount',
                                  'h': 'The initial amount of food in each cell of the grid.'},
                'distribution': {'t': str, 'v': 'uniform', 'vs': ['uniform'],
                                 'h': 'The distribution of food in the grid.'},
                'default_color': pCol('green', 'food grid'),
                **d['substrate']
            }

            d['epoch'] = {
                'start': {'lim': (0.0, 250.0), 'h': 'The beginning of the epoch in hours post-hatch.'},
                'stop': {'lim': (0.0, 250.0), 'h': 'The end of the epoch in hours post-hatch.'},
                'substrate': d['substrate']

            }

            d['life_history'] = {
                'age': {'v': 0.0, 'lim': (0.0, 250.0), 'dv': 1.0,
                        'h': 'The larva age in hours post-hatch.'},
                'epochs': {'t': TypedDict, 'v': {}, 'entry': 'epoch', 'disp': 'life epochs',
                           'h': 'The feeding epochs comprising life-history.'}

            }

            d['food_params'] = {'source_groups': {'t': dict, 'v': {}},
                                'food_grid': {'t': dict},
                                'source_units': {'t': dict, 'v': {}}
                                }

            d['spatial_distro'] = {
                'mode': {'t': str, 'v': 'normal', 'vs': ['normal', 'periphery', 'uniform'],
                         'disp': 'placing',
                         'h': 'The wa to place agents in the distribution shape.'},
                'shape': {'t': str, 'v': 'circle', 'vs': ['circle', 'rect', 'oval'],
                          'h': 'The space of the spatial distribution.'},
                'N': {'t': int, 'v': 10, 'lim': (0, 1000),
                      'h': 'The number of agents in the group.'},
                'loc': d['xy'],
                'scale': d['xy'],
            }

            d['larva_distro'] = {
                **d['spatial_distro'],
                'orientation_range': {'t': Tuple[float], 'v': (0.0, 360.0), 'lim': (0.0, 360.0),
                                      'dv': 1.0,
                                      'disp': 'heading',
                                      'h': 'The range of larva body orientations to sample from, in degrees.'}
            }

            d['larva_model']=confID_entry('Model', default='explorer')

            d['Larva_DISTRO'] = {
                'model': d['larva_model'],
                **d['larva_distro'],
            }

            d['LarvaGroup'] = {
                'model': d['larva_model'],
                'sample': {'t': str, 'v': 'None.50controls'},
                'default_color': pCol('black', 'larva group'),
                'imitation': bF,
                'distribution': d['larva_distro'],
                'life_history': d['life_history'],
                'odor': d['odor']
            }

            d['agent'] = {
                'group': {'t': str, 'v': '', 'h': 'The unique ID of the agent group.'},

            }

            d['source'] = {
                **d['agent'],
                'default_color': pCol('green', 'source'),
                'pos': d['xy'],
                **d['food'],
                'odor': d['odor']
            }

            d['SourceGroup'] = {
                'distribution': d['spatial_distro'],
                'default_color': pCol('green', 'source group'),
                **d['food'],
                'odor': d['odor'],
                'regeneration': {**bF, 'h': 'Whether to regenerate a source when depleted.'},
                'regeneration_pos': {
                    'loc': d['xy'],
                    'scale': d['xy'],
                }
            }

            d['Border'] = {
                'default_color': pCol('black', 'border'),
                'width': {'v': 0.001, 'lim': (0.0, 10.0), 'h': 'The width of the border.'},
                'points': {'t': List[Tuple[float]], 'lim': (-10.0, 10.0),
                           'h': 'The XY coordinates of the consecutive border segments.'},
            }

            d['border_list'] = {
                'default_color': pCol('black', 'border'),
                'points': {'t': List[Tuple[float]], 'lim': (-10.0, 10.0),
                           'h': 'The XY coordinates of the consecutive border segments.'},
            }
            d['Source_DISTRO'] = d['spatial_distro']

            return d

        def init_pars0():
            d = dNl.NestDict({

                'odorscape': {
                    'odorscape': {'t': str, 'v': 'Gaussian', 'vs': ['Gaussian', 'Diffusion'],
                                  'k': 'odorscape_mod',
                                  'h': 'The algorithm used for odorscape generation.'},
                    'grid_dims': {'t': Tuple[int], 'v': (51, 51), 'lim': (10, 100), 'vfunc': param.Tuple,
                                  'k': 'grid_dims',
                                  'h': 'The odorscape grid resolution.'},
                    'evap_const': {'lim': (0.0, 1.0), 'k': 'c_evap',
                                   'h': 'The evaporation constant of the diffusion algorithm.'},
                    'gaussian_sigma': {'t': Tuple[float], 'lim': (0.0, 1.0), 'vfunc': param.NumericTuple,
                                       'k': 'gau_sigma',
                                       'h': 'The sigma of the gaussian difusion algorithm.'}
                },
                'thermoscape': {
                    'thermo_sources': {'v': [(0.5, 0.05), (0.05, 0.5), (0.5, 0.95), (0.95, 0.5)],
                                       't': List[Tuple[float]],
                                       'lim': (-100.0, 100.0), 'h': 'The xy coordinates of the thermal sources',
                                       'disp': 'thermal sources',
                                       'k': 'temp_sources'},
                    'plate_temp': {'v': 22.0, 'lim': (0.0, 100.0), 'h': 'reference temperature',
                                   'disp': 'reference temperature',
                                   'k': 'temp_0'},
                    'thermo_source_dTemps': {'v': [8.0, -8.0, 8.0, -8.0], 't': List[float],
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
                    'N': {'t': int, 'lim': (0, 10000),
                          'h': 'The number of repetitions of the puff. If N>1 an interval must be provided'},
                    'interval': {'v': 5.0, 'lim': (0.0, 10000.0),
                                 'h': 'Whether the puff will reoccur at constant time intervals in seconds. Ignored if N=1'},
                },
                'windscape': {
                    'wind_direction': {'t': float, 'v': np.pi, 'lim': (0.0, 2 * np.pi), 'dv': 0.1,
                                       'h': 'The absolute polar direction of the wind/air puff.'},
                    'wind_speed': {'t': float, 'v': 0.0, 'lim': (0.0, 100.0), 'dv': 1.0,
                                   'h': 'The speed of the wind/air puff.'},
                    'puffs': {'t': TypedDict, 'v': {}, 'entry': 'air_puff', 'disp': 'air-puffs',
                              'h': 'Repetitive or single air-puff stimuli.'}
                },
                'odor_gains': {
                    'unique_id': pID('odorant'),
                    'mean': {'lim': (0.0, 1000.0), 'dv': 10.0,
                             'h': 'The mean gain/valence for the odorant. Positive/negative for appettitive/aversive valence.'},
                    'std': {'lim': (0.0, 10.0), 'dv': 1.0, 'h': 'The standard deviation for the odorant gain/valence.'}
                },

                'arena': {
                    'arena_dims': {'t': Tuple[float], 'v': (0.1, 0.1), 'lim': (0.0, 2.0), 'dv': 0.01, 'disp': 'X,Y (m)',
                                   'vfunc': param.NumericTuple,
                                   'h': 'The arena dimensions in meters.'},
                    'arena_shape': {'t': str, 'v': 'circular', 'vs': ['circular', 'rectangular'], 'disp': 'shape',
                                    'h': 'The arena shape.'}
                },

                'essay_params': {
                    'essay_ID': pID('essay'),
                    'path': pPath('essay'),
                    'N': {'t': int, 'lim': (1, 100), 'disp': '# larvae', 'h': 'The number of larvae per larva-group.'}
                },
                'sim_params': {
                    'sim_ID': pID('simulation', k='id'),
                    'path': pPath('simulation', k='path'),
                    'duration': {'lim': (0.0, 100000.0), 'h': 'The duration of the simulation in minutes.',
                                 'k': 't'},
                    'timestep': {'v': 0.1, 'lim': (0.0, 0.4), 'dv': 0.05,
                                 'h': 'The timestep of the simulation in seconds.',
                                 'k': 'dt'},
                    'Box2D': {**bF, 'h': 'Whether to use the Box2D physics engine or not.', 'k': 'Box2D'},
                    'store_data': {**bT, 'h': 'Whether to store the simulation data or not.', 'k': 'no_store'},
                },
                'logn_dist': {
                    'range': {'t': Tuple[float], 'v': (0.0, 2.0), 'lim': (0.0, 10.0), 'dv': 1.0},
                    'name': {'t': str, 'v': 'lognormal', 'vs': ['lognormal']},
                    'mu': {'v': 1.0, 'lim': (0.0, 10.0)},
                    'sigma': {'v': 0.0, 'lim': (0.0, 10.0)},
                    'fit': bF
                },
                'par': {
                    'p': {'t': str},
                    'u': {'t': str},
                    'k': {'t': str},
                    's': {'t': str},
                    'symbol': {'t': str},
                    'codename': {'t': str},
                    'o': {'t': type},
                    'lim': {'t': Tuple[float]},
                    'd': {'t': str},
                    'l': {'t': str},
                    'exists': bF,
                    'func': {'t': str},
                    'const': {'t': str},
                    'operator': {'t': str, 'vs': [None, 'diff', 'cum', 'max', 'min', 'mean', 'std', 'final']},
                    'k0': {'t': str},
                    'k_num': {'t': str},
                    'k_den': {'t': str},
                    'dst2source': {'t': Tuple[float], 'min': -100.0, 'max': 100.0},
                    'or2source': {'t': Tuple[float], 'min': -180.0, 'max': 180.0},
                    'dispersion': {'t': bool, 'v': False},
                    'wrap_mode': {'t': str, 'vs': [None, 'zero', 'positive']}
                },
                'build_conf': {
                    'min_duration_in_sec': {'v': 170.0, 'lim': (0.0, 3600.0), 'dv': 1.0,
                                            'disp': 'min track duration (sec)'},
                    'min_end_time_in_sec': {'v': 0.0, 'lim': (0.0, 3600.0), 'dv': 1.0,
                                            'disp': 'min track termination time (sec)'},
                    'start_time_in_sec': {'v': 0.0, 'lim': (0.0, 3600.0), 'dv': 1.0,
                                          'disp': 'track initiation time (sec)'},
                    'max_Nagents': {'t': int, 'v': 500, 'lim': (0, 5000), 'disp': 'max # larvae'},
                    'save_mode': {'t': str, 'v': 'semifull', 'vs': ['minimal', 'semifull', 'full', 'points']},
                },
                'output': {n: bF for n in output_keys}})
            return d

        def init_mods():
            from lib.aux.par_aux import subsup, sub, tilde, bar, circle, sup
            from lib.registry.units import ureg
            d = dNl.NestDict({
                'bout_distro': {
                    'fit': {**bT, 'combo': 'distro',
                            'h': 'Whether the distribution is sampled from a reference dataset. Once this is set to "ON" no other parameter is taken into account.'},
                    'range': {'t': Tuple[float], 'max': 500.0, 'combo': 'distro', 'h': 'The distribution range.'},
                    'name': {'t': str,
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
                                    'symbol': sub('c', 'T'), 'u_name': sup('sec', -2), 'u': ureg.s ** -2,
                                    'h': 'Conversion coefficient from TURNER output to torque-per-inertia-unit.'},
                    'ang_vel_coef': {'v': 1.0, 'lim': (0.0, 5.0), 'dv': 0.01, 'label': 'angular velocity coefficient',
                                     'h': 'Conversion coefficient from TURNER output to angular velocity.'},
                    'ang_damping': {'v': 1.0, 'lim': (0.1, 2.0), 'label': 'angular damping', 'symbol': 'z',
                                    'u_name': sup('sec', -1), 'u': ureg.s ** -1,
                                    'h': 'Angular damping exerted on angular velocity.'},
                    'lin_damping': {'v': 1.0, 'lim': (0.0, 10.0), 'label': 'linear damping', 'symbol': 'zl',
                                    'u_name': sup('sec', -1), 'u': ureg.s ** -1,
                                    'h': 'Linear damping exerted on forward velocity.'},
                    'body_spring_k': {'v': 1.0, 'lim': (0.0, 10.0), 'dv': 0.1, 'label': 'body spring constant',
                                      'symbol': 'k', 'u_name': sup('sec', -2), 'u': ureg.s ** -2,
                                      'h': 'Larva-body torsional spring constant reflecting deformation resistance.'},
                    'bend_correction_coef': {'v': 1.0, 'lim': (0.8, 1.5), 'label': 'bend correction coefficient',
                                             'symbol': sub('c', 'b'),
                                             'h': 'Correction coefficient of bending angle during forward motion.'},
                    'ang_mode': {'t': str, 'v': 'torque', 'vs': ['torque', 'velocity'], 'label': 'angular mode',
                                 'h': 'Whether the Turner module output is equivalent to torque or angular velocity.'},
                },
                'crawler': {
                    'mode': {'t': str, 'v': 'realistic', 'k': 'Cr_mod',
                             'vs': ['realistic', 'square', 'gaussian', 'constant'],
                             'symbol': subsup('A', 'C', 'mode'),
                             'label': 'crawler waveform',
                             'h': 'The waveform of the repetitive crawling oscillator (CRAWLER) module.'},
                    'initial_freq': {'v': 1.418, 'lim': (0.5, 2.5), 'dv': 0.1, 'aux_vs': ['sample'], 'disp': 'initial',
                                     'k': 'f_C0',
                                     'label': 'crawling frequency', 'symbol': sub('f', 'C'), 'u': ureg.Hz,
                                     'combo': 'frequency', 'codename': 'scaled_velocity_freq',
                                     'h': 'The initial frequency of the repetitive crawling behavior.'},
                    'max_scaled_vel': {'v': 0.6, 'lim': (0.0, 1.5), 'label': 'maximum scaled velocity',
                                       'codename': 'stride_scaled_velocity_max', 'k': 'sstr_v_max', 'dv': 0.1,
                                       'symbol': sub(circle('v'), 'max'), 'u': ureg.s ** -1,
                                       'u_name': '$body-lengths/sec$',
                                       'h': 'The maximum scaled forward velocity.'},
                    'stride_dst_mean': {'v': 0.224, 'lim': (0.0, 1.0), 'dv': 0.01, 'aux_vs': ['sample'], 'disp': 'mean',
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
                                      'symbol': subsup('$\phi$', 'C', 'v'), 'u_name': 'rad', 'u': ureg.rad,
                                      'codename': 'phi_scaled_velocity_max',
                                      'h': 'The phase of the crawling oscillation cycle where forward velocity is maximum.'}
                },
                'olfactor': {
                    'perception': {'t': str, 'v': 'log', 'vs': ['log', 'linear', 'null'],
                                   'label': 'olfaction sensing transduction mode',
                                   'k': 'mod_O',
                                   'symbol': sub('mod', 'O'), 'u_name': None,
                                   'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input.'},
                    'input_noise': {'v': 0.0, 'lim': (0.0, 1.0), 'h': 'The intrinsic noise of the sensory input.'},
                    'decay_coef': {'v': 0.0, 'lim': (0.0, 2.0), 'label': 'olfactory decay coef',
                                   'symbol': sub('c', 'O'), 'k': 'c_O',
                                   'h': 'The linear decay coefficient of the olfactory sensory activation.'},
                    'brute_force': {**bF, 'h': 'Whether to apply direct rule-based modulation on locomotion or not.'}
                },
                'thermosensor': {
                    'perception': {'t': str, 'v': 'linear', 'vs': ['log', 'linear', 'null'],
                                   'label': 'thermosensing transduction mode',
                                   'k': 'mod_th',
                                   'symbol': sub('mod', 'th'), 'u_name': None,
                                   'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input.'},
                    'input_noise': {'v': 0.0, 'lim': (0.0, 1.0), 'h': 'The intrinsic noise of the sensory input.'},
                    'decay_coef': {'v': 0.0, 'lim': (0.0, 2.0), 'label': 'thermosensation decay coef',
                                   'symbol': sub('c', 'th'), 'k': 'c_th',
                                   'h': 'The linear decay coefficient of the thermosensory activation.'},
                    'brute_force': {**bF, 'h': 'Whether to apply direct rule-based modulation on locomotion or not.'},
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
                    'perception': {'t': str, 'v': 'linear', 'vs': ['log', 'linear'],
                                   'symbol': sub('mod', 'T'),
                                   'k': 'mod_T', 'label': 'tactile sensing transduction mode',
                                   'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input.'},
                    'input_noise': {'v': 0.0, 'lim': (0.0, 1.0), 'h': 'The intrinsic noise of the sensory input.'},
                    'decay_coef': {'v': 0.1, 'lim': (0.0, 2.0), 'label': 'tactile decay coef',
                                   'symbol': sub('c', 'T'), 'k': 'c_T',
                                   'h': 'The exponential decay coefficient of the tactile sensory activation.'},
                    'state_specific_best': {**bT,
                                            'h': 'Whether to use the state-specific or the global highest evaluated gain after the end of the memory training period.'},
                    'brute_force': {**bF, 'h': 'Whether to apply direct rule-based modulation on locomotion or not.'},
                    'initial_gain': {'v': 40.0, 'lim': (-100.0, 100.0),
                                     'label': 'tactile sensitivity coef', 'symbol': sub('G', 'T'), 'k': 'G_T',
                                     'h': 'The initial gain of the tactile sensor.'},
                    'touch_sensors': {'t': List[int], 'lim': (0, 8), 'k': 'sens_touch',
                                      'symbol': sub('N', 'T'), 'label': 'tactile sensor contour locations',
                                      'h': 'The number of touch sensors existing on the larva body.'},
                },
                'feeder': {
                    'freq_range': {'t': Tuple[float], 'v': (1.0, 3.0), 'lim': (0.0, 4.0), 'disp': 'range',
                                   'combo': 'frequency',
                                   'h': 'The frequency range of the repetitive feeding behavior.'},
                    'initial_freq': {'v': 2.0, 'lim': (0.0, 4.0), 'disp': 'initial', 'combo': 'frequency',
                                     'k': 'f_F0',
                                     'label': 'feeding frequency', 'symbol': sub('f', 'F'), 'u': ureg.Hz,
                                     'h': 'The initial default frequency of the repetitive feeding behavior'},
                    'feed_radius': {'v': 0.1, 'lim': (0.1, 10.0), 'symbol': sub('rad', 'F'),
                                    'label': 'feeding radius', 'k': 'rad_F',
                                    'h': 'The radius around the mouth in which food is consumable as a fraction of the body length.'},
                    'V_bite': {'v': 0.0005, 'lim': (0.0001, 0.01), 'dv': 0.0001,
                               'symbol': sub('V', 'F'), 'label': 'feeding volume ratio', 'k': 'V_F',
                               'h': 'The volume of food consumed on a single feeding motion as a fraction of the body volume.'}
                },
                'memory': {
                    'modality': {'t': str, 'v': 'olfaction', 'vs': ['olfaction', 'touch'],
                                 'h': 'The modality for which the memory module is used.'},
                    'Delta': {'v': 0.1, 'lim': (0.0, 10.0), 'h': 'The input sensitivity of the memory.'},
                    'state_spacePerSide': {'t': int, 'v': 0, 'max': 20, 'disp': 'state space dim',
                                           'h': 'The number of discrete states to parse the state space on either side of 0.'},
                    'gain_space': {'t': List[float], 'v': [-300.0, -50.0, 50.0, 300.0], 'lim': (-1000.0, 1000.0),
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
                    'crawler_phi_range': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': 2 * np.pi,
                                          'label': 'suppression relief phase interval',
                                          'symbol': '$[\phi_{C}^{\omega_{0}},\phi_{C}^{\omega_{1}}]$',
                                          'u': ureg.rad,
                                          'h': 'CRAWLER phase range for TURNER suppression lift.'},
                    'feeder_phi_range': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': 2 * np.pi,
                                         'label': 'feeder suppression relief phase interval',
                                         'symbol': '$[\phi_{F}^{\omega_{0}},\phi_{F}^{\omega_{1}}]$',
                                         'u': ureg.rad,
                                         'h': 'FEEDER phase range for TURNER suppression lift.'}

                },
                'phasic_interference': {
                    'max_attenuation_phase': {'v': 3.4, 'max': 2 * np.pi, 'label': 'max relief phase',
                                              'lim': (0.0, 2 * np.pi),
                                              'symbol': '$\phi_{C}^{\omega}$', 'u': ureg.rad, 'k': 'phi_fov_max',
                                              'h': 'CRAWLER phase of minimum TURNER suppression.'}
                },
                'base_interference': {
                    'mode': {'t': str, 'v': 'square', 'k': 'IF_mod', 'vs': ['default', 'square', 'phasic'],
                             'h': 'CRAWLER:TURNER suppression phase mode.'},
                    'suppression_mode': {'t': str, 'v': 'amplitude', 'vs': ['amplitude', 'oscillation', 'both'],
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
                'base_activation': {'v': 20.0, 'min': 10.0, 'max': 40.0, 'lim': (10.0, 40.0), 'dv': 0.1, 'disp': 'mean',
                                    'combo': 'activation',
                                    'label': 'tonic input', 'symbol': '$I_{T}^{0}$', 'k': 'I_T0',
                                    'h': 'The baseline activation/input of the TURNER module.'},
                'activation_range': {'t': Tuple[float], 'v': (10.0, 40.0), 'lim': (0.0, 100.0), 'dv': 0.1,
                                     'disp': 'range',
                                     'k': 'I_T_r',
                                     'label': 'input range', 'symbol': r'$[I_{T}^{min},I_{T}^{max}]$',
                                     'combo': 'activation', 'h': 'The activation/input range of the TURNER module.'},

                'tau': {'v': 0.1, 'min': 0.05, 'max': 0.5, 'dv': 0.01, 'label': 'time constant',
                        'symbol': r'$\tau_{T}$',
                        'u': ureg.s,
                        'h': 'The time constant of the neural oscillator.'},
                'm': {'t': int, 'v': 100, 'min': 50, 'max': 200, 'label': 'maximum spike-rate', 'symbol': '$SR_{max}$',
                      'h': 'The maximum allowed spike rate.'},
                'n': {'v': 2.0, 'min': 1.5, 'max': 3.0, 'dv': 0.01, 'label': 'spike response steepness',
                      'symbol': '$n_{T}$',
                      'h': 'The neuron spike-rate response steepness coefficient.'}

            }

            d['sinusoidal_turner'] = {
                'initial_amp': {'v': 19.27, 'lim': (0.0, 100.0), 'disp': 'initial',
                                'combo': 'amplitude',
                                'label': 'output amplitude', 'symbol': '$A_{T}^{0}$', 'k': 'A_T0',
                                'h': 'The initial activity amplitude of the TURNER module.'},
                'amp_range': {'t': Tuple[float], 'lim': (0.0, 1000.0), 'v': (0.0, 100.0),
                              'disp': 'range', 'combo': 'amplitude',
                              'label': 'output amplitude range', 'symbol': r'$[A_{T}^{min},A_{T}^{max}]$',
                              'k': 'A_T_r',
                              'h': 'The activity amplitude range of the TURNER module.'},
                'initial_freq': {'v': 0.58, 'lim': (0.01, 2.0), 'dv': 0.01, 'disp': 'initial', 'combo': 'frequency',
                                 'k': 'f_T0',
                                 'label': 'bending frequency', 'symbol': sub('f', 'T'), 'u_name': '$Hz$', 'u': ureg.Hz,
                                 'h': 'The initial frequency of the repetitive lateral bending behavior if this is hardcoded (e.g. sinusoidal mode).'},
                'freq_range': {'t': Tuple[float], 'lim': (0.01, 2.0), 'dv': 0.01, 'disp': 'range',
                               'combo': 'frequency',
                               'label': 'bending frequency range', 'k': 'f_T_r', 'v': (0.1, 0.8),
                               'symbol': r'$[f_{T}^{min},f_{T}^{max}]$', 'u_name': '$Hz$', 'u': ureg.Hz,
                               'h': 'The frequency range of the repetitive lateral bending behavior.'}
            }

            d['constant_turner'] = {
                'initial_amp': {'lim': (0.1, 20.0), 'disp': 'initial', 'combo': 'amplitude', 'k': 'A_T0',
                                'label': 'output amplitude', 'symbol': '$A_{T}^{0}$', 'u_name': None,
                                'h': 'The initial activity amplitude of the TURNER module.'},
            }

            d['base_turner'] = {
                'mode': {'t': str, 'v': 'neural', 'vs': ['', 'neural', 'sinusoidal', 'constant'], 'k': 'Tur_mod',
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

            d['Box2D_joint_N'] = {'t': int, 'v': 0, 'max': 2}

            d['friction_joint'] = {'N': d['Box2D_joint_N'], 'args': {'maxForce': {'v': 10 ** 0, 'max': 10 ** 5},
                                                                     'maxTorque': {'v': 10 ** 0, 'max': 10 ** 5}
                                                                     }}
            d['revolute_joint'] = {'N': d['Box2D_joint_N'], 'args': {
                'enableMotor': bT,  # )
                'maxMotorTorque': {'v': 0.0, 'max': 10 ** 5},
                'motorSpeed': {'v': 0.0, 'max': 10 ** 5}
            }}
            d['distance_joint'] = {'N': d['Box2D_joint_N'], 'args': {
                'frequencyHz': {'v': 5.0, 'max': 20.0},
                'dampingRatio': {'v': 1.0, 'max': 10 ** 5},
            }}

            d['Box2D_params'] = {
                'joint_types': {
                    'friction': d['friction_joint'],
                    'revolute': d['revolute_joint'],
                    'distance': d['distance_joint']
                }
            }

            d['body_shape'] = {
                'symmetry': {'t': str, 'v': 'bilateral', 'vs': ['bilateral', 'radial'],
                             'h': 'The body symmetry.'},
                'Nsegs': {'t': int, 'v': 2, 'min': 1, 'max': 12,
                          'h': 'The number of segments comprising the larva body.'},
                'seg_ratio': {'lim': (0.0, 1.0),
                              'h': 'The length ratio of the body segments. If null, equal-length segments are generated.'},

                'olfaction_sensors': {'t': List[int], 'lim': (0, 16), 'v': [0], 'disp': 'olfaction',
                                      'h': 'The indexes of the contour points bearing olfaction sensors.'},

                'touch_sensors': {'t': List[int], 'lim': (0, 16), 'disp': 'touch',
                                  'h': 'The indexes of the contour points bearing touch sensors.'},
                'points': {'t': List[Tuple[float]], 'lim': (-1.0, 1.0), 'disp': 'contour',
                           'h': 'The XY coordinates of the body contour.'},
            }

            d['body'] = {
                'initial_length': {'v': 0.004, 'lim': (0.0, 0.01), 'dv': 0.0001, 'aux_vs': ['sample'],
                                   'disp': 'initial',
                                   'label': 'length', 'symbol': '$l$', 'u': ureg.m, 'k': 'l0',
                                   'combo': 'length', 'h': 'The initial body length.'},
                'length_std': {'v': 0.0, 'lim': (0.0, 0.001), 'dv': 0.0001, 'u': ureg.m, 'aux_vs': ['sample'],
                               'disp': 'std', 'k': 'l_std',
                               'combo': 'length', 'h': 'The standard deviation of the initial body length.'},
                'Nsegs': {'t': int, 'v': 2, 'lim': (1, 12), 'label': 'number of body segments', 'symbol': '-',
                          'u_name': '# $segments$', 'k': 'Nsegs',
                          'h': 'The number of segments comprising the larva body.'},
                'seg_ratio': {'k': 'seg_r', 'lim': (0.0, 1.0),
                              'h': 'The length ratio of the body segments. If null, equal-length segments are generated.'},

                'shape': {'t': str, 'v': 'drosophila_larva', 'vs': ['drosophila_larva', 'zebrafish_larva'],
                          'k': 'body_shape', 'h': 'The body shape.'},
            }

            d['intermitter'] = {
                'mode': {'t': str, 'v': 'default', 'vs': ['', 'default', 'branch', 'nengo'],
                         'h': 'The implementation mode of the intermittency (INTERMITTER) module.'},
                'run_mode': {'t': str, 'v': 'stridechain', 'vs': ['stridechain', 'run'],
                             'h': 'The generation mode of run epochs.'},
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
                'feed_bouts': {**bF, 'disp': 'feeding bouts', 'h': 'Whether feeding bouts (feedchains) are generated.'},
                'crawl_freq': {'v': 1.43, 'lim': (0.5, 2.5), 'k': 'f_C', 'dv': 0.01, 'u': ureg.Hz,
                               'symbol': sub('f', 'C'),
                               'disp': 'crawling frequency',
                               'h': 'The default frequency of the CRAWLER oscillator when simulating offline.'},
                'feed_freq': {'v': 2.0, 'lim': (0.5, 4.0), 'dv': 0.01, 'k': 'f_F', 'u': ureg.Hz,
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
                'M_gm': {'v': 10 ** -2, 'min': 0.0, 'disp': 'gut scaled capacity',
                         'symbol': 'M_gm',
                         'k': 'M_gm',
                         'h': 'Gut capacity in C-moles per unit of gut volume.'},
                'y_P_X': {'v': 0.9, 'disp': 'food->product yield',
                          'symbol': 'y_P_X', 'k': 'y_P_X',
                          'h': 'Yield of product per unit of food.'},
                'J_g_per_cm2': {'v': 10 ** -2 / (24 * 60 * 60), 'min': 0.0, 'disp': 'digestion secretion rate',
                                'symbol': 'J_g_per_cm2', 'k': 'J_g_per_cm2',
                                'h': 'Secretion rate of enzyme per unit of gut surface per second.'},
                'k_g': {'v': 1.0, 'min': 0.0, 'disp': 'digestion decay rate', 'symbol': 'k_g',
                        'k': 'k_g',
                        'h': 'Decay rate of digestive enzyme.'},
                'k_dig': {'v': 1.0, 'min': 0.0, 'disp': 'digestion rate', 'symbol': 'k_dig',
                          'k': 'k_dig',
                          'h': 'Rate constant for digestion : k_X * y_Xg.'},
                'f_dig': {'v': 1.0, 'disp': 'digestion response',
                          'symbol': 'f_dig', 'k': 'f_dig',
                          'h': 'Scaled functional response for digestion : M_X/(M_X+M_K_X)'},
                'M_c_per_cm2': {'v': 5 * 10 ** -8, 'min': 0.0, 'disp': 'carrier density',
                                'symbol': 'M_c_per_cm2', 'k': 'M_c_per_cm2',
                                'h': 'Area specific amount of carriers in the gut per unit of gut surface.'},
                'constant_M_c': {**bT, 'disp': 'constant carrier density', 'symbol': 'constant_M_c',
                                 'k': 'constant_M_c',
                                 'h': 'Whether to assume a constant amount of carrier enzymes on the gut surface.'},
                'k_c': {'v': 1.0, 'min': 0.0, 'disp': 'carrier release rate', 'symbol': 'k_c',
                        'k': 'gut_k_c',
                        'h': 'Release rate of carrier enzymes.'},
                'k_abs': {'v': 1.0, 'min': 0.0, 'disp': 'absorption rate', 'symbol': 'k_abs',
                          'k': 'gut_k_abs',
                          'h': 'Rate constant for absorption : k_P * y_Pc.'},
                'f_abs': {'v': 1.0, 'lim': (0.0, 1.0), 'disp': 'absorption response',
                          'symbol': 'f_abs', 'k': 'f_abs',
                          'h': 'Scaled functional response for absorption : M_P/(M_P+M_K_P)'},
            }

            d['DEB'] = {'species': {'t': str, 'v': 'default', 'vs': ['default', 'rover', 'sitter'], 'disp': 'phenotype',
                                    'k': 'species',
                                    'h': 'The phenotype/species-specific fitted DEB model to use.'},
                        'f_decay': {'v': 0.1, 'max': 1.0, 'dv': 0.1, 'symbol': sub('c', 'DEB'), 'k': 'c_DEB',
                                    'label': 'DEB functional response decay coef',
                                    'h': 'The exponential decay coefficient of the DEB functional response.'},
                        'absorption': {'v': 0.5, 'lim': (0.0, 1.0), 'symbol': sub('c', 'abs'),
                                       'k': 'c_abs',
                                       'h': 'The absorption ration for consumed food.'},
                        'V_bite': {'v': 0.0005, 'max': 0.01, 'dv': 0.0001,
                                   'symbol': sub('V', 'bite'),
                                   'k': 'V_bite',
                                   'h': 'The volume of food consumed on a single feeding motion as a fraction of the body volume.'},
                        'hunger_as_EEB': {**bT,
                                          'h': 'Whether the DEB-generated hunger drive informs the exploration-exploitation balance.',
                                          'symbol': 'hunger_as_EEB', 'k': 'hunger_as_EEB'},
                        'hunger_gain': {'v': 0.0, 'lim': (0.0, 1.0), 'symbol': sub('G', 'hunger'),
                                        'k': 'G_hunger', 'label': 'hunger sensitivity to reserve reduction',
                                        'h': 'The sensitivy of the hunger drive in deviations of the DEB reserve density.'},
                        'assimilation_mode': {'t': str, 'v': 'gut', 'vs': ['sim', 'gut', 'deb'],
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
            d['larva_conf'] = {
                'brain': d['brain'],
                'body': d['body'],
                'energetics': d['energetics'],
                'physics': d['physics'],
                'Box2D_params': d['Box2D_params'],
            }

            d['model_conf'] = d['larva_conf']

            return d

        def buildInitDict():
            d0 = init_pars0()
            dvis = init_vis()
            dmod = init_mods()
            dbatch = init_batch_pars()
            denr = init_enr_pars()
            ddist = init_distpars()
            d = dNl.NestDict({**d0, **dvis, **dmod, **dbatch, **denr, **ddist})

            d['batch_conf'] = {'exp': {'t': str},
                               'space_search': d['space_search'],
                               'batch_methods': d['batch_methods'],
                               'optimization': d['optimization'],
                               'exp_kws': {'t': dict, 'v': {'enrichment': d['enrichment']}},
                               'post_kws': {'t': dict, 'v': {}},
                               'proc_kws': {'t': dict, 'v': {}},
                               'save_hdf5': {**bF, 'h': 'Whether to store the sur datasets.'}
                               }

            d['env_conf'] = {'arena': d['arena'],
                             'border_list': {'t': dict, 'v': {}},
                             'food_params': d['food_params'],
                             'odorscape': {'t': dict},
                             'windscape': {'t': dict},
                             'thermoscape': {'t': dict},
                             }

            d['exp_conf'] = {
                'env_params': confID_entry('Env'),
                             'larva_groups': {'t': dict, 'v': {}},
                             'sim_params': d['sim_params'],
                             'trials': confID_entry('Trial', default='default'),
                             'collections': {'t': List[str], 'v': ['pose']},
                             'enrichment': d['enrichment'],
                             'experiment': confID_entry('Exp'),
                             }

            d['tracker_conf'] = {
                'resolution': {
                    'fr': {'v': 10.0, 'max': 100.0, 'disp': 'framerate (Hz)',
                           'h': 'The framerate of the tracker recordings.'},
                    'Npoints': {'t': int, 'v': 1, 'max': 20, 'disp': '# midline xy',
                                'h': 'The number of points tracked along the larva midline.'},
                    'Ncontour': {'t': int, 'v': 0, 'max': 100, 'disp': '# contour xy',
                                 'h': 'The number of points tracked around the larva contour.'}
                },
                'arena': d['arena'],
                'filesystem': {
                    'read_sequence': {'t': List[str], 'disp': 'columns',
                                      'h': 'The sequence of columns in the tracker-exported files.'},
                    'read_metadata': {**bF, 'disp': 'metadata',
                                      'h': 'Whether metadata files are available for the tracker-exported files/folders.'},
                    'folder': {
                        'pref': {'t': str, 'h': 'A prefix for detecting a raw-data folder.'},
                        'suf': {'t': str, 'h': 'A suffix for detecting a raw-data folder.'}},
                    'file': {'pref': {'t': str, 'h': 'A prefix for detecting a raw-data file.'},
                             'suf': {'t': str, 'h': 'A suffix for detecting a raw-data file.'},
                             'sep': {'t': str, 'h': 'A separator for detecting a raw-data file.'}}
                },

            }


            d['replay'] = {
                'env_params': confID_entry('Env'),
                'transposition': {'t': str, 'vs': [None, 'origin', 'arena', 'center'],
                                  'h': 'Whether to transpose the dataset spatial coordinates.'},
                'agent_ids': {'t': List[str],
                              'h': 'Whether to only display some larvae of the dataset, defined by their indexes.'},
                'dynamic_color': {'t': str, 'vs': [None, 'lin_color', 'ang_color'],
                                  'h': 'Whether to display larva tracks according to the instantaneous forward or angular velocity.'},
                'time_range': {'t': Tuple[float], 'max': 1000.0, 'dv': 1.0,
                               'h': 'Whether to only replay a defined temporal slice of the dataset.'},
                'track_point': {'t': int, 'v': -1, 'min': -1, 'max': 12,
                                'h': 'The midline point to use for defining the larva position.'},
                'draw_Nsegs': {'t': int, 'min': 1, 'max': 12,
                               'h': 'Whether to artificially simplify the experimentally tracked larva body to a segmented virtual body of the given number of segments.'},
                'fix_point': {'t': int, 'min': 1, 'max': 12,
                              'h': 'Whether to fixate a specific midline point to the center of the screen. Relevant when replaying a single larva track.'},
                'fix_segment': {'t': int, 'vs': [-1, 1],
                                'h': 'Whether to additionally fixate the above or below body segment.'},
                'use_background': {**bF,
                                   'h': 'Whether to use a virtual moving background when replaying a fixated larva.'}
            }

            d['ga_select_kws'] = {
                'Nagents': {'t': int, 'v': 30, 'min': 2, 'max': 1000,
                            'h': 'Number of agents per generation', 'k': 'N'},
                'Nelits': {'t': int, 'v': 3, 'min': 0, 'max': 1000,
                           'h': 'Number of elite agents preserved per generation', 'k': 'Nel'},
                'Ngenerations': {'t': int, 'max': 1000, 'h': 'Number of generations to run',
                                 'k': 'Ngen'},
                'Pmutation': {'v': 0.3, 'lim': (0.0, 1.0), 'h': 'Probability of genome mutation',
                              'k': 'Pmut'},
                'Cmutation': {'v': 0.1, 'lim': (0.0, 1.0), 'h': 'Mutation coefficient', 'k': 'Cmut'},
                'selection_ratio': {'v': 0.3, 'lim': (0.0, 1.0),
                                    'h': 'Fraction of agents to be selected for the next generation', 'k': 'Rsel'},
                'verbose': {'t': int, 'v': 0, 'vs': [0, 1, 2, 3],
                            'h': 'Verbose argument for GA launcher', 'k': 'verb'}
            }

            d['ga_build_kws'] = {
                'space_dict': {'t': dict, 'h': 'The parameter state space'},
                'robot_class': {'t': type, 'h': 'The agent class to use in the simulations'},
                'base_model': confID_entry('Model', default='navigator', k='mID0', symbol=subsup('ID', 'mod', 0)),
                'bestConfID': {'t': str,
                               'h': 'The model configuration ID to store the best genome',
                               'k': 'mID1'},
                'init_mode': {'t': str, 'v': 'random', 'vs': ['default', 'random', 'model'],
                              'h': 'The initialization mode for the first generation', 'k': 'mGA'},
                'multicore': {**bF, 'h': 'Whether to use multiple cores', 'k': 'multicore'},
                'fitness_target_refID': confID_entry('Ref'),
                'fitness_target_kws': {'t': dict, 'v': {},
                                       'h': 'The target data to derive from the reference dataset for evaluation'},
                'fitness_func': {'t': FunctionType, 'h': 'The method for fitness evaluation'},
                'plot_func': {'t': FunctionType,
                              'h': 'The method for real-time simulation and plotting of the best genome'},
                'exclude_func': {'t': FunctionType,
                                 'h': 'The method for real-time excluding agents'},
            }

            d['GAconf'] = {
                'scene': {'t': str, 'v': 'no_boxes', 'h': 'The name of the scene to load'},
                'scene_speed': {'t': int, 'v': 0, 'max': 100,
                                'h': 'The rendering speed of the scene'},
                'env_params': confID_entry('Env'),
                'sim_params': d['sim_params'],
                'experiment': confID_entry('Ga', default='exploration'),
                'caption': {'t': str, 'h': 'The screen caption'},
                'save_to': pPath(v=None, h='The directory to save data and plots'),
                'show_screen': {**bT, 'h': 'Whether to render the screen visualization', 'k': 'hide'},
                'offline': {**bF, 'h': 'Whether to run a full LarvaworldSim environment', 'k': 'offline'},
                'ga_build_kws': d['ga_build_kws'],
                'ga_select_kws': d['ga_select_kws'],
                # 'ga_kws': {**d['GAbuilder'], **d['GAselector']},
            }

            d['obstacle_avoidance'] = {
                'sensor_delta_direction': {'v': 0.4, 'dv': 0.01, 'min': 0.2, 'max': 1.2,
                                           'h': 'Sensor delta_direction'},
                'sensor_saturation_value': {'t': int, 'v': 40, 'min': 0, 'max': 200,
                                            'h': 'Sensor saturation value'},
                'obstacle_sensor_error': {'v': 0.35, 'dv': 0.01, 'max': 1.0,
                                          'h': 'Proximity sensor error'},
                'sensor_max_distance': {'v': 0.9, 'dv': 0.01, 'min': 0.1, 'max': 1.5,
                                        'h': 'Sensor max_distance'},
                'motor_ctrl_coefficient': {'t': int, 'v': 8770, 'max': 10000,
                                           'h': 'Motor ctrl_coefficient'},
                'motor_ctrl_min_actuator_value': {'t': int, 'v': 35, 'min': 0, 'max': 50,
                                                  'h': 'Motor ctrl_min_actuator_value'},
            }

            d['eval_conf'] = {
                'refID': confID_entry('Ref', default='None.150controls'),
                'modelIDs': confID_entry('Model', single_choice=False, k='mIDs'),
                'dataset_ids': {'t': List[str], 'h': 'The ids for the generated datasets', 'k': 'dIDs'},
                'offline': {**bF, 'h': 'Whether to run a full LarvaworldSim environment', 'k': 'offline'},
                'N': {'t': int, 'v': 5, 'min': 2, 'max': 1000,
                      'h': 'Number of agents per model ID',
                      'k': 'N'},
                'id': pID('evaluation run', k='id'),

            }

            return dNl.NestDict(d)

        self.dict = buildInitDict()
