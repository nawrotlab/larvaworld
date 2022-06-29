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


class ParInitDict:
    def __init__(self):
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        from typing import List, Tuple, Union, TypedDict
        from types import FunctionType
        import numpy as np
        import param
        from lib.registry.units import ureg
        from lib.aux import dictsNlists as dNl
        from lib.aux.collecting import output_keys
        from lib.conf.stored.conf import ConfSelector, kConfDict
        from lib.aux.par_aux import sub, sup, bar, circle, wave, tilde, subsup
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

            d['larva_model'] = {'t': str, 'vparfunc': ConfSelector('Model', default='explorer'), 'v': 'explorer',
                                'vs': kConfDict('Model'), 'symbol': sub('ID', 'mod'),
                                'k': 'mID', 'h': 'The stored larva model configurations as a list of IDs',
                                'disp': 'larva-model ID'}

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

        def buildInitDict():
            from lib.registry.modConfs import init_mods
            # raise
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

            d['exp_conf'] = {'env_params': {'t': str, 'vparfunc': ConfSelector('Env'), 'vs': kConfDict('Env')},
                             'larva_groups': {'t': dict, 'v': {}},
                             'sim_params': d['sim_params'],
                             'trials': {'t': str, 'vparfunc': ConfSelector('Trial', default='default'), 'v': 'default',
                                        'vs': kConfDict('Trial')},
                             'collections': {'t': List[str], 'v': ['pose']},
                             'enrichment': d['enrichment'],
                             'experiment': {'t': str, 'vparfunc': ConfSelector('Exp'), 'vs': kConfDict('Exp')},

                             }

            d['tracker'] = {
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

            # d.update(init_vis())

            d['replay'] = {
                'env_params': {'t': str, 'vparfunc': ConfSelector('Env'), 'vs': kConfDict('Env'), 'aux_vs': [''],
                               'h': 'The arena configuration to display the replay on, if not the default one in the dataset configuration.'},
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
                # 'max_Nticks': {'t': int, 'max': 100000, 'h': 'Maximum number of ticks per generation'},
                # 'max_dur': {'v': 3, 'max': 100, 'h': 'Maximum duration per generation in minutes'},
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
                'base_model': {'t': str, 'v': 'navigator', 'vs': kConfDict('Model'),
                               'vparfunc': ConfSelector('Model', default='navigator'),
                               'h': 'The model configuration to optimize', 'k': 'mID0'},
                'bestConfID': {'t': str,
                               'h': 'The model configuration ID to store the best genome',
                               'k': 'mID1'},
                'init_mode': {'t': str, 'v': 'random', 'vs': ['default', 'random', 'model'],
                              'h': 'The initialization mode for the first generation', 'k': 'mGA'},
                'multicore': {**bF, 'h': 'Whether to use multiple cores', 'k': 'multicore'},
                'fitness_target_refID': {'t': str, 'vparfunc': ConfSelector('Ref'), 'vs': kConfDict('Ref'),
                                         'h': 'The ID of the reference dataset for comparison', 'k': 'refID'},
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
                'env_params': {'t': str, 'vparfunc': ConfSelector('Env'), 'vs': kConfDict('Env'),
                               'h': 'The environment configuration ID to use in the simulation'},
                'sim_params': d['sim_params'],
                'experiment': {'t': str, 'v': 'exploration', 'vs': kConfDict('Ga'),
                               'vparfunc': ConfSelector('Ga', default='exploration'),
                               'h': 'The GA experiment configuration'},
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
                'refID': {'t': str, 'v': 'None.150controls', 'vs': kConfDict('Ref'),
                          'vparfunc': ConfSelector('Ref', default='None.150controls'),
                          'h': 'The ID of the reference dataset for comparison', 'k': 'refID'},
                'modelIDs': {'t': List[str], 'vs': kConfDict('Model'),
                             'vparfunc': ConfSelector('Model', single_choice=False),
                             'h': 'The model configurations to evaluate', 'k': 'mIDs'},
                'dataset_ids': {'t': List[str], 'h': 'The ids for the generated datasets', 'k': 'dIDs'},
                'offline': {**bF, 'h': 'Whether to run a full LarvaworldSim environment', 'k': 'offline'},
                'N': {'t': int, 'v': 5, 'min': 2, 'max': 1000,
                      'h': 'Number of agents per model ID',
                      'k': 'N'},
                'id': pID('evaluation run', k='id'),

            }

            return d

        self.dict = buildInitDict()
