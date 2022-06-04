import warnings
from types import FunctionType

import param
import siunits as siu



warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

from lib.aux.collecting import output_keys
from lib.aux.par_aux import sub, sup, bar, circle, wave, tilde, subsup

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


siu.day = siu.s * 24 * 60 * 60
siu.cm = siu.m * 10 ** -2
siu.mm = siu.m * 10 ** -3
siu.g = siu.kg * 10 ** -3
siu.deg = siu.I.rename("deg", "deg", "plain angle")
siu.microM = siu.mol * 10 ** -6
unit_dic = {None: 1 * siu.I,'# $segments$': 1 * siu.I, 'rad': 1 * siu.rad, 'deg': 1 * siu.deg,
             'sec': 1 * siu.s, 'm': 1 * siu.m,'cm': 1 * siu.cm, '$mm$': 1 * siu.mm,'mm': 1 * siu.mm}

def init_vis():
    d = {}
    d['render'] = {
        'mode': {'t': str, 'v': None, 'vs': [None, 'video', 'image'], 'h': 'The visualization mode', 'k': 'm'},
        'image_mode': {'t': str, 'vs': [None, 'final', 'snapshots', 'overlap'], 'h': 'The image-render mode',
                       'k': 'im'},
        'video_speed': {'t': int, 'v': 60, 'min': 1, 'max': 100, 'h': 'The video speed', 'k': 'fps'},
        'media_name': {'t': str, 'h': 'Filename for the saved video/image. File extension mp4/png sutomatically added.',
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
        'color_behavior': {'t': bool, 'v': False, 'h': 'Color the larvae according to their instantaneous behavior'},
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


def init_pars():
    from typing import List, Tuple, Union, TypedDict
    from lib.conf.stored.conf import kConfDict, ConfSelector
    import param

    bF, bT = {'t': bool, 'v': False, 'vfunc': param.Boolean}, {'t': bool, 'v': True, 'vfunc': param.Boolean}

    d = {
        'bout_distro': {
            'fit': {**bT, 'combo': 'distro',
                    'h': 'Whether the distribution is sampled from a reference dataset. Once this is set to "ON" no other parameter is taken into account.'},
            'range': {'t': Tuple[float], 'max': 100.0, 'combo': 'distro', 'h': 'The distribution range.', 'vfunc': param.Range},
            'name': {'t': str, 'vfunc': param.Selector,
                     'vs': ['powerlaw', 'exponential', 'lognormal', 'lognormal-powerlaw', 'levy', 'normal', 'uniform'],
                     'combo': 'distro', 'h': 'The distribution name.'},
            'mu': {'max': 10.0, 'disp': 'mean', 'combo': 'distro', 'vfunc': param.Number,
                   'h': 'The "mean" argument for constructing the distribution.'},
            'sigma': {'max': 10.0, 'disp': 'std', 'combo': 'distro','vfunc': param.Number,
                      'h': 'The "sigma" argument for constructing the distribution.'},
        },
        'xy': {'t': Tuple[float], 'v': (0.0, 0.0),'k': 'xy','lim': (-1.0, 1.0), 'min': -1.0, 'max': 1.0,'vfunc': param.XYCoordinates,
               'h': 'The xy spatial position coordinates.'},
        'odor': {
            'odor_id': {'t': str, 'disp': 'ID', 'h': 'The unique ID of the odorant.','vfunc': param.String},
            'odor_intensity': {'max': 10.0, 'disp': 'C peak','vfunc': param.Number,
                               'h': 'The peak concentration of the odorant in micromoles.'},
            'odor_spread': {'max': 10.0, 'disp': 'spread','vfunc': param.Number,
                            'h': 'The spread of the concentration gradient around the peak.'}
        },

        'odorscape': {'odorscape': {'t': str, 'v': 'Gaussian', 'vs': ['Gaussian', 'Diffusion'], 'vfunc': param.Selector,'k':'odorscape_mod',
                                    'h': 'The algorithm used for odorscape generation.'},
                      'grid_dims': {'t': Tuple[int], 'v': (51, 51), 'min': 10, 'max': 100,'vfunc': param.Tuple,'k':'grid_dims',
                                    'h': 'The odorscape grid resolution.'},
                      'evap_const': {'max': 1.0,'k':'c_evap', 'h': 'The evaporation constant of the diffusion algorithm.','vfunc': param.Magnitude},
                      'gaussian_sigma': {'t': Tuple[float], 'lim': (0.0,1.0),'max': 1.0,'vfunc': param.NumericTuple,'k':'gau_sigma',
                                         'h': 'The sigma of the gaussian difusion algorithm.'}
                      },
        'air_puff': {
            'duration': {'v': 1.0, 'max': 100.0, 'h': 'The duration of the air-puff in seconds.','vfunc': param.Number},
            'speed': {'v': 10.0, 'max': 1000.0, 'h': 'The wind speed of the air-puff.','vfunc': param.Number},
            'direction': {'v': 0.0, 'max': 100.0, 'h': 'The directions of the air puff in radians.','vfunc': param.Number},
            'start_time': {'v': 0.0, 'max': 10000.0, 'dv': 1.0, 'h': 'The starting time of the air-puff in seconds.','vfunc': param.Number},
            'N': {'t': int,'max': 10000,'lim' : (0,10000), 'h': 'The number of repetitions of the puff. If N>1 an interval must be provided',
                  'vfunc': param.Integer},
            'interval': {'v': 5.0, 'max': 10000.0,'vfunc': param.Number,
                         'h': 'Whether the puff will reoccur at constant time intervals in seconds. Ignored if N=1'},
        },
        'windscape': {'wind_direction': {'t': float, 'v': np.pi, 'min': 0.0, 'max': 2 * np.pi, 'dv': 0.1,'vfunc': param.Number,
                                         'h': 'The absolute polar direction of the wind/air puff.'},
                      'wind_speed': {'t': float, 'v': 0.0, 'min': 0.0, 'max': 100.0, 'dv': 1.0,'vfunc': param.Number,
                                     'h': 'The speed of the wind/air puff.'},
                      'puffs': {'t': TypedDict, 'v': {}, 'entry': 'air_puff', 'disp': 'air-puffs',
                                'h': 'Repetitive or single air-puff stimuli.'}
                      },
        'odor_gains': {
            'unique_id': {'t': str, 'h': 'The unique ID of the odorant.','vfunc': param.String},
            'mean': {'max': 1000.0, 'dv': 10.0,'vfunc': param.Number,
                     'h': 'The mean gain/valence for the odorant. Positive/negative for appettitive/aversive valence.'},
            'std': {'max': 10.0, 'dv': 1.0, 'h': 'The standard deviation for the odorant gain/valence.','vfunc': param.Number}
        },
        'optimization': {
            'fit_par': {'t': str, 'disp': 'Utility metric', 'h': 'The utility parameter optimized.','vfunc': param.String},
            'minimize': {**bT, 'h': 'Whether to minimize or maximize the utility parameter.'},
            'threshold': {'v': 0.001, 'max': 0.01, 'dv': 0.0001,'vfunc': param.Number,
                          'h': 'The utility threshold to reach before terminating the batch-run.'},
            'max_Nsims': {'t': int, 'v': 7, 'max': 100,'vfunc': param.Integer,
                          'h': 'The maximum number of single runs before terminating the batch-run.'},
            'Nbest': {'t': int, 'v': 3, 'max': 10,'vfunc': param.Integer,
                      'h': 'The number of best parameter combinations to use for generating the next generation.'},
            'operations': {
                'mean': {**bT, 'h': 'Whether to use the mean of the utility across individuals'},
                'std': {**bF, 'h': 'Whether to use the standard deviation of the utility across individuals'},
                'abs': {**bF, 'h': 'Whether to use the absolute value of the utility'}
            },
        },
        'batch_methods': {
            'run': {'t': str, 'v': 'default','vfunc': param.Selector, 'vs': ['null', 'default', 'deb', 'odor_preference', 'exp_fit'],
                    'h': 'The method to be applied on simulated data derived from every individual run'},
            'post': {'t': str, 'v': 'default','vfunc': param.Selector, 'vs': ['null', 'default'],
                     'h': 'The method to be applied after a generation of runs is completed to judge whether space-search will continue or batch-run will be terminated.'},
            'final': {'t': str, 'v': 'null','vfunc': param.Selector, 'vs': ['null', 'scatterplots', 'deb', 'odor_preference'],
                      'h': 'The method to be applied once the batch-run is complete to plot/save the results.'}
        },
        'space_search_par': {
            'range': {'t': Tuple[float], 'max': 100.0, 'min': -100.0, 'dv': 1.0,'vfunc': param.Range,'k': 'ss.range',
                      'h': 'The parameter range to perform the space-search.'},
            'Ngrid': {'t': int, 'max': 100, 'disp': '# steps','vfunc': param.Integer,'k': 'ss.Ngrid',
                      'h': 'The number of equally-distanced values to parse the parameter range.'},
            'values': {'t': List[float], 'min': -100.0, 'max': 100.0,'vfunc': param.List,'k': 'ss.vs',
                       'h': 'A list of values of the parameter to space-search. Once this is filled no range/# steps parameters are taken into account.'}
        },
        'space_search': {'pars': {'t': List[str], 'h': 'The parameters for space search.', 'k': 'ss.pars','vfunc': param.List},
                         'ranges': {'t': List[Tuple[float]], 'max': 100.0, 'min': -100.0, 'dv': 1.0,'vfunc': param.List,
                                    'h': 'The range of the parameters for space search.', 'k': 'ss.ranges'},
                         'Ngrid': {'t': int, 'max': 100, 'h': 'The number of steps for space search.','vfunc': param.Integer,
                                   'k': 'ss.Ngrid'}},

        'arena': {'arena_dims': {'t': Tuple[float], 'v': (0.1, 0.1), 'max': 1.0, 'dv': 0.01, 'disp': 'X,Y (m)','vfunc': param.NumericTuple,
                                 'h': 'The arena dimensions in meters.'},
                  'arena_shape': {'t': str, 'v': 'circular', 'vs': ['circular', 'rectangular'], 'disp': 'shape','vfunc': param.Selector,
                                  'h': 'The arena shape.'}
                  },
        'physics': {
            'torque_coef': {'v': 0.5, 'min': 0.5, 'max': 0.5, 'dv': 0.01, 'label': 'torque coefficient','vfunc': param.Number,
                            'symbol': sub('c', 'T'),'u_name': sup('sec', -2),'u' : 1*siu.s * 10**-2,
                            'h': 'Conversion coefficient from TURNER output to torque-per-inertia-unit.'},
            'ang_vel_coef': {'v': 1.0, 'max': 5.0, 'dv': 0.01, 'label': 'angular velocity coefficient','vfunc': param.Number,
                             'h': 'Conversion coefficient from TURNER output to angular velocity.'},
            'ang_damping': {'v': 1.0, 'min': 1.0, 'max': 1.0, 'label': 'angular damping', 'symbol': 'z','vfunc': param.Number,
                            'u_name': sup('sec', -1), 'u': 1 * siu.s * 10 ** -1,
                            'h': 'Angular damping exerted on angular velocity.'},
            'lin_damping': {'v': 1.0, 'min': 0.0, 'max': 10.0, 'label': 'linear damping', 'symbol': 'zl','vfunc': param.Number,
                            'u_name': sup('sec', -1),'u' : 1*siu.s * 10**-1,
                            'h': 'Linear damping exerted on forward velocity.'},
            'body_spring_k': {'v': 1.0, 'min': 1.0, 'max': 1.0, 'dv': 0.1, 'label': 'body spring constant','vfunc': param.Number,
                              'symbol': 'k', 'u_name': sup('sec', -2),'u' : 1*siu.s * 10**-2,
                              'h': 'Larva-body torsional spring constant reflecting deformation resistance.'},
            'bend_correction_coef': {'v': 1.0, 'min': 0.8, 'max': 1.5, 'label': 'bend correction coefficient','vfunc': param.Number,
                                     'symbol': sub('c', 'b'), 'u_name':None,
                                     'h': 'Correction coefficient of bending angle during forward motion.'},
            'ang_mode': {'t': str, 'v': 'torque', 'vs': ['torque', 'velocity'], 'label': 'angular mode','vfunc': param.Selector,
                         'h': 'Whether the Turner module output is equivalent to torque or angular velocity.'},
        },

        'crawler': {'waveform': {'t': str, 'v': 'realistic','k' : 'Cr_mod', 'vs': ['realistic', 'square', 'gaussian', 'constant'],'vfunc': param.Selector,
                                 'h': 'The waveform of the repetitive crawling oscillator (CRAWLER) module.'},
                    'freq_range': {'t': Tuple[float], 'v': (0.5, 2.5), 'max': 2.0, 'disp': 'range','vfunc': param.Range,
                                   'combo': 'frequency','k' : 'f_C_r','lim': (0.0,5.0),'u': 1 * siu.hz,
                                   'h': 'The frequency range of the repetitive crawling behavior.'},
                    'initial_freq': {'v': 1.418, 'max': 5.0,'lim': (0.0,5.0), 'aux_vs': ['sample'], 'disp': 'initial','k' : 'f_C0','vfunc': param.Number,
                                     'label': 'crawling frequency', 'symbol': sub('f', 'C'), 'u': 1 * siu.hz,
                                     'combo': 'frequency', 'codename': 'scaled_velocity_freq',
                                     'h': 'The initial frequency of the repetitive crawling behavior.'},  # From D1 fit
                    'freq_std': {'v': 0.0, 'max': 1.0,'lim': (0.0,1.0),'k' : 'f_C_std', 'disp': 'std', 'combo': 'frequency','u': 1 * siu.hz,'vfunc': param.Number,
                                 'h': 'The standard deviation of the frequency of the repetitive crawling behavior.'},
                    'max_scaled_vel': {'v': 0.6, 'max': 1.5,'lim': (0.0,1.5), 'label': 'maximum scaled velocity','vfunc': param.Number,
                                       'codename': 'stride_scaled_velocity_max','k' : 'sstr_v_max',
                                       'symbol': sub(circle('v'), 'max'), 'u': siu.s * 10**-1,'u_name' : '$body-lengths/sec$',
                                       'h': 'The maximum scaled forward velocity.'},
                    'stride_dst_mean': {'v': 0.224, 'max': 1.0,'lim': (0.0,1.0), 'dv': 0.01, 'aux_vs': ['sample'], 'disp': 'mean','k' : 'sstr_d_mu','vfunc': param.Number,
                                        'label': r'stride distance mean', 'symbol': sub(bar(circle('d')), 'S'),
                                        'u': 1*siu.I,'u_name' : '$body-lengths$',
                                        'combo': 'scaled distance / stride', 'codename': 'scaled_stride_dst_mean',
                                        'h': 'The mean displacement achieved in a single peristaltic stride as a fraction of the body length.'},
                    'stride_dst_std': {'v': 0.033, 'max': 1.0,'lim': (0.0,1.0), 'aux_vs': ['sample'], 'disp': 'std','k' : 'sstr_d_std','vfunc': param.Number,
                                       'label': 'stride distance std', 'symbol': sub(tilde(circle('d')), 'S'),
                                       'u': 1*siu.I,'u_name' : '$body-lengths$',
                                       'combo': 'scaled distance / stride', 'codename': 'scaled_stride_dst_std',
                                       'h': 'The standard deviation of the displacement achieved in a single peristaltic stride as a fraction of the body length.'},
                    'initial_amp': {'max': 2.0,'lim': (0.0,2.0), 'disp': 'initial', 'combo': 'amplitude','k' : 'A_C0','vfunc': param.Number,
                                    'h': 'The initial amplitude of the CRAWLER-generated forward velocity if this is hardcoded (e.g. constant waveform).'},
                    'noise': {'v': 0.0, 'max': 1.0, 'lim': (0.0,1.0),'dv': 0.01, 'disp': 'noise', 'combo': 'amplitude','k' : 'A_Cnoise','vfunc': param.Magnitude,
                              'h': 'The intrinsic output noise of the CRAWLER-generated forward velocity.'},
                    'max_vel_phase': {'v': 3.6, 'max': 2 * np.pi,'lim': (0.0,2 * np.pi), 'label': 'max velocity phase','k' : 'phi_v_max','vfunc': param.Number,
                                      'symbol': subsup('$\phi$', 'C', 'v'), 'u_name': 'rad','u' : 1*siu.rad,
                                      'codename': 'phi_scaled_velocity_max',
                                      'h': 'The phase of the crawling oscillation cycle where forward velocity is maximum.'}
                    },



        'olfactor': {
            'perception': {'t': str, 'v': 'log', 'vs': ['log', 'linear', 'null'], 'label': 'sensory transduction mode','vfunc': param.Selector,
                           'symbol': '-', 'u_name': None,
                           'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input.'},
            'input_noise': {'v': 0.0, 'max': 1.0, 'h': 'The intrinsic noise of the sensory input.','vfunc': param.Magnitude},
            'decay_coef': {'v': 0.0, 'max': 2.0, 'label': 'decay coefficient', 'symbol': sub('c', 'O'), 'u_name':None,'vfunc': param.Number,
                           'h': 'The linear decay coefficient of the olfactory sensory activation.'},
            'brute_force': {**bF, 'h': 'Whether to apply direct rule-based modulation on locomotion or not.'}
        },
        'windsensor': {
            'weights': {
                'hunch_lin': {'v': 10.0, 'min': -100.0, 'max': 100.0, 'disp': 'HUNCH->CRAWLER','vfunc': param.Number,
                              'h': 'The connection weight between the HUNCH neuron ensemble and the CRAWLER module.'},
                'hunch_ang': {'v': 0.0, 'min': -100.0, 'max': 100.0, 'disp': 'HUNCH->TURNER','vfunc': param.Number,
                              'h': 'The connection weight between the HUNCH neuron ensemble and the TURNER module.'},
                'bend_lin': {'v': 0.0, 'min': -100.0, 'max': 100.0, 'disp': 'BEND->CRAWLER','vfunc': param.Number,
                             'h': 'The connection weight between the BEND neuron ensemble and the CRAWLER module.'},
                'bend_ang': {'v': -10.0, 'min': -100.0, 'max': 100.0, 'disp': 'BEND->TURNER','vfunc': param.Number,
                             'h': 'The connection weight between the BEND neuron ensemble and the TURNER module.'},
            }
        },
        'toucher': {
            'perception': {'t': str, 'v': 'linear', 'vs': ['log', 'linear'],'vfunc': param.Selector,
                           'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input.'},
            'input_noise': {'v': 0.0, 'max': 1.0, 'h': 'The intrinsic noise of the sensory input.','vfunc': param.Magnitude},
            'decay_coef': {'v': 0.1, 'max': 2.0,'vfunc': param.Number,
                           'h': 'The exponential decay coefficient of the tactile sensory activation.'},
            'state_specific_best': {**bT,
                                    'h': 'Whether to use the state-specific or the global highest evaluated gain after the end of the memory training period.'},
            'brute_force': {**bF, 'h': 'Whether to apply direct rule-based modulation on locomotion or not.'},
            'initial_gain': {'v': 40.0,'vfunc': param.Number, 'min': -100.0, 'max': 100.0, 'h': 'The initial gain of the tactile sensor.'}
        },
        'feeder': {
            'freq_range': {'t': Tuple[float], 'v': (1.0, 3.0), 'lim': (0.0, 4.0), 'max': 4.0, 'disp': 'range', 'combo': 'frequency','vfunc': param.Range,
                           'h': 'The frequency range of the repetitive feeding behavior.'},
            'initial_freq': {'v': 2.0, 'max': 4.0, 'disp': 'initial', 'combo': 'frequency','vfunc': param.Number,
                             'h': 'The initial default frequency of the repetitive feeding behavior.'},
            'feed_radius': {'v': 0.1, 'max': 10.0,'vfunc': param.Number,
                            'h': 'The radius around the mouth in which food is consumable as a fraction of the body length.'},
            'V_bite': {'v': 0.0005, 'max': 0.01, 'dv': 0.0001,'vfunc': param.Number,
                       'h': 'The volume of food consumed on a single feeding motion as a fraction of the body volume.'}
        },
        'memory': {'modality': {'t': str, 'v': 'olfaction', 'vs': ['olfaction', 'touch'],'vfunc': param.Selector,
                                'h': 'The modality for which the memory module is used.'},
                   'Delta': {'v': 0.1, 'max': 10.0, 'h': 'The input sensitivity of the memory.','vfunc': param.Number},
                   'state_spacePerSide': {'t': int, 'v': 0, 'max': 20, 'disp': 'state space dim','vfunc': param.Integer,
                                          'h': 'The number of discrete states to parse the state space on either side of 0.'},
                   'gain_space': {'t': List[float], 'v': [-300.0, -50.0, 50.0, 300.0], 'min': 1000.0, 'max': 1000.0,'vfunc': param.List,
                                  'dv': 1.0, 'h': 'The possible values for memory gain to choose from.'},
                   'update_dt': {'v': 1.0, 'max': 10.0, 'dv': 1.0, 'h': 'The interval duration between gain switches.','vfunc': param.Number},
                   'alpha': {'v': 0.05, 'max': 1.0, 'dv': 0.01,'vfunc': param.Number,
                             'h': 'The alpha parameter of reinforcement learning algorithm.'},
                   'gamma': {'v': 0.6, 'max': 1.0,'vfunc': param.Number,
                             'h': 'The probability of sampling a random gain rather than exploiting the currently highest evaluated gain for the current state.'},
                   'epsilon': {'v': 0.3, 'max': 1.0,'vfunc': param.Number, 'h': 'The epsilon parameter of reinforcement learning algorithm.'},
                   'train_dur': {'v': 20.0, 'max': 100.0,'vfunc': param.Number,
                                 'h': 'The duration of the training period after which no further learning will take place.'}
                   },
        'modules': {'turner': bF,
                    'crawler': bF,
                    'interference': bF,
                    'intermitter': bF,
                    'olfactor': bF,
                    'windsensor': bF,
                    'toucher': bF,
                    'feeder': bF,
                    'memory': bF},

        'essay_params': {
            'essay_ID': {'t': str, 'h': 'The unique ID of the essay.','vfunc': param.String},
            'path': {'t': str, 'h': 'The relative path to store the essay datasets.', 'vfunc': param.Foldername},
            'N': {'t': int, 'min': 1, 'max': 100, 'disp': '# larvae', 'h': 'The number of larvae per larva-group.','vfunc': param.Integer}
        },

        'sim_params': {
            'sim_ID': {'t': str, 'h': 'The unique ID of the simulation.', 'k': 'id','vfunc': param.String},
            'path': {'t': str, 'h': 'The relative path to save the simulation dataset.', 'k': 'path', 'vfunc': param.Foldername},
            'duration': {'max': 100.0, 'h': 'The duration of the simulation in minutes.','vfunc': param.Number, 'k': 't'},
            'timestep': {'v': 0.1, 'max': 0.4, 'dv': 0.05, 'h': 'The timestep of the simulation in seconds.','vfunc': param.Number,
                         'k': 'dt'},
            'Box2D': {**bF, 'h': 'Whether to use the Box2D physics engine or not.', 'k': 'Box2D'},
            'store_data': {**bT, 'h': 'Whether to store the simulation data or not.', 'k': 'no_store'},
            # 'analysis': {'t': bool, 'v': True, 'h': 'Whether to analyze the simulation data', 'k' : 'no_analysis'},
        },

        'logn_dist': {
            'range': {'t': Tuple[float], 'v': (0.0, 2.0), 'max': 10.0, 'dv': 1.0,'vfunc': param.Range},
            'name': {'t': str, 'v': 'lognormal', 'vs': ['lognormal'],'vfunc': param.Selector},
            'mu': {'v': 1.0, 'max': 10.0,'vfunc': param.Number},
            'sigma': {'v': 0.0, 'max': 10.0,'vfunc': param.Number},
            'fit': bF
        },
        'par': {
            'p': {'t': str},
            'u': {'t': Union[siu.BaseUnit, siu.Composite, siu.DerivedUnit]},
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
            'min_duration_in_sec': {'v': 170.0, 'max': 3600.0, 'dv': 1.0, 'disp': 'min track duration (sec)','vfunc': param.Number},
            'min_end_time_in_sec': {'v': 0.0, 'max': 3600.0, 'dv': 1.0, 'disp': 'min track termination time (sec)','vfunc': param.Number},
            'start_time_in_sec': {'v': 0.0, 'max': 3600.0, 'dv': 1.0, 'disp': 'track initiation time (sec)','vfunc': param.Number},
            'max_Nagents': {'t': int, 'v': 500, 'max': 5000, 'disp': 'max # larvae','vfunc': param.Integer},
            'save_mode': {'t': str, 'v': 'semifull', 'vs': ['minimal', 'semifull', 'full', 'points'],'vfunc': param.Selector},
        },
        # 'substrate': {k: {'v': v, 'max': 1000.0, 'dv': 5.0} for k, v in substrate_dict['standard'].items()},
        'output': {n: bF for n in output_keys}
    }

    d['square_interference'] = {
                            'crawler_phi_range': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': 2 * np.pi,'vfunc': param.Range,
                                                  'label': 'suppression relief phase interval',
                                                  'symbol': '$[\phi_{C}^{\omega_{0}},\phi_{C}^{\omega_{1}}]$',
                                                  'u_name': '$rads$','u' : 1*siu.rad,
                                                  'h': 'CRAWLER phase range for TURNER suppression lift.'},
                            'feeder_phi_range': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': 2 * np.pi,'vfunc': param.Range,
                                                 'label': 'feeder suppression relief phase interval',
                                                 'symbol': '$[\phi_{F}^{\omega_{0}},\phi_{F}^{\omega_{1}}]$',
                                                 'u_name': '$rads$','u' : 1*siu.rad,
                                                 'h': 'FEEDER phase range for TURNER suppression lift.'}

                        }

    d['phasic_interference'] = {
         'max_attenuation_phase': {'v': 3.4, 'max': 2 * np.pi, 'label': 'max relief phase','lim': (0.0, 2 * np.pi),'vfunc': param.Number,
                                      'symbol': '$\phi_{C}^{\omega}$', 'u_name': 'rad','u' : 1*siu.rad,'k' : 'phi_fov_max',
                                      'h': 'CRAWLER phase of minimum TURNER suppression.'}
    }

    d['base_interference'] = {
            'mode': {'t': str, 'v': 'square','k' : 'IF_mod', 'vs': ['default', 'square', 'phasic'],'vfunc': param.Selector,
                     'h': 'CRAWLER:TURNER suppression phase mode.'},
            'suppression_mode': {'t': str, 'v': 'amplitude', 'vs': ['amplitude', 'oscillation', 'both'],'k' : 'IF_target','vfunc': param.Selector,
                                 'label': 'suppression target', 'symbol': '-', 'u_name': None,
                                 'h': 'CRAWLER:TURNER suppression target.'},
            'attenuation': {'v': 1.0, 'max': 1.0,'lim': (0.0, 1.0),  'label': 'suppression coefficient', 'symbol': '$c_{CT}^{0}$',
                            'u_name': None,'k' : 'c_CT0','vfunc': param.Magnitude,
                            'h': 'CRAWLER:TURNER baseline suppression coefficient'},
            'attenuation_max': {'v': 0.31, 'max': 1.0,'lim': (0.0, 1.0), 'label': 'suppression relief coefficient.','vfunc': param.Magnitude,
                                'symbol': '$c_{CT}^{1}$','u_name': None,'k' : 'c_CT1',
                                'h': 'CRAWLER:TURNER suppression relief coefficient.'},
        }

    d['interference'] = {
        **d['base_interference'],
        **d['square_interference'],
        **d['phasic_interference']
        }

    d['neural_turner'] = {
        'base_activation': {'v': 20.0, 'min': 10.0, 'max': 40.0,'lim': (10.0,40.0), 'dv': 0.1, 'disp': 'mean', 'combo': 'activation',
                            'label': 'tonic input', 'symbol': '$I_{T}^{0}$','u_name': None,'u' : 1*siu.I,'k' : 'I_T0','vfunc': param.Number,
                            'h': 'The baseline activation/input of the TURNER module.'},
        'activation_range': {'t': Tuple[float], 'v': (10.0, 40.0), 'max': 100.0,'lim': (0.0,100.0), 'dv': 0.1, 'disp': 'range','k' : 'I_T_r',
                             'label': 'input range', 'symbol': r'$[I_{T}^{min},I_{T}^{max}]$','u_name': None,'u' : 1*siu.I,'vfunc': param.Range,
                             'combo': 'activation', 'h': 'The activation/input range of the TURNER module.'},

        'tau': {'v': 0.1, 'min': 0.05, 'max': 0.5, 'dv': 0.01, 'label': 'time constant', 'symbol': r'$\tau_{T}$','vfunc': param.Number,
                'u_name': 'sec','u' : 1*siu.s,
                'h': 'The time constant of the neural oscillator.'},
        'm': {'v': 100, 'min': 50, 'max': 200, 'label': 'maximum spike-rate', 'symbol': '$SR_{max}$','u_name': None,'vfunc': param.Integer,
              'h': 'The maximum allowed spike rate.'},
        'n': {'v': 2.0, 'min': 1.5, 'max': 3.0, 'dv': 0.01, 'label': 'spike response steepness', 'symbol': '$n_{T}$','vfunc': param.Number,
              'u_name': None,
              'h': 'The neuron spike-rate response steepness coefficient.'}

    }

    d['sinusoidal_turner'] = {'initial_amp': {'v': 19.27, 'min': 0.1,'max': 100.0,'lim': (0.1,100.0),'disp': 'initial', 'combo': 'amplitude',
                                              'label': 'output amplitude', 'symbol': '$A_{T}^{0}$', 'u_name': None,'k' : 'A_T0','vfunc': param.Number,
                                              'h': 'The initial activity amplitude of the TURNER module.'},
                              'amp_range': {'t': Tuple[float],'min': 0.1, 'max': 100.0,'lim': (0.1,100.0),'v': (10.0,30.0), 'disp': 'range', 'combo': 'amplitude',
                                            'label': 'output amplitude range', 'symbol': r'$[A_{T}^{min},A_{T}^{max}]$','vfunc': param.Range,
                                            'u_name': None,'k' : 'A_T_r',
                                            'h': 'The activity amplitude range of the TURNER module.'},
                              'initial_freq': {'v': 0.58, 'min': 0.01, 'max': 2.0,'dv': 0.01,  'disp': 'initial', 'combo': 'frequency','k' : 'f_T0',
                                               'label': 'bending frequency', 'symbol': sub('f', 'T'), 'u_name': '$Hz$','u' : 1*siu.hz,'vfunc': param.Number,
                                               'h': 'The initial frequency of the repetitive lateral bending behavior if this is hardcoded (e.g. sinusoidal mode).'},
                              'freq_range': {'t': Tuple[float],'min': 0.01, 'max': 2.0,'lim': (0.01,2.0),'dv': 0.01,  'disp': 'range', 'combo': 'frequency',
                                             'label': 'bending frequency range','k' : 'f_T_r','v': (0.1,0.8),'vfunc': param.Range,
                                             'symbol': r'$[f_{T}^{min},f_{T}^{max}]$','u_name': '$Hz$','u' : 1*siu.hz,
                                             'h': 'The frequency range of the repetitive lateral bending behavior.'}
                              }

    d['constant_turner'] = {'initial_amp': {'min': 0.1,'max': 20.0,'disp': 'initial', 'combo': 'amplitude','k' : 'A_T0',
                                              'label': 'output amplitude', 'symbol': '$A_{T}^{0}$','u_name': None,'vfunc': param.Number,
                                              'h': 'The initial activity amplitude of the TURNER module.'},
                              'amp_range': {'t': Tuple[float],'min': 0.1, 'max': 20.0,'lim' : (0.1,20.0), 'disp': 'range', 'combo': 'amplitude',
                                            'label': 'output amplitude range', 'symbol': '$[A_{T}^{min},A_{T}^{max}]$','k' : 'A_T_r','vfunc': param.Range,
                                            'u_name': None,'u' : 1*siu.I,
                                            'h': 'The activity amplitude range of the TURNER module.'}
                            }

    d['base_turner'] = {'mode': {'t': str, 'v': 'neural', 'vs': ['', 'neural', 'sinusoidal', 'constant'],'k' : 'Tur_mod','vfunc': param.Selector,
                            'h': 'The implementation mode of the lateral oscillator (TURNER) module.'},

                   'noise': {'v': 0.0, 'max': 1.0, 'disp': 'noise', 'combo': 'amplitude','k' : 'A_Tnoise','u_name': None,'u' : 1*siu.I,'vfunc': param.Magnitude,
                             'h': 'The intrinsic output noise of the TURNER activity amplitude.'},
                   'activation_noise': {'v': 0.0, 'max': 1.0, 'disp': 'noise', 'combo': 'activation','k' : 'I_Tnoise','u_name': None,'u' : 1*siu.I,'vfunc': param.Magnitude,
                                        'h': 'The intrinsic input noise of the TURNER module.'}
                            }

    d['turner'] = {
                   **d['base_turner'],
                   **d['neural_turner'],
                   **d['sinusoidal_turner']
                   }

    d['Box2D_joint_N'] = {'t': int, 'v': 0, 'max': 2,'vfunc': param.Integer}

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
        'symmetry': {'t': str, 'v': 'bilateral', 'vs': ['bilateral', 'radial'],'vfunc': param.Selector,
                     'h': 'The body symmetry.'},
        'Nsegs': {'t': int, 'v': 2, 'min': 1, 'max': 12,'vfunc': param.Integer,
                  'h': 'The number of segments comprising the larva body.'},
        'seg_ratio': {'max': 1.0,'vfunc': param.Magnitude,
                      'h': 'The length ratio of the body segments. If null, equal-length segments are generated.'},

        'olfaction_sensors': {'t': List[int], 'min': 0, 'max': 16, 'v': [0], 'disp': 'olfaction','vfunc': param.List,
                              'h': 'The indexes of the contour points bearing olfaction sensors.'},

        'touch_sensors': {'t': List[int], 'min': 0, 'max': 16, 'disp': 'touch','vfunc': param.List,
                          'h': 'The indexes of the contour points bearing touch sensors.'},
        'points': {'t': List[Tuple[float]], 'min': -1.0, 'max': 1.0, 'disp': 'contour','vfunc': param.List,
                   'h': 'The XY coordinates of the body contour.'},
    }

    d['body'] = {'initial_length': {'v': 0.004, 'max': 0.01, 'lim': (0.0,0.01),'dv': 0.0001, 'aux_vs': ['sample'], 'disp': 'initial',
                                    'label': 'length', 'symbol': '$l$', 'u_name': '$mm$','k' : 'l0','vfunc': param.Number,
                                    'combo': 'length', 'h': 'The initial body length.'},
                 'length_std': {'v': 0.0, 'max': 0.001,'lim': (0.0,0.001), 'dv': 0.0001, 'u_name': '$mm$', 'aux_vs': ['sample'],'vfunc': param.Number, 'disp': 'std','k' : 'l_std',
                                'combo': 'length', 'h': 'The standard deviation of the initial body length.'},
                 'Nsegs': {'t': int, 'v': 2, 'min': 1, 'max': 12, 'label': 'number of body segments', 'symbol': '-','vfunc': param.Integer,
                           'u_name': '# $segments$', 'u' : 1*siu.I,'k' : 'Nsegs',
                           'h': 'The number of segments comprising the larva body.'},
                 'seg_ratio': {'max': 1.0,'k' : 'seg_r','lim': (0.0,1.0),'u' : 1*siu.I,'vfunc': param.Magnitude,
                               'h': 'The length ratio of the body segments. If null, equal-length segments are generated.'},
                 # [5 / 11, 6 / 11]
                 'touch_sensors': {'t': int, 'min': 0, 'max': 8,'u' : 1*siu.I,'k' : 'sens_touch','vfunc': param.Integer,
                                   'h': 'The number of touch sensors existing on the larva body.'},
                 'shape': {'t': str, 'v': 'drosophila_larva', 'vs': ['drosophila_larva', 'zebrafish_larva'],'k' : 'body_shape','u' : None,'vfunc': param.Selector,
                           'h': 'The body shape.'},
                 }

    d['intermitter'] = {
        'mode': {'t': str, 'v': 'default', 'vs': ['', 'default', 'branch', 'nengo'],'vfunc': param.Selector,
                 'h': 'The implementation mode of the intermittency (INTERMITTER) module.'},
        'run_mode': {'t': str, 'v': 'stridechain', 'vs': ['stridechain', 'run'],'vfunc': param.Selector,
                     'h': 'The generation mode of run epochs.'},
        'stridechain_dist': d['bout_distro'],
        'run_dist': d['bout_distro'],
        'pause_dist': d['bout_distro'],
        'EEB': {'v': 0.0, 'max': 1.0,'vfunc': param.Magnitude,
                'h': 'The baseline exploitation-exploration balance. 0 means only exploitation, 1 only exploration.'},
        'EEB_decay': {'v': 1.0, 'max': 2.0,'vfunc': param.Number,
                      'h': 'The exponential decay coefficient of the exploitation-exploration balance when no food is detected.'},
        'crawl_bouts': {**bT, 'disp': 'crawling bouts',
                        'h': 'Whether crawling bouts (runs/stridechains) are generated.'},
        'feed_bouts': {**bF, 'disp': 'feeding bouts', 'h': 'Whether feeding bouts (feedchains) are generated.'},
        'crawl_freq': {'v': 1.43, 'max': 2.0, 'dv': 0.01, 'disp': 'crawling frequency','vfunc': param.Number,
                       'h': 'The default frequency of the CRAWLER oscillator when simulating offline.'},
        'feed_freq': {'v': 2.0, 'max': 4.0, 'dv': 0.01, 'disp': 'feeding frequency','vfunc': param.Number,
                      'h': 'The default frequency of the FEEDER oscillator when simulating offline.'},
        'feeder_reoccurence_rate': {'max': 1.0, 'disp': 'feed reoccurence','vfunc': param.Magnitude,
                                    'h': 'The default reoccurence rate of the feeding motion.'}

    }

    d['substrate_composition'] = {n: {'v': 0.0, 'max': 10.0, 'h': f'{n} density in g/cm**3.','vfunc': param.Number} for n in
                                  ['glucose', 'dextrose', 'saccharose', 'yeast', 'agar', 'cornmeal']}

    d['substrate'] = {
        'type': {'t': str, 'v': 'standard', 'vs': list(substrate_dict.keys()),'vfunc': param.Selector, 'h': 'The type of substrate.'},
        'quality': {'v': 1.0, 'max': 1.0,'vfunc': param.Magnitude,
                    'h': 'The substrate quality as percentage of nutrients relative to the intact substrate type.'}

    }

    d['food'] = {
        'radius': {'v': 0.003, 'max': 0.1, 'dv': 0.001,'vfunc': param.Number, 'h': 'The spatial radius of the source in meters.'},
        'amount': {'v': 0.0, 'max': 1.0,'vfunc': param.Number, 'h': 'The food amount in the source.'},
        'can_be_carried': {**bF, 'disp': 'carriable', 'h': 'Whether the source can be carried around.'},
        'can_be_displaced': {**bF, 'disp': 'displaceable', 'h': 'Whether the source can be displaced by wind/water.'},
        **d['substrate']
    }
    d['food_grid'] = {
        'unique_id': {'t': str, 'v': 'Food_grid','vfunc': param.String, 'disp': 'ID', 'h': 'The unique ID of the food grid.'},
        'grid_dims': {'t': Tuple[int], 'v': (50, 50), 'min': 10, 'max': 200, 'disp': 'XY dims','vfunc': param.Tuple,
                      'h': 'The spatial resolution of the food grid.'},
        'initial_value': {'v': 0.1, 'max': 1.0, 'dv': 0.01, 'disp': 'Initial amount','vfunc': param.Number,
                          'h': 'The initial amount of food in each cell of the grid.'},
        'distribution': {'t': str, 'v': 'uniform','vfunc': param.Selector, 'vs': ['uniform'], 'h': 'The distribution of food in the grid.'},
        'default_color': {'t': str, 'v': 'green','vfunc': param.Color, 'disp': 'color', 'h': 'The default color of the food grid.'},
        **d['substrate']
    }

    d['epoch'] = {
        'start': {'max': 200.0, 'h': 'The beginning of the epoch in hours post-hatch.','vfunc': param.Number},
        'stop': {'max': 200.0, 'h': 'The end of the epoch in hours post-hatch.','vfunc': param.Number},
        'substrate': d['substrate']

    }

    d['life_history'] = {
        'age': {'v': 0.0, 'max': 250.0,'vfunc': param.Number, 'dv': 1.0, 'h': 'The larva age in hours post-hatch.'},
        'epochs': {'t': TypedDict, 'v': {}, 'entry': 'epoch', 'disp': 'life epochs',
                   'h': 'The feeding epochs comprising life-history.'}

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
        'nengo': bF
    }

    d['gut'] = {
        'M_gm': {'v': 10 ** -2, 'min': 0.0, 'disp': 'gut scaled capacity','vfunc': param.Number,
                 'h': 'Gut capacity in C-moles per unit of gut volume.'},
        'y_P_X': {'v': 0.9, 'min': 0.0, 'max': 1.0, 'disp': 'food->product yield','vfunc': param.Magnitude,
                  'h': 'Yield of product per unit of food.'},
        'J_g_per_cm2': {'v': 10 ** -2 / (24 * 60 * 60), 'min': 0.0, 'disp': 'digestion secretion rate','vfunc': param.Number,
                        'h': 'Secretion rate of enzyme per unit of gut surface per second.'},
        'k_g': {'v': 1.0, 'min': 0.0, 'disp': 'digestion decay rate','vfunc': param.Number, 'h': 'Decay rate of digestive enzyme.'},
        'k_dig': {'v': 1.0, 'min': 0.0, 'disp': 'digestion rate','vfunc': param.Number, 'h': 'Rate constant for digestion : k_X * y_Xg.'},
        'f_dig': {'v': 1.0, 'min': 0.0, 'max': 1.0, 'disp': 'digestion response','vfunc': param.Magnitude,
                  'h': 'Scaled functional response for digestion : M_X/(M_X+M_K_X)'},
        'M_c_per_cm2': {'v': 5 * 10 ** -8, 'min': 0.0, 'disp': 'carrier density','vfunc': param.Number,
                        'h': 'Area specific amount of carriers in the gut per unit of gut surface.'},
        'constant_M_c': {**bT, 'disp': 'constant carrier density',
                         'h': 'Whether to assume a constant amount of carrier enzymes on the gut surface.'},
        'k_c': {'v': 1.0, 'min': 0.0,'vfunc': param.Number, 'disp': 'carrier release rate', 'h': 'Release rate of carrier enzymes.'},
        'k_abs': {'v': 1.0, 'min': 0.0,'vfunc': param.Number, 'disp': 'absorption rate', 'h': 'Rate constant for absorption : k_P * y_Pc.'},
        'f_abs': {'v': 1.0, 'min': 0.0, 'max': 1.0, 'disp': 'absorption response','vfunc': param.Magnitude,
                  'h': 'Scaled functional response for absorption : M_P/(M_P+M_K_P)'},
    }

    d['DEB'] = {'species': {'t': str, 'v': 'default', 'vs': ['default', 'rover', 'sitter'], 'disp': 'phenotype','vfunc': param.Selector,
                            'h': 'The phenotype/species-specific fitted DEB model to use.'},
                'f_decay': {'v': 0.1, 'max': 1.0, 'dv': 0.1,'vfunc': param.Number,
                            'h': 'The exponential decay coefficient of the DEB functional response.'},
                'absorption': {'v': 0.5, 'max': 1.0,'vfunc': param.Magnitude, 'h': 'The absorption ration for consumed food.'},
                'V_bite': {'v': 0.0005, 'max': 0.01, 'dv': 0.0001,'vfunc': param.Number,
                           'h': 'The volume of food consumed on a single feeding motion as a fraction of the body volume.'},
                'hunger_as_EEB': {**bT,
                                  'h': 'Whether the DEB-generated hunger drive informs the exploration-exploitation balance.'},
                'hunger_gain': {'v': 0.0, 'max': 1.0,'vfunc': param.Magnitude,
                                'h': 'The sensitivy of the hunger drive in deviations of the DEB reserve density.'},
                'assimilation_mode': {'t': str, 'v': 'gut', 'vs': ['sim', 'gut', 'deb'],'vfunc': param.Selector,
                                      'h': 'The method used to calculate the DEB assimilation energy flow.'},
                'DEB_dt': {'max': 1.0, 'disp': 'DEB timestep (sec)','vfunc': param.Number,
                           'h': 'The timestep of the DEB energetics module in seconds.'},
                # 'gut_params':d['gut_params']
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
    d['ang_definition'] = {
        'bend': {'t': str, 'v': 'from_vectors', 'vs': ['from_angles', 'from_vectors'],'vfunc': param.Selector,
                 'h': 'Whether bending angle is computed as a sum of sequential segmental angles or as the angle between front and rear body vectors.'},
        'front_vector': {'t': Tuple[int], 'v': (1, 2), 'min': -12, 'max': 12,'vfunc': param.Tuple,
                         'h': 'The initial & final segment of the front body vector.'},
        'rear_vector': {'t': Tuple[int], 'v': (-2, -1), 'min': -12, 'max': 12,'vfunc': param.Tuple,
                        'h': 'The initial & final segment of the rear body vector.'},
        'front_body_ratio': {'v': 0.5, 'max': 1.0, 'disp': 'front_ratio','vfunc': param.Magnitude,
                             'h': 'The fraction of the body considered front, relevant for bend computation from angles.'}
    }
    d['spatial_definition'] = {
        'point_idx': {'t': int, 'min': -1, 'max': 12,'vfunc': param.Integer,
                      'h': 'The index of the segment used as the larva spatial position (-1 means using the centroid).'},
        'use_component_vel': {**bF, 'disp': 'vel_component',
                              'h': 'Whether to use the component velocity ralative to the axis of forward motion.'}
    }

    d['metric_definition'] = {
        'angular': d['ang_definition'],
        'spatial': d['spatial_definition'],
        'dispersion': {
            'dsp_starts': {'t': List[int], 'v': [0], 'max': 200, 'dv': 1, 'disp': 'starts','vfunc': param.List,
                           'h': 'The timepoints to start calculating dispersion in seconds.'},
            'dsp_stops': {'t': List[int], 'v': [40], 'max': 200, 'dv': 1, 'disp': 'stops','vfunc': param.List,
                          'h': 'The timepoints to stop calculating dispersion in seconds.'},
        },
        'tortuosity': {
            'tor_durs': {'t': List[int], 'v': [5, 10], 'max': 100, 'dv': 1, 'disp': 't (sec)','vfunc': param.List,
                         'h': 'The time windows to use when calculating tortuosity in seconds.'}
        },
        'stride': {
            'track_point': {'t': str,'vfunc': param.String,
                            'h': 'The midline point to use when detecting the strides. When none is provided, the default position of the larva is used (see spatial definition).'},
            'use_scaled_vel': {**bT, 'disp': 'vel_scaled',
                               'h': 'Whether to use the velocity scaled to the body length.'},
            'vel_threshold': {'v': 0.3, 'max': 1.0, 'disp': 'vel_thr','vfunc': param.Number,
                              'h': 'The velocity threshold to be reached in every stride cycle.'},
        },
        # 'pause': {
        #     'stride_non_overlap': {**bT, 'disp': 'excl. strides',
        #                            'h': 'Whether pause bouts are required not to overlap with strides.'},
        #     'min_dur': {'v': 0.4, 'max': 2.0, 'h': 'The minimum duration for detecting a pause, in seconds.'},
        # },
        'turn': {
            'min_ang': {'v': 30.0, 'max': 180.0, 'dv': 1.0,'vfunc': param.Number,
                        'h': 'The minimum orientation angle change required to detect a turn.'},
            'min_ang_vel': {'v': 0.0, 'max': 1000.0, 'dv': 1.0,'vfunc': param.Number,
                            'h': 'The minimum angular velocity maximum required to detect a turn.'},
            'chunk_only': {'t': str, 'vs': ['', 'stride', 'pause'],'vfunc': param.Selector,
                           'h': 'Whether to only detect turns whithin a given bout type.'},
        }
    }

    d['preprocessing'] = {
        'rescale_by': {'max': 10.0,'vfunc': param.Number, 'h': 'Whether to rescale spatial coordinates by a scalar in meters.'},
        'drop_collisions': {**bF, 'h': 'Whether to drop timepoints where larva collisions are detected.'},
        'interpolate_nans': {**bF, 'h': 'Whether to interpolate missing values.'},
        'filter_f': {'max': 10.0, 'vfunc': param.Number,'disp': 'filter frequency',
                     'h': 'Whether to filter spatial coordinates by a grade-1 low-pass filter of the given cut-off frequency.'},
        'transposition': {'t': str, 'vs': ['', 'origin', 'arena', 'center'],'vfunc': param.Selector,
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
                       'mode': {'t': str,'vfunc': param.Selector, 'v': 'minimal', 'vs': ['minimal', 'full']}
                       }

    d['food_params'] = {'source_groups': {'t': dict,'vfunc': param.Dict, 'v': {}},
                        'food_grid': {'t': dict,'vfunc': param.Dict},
                        'source_units': {'t': dict,'vfunc': param.Dict, 'v': {}}
                        }

    d['env_conf'] = {'arena': d['arena'],
                     'border_list': {'t': dict, 'vfunc': param.Dict,'v': {}},
                     'food_params': d['food_params'],
                     # 'odorscape': d['odorscape'],
                     'odorscape': {'t': dict,'vfunc': param.Dict},
                     'windscape': {'t': dict,'vfunc': param.Dict}
                     # 'windscape': d['windscape']
                     }

    d['exp_conf'] = {'env_params': {'t': str, 'vparfunc': ConfSelector('Env'),'vs': kConfDict('Env')},
                     'larva_groups': {'t': dict,'vfunc': param.Dict, 'v': {}},
                     'sim_params': d['sim_params'],
                     'trials': {'t': str, 'vparfunc': ConfSelector('Trial', default='default'),'v': 'default', 'vs': kConfDict('Trial')},
                     'collections': {'t': List[str],'vfunc': param.List, 'v': ['pose']},
                     'enrichment': d['enrichment'],
                     'experiment': {'t': str,'vparfunc': ConfSelector('Exp'), 'vs': kConfDict('Exp')},

                     }
    d['batch_setup'] = {
        'batch_id': {'t': str,'vfunc': param.String, 'h': 'The id of the batch-run', 'k': 'b_id'},
        'save_hdf5': {**bF, 'h': 'Whether to store the batch-run data', 'k': 'store_batch'}
    }
    d['batch_conf'] = {'exp': {'t': str,'vfunc': param.String},
                       'space_search': d['space_search'],
                       'batch_methods': d['batch_methods'],
                       'optimization': d['optimization'],
                       'exp_kws': {'t': dict,'vfunc': param.Dict, 'v': {'enrichment': d['enrichment']}},
                       'post_kws': {'t': dict, 'vfunc': param.Dict,'v': {}},
                       'proc_kws': {'t': dict,'vfunc': param.Dict, 'v': {}},
                       'save_hdf5': {**bF, 'h': 'Whether to store the sur datasets.'}
                       }

    d['tracker'] = {
        'resolution': {
            'fr': {'v': 10.0, 'max': 100.0,'vfunc': param.Number, 'disp': 'framerate (Hz)', 'h': 'The framerate of the tracker recordings.'},
            'Npoints': {'t': int,'vfunc': param.Integer, 'v': 1, 'max': 20, 'disp': '# midline xy',
                        'h': 'The number of points tracked along the larva midline.'},
            'Ncontour': {'t': int,'vfunc': param.Integer, 'v': 0, 'max': 100, 'disp': '# contour xy',
                         'h': 'The number of points tracked around the larva contour.'}
        },
        'arena': d['arena'],
        'filesystem': {
            'read_sequence': {'t': List[str], 'disp': 'columns','vfunc': param.List,
                              'h': 'The sequence of columns in the tracker-exported files.'},
            'read_metadata': {**bF, 'disp': 'metadata',
                              'h': 'Whether metadata files are available for the tracker-exported files/folders.'},
            'folder': {'pref': {'t': str,'vfunc': param.String, 'h': 'A prefix for detecting a raw-data folder.'},
                       'suf': {'t': str,'vfunc': param.String, 'h': 'A suffix for detecting a raw-data folder.'}},
            'file': {'pref': {'t': str,'vfunc': param.String, 'h': 'A prefix for detecting a raw-data file.'},
                     'suf': {'t': str,'vfunc': param.String, 'h': 'A suffix for detecting a raw-data file.'},
                     'sep': {'t': str,'vfunc': param.String, 'h': 'A separator for detecting a raw-data file.'}}
        },

    }

    d['spatial_distro'] = {
        'mode': {'t': str,'vfunc': param.Selector, 'v': 'normal', 'vs': ['normal', 'periphery', 'uniform'], 'disp': 'placing',
                 'h': 'The wa to place agents in the distribution shape.'},
        'shape': {'t': str, 'vfunc': param.Selector,'v': 'circle', 'vs': ['circle', 'rect', 'oval'],
                  'h': 'The space of the spatial distribution.'},
        'N': {'t': int,'vfunc': param.Integer, 'v': 10, 'max': 1000, 'h': 'The number of agents in the group.'},
        'loc': d['xy'],
        'scale': d['xy'],
    }

    d['larva_distro'] = {
        **d['spatial_distro'],
        'orientation_range': {'t': Tuple[float],'vfunc': param.Range, 'v': (0.0, 360.0), 'min': 0.0, 'max': 360.0, 'dv': 1.0,
                              'disp': 'heading',
                              'h': 'The range of larva body orientations to sample from, in degrees.'}
    }

    d['larva_model'] = {'t': str, 'vparfunc': ConfSelector('Model', default='explorer'), 'v': 'explorer', 'vs': kConfDict('Model')}

    d['Larva_DISTRO'] = {
        'model': d['larva_model'],
        **d['larva_distro'],
    }

    d['LarvaGroup'] = {
        'model': d['larva_model'],
        'sample': {'t': str,'vfunc': param.String, 'v': 'None.50controls'},
        'default_color': {'t': str, 'vfunc': param.Color,'v': 'black', 'disp': 'color', 'h': 'The default color of the larva group.'},
        'imitation': bF,
        'distribution': d['larva_distro'],
        'life_history': d['life_history'],
        'odor': d['odor']
    }

    d['agent'] = {
        'group': {'t': str, 'vfunc': param.String,'v': '', 'h': 'The unique ID of the agent group.'},

    }

    d['source'] = {
        **d['agent'],
        'default_color': {'t': str, 'vfunc': param.Color,'v': 'green', 'disp': 'color', 'h': 'The default color of the source.'},
        'pos': d['xy'],
        **d['food'],
        'odor': d['odor']
    }

    d['SourceGroup'] = {
        'distribution': d['spatial_distro'],
        'default_color': {'t': str, 'vfunc': param.Color,'v': 'green', 'disp': 'color', 'h': 'The default color of the source group.'},
        **d['food'],
        'odor': d['odor'],
        'regeneration': {**bF, 'h': 'Whether to regenerate a source when depleted.'},
        'regeneration_pos': {
            'loc': d['xy'],
            'scale': d['xy'],
        }
    }

    d['Border'] = {
        'default_color': {'t': str, 'vfunc': param.Color,'v': 'black', 'disp': 'color', 'h': 'The default color of the border.'},
        'width': {'v': 0.001, 'min': 0.0,'vfunc': param.Number, 'h': 'The width of the border.'},
        'points': {'t': List[Tuple[float]],'vfunc': param.List, 'min': -1.0, 'max': 1.0,
                   'h': 'The XY coordinates of the consecutive border segments.'},
    }

    d['border_list'] = {
        'default_color': {'t': str,'vfunc': param.Color, 'v': 'black', 'disp': 'color', 'h': 'The default color of the border.'},
        'points': {'t': List[Tuple[float]],'vfunc': param.List, 'min': -1.0, 'max': 1.0,
                   'h': 'The XY coordinates of the consecutive border segments.'},
    }
    d['Source_DISTRO'] = d['spatial_distro']

    d.update(init_vis())

    d['replay'] = {
        'env_params': {'t': str, 'vparfunc': ConfSelector('Env'),'vs': kConfDict('Env'), 'aux_vs': [''],
                       'h': 'The arena configuration to display the replay on, if not the default one in the dataset configuration.'},
        'transposition': {'t': str,  'vfunc': param.Selector,'vs': [None, 'origin', 'arena', 'center'],
                          'h': 'Whether to transpose the dataset spatial coordinates.'},
        'agent_ids': {'t': List[str], 'vfunc': param.List,
                      'h': 'Whether to only display some larvae of the dataset, defined by their indexes.'},
        'dynamic_color': {'t': str, 'vfunc': param.Selector, 'vs': [None, 'lin_color', 'ang_color'],
                          'h': 'Whether to display larva tracks according to the instantaneous forward or angular velocity.'},
        'time_range': {'t': Tuple[float], 'vfunc': param.Range, 'max': 1000.0, 'dv': 1.0,
                       'h': 'Whether to only replay a defined temporal slice of the dataset.'},
        'track_point': {'t': int, 'vfunc': param.Integer, 'v': -1, 'min': -1, 'max': 12,
                        'h': 'The midline point to use for defining the larva position.'},
        'draw_Nsegs': {'t': int, 'min': 1, 'max': 12, 'vfunc': param.Integer,
                       'h': 'Whether to artificially simplify the experimentally tracked larva body to a segmented virtual body of the given number of segments.'},
        'fix_point': {'t': int, 'min': 1, 'max': 12,'vfunc': param.Integer,
                      'h': 'Whether to fixate a specific midline point to the center of the screen. Relevant when replaying a single larva track.'},
        'fix_segment': {'t': int, 'vs': [-1, 1],'vfunc': param.Selector,
                        'h': 'Whether to additionally fixate the above or below body segment.'},
        'use_background': {**bF, 'h': 'Whether to use a virtual moving background when replaying a fixated larva.'}
    }

    d['ga_select_kws'] = {
        'Nagents': {'t': int,'vfunc': param.Integer, 'v': 30, 'min': 2, 'max': 1000, 'h': 'Number of agents per generation', 'k': 'N'},
        'Nelits': {'t': int,'vfunc': param.Integer, 'v': 3, 'min': 0, 'max': 1000, 'h': 'Number of elite agents preserved per generation', 'k': 'Nel'},
        'Ngenerations': {'t': int, 'vfunc': param.Integer,'max': 1000, 'h': 'Number of generations to run', 'k': 'Ngen'},
        # 'max_Nticks': {'t': int, 'max': 100000, 'h': 'Maximum number of ticks per generation'},
        # 'max_dur': {'v': 3, 'max': 100, 'h': 'Maximum duration per generation in minutes'},
        'Pmutation': {'v': 0.3,'vfunc': param.Magnitude, 'max': 1.0, 'h': 'Probability of genome mutation', 'k': 'Pmut'},
        'Cmutation': {'v': 0.1, 'vfunc': param.Magnitude,'max': 1.0, 'h': 'Mutation coefficient', 'k': 'Cmut'},
        'selection_ratio': {'v': 0.3,'vfunc': param.Magnitude, 'max': 1.0, 'h': 'Fraction of agents to be selected for the next generation', 'k': 'Rsel'},
        'verbose': {'t': int,'vfunc': param.Selector, 'v': 0, 'vs': [0,1,2,3], 'h': 'Verbose argument for GA launcher', 'k': 'verb'}
    }

    d['ga_build_kws'] = {
        'space_dict': {'t': dict,'vfunc': param.Dict, 'h': 'The parameter state space'},
        'robot_class': {'t': type,'vfunc': param.ClassSelector, 'h': 'The agent class to use in the simulations'},
        'base_model': {'t': str, 'v': 'navigator', 'vs': kConfDict('Model'),'vparfunc': ConfSelector('Model', default='navigator'),
                       'h': 'The model configuration to optimize', 'k': 'mID0'},
        'bestConfID': {'t': str,'vfunc': param.String, 'h': 'The model configuration ID to store the best genome', 'k': 'mID1'},
        'init_mode': {'t': str, 'v': 'random', 'vs': ['default', 'random', 'model'],'vfunc': param.Selector,
                      'h': 'The initialization mode for the first generation', 'k': 'mGA'},
        'multicore': {**bF, 'h': 'Whether to use multiple cores', 'k': 'multicore'},
        'fitness_target_refID': {'t': str, 'vparfunc': ConfSelector('Ref'),'vs': kConfDict('Ref'),
                                 'h': 'The ID of the reference dataset for comparison', 'k': 'refID'},
        'fitness_target_kws': {'t': dict,'vfunc': param.Dict, 'v': {},
                               'h': 'The target data to derive from the reference dataset for evaluation'},
        'fitness_func': {'t': FunctionType,'vfunc': param.Callable, 'h': 'The method for fitness evaluation'},
        'plot_func': {'t': FunctionType,'vfunc': param.Callable, 'h': 'The method for real-time simulation and plotting of the best genome'},
        'exclude_func': {'t': FunctionType,'vfunc': param.Callable, 'h': 'The method for real-time excluding agents'},
    }

    d['GAconf'] = {
        'scene': {'t': str,'vfunc': param.String, 'v': 'no_boxes', 'h': 'The name of the scene to load'},
        'scene_speed': {'t': int, 'vfunc': param.Integer,'v': 0, 'max': 100, 'h': 'The rendering speed of the scene'},
        # 'id': {'t': str, 'h': 'The GA simulation ID'},
        'env_params': {'t': str, 'vparfunc': ConfSelector('Env'),'vs': kConfDict('Env'),
                       'h': 'The environment configuration ID to use in the simulation'},
        'sim_params': d['sim_params'],
        'experiment': {'t': str, 'v': 'exploration', 'vs': kConfDict('Ga'), 'vparfunc': ConfSelector('Ga', default='exploration'),
                       'h': 'The GA experiment configuration'},
        'caption': {'t': str,'vfunc': param.String, 'h': 'The screen caption'},
        'save_to': {'t': str,'vfunc': param.Foldername, 'h': 'The directory to save data and plots'},
        'show_screen': {**bT, 'h': 'Whether to render the screen visualization', 'k' : 'hide'},
        'offline': {**bF, 'h': 'Whether to run a full LarvaworldSim environment', 'k': 'offline'},
        'ga_build_kws': d['ga_build_kws'],
        'ga_select_kws': d['ga_select_kws'],
        # 'ga_kws': {**d['GAbuilder'], **d['GAselector']},
    }

    d['obstacle_avoidance'] = {
        'sensor_delta_direction': {'v': 0.4,'vfunc': param.Number, 'dv': 0.01, 'min': 0.2, 'max': 1.2, 'h': 'Sensor delta_direction'},
        'sensor_saturation_value': {'t': int, 'vfunc': param.Integer, 'v': 40, 'min': 0, 'max': 200, 'h': 'Sensor saturation value'},
        'obstacle_sensor_error': {'v': 0.35,  'vfunc': param.Number,'dv': 0.01, 'max': 1.0, 'h': 'Proximity sensor error'},
        'sensor_max_distance': {'v': 0.9, 'vfunc': param.Number, 'dv': 0.01, 'min': 0.1, 'max': 1.5, 'h': 'Sensor max_distance'},
        'motor_ctrl_coefficient': {'t': int,  'vfunc': param.Integer,'v': 8770, 'max': 10000, 'h': 'Motor ctrl_coefficient'},
        'motor_ctrl_min_actuator_value': {'t': int, 'vfunc': param.Integer, 'v': 35, 'min': 0, 'max': 50, 'h': 'Motor ctrl_min_actuator_value'},
    }

    d['eval_conf'] = {
        'refID': {'t': str, 'v' : 'None.150controls', 'vs': kConfDict('Ref'), 'vparfunc': ConfSelector('Ref', default='None.150controls'),
                                 'h': 'The ID of the reference dataset for comparison', 'k': 'refID'},
        'modelIDs': {'t': List[str], 'vs': kConfDict('Model'),'vparfunc': ConfSelector('Model', single_choice=False),
                       'h': 'The model configurations to evaluate', 'k': 'mIDs'},
        'dataset_ids': {'t': List[str],'vfunc': param.List,
                     'h': 'The ids for the generated datasets', 'k': 'dIDs'},
        'offline': {**bF, 'h': 'Whether to run a full LarvaworldSim environment', 'k': 'offline'},
        'N': {'t': int, 'vfunc': param.Integer, 'v': 5, 'min': 2, 'max': 1000, 'h': 'Number of agents per model ID', 'k': 'N'},
        'id': {'t': str,'vfunc': param.String,  'h': 'The id of the evaluation run', 'k': 'id'}


    }

    return d


if __name__ == '__main__':
    dic = init_pars()
