import numpy as np
from typing import List, Tuple, Union
import pandas as pd
from siunits import BaseUnit, Composite, DerivedUnit

from lib.aux.collecting import output_keys
from lib.aux.par_aux import sub

from lib.gui.aux.functions import get_pygame_key


def maxNdigits(array, Min=None):
    N = len(max(array.astype(str), key=len))
    if Min is not None:
        N = max([N, Min])
    return N


def base_dtype(t):
    if t in [float, Tuple[float], List[float], List[Tuple[float]]]:
        base_t = float
    elif t in [int, Tuple[int], List[int], List[Tuple[int]]]:
        base_t = int
    else:
        base_t = t
    return base_t


def par(name, t=float, v=None, vs=None, min=None, max=None, dv=None, aux_vs=None, disp=None, Ndigits=None, h='', s='',
        combo=None, argparser=False):
    if not argparser:
        cur_dtype = base_dtype(t)
        if cur_dtype in [float, int]:
            if any([arg is not None for arg in [min, max, dv]]):
                if vs is None:
                    if min is None:
                        min = 0
                    if max is None:
                        max = 1
                    if dv is None:
                        if cur_dtype == int:
                            dv = 1
                        elif cur_dtype == float:
                            dv = 0.1

                    ar = np.arange(min, max + dv, dv)
                    if cur_dtype == float:
                        Ndec = len(str(format(dv, 'f')).split('.')[1])
                        ar = np.round(ar, Ndec)
                    vs = ar.astype(cur_dtype)

                    vs = vs.tolist()
        if vs is not None:
            Ndigits = maxNdigits(np.array(vs), 3)
        if aux_vs is not None and vs is not None:
            vs += aux_vs
        d = {'initial_value': v, 'values': vs, 'Ndigits': Ndigits, 'dtype': t,
             'disp': disp if disp is not None else name, 'combo': combo, 'tooltip': h}

        return {name: d}
    else:
        d = {
            'key': name,
            'short': s if s != '' else name,
            'help': h,
        }
        if t == bool:
            d['action'] = 'store_true' if not v else 'store_false'
        else:
            d['type'] = t
            if vs is not None:
                d['choices'] = vs
            if v is not None:
                d['default'] = v
                d['nargs'] = '?'
        return {name: d}


def par_dict(name, d0=None, **kwargs):
    if d0 is None:
        d0 = init_pars()[name]
    d = {}
    # print(name,d0)
    for n, v in d0.items():
        # print(n,v)
        try:
            entry = par(n, **v, **kwargs)
        except:
            entry = {n: {'dtype': dict, 'content': par_dict(n, d0=d0[n], **kwargs)}}
        d.update(entry)
    return d


def par_dict_from_df(name, df):
    df = df.where(pd.notnull(df), None)
    d = {}
    for n in df.index:
        entry = par(n, **df.loc[n])
        d.update(entry)
    return {name: d}


def pars_to_df(d):
    df = pd.DataFrame.from_dict(d, orient='index',
                                columns=['dtype', 'initial_value', 'value_list', 'min', 'max', 'interval'])
    df.index.name = 'name'
    df = df.where(pd.notnull(df), None)


def init_vis():
    d = {}
    d['render'] = {
        'mode': {'t': str, 'v': None, 'vs': [None, 'video', 'image'], 'h': 'The visualization mode', 's': 'm'},
        'image_mode': {'t': str, 'vs': [None, 'final', 'snapshots', 'overlap'], 'h': 'The image-render mode',
                       's': 'im'},
        'video_speed': {'t': int, 'v': 60, 'min': 1, 'max': 100, 'h': 'The video speed', 's': 'vid'},
        'media_name': {'t': str, 'h': 'Filename for the saved video/image', 's': 'media'},
        'show_display': {'t': bool, 'v': True, 'h': 'Hide display', 's': 'hide'},
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
    'standard': {
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
null_bout_dist = {
    'fit': True,
    'range': None,
    'name': None,
    'mu': None,
    'sigma': None,
}


def init_pars():
    from lib.conf.stored.conf import kConfDict
    bF, bT = {'t': bool, 'v': False}, {'t': bool, 'v': True}

    d = {
        'bout_distro': {
            'fit': {**bT, 'combo': 'distro'},
            'range': {'t': Tuple[float], 'max': 100.0, 'combo': 'distro'},
            'name': {'t': str,
                     'vs': ['powerlaw', 'exponential', 'lognormal', 'lognormal-powerlaw', 'levy', 'normal', 'uniform'],
                     'combo': 'distro'},
            'mu': {'max': 10.0, 'disp': 'mean', 'combo': 'distro'},
            'sigma': {'max': 10.0, 'disp': 'std', 'combo': 'distro'},
        },
        'xy': {'t': Tuple[float], 'v': (0.0, 0.0), 'min': -1.0, 'max': 1.0},
        'odor': {
            'odor_id': {'t': str, 'disp': 'ID','h': 'The unique ID of the odorant'},
            'odor_intensity': {'max': 10.0, 'disp': 'C peak (μmole)','h': 'The peak concentration of the odorant in micromoles'},
            'odor_spread': {'max': 10.0, 'disp': 'spread','h': 'The spread of the concentration gradient around the peak'}
        },

        'odorscape': {'odorscape': {'t': str, 'v': 'Gaussian', 'vs': ['Gaussian', 'Diffusion'],'h': 'The algorithm used for odorscape generation'},
                      'grid_dims': {'t': Tuple[int], 'min': 10, 'max': 100,'h': 'The odorscape grid resolution'},
                      'evap_const': {'max': 1.0,'h': 'The evaporation constant of the diffusion algorithm'},
                      'gaussian_sigma': {'t': Tuple[float], 'max': 1.0,'h': 'The sigma of the gaussian difusion algorithm'}
                      },
        'windscape': {'wind_direction': {'t': float, 'min': 0.0, 'max': 360.0, 'dv': 1.0,'h': 'The absolute polar direction of the wind/air puff'},
                      'wind_speed': {'t': float, 'min': 0.0, 'max': 100.0, 'dv': 1.0,'h': 'The speed of the wind/air puff'},
                      },
        'odor_gains': {
            'unique_id': {'t': str},
            'mean': {'max': 1000.0, 'dv': 10.0},
            'std': {'max': 10.0, 'dv': 1.0}
        },
        'optimization': {
            'fit_par': {'t': str,'h': 'The utility parameter optimized'},
            'minimize': {**bT,'h': 'Whether to minimize or maximize the utility parameter'},
            'threshold': {'v': 0.001, 'max': 0.01, 'dv': 0.0001,'h': 'The utility threshold to reach before terminating the batch-run'},
            'max_Nsims': {'t': int, 'v': 7, 'max': 100,'h': 'The maximum number of single runs before terminating the batch-run'},
            'Nbest': {'t': int, 'v': 3, 'max': 10,'h': 'The number of best parameter combinations to use for generating the next generation'},
            'operations': {
                'mean': {**bT, 'h': 'Whether to use the mean of the utility across individuals'},
                'std': {**bF, 'h': 'Whether to use the standard deviation of the utility across individuals'},
                'abs': {**bF, 'h': 'Whether to use the absolute value of the utility'}
            },
        },
        'batch_methods': {
            'run': {'t': str, 'v': 'default', 'vs': ['null', 'default', 'deb', 'odor_preference', 'exp_fit']},
            'post': {'t': str, 'v': 'default', 'vs': ['null', 'default']},
            'final': {'t': str, 'v': 'null', 'vs': ['null', 'scatterplots', 'deb', 'odor_preference']}
        },
        'space_search_par': {
            'range': {'t': Tuple[float], 'max': 100.0, 'min': -100.0, 'dv': 1.0},
            'Ngrid': {'t': int, 'max': 100},
            'values': {'t': List[float], 'min': -100.0, 'max': 100.0}
        },
        'space_search': {'pars': {'t': List[str], 'h': 'The parameters for space search', 's': 'ss.pars'},
                         'ranges': {'t': List[Tuple[float]], 'max': 100.0, 'min': -100.0, 'dv': 1.0,
                                    'h': 'The range of the parameters for space search', 's': 'ss.ranges'},
                         'Ngrid': {'t': int, 'max': 100, 'h': 'The number of steps for space search', 's': 'ss.Ngrid'}},
        'body': {'initial_length': {'v': 0.004, 'max': 0.01, 'dv': 0.0001, 'aux_vs': ['sample'], 'disp': 'initial',
                                    'combo': 'length','h': 'The initial body length'},
                 'length_std': {'v': 0.0004, 'max': 0.001, 'dv': 0.0001, 'aux_vs': ['sample'], 'disp': 'std',
                                'combo': 'length','h': 'The standard deviation of the initial body length'},
                 'Nsegs': {'t': int, 'v': 2, 'min': 1, 'max': 12,'h': 'The number of segments comprising the larva body'},
                 'seg_ratio': {'max': 1.0,'h': 'The length ratio of the body segments. If null, equal-length segments are generated'},  # [5 / 11, 6 / 11]
                 'touch_sensors': {'t': int, 'min': 0, 'max': 8,'h': 'The number of touch sensors existing on the larva body'},
                 },
        'arena': {'arena_dims': {'t': Tuple[float], 'v': (0.1, 0.1), 'max': 1.0, 'dv': 0.01, 'disp': 'X,Y (m)','h': 'The arena dimensions in meters'},
                  'arena_shape': {'t': str, 'v': 'circular', 'vs': ['circular', 'rectangular'], 'disp': 'shape','h': 'The arena shape'}
                  },
        'physics': {
            'torque_coef': {'v': 0.41, 'max': 5.0, 'dv': 0.01,'h': 'The coefficient converting the lateral oscillator (TURNER) activity to bending torque'},
            'ang_damping': {'v': 2.5, 'max': 10.0,'h': 'The environmental angular damping exerted on bending angular velocity'},
            'body_spring_k': {'v': 0.02, 'max': 1.0, 'dv': 0.01,'h': 'The torsional spring constant of the larva body restoring the bending angle to 0'},
            'bend_correction_coef': {'v': 1.4, 'max': 10.0,'h': 'The correction coefficient restoring the bending angle during forward motion by aligning the rear body segments to the front heading axis'},
        },
        'energetics': {'species': {'t': str, 'v': 'default', 'vs': ['default', 'rover', 'sitter'], 'disp': 'phenotype','h': 'The phenotype/species-specific fitted DEB model to use'},
                       'f_decay': {'v': 0.1, 'max': 1.0, 'dv': 0.1,'h': 'The exponential decay coefficient of the DEB functional response'},
                       'absorption': {'max': 1.0,'h': 'The absorption ration for consumed food'},
                       'V_bite': {'v': 0.001, 'max': 0.01, 'dv': 0.0001,'h': 'The volume of food consumed on a single feeding motion as a fraction of the body volume'},
                       'hunger_as_EEB': {**bT,'h': 'Whether the DEB-generated hunger drive informs the exploration-exploitation balance'},
                       'hunger_gain': {'v': 0.0, 'max': 1.0, 'h': 'The sensitivy of the hunger drive in deviations of the DEB reserve density'},
                       'assimilation_mode': {'t': str, 'v': 'gut', 'vs': ['sim', 'gut', 'deb'], 'h': 'The method used to calculate the DEB assimilation energy flow'},
                       'DEB_dt': {'max': 1.0, 'disp' : 'DEB timestep', 'h': 'The timestep of the DEB energetics module in seconds'},
                       },
        'crawler': {'waveform': {'t': str, 'v': 'realistic', 'vs': ['realistic', 'square', 'gaussian', 'constant'], 'h': 'The waveform of the repetitive crawling oscillator (CRAWLER) module'},
                    'freq_range': {'t': Tuple[float], 'v': (0.5, 2.5), 'max': 2.0, 'disp': 'range',
                                   'combo': 'frequency', 'h': 'The frequency range of the repetitive crawling behavior'},
                    'initial_freq': {'v': 1.418, 'max': 10.0, 'aux_vs': ['sample'], 'disp': 'initial',
                                     'combo': 'frequency', 'h': 'The initial frequency of the repetitive crawling behavior'},  # From D1 fit
                    'freq_std': {'v': 0.184, 'max': 1.0, 'disp': 'std', 'combo': 'frequency', 'h': 'The standard deviation of the frequency of the repetitive crawling behavior'},  # From D1 fit
                    'step_to_length_mu': {'v': 0.224, 'max': 1.0, 'dv': 0.01, 'aux_vs': ['sample'], 'disp': 'mean',
                                          'combo': 'scaled distance / stride', 'h': 'The mean displacement achieved in a single peristaltic stride as a fraction of the body length'},
                    # From D1 fit
                    'step_to_length_std': {'v': 0.033, 'max': 1.0, 'aux_vs': ['sample'], 'disp': 'std',
                                           'combo': 'scaled distance / stride', 'h': 'The standard deviation of the displacement achieved in a single peristaltic stride as a fraction of the body length'},  # From D1 fit
                    'initial_amp': {'max': 2.0, 'disp': 'initial', 'combo': 'amplitude', 'h': 'The initial amplitude of the CRAWLER-generated forward velocity if this is hardcoded (e.g. constant waveform)'},
                    'noise': {'v': 0.1, 'max': 1.0, 'dv': 0.01, 'disp': 'noise', 'combo': 'amplitude', 'h': 'The intrinsic output noise of the CRAWLER-generated forward velocity'},
                    'max_vel_phase': {'v': 1.0, 'max': 2.0, 'h': 'The phase of the crawling oscillation cycle where the output (forward velocity) is maximum'}
                    },
        'turner': {'mode': {'t': str, 'v': 'neural', 'vs': ['', 'neural', 'sinusoidal'], 'h': 'The implementation mode of the lateral oscillator (TURNER) module'},
                   'base_activation': {'v': 20.0, 'max': 100.0, 'dv': 1.0, 'disp': 'mean', 'combo': 'activation', 'h': 'The baseline activation/input of the TURNER module'},
                   'activation_range': {'t': Tuple[float], 'v': (10.0, 40.0), 'max': 100.0, 'dv': 1.0, 'disp': 'range',
                                        'combo': 'activation', 'h': 'The activation/input range of the TURNER module'},
                   'noise': {'v': 0.15, 'max': 10.0, 'disp': 'noise', 'combo': 'amplitude', 'h': 'The intrinsic output noise of the TURNER activity amplitude'},
                   'activation_noise': {'v': 0.5, 'max': 10.0, 'disp': 'noise', 'combo': 'activation', 'h': 'The intrinsic input noise of the TURNER module'},
                   'initial_amp': {'max': 20.0, 'disp': 'initial', 'combo': 'amplitude', 'h': 'The initial activity amplitude of the TURNER module'},
                   'amp_range': {'t': Tuple[float], 'max': 20.0, 'disp': 'range', 'combo': 'amplitude', 'h': 'The activity amplitude range of the TURNER module'},
                   'initial_freq': {'max': 2.0, 'disp': 'initial', 'combo': 'frequency', 'h': 'The initial frequency of the repetitive lateral bending behavior if this is hardcoded (e.g. sinusoidal mode)'},
                   'freq_range': {'t': Tuple[float], 'max': 2.0, 'disp': 'range', 'combo': 'frequency', 'h': 'The frequency range of the repetitive lateral bending behavior'},
                   },
        'interference': {
            'crawler_phi_range': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': 2.0, 'h': 'The CRAWLER oscillator cycle range during which it interferes with the TURNER'},  # np.pi * 0.55,  # 0.9, #,
            'feeder_phi_range': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': 2.0, 'h': 'The FEEDER oscillator cycle range during which it interferes with the TURNER'},
            'attenuation': {'v': 1.0, 'max': 1.0, 'h': 'The activity attenuation exerted on the TURNER module due to interference by other oscillators'}
        },

        'olfactor': {
            'perception': {'t': str, 'v': 'log', 'vs': ['log', 'linear', 'null'], 'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input'},
            'input_noise': {'v': 0.0, 'max': 1.0, 'h': 'The intrinsic noise of the sensory input'},
            'decay_coef': {'v': 0.0, 'max': 2.0, 'h': 'The linear decay coefficient of the olfactory sensory activation'}
        },
        'windsensor': {
            'weights': {
                'hunch_lin': {'v': -1.0, 'min': -100.0, 'max': 100.0, 'disp': 'HUNCH->CRAWLER', 'h': 'The connection weight between the HUNCH neuron ensemble and the CRAWLER module'},
                'hunch_ang': {'v': 0.0, 'min': -100.0, 'max': 100.0, 'disp': 'HUNCH->TURNER', 'h': 'The connection weight between the HUNCH neuron ensemble and the TURNER module'},
                'bend_lin': {'v': 0.0, 'min': -100.0, 'max': 100.0, 'disp': 'BEND->CRAWLER', 'h': 'The connection weight between the BEND neuron ensemble and the CRAWLER module'},
                'bend_ang': {'v': 1.0, 'min': -100.0, 'max': 100.0, 'disp': 'BEND->TURNER', 'h': 'The connection weight between the BEND neuron ensemble and the TURNER module'},
            }
        },
        'toucher': {
            'perception': {'t': str, 'v': 'linear', 'vs': ['log', 'linear'], 'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input'},
            'input_noise': {'v': 0.0, 'max': 1.0, 'h': 'The intrinsic noise of the sensory input'},
            'decay_coef': {'v': 0.1, 'max': 2.0, 'h': 'The exponential decay coefficient of the tactile sensory activation'},
            'state_specific_best': {**bT,'h': 'Whether to use the state-specific or the global highest evaluated gain after the end of the memory training period'},
            'brute_force': {**bF, 'h': 'Whether to apply direct rule-based modulation on locomotion or not'},
            'initial_gain': {'v': 40.0, 'min': -100.0, 'max': 100.0, 'h': 'The initial gain of the tactile sensor'}
        },
        'feeder': {
            'freq_range': {'t': Tuple[float], 'v': (1.0, 3.0), 'max': 4.0, 'disp': 'range', 'combo': 'frequency', 'h': 'The frequency range of the repetitive feeding behavior'},
            'initial_freq': {'v': 2.0, 'max': 4.0, 'disp': 'initial', 'combo': 'frequency', 'h': 'The initial default frequency of the repetitive feeding behavior'},
            'feed_radius': {'v': 0.1, 'max': 10.0, 'h': 'The radius around the mouth in which food is consumable as a fraction of the body length'},
            'V_bite': {'v': 0.001, 'max': 0.01, 'dv': 0.0001, 'h': 'The volume of food consumed on a single feeding motion as a fraction of the body volume'}
        },
        'memory': {'modality': {'t': str, 'v': 'olfaction', 'vs': ['olfaction', 'touch'], 'h': 'The modality for which the memory module is used'},
                   'Delta': {'v': 0.1, 'max': 10.0, 'h': 'The input sensitivity of the memory'},
                   'state_spacePerSide': {'t': int, 'v': 0, 'max': 20, 'disp': 'state space dim', 'h': 'The number of discrete states to parse the state space on either side of 0'},
                   'gain_space': {'t': List[float], 'v': [-300.0, -50.0, 50.0, 300.0], 'min': 1000.0, 'max': 1000.0,
                                  'dv': 1.0, 'h': 'The possible values for memory gain to choose from'},
                   'update_dt': {'v': 1.0, 'max': 10.0, 'dv': 1.0, 'h': 'The interval duration between gain switches'},
                   'alpha': {'v': 0.05, 'max': 1.0, 'dv': 0.01, 'h': 'The alpha parameter of reinforcement learning algorithm'},
                   'gamma': {'v': 0.6, 'max': 1.0, 'h': 'The probability of sampling a random gain rather than exploiting the currently highest evaluated gain for the current state'},
                   'epsilon': {'v': 0.3, 'max': 1.0, 'h': 'The epsilon parameter of reinforcement learning algorithm'},
                   'train_dur': {'v': 20.0, 'max': 100.0, 'h': 'The duration of the training period after which no further learning will take place'}
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
            'essay_ID': {'t': str, 'h': 'The unique ID of the essay'},
            'path': {'t': str, 'h': 'The relative path to store the essay datasets'},
            'N': {'t': int, 'min': 1, 'max': 100, 'disp': '# larvae', 'h': 'The number of larvae per larva-group'}
        },

        'sim_params': {
            'sim_ID': {'t': str, 'h': 'The unique ID of the simulation', 's': 'id'},
            'path': {'t': str, 'h': 'The relative path to save the simulation dataset', 's': 'path'},
            'duration': {'max': 100.0, 'h': 'The duration of the simulation in minutes', 's': 't'},
            'timestep': {'v': 0.1, 'max': 0.4, 'dv': 0.05, 'h': 'The timestep of the simulation in seconds', 's': 'dt'},
            'Box2D': {'t': bool, 'v': False, 'h': 'Whether to use the Box2D physics engine or not'},
            'store_data': {'t': bool, 'v': True, 'h': 'Whether to store the simulation data or not', 's': 'no_store'},
            # 'analysis': {'t': bool, 'v': True, 'h': 'Whether to analyze the simulation data', 's' : 'no_analysis'},
        },

        'logn_dist': {
            'range': {'t': Tuple[float], 'v': (0.0, 2.0), 'max': 10.0, 'dv': 1.0},
            'name': {'t': str, 'v': 'lognormal', 'vs': ['lognormal']},
            'mu': {'v': 1.0, 'max': 10.0},
            'sigma': {'v': 0.0, 'max': 10.0},
            'fit': bF
        },
        'par': {
            'p': {'t': str},
            'u': {'t': Union[BaseUnit, Composite, DerivedUnit]},
            'k': {'t': str},
            's': {'t': str},
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
            'min_duration_in_sec': {'v': 170.0, 'max': 3600.0, 'dv': 1.0, 'disp': 'min track duration (sec)'},
            'min_end_time_in_sec': {'v': 0.0, 'max': 3600.0, 'dv': 1.0, 'disp': 'min track termination time (sec)'},
            'start_time_in_sec': {'v': 0.0, 'max': 3600.0, 'dv': 1.0, 'disp': 'track initiation time (sec)'},
            'max_Nagents': {'t': int, 'v': 500, 'max': 5000, 'disp': 'max # larvae'},
            'save_mode': {'t': str, 'v': 'semifull', 'vs': ['minimal', 'semifull', 'full', 'points']},
        },
        # 'substrate': {k: {'v': v, 'max': 1000.0, 'dv': 5.0} for k, v in substrate_dict['standard'].items()},
        'output': {n: bF for n in output_keys}
    }

    d['intermitter'] = {
        'stridechain_dist': d['bout_distro'],
        'pause_dist': d['bout_distro'],
        'crawl_bouts': bT,
        'feed_bouts': bF,
        'crawl_freq': {'v': 1.43, 'max': 2.0, 'dv': 0.01},
        'feed_freq': {'v': 2.0, 'max': 4.0, 'dv': 0.01},
        'feeder_reoccurence_rate': {'max': 1.0, 'disp': 'feed reoccurence'},
        'EEB_decay': {'v': 1.0, 'max': 2.0},
        'EEB': {'v': 0.0, 'max': 1.0},
    }

    d['substrate_composition'] = {n: {'v': 0.0, 'max': 10.0} for n in
                                  ['glucose', 'dextrose', 'saccharose', 'yeast', 'agar', 'cornmeal']}

    d['substrate'] = {
        'type': {'t': str, 'v': 'standard', 'vs': list(substrate_dict.keys())},
        'quality': {'v': 1.0, 'max': 1.0}

    }

    d['food'] = {
        'radius': {'v': 0.003, 'max': 0.1, 'dv': 0.001},
        'amount': {'v': 0.0, 'max': 1.0},
        'can_be_carried': {**bF, 'disp': 'carriable'},
        **d['substrate']
    }
    d['food_grid'] = {
        'unique_id': {'t': str, 'v': 'Food_grid'},
        'grid_dims': {'t': Tuple[int], 'v': (50, 50), 'min': 10, 'max': 200},
        'initial_value': {'v': 0.1, 'max': 1.0, 'dv': 0.01},
        'distribution': {'t': str, 'v': 'uniform', 'vs': ['uniform']},
        'default_color': {'t': str, 'v': 'green'},
        **d['substrate']
        # 'substrate' : d['substrate']
    }

    d['epoch'] = {
        'start': {'max': 200.0},
        'stop': {'max': 200.0},
        # 'duration': {'max': 200.0},
        'substrate': d['substrate']

    }

    d['life_history'] = {
        'epochs': {'t': dict},
        'age': {'v': 0.0, 'max': 250.0, 'dv': 1.0},
    }

    d['brain'] = {
        'modules': d['modules'],
        **{f'{m}_params': d[m] for m in d['modules'].keys()},
        'nengo': bF
    }

    d['larva_conf'] = {
        'brain': d['brain'],
        'body': d['body'],
        'energetics': d['energetics'],
        'physics': d['physics'],
    }
    # d['parameterization'] = {'bend': {'t': str, 'v': 'from_angles', 'vs': ['from_angles', 'from_vectors']},
    #                          'front_vector': {'t': Tuple[int], 'v': (1, 2), 'min': -12, 'max': 12},
    #                          'rear_vector': {'t': Tuple[int], 'v': (-2, -1), 'min': -12, 'max': 12},
    #                          'front_body_ratio': {'v': 0.5, 'max': 1.0},
    #                          'point_idx': {'t': int, 'min': -1, 'max': 12},
    #
    #                          'use_component_vel': bF}
    d['ang_definition'] = {
        'bend': {'t': str, 'v': 'from_angles', 'vs': ['from_angles', 'from_vectors']},
        'front_vector': {'t': Tuple[int], 'v': (1, 2), 'min': -12, 'max': 12},
        'rear_vector': {'t': Tuple[int], 'v': (-2, -1), 'min': -12, 'max': 12},
        'front_body_ratio': {'v': 0.5, 'max': 1.0, 'disp': 'front_ratio'}
    }
    d['spatial_definition'] = {
        'point_idx': {'t': int, 'min': -1, 'max': 12},
        'use_component_vel': {**bF, 'disp': 'vel_component'}
    }

    d['metric_definition'] = {
        'angular': d['ang_definition'],
        'spatial': d['spatial_definition'],
        'dispersion': {
            'dsp_starts': {'t': List[float], 'v': [0.0], 'max': 200.0, 'dv': 1.0, 'disp': 'starts'},
            'dsp_stops': {'t': List[float], 'v': [40.0], 'max': 200.0, 'dv': 1.0, 'disp': 'stops'},
        },
        'tortuosity': {
            'tor_durs': {'t': List[int], 'v': [5, 10, 20], 'max': 100, 'dv': 1, 'disp': 't (sec)'}
        },
        'stride': {
            'track_point': {'t': str},
            'use_scaled_vel': {**bT, 'disp': 'vel_scaled'},
            'vel_threshold': {'v': 0.2, 'max': 1.0, 'disp': 'vel_thr'},
        },
        'pause': {
            'stride_non_overlap': {**bT, 'disp': 'excl. strides'},
            'min_dur': {'v': 0.4, 'max': 2.0},
        },
        'turn': {
            'min_ang': {'v': 30.0, 'max': 180.0, 'dv': 1.0},
            'min_ang_vel': {'v': 0.0, 'max': 1000.0, 'dv': 1.0},
            'chunk_only': {'t': str},
        }
    }

    d['preprocessing'] = {
        'rescale_by': {'max': 10.0},
        'drop_collisions': bF,
        'interpolate_nans': bF,
        'filter_f': {'max': 10.0},
        'transposition': {'t': str, 'vs': ['', 'origin', 'arena', 'center']}
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

    d['food_params'] = {'source_groups': {'t': dict, 'v': {}},
                        'food_grid': {'t': dict},
                        'source_units': {'t': dict, 'v': {}}
                        }

    d['env_conf'] = {'arena': d['arena'],
                     'border_list': {'t': dict, 'v': {}},
                     'food_params': d['food_params'],
                     'odorscape': d['odorscape'],
                     'windscape': d['windscape']}

    d['exp_conf'] = {'env_params': {'t': str, 'vs': kConfDict('Env')},
                     'larva_groups': {'t': dict, 'v': {}},
                     'sim_params': d['sim_params'],
                     'trials': {'t': str, 'v': 'default', 'vs': kConfDict('Trial')},
                     'collections': {'t': List[str], 'v': ['pose']},
                     'enrichment': d['enrichment'],
                     'experiment': {'t': str, 'vs': kConfDict('Exp')},
                     }
    d['batch_setup'] = {
        'batch_id': {'t': str, 'h': 'The id of the batch-run', 's': 'b_id'},
        'save_hdf5': {'t': bool, 'v': False, 'h': 'Whether to store the batch-run data', 's': 'store_batch'}
    }
    d['batch_conf'] = {'exp': {'t': str},
                       'space_search': d['space_search'],
                       'batch_methods': d['batch_methods'],
                       'optimization': d['optimization'],
                       'exp_kws': {'t': dict, 'v': {'enrichment': d['enrichment']}},
                       'post_kws': {'t': dict, 'v': {}},
                       'proc_kws': {'t': dict, 'v': {}},
                       'save_hdf5': bF
                       }

    d['tracker'] = {
        'resolution': {'fr': {'v': 10.0, 'max': 100.0, 'disp': 'framerate (Hz)'},
                       'Npoints': {'t': int, 'v': 1, 'max': 20, 'disp': '# midline xy'},
                       'Ncontour': {'t': int, 'v': 0, 'max': 100, 'disp': '# contour xy'}
                       },
        'arena': d['arena'],
        'filesystem': {
            'read_sequence': {'t': List[str], 'disp': 'columns'},
            'read_metadata': {**bF, 'disp': 'metadata'},
            'folder': {'pref': {'t': str}, 'suf': {'t': str}},
            'file': {'pref': {'t': str}, 'suf': {'t': str}, 'sep': {'t': str}}
        },

    }

    d['spatial_distro'] = {
        'mode': {'t': str, 'v': 'normal', 'vs': ['normal', 'periphery', 'uniform']},
        'shape': {'t': str, 'v': 'circle', 'vs': ['circle', 'rect', 'oval']},
        'N': {'t': int, 'v': 10, 'max': 1000},
        'loc': d['xy'],
        'scale': d['xy'],
    }

    d['larva_distro'] = {
        **d['spatial_distro'],
        'orientation_range': {'t': Tuple[float], 'v': (0.0, 360.0), 'min': 0.0, 'max': 360.0, 'dv': 1.0,
                              'disp': 'heading'}
    }

    d['larva_model'] = {'t': str, 'v': 'explorer', 'vs': kConfDict('Model')}

    d['Larva_DISTRO'] = {
        'model': d['larva_model'],
        **d['larva_distro'],
    }

    d['LarvaGroup'] = {
        'model': d['larva_model'],
        'sample': {'t': str, 'v': 'None.200_controls'},
        'default_color': {'t': str, 'v': 'black'},
        'imitation': bF,
        'distribution': d['larva_distro'],
        'life_history': d['life_history'],
        'odor': d['odor']
    }

    d['agent'] = {
        'group': {'t': str, 'v': ''},

    }

    d['source'] = {
        **d['agent'],
        'default_color': {'t': str, 'v': 'green'},
        'pos': d['xy'],
        **d['food'],
        'odor': d['odor']
    }

    d['SourceGroup'] = {
        'distribution': d['spatial_distro'],
        'default_color': {'t': str, 'v': 'green'},
        **d['food'],
        'odor': d['odor']
    }

    d['Border'] = {
        'default_color': {'t': str, 'v': 'black'},
        'width': {'v': 0.01, 'min': 0.0},
        'points': {'t': List[Tuple[float]], 'min': -1.0, 'max': 1.0},
    }

    d['border_list'] = {
        'default_color': {'t': str, 'v': 'black'},
        'points': {'t': List[Tuple[float]], 'min': -1.0, 'max': 1.0},
    }
    d['Source_DISTRO'] = d['spatial_distro']

    d.update(init_vis())

    d['replay'] = {
        'env_params': {'t': str, 'vs': kConfDict('Env'), 'aux_vs': ['']},
        'transposition': {'t': str, 'vs': [None, 'origin', 'arena', 'center']},
        'agent_ids': {'t': List[str]},
        'dynamic_color': {'t': str, 'vs': [None, 'lin_color', 'ang_color']},
        'time_range': {'t': Tuple[float], 'max': 1000.0, 'dv': 1.0},
        'track_point': {'t': int, 'v': -1, 'min': -1, 'max': 12},
        'draw_Nsegs': {'t': int, 'min': 1, 'max': 12},
        'fix_point': {'t': int, 'min': 1, 'max': 12},
        'fix_segment': {'t': int, 'vs': [-1, 1]},
        'use_background': bF
    }

    return d


def null_dict(n, key='initial_value', **kwargs):
    def v0(d):
        null = {}
        for k, v in d.items():
            if key in v:
                null[k] = v[key]
            else:
                null[k] = v0(v['content'])
        return null

    dic = par_dict(n)
    dic2 = v0(dic)
    if n not in ['visualization', 'enrichment']:
        dic2.update(kwargs)
        return dic2
    else:
        for k, v in dic2.items():
            if k in list(kwargs.keys()):
                dic2[k] = kwargs[k]
            elif type(v) == dict:
                for k0, v0 in v.items():
                    if k0 in list(kwargs.keys()):
                        dic2[k][k0] = kwargs[k0]
        return dic2


def ang_def(b='from_angles', fv=(1, 2), rv=(-2, -1), **kwargs):
    return null_dict('ang_definition', bend=b, front_vector=fv, rear_vector=rv, **kwargs)


def metric_def(ang={}, sp={}, **kwargs):
    # def metric_def(ang={}, sp={}, dsp={}, tor={}, str={}, pau={}, tur={}) :
    return null_dict('metric_definition',
                     angular=ang_def(**ang),
                     spatial=null_dict('spatial_definition', **sp),
                     **kwargs
                     )


def enr_dict(proc=[], bouts=[], to_keep=[], pre_kws={}, fits=True, on_food=False, def_kws={}, **kwargs):
    metrdef = metric_def(**def_kws)
    pre = null_dict('preprocessing', **pre_kws)
    proc = null_dict('processing', **{k: True if k in proc else False for k in proc_type_keys})
    annot = null_dict('annotation', **{k: True if k in bouts else False for k in bout_keys}, fits=fits,
                      on_food=on_food)
    to_drop = null_dict('to_drop', **{k: True if k not in to_keep else False for k in to_drop_keys})
    dic = null_dict('enrichment', metric_definition=metrdef, preprocessing=pre, processing=proc, annotation=annot,
                    to_drop=to_drop, **kwargs)
    return dic


def base_enrich(**kwargs):
    return enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
                    bouts=['stride', 'pause', 'turn'],
                    to_keep=['midline', 'contour'], **kwargs)


def arena(x, y=None):
    if y is None:
        return null_dict('arena', arena_shape='circular', arena_dims=(x, x))
    else:
        return null_dict('arena', arena_shape='rectangular', arena_dims=(x, y))


def border(ps, c='black', w=0.01, id=None):
    b = null_dict('Border', points=ps, default_color=c, width=w)
    if id is not None:
        return {id: b}
    else:
        return b


def hborder(y, xs, **kwargs):
    ps = [(x, y) for x in xs]
    return border(ps, **kwargs)


def vborder(x, ys, **kwargs):
    ps = [(x, y) for y in ys]
    return border(ps, **kwargs)


def prestarved(h=0.0, age=0.0, q=1.0, substrate_type='standard'):
    sub0 = null_dict('substrate', type=substrate_type, quality=q)
    ep0 = {0: null_dict('epoch', start=0.0, stop=age - h, substrate=sub0)}
    if h == 0.0:
        return ep0
    else:
        sub1 = null_dict('substrate', type=substrate_type, quality=0.0)
        ep1 = {1: null_dict('epoch', start=age - h, stop=age, substrate=sub1)}
    return {**ep0, **ep1}


def init_shortcuts():
    draw = {
        'visible trail': 'p',
        '▲ trail duration': '+',
        '▼ trail duration': '-',

        'draw_head': 'h',
        'draw_centroid': 'e',
        'draw_midline': 'm',
        'draw_contour': 'c',
        'draw_sensors': 'j',
    }

    inspect = {
        'focus_mode': 'f',
        'odor gains': 'z',
        'dynamic graph': 'q',
    }

    color = {
        'black_background': 'g',
        'random_colors': 'r',
        'color_behavior': 'b',
    }

    aux = {
        'visible_clock': 't',
        'visible_scale': 'n',
        'visible_state': 's',
        'visible_ids': 'tab',
    }

    screen = {
        'move up': 'UP',
        'move down': 'DOWN',
        'move left': 'LEFT',
        'move right': 'RIGHT',
    }

    sim = {
        'larva_collisions': 'y',
        'pause': 'space',
        'snapshot': 'i',
        'delete item': 'del',

    }

    odorscape = {
        'windscape': 'w',
        'plot odorscapes': 'o',
        **{f'odorscape {i}': i for i in range(10)},
        # 'move_right': 'RIGHT',
    }

    d = {
        'draw': draw,
        'color': color,
        'aux': aux,
        'screen': screen,
        'simulation': sim,
        'inspect': inspect,
        'landscape': odorscape,
    }

    return d


def init_controls():
    k = init_shortcuts()
    d = {'keys': {}, 'pygame_keys': {}, 'mouse': {
        'select item': 'left click',
        'add item': 'left click',
        'select item mode': 'right click',
        'inspect item': 'right click',
        'screen zoom in': 'scroll up',
        'screen zoom out': 'scroll down',
    }}
    ds = {}
    for title, dic in k.items():
        ds.update(dic)
        d['keys'][title] = dic
    d['pygame_keys'] = {k: get_pygame_key(v) for k, v in ds.items()}
    return d


def store_controls():
    d = init_controls()
    from lib.conf.stored.conf import saveConfDict
    saveConfDict(d, 'Settings')


def store_RefPars():
    from lib.aux.dictsNlists import save_dict
    from lib.conf.base import paths
    import lib.aux.naming as nam
    d = {
        'length': 'body.initial_length',
        nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.initial_freq',
        'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
        nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.step_to_length_mu',
        nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.step_to_length_std',
        nam.freq('feed'): 'brain.feeder_params.initial_freq',
    }
    save_dict(d, paths.path('ParRef'), use_pickle=False)


def odor(i, s, id='Odor'):
    return null_dict('odor', odor_id=id, odor_intensity=i, odor_spread=s)


def oG(c=1, id='Odor'):
    return odor(i=2.0 * c, s=0.0002 * np.sqrt(c), id=id)


def oD(c=1, id='Odor'):
    return odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)


if __name__ == '__main__':
    store_controls()
    store_RefPars()
