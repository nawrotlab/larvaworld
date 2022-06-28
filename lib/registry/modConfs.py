import copy
from typing import List, Tuple
import numpy as np
from scipy import signal

from lib.aux.par_aux import sub, subsup, circle, bar, tilde, sup

from lib.registry.units import ureg
from lib.aux import dictsNlists as dNl

bF, bT = {'dtype': bool, 'v0': False, 'v': False}, {'dtype': bool, 'v0': True, 'v': True}


def init_mods():
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

    return d


def Tur0():
    from lib.model.modules.basic import StepEffector, StepOscillator
    from lib.model.modules.turner import NeuralOscillator

    NEUargs = {
        'base_activation': {'dtype': float, 'v0': 20.0, 'lim': (10.0, 40.0), 'dv': 0.1,
                            'disp': 'tonic input', 'sym': '$I_{T}^{0}$', 'k': 'I_T0','codename':'turner_input_constant',
                            'h': 'The baseline activation/input of the TURNER module.'},
        'activation_range': {'dtype': Tuple[float], 'v0': (10.0, 40.0),
                             'lim': (0.0, 100.0), 'dv': 0.1,
                             'k': 'I_T_r',
                             'disp': 'input range',
                             'sym': r'$[I_{T}^{min},I_{T}^{max}]$',
                             'h': 'The activation/input range of the TURNER module.'},

        'tau': {'v0': 0.1, 'lim': (0.05, 0.5), 'dv': 0.01, 'disp': 'time constant', 'k': 'tau_T',
                'sym': r'$\tau_{T}$',
                'u': ureg.s,
                'h': 'The time constant of the neural oscillator.'},
        'm': {'dtype': int, 'v0': 100, 'lim': (50, 200), 'dv': 1, 'disp': 'maximum spike-rate',
              'sym': '$SR_{max}$', 'k': 'SR_T',
              'h': 'The maximum allowed spike rate.'},
        'n': {'v0': 2.0, 'lim': (1.5, 3.0), 'dv': 0.01, 'disp': 'spike response steepness', 'k': 'n_T',
              'sym': '$n_{T}$',
              'h': 'The neuron spike-rate response steepness coefficient.'},

    }

    Tamp = {'initial_amp': {'lim': (0.1, 20.0), 'dv': 0.1, 'v0': 20.0,
                            'k': 'A_T0','codename': 'pause_front_orientation_velocity_mean',
                            'disp': 'output amplitude', 'sym': '$A_{T}^{0}$',
                            'h': 'The initial activity amplitude of the TURNER module.'},
            }



    SINargs = {**Tamp,
               'initial_freq': {'v0': 0.58, 'lim': (0.01, 2.0), 'dv': 0.01,
                                'k': 'f_T0',
                                'disp': 'bending frequency', 'sym': sub('f', 'T'), 'u_name': '$Hz$',
                                'u': ureg.Hz,'codename': 'front_orientation_velocity_freq',
                                'h': 'The initial frequency of the repetitive lateral bending behavior if this is hardcoded (e.g. sinusoidal mode).'},
               }


    d = {'neural': {'args': NEUargs, 'class_func': NeuralOscillator},
         'sinusoidal': {'args': SINargs, 'class_func': StepOscillator},
         'constant': {'args': Tamp, 'class_func': StepEffector}
         }
    return dNl.NestDict(d)


def Cr0():
    from lib.model.modules.basic import StepEffector
    from lib.model.modules.crawler import SquareOscillator, PhaseOscillator, GaussOscillator
    str_kws = {'stride_dst_mean': {'v0': 0.224, 'lim': (0.0, 1.0), 'dv': 0.01,
                                   'k': 'str_sd_mu',
                                   'disp': r'stride distance mean', 'sym': sub(bar(circle('d')), 'S'),
                                   'u_name': '$body-lengths$', 'codename': 'scaled_stride_dst_mean',
                                   'h': 'The mean displacement achieved in a single peristaltic stride as a fraction of the body length.'},
               'stride_dst_std': {'v0': 0.033, 'lim': (0.0, 1.0),
                                  'k': 'str_sd_std',
                                  'disp': 'stride distance std', 'sym': sub(tilde(circle('d')), 'S'),
                                  'u_name': '$body-lengths$','codename': 'scaled_stride_dst_std',
                                  'h': 'The standard deviation of the displacement achieved in a single peristaltic stride as a fraction of the body length.'}}

    Camp = {'initial_amp': {'lim': (0.0, 2.0), 'dv': 0.1, 'v0': 0.5,
                            'k': 'A_C0','codename': 'stride_scaled_velocity_mean',
                            'disp': 'output amplitude', 'sym': subsup('A', 'C', 0),
                            'h': 'The initial output amplitude of the CRAWLER module.'}}
    Cfr = {'initial_freq': {'v0': 1.418, 'lim': (0.5, 2.5), 'dv': 0.1,
                            'k': 'f_C0',
                            'disp': 'crawling frequency', 'sym': subsup('f', 'C', 0), 'u': ureg.Hz,
                            'codename': 'scaled_velocity_freq',
                            'h': 'The initial frequency of the repetitive crawling behavior.'}}

    SQargs = {
        'duty': {'v0': 0.6, 'lim': (0.0, 1.0), 'dv': 0.1,
                               'k': 'r_C',
                               'disp': 'square signal duty', 'sym': sub('r', 'C'),
                               'h': 'The duty parameter(%time at the upper end) of the square signal.'},
        **Cfr, **Camp

    }
    GAUargs = {
        'std': {'v0': 0.6, 'lim': (0.0, 1.0), 'dv': 0.1,
                                'k': 'r_C',
                                'disp': 'gaussian window std', 'sym': sub('r', 'C'),
                                'h': 'The std of the gaussian window.'},

        **Cfr, **Camp
    }
    Rargs = {'max_scaled_vel': {'v0': 0.6, 'lim': (0.0, 1.5), 'disp': 'maximum scaled velocity',
                                'codename': 'stride_scaled_velocity_max', 'k': 'str_sv_max', 'dv': 0.1,
                                'sym': sub(circle('v'), 'max'), 'u': ureg.s ** -1,
                                'u_name': '$body-lengths/sec$',
                                'h': 'The maximum scaled forward velocity.'},

             'max_vel_phase': {'v0': 3.6, 'lim': (0.0, 2 * np.pi), 'disp': 'max velocity phase',
                               'k': 'phi_v_max', 'dv': 0.1,
                               'sym': subsup('$\phi$', 'C', 'v'), 'u_name': 'rad', 'u': ureg.rad,
                               'codename': 'phi_scaled_velocity_max',
                               'h': 'The phase of the crawling oscillation cycle where forward velocity is maximum.'},
             **Cfr
             }

    d = {

        'gaussian': {'args': {**GAUargs, **str_kws}, 'class_func': GaussOscillator},
        'square': {'args': {**SQargs, **str_kws}, 'class_func': SquareOscillator},
        'realistic': {'args': {**Rargs, **str_kws}, 'class_func': PhaseOscillator},
        'constant': {'args': Camp, 'class_func': StepEffector}
    }
    return dNl.NestDict(d)


def If0():
    from lib.model.modules.crawl_bend_interference import DefaultCoupling, SquareCoupling, PhasicCoupling

    IFargs = {
        'suppression_mode': {'dtype': str, 'v0': 'amplitude', 'vs': ['amplitude', 'oscillation', 'both'],
                             'k': 'IF_target',
                             'disp': 'suppression target', 'sym': '-',
                             'h': 'CRAWLER:TURNER suppression target.'},
        'attenuation': {'v0': 1.0, 'lim': (0.0, 1.0), 'disp': 'suppression coefficient',
                        'sym': '$c_{CT}^{0}$', 'k': 'c_CT0','codename': 'attenuation_min',
                        'h': 'CRAWLER:TURNER baseline suppression coefficient'},
        'attenuation_max': {'v0': 0.0, 'lim': (0.0, 1.0),
                            'disp': 'suppression relief coefficient',
                            'sym': '$c_{CT}^{1}$', 'k': 'c_CT1','codename': 'attenuation_max',
                            'h': 'CRAWLER:TURNER suppression relief coefficient.'}
    }

    PHIargs = {
        'max_attenuation_phase': {'v0': 3.4, 'lim': (0.0, 2 * np.pi), 'disp': 'max relief phase','codename':'phi_attenuation_max',
                                  'sym': '$\phi_{C}^{\omega}$', 'u': ureg.rad, 'k': 'phi_fov_max',
                                  'h': 'CRAWLER phase of minimum TURNER suppression.'},
        **IFargs
    }
    SQargs = {
        'crawler_phi_range': {'dtype': Tuple[float], 'v0': (0.0, 0.0), 'lim': (0.0, 2 * np.pi),
                              'disp': 'suppression relief phase interval',
                              'sym': '$[\phi_{C}^{\omega_{0}},\phi_{C}^{\omega_{1}}]$',
                              'u': ureg.rad,
                              'h': 'CRAWLER phase range for TURNER suppression lift.'},
        'feeder_phi_range': {'dtype': Tuple[float], 'v0': (0.0, 0.0), 'lim': (0.0, 2 * np.pi),
                             'disp': 'feeder suppression relief phase interval',
                             'sym': '$[\phi_{F}^{\omega_{0}},\phi_{F}^{\omega_{1}}]$',
                             'u': ureg.rad,
                             'h': 'FEEDER phase range for TURNER suppression lift.'},
        **IFargs

    }

    d = {'default': {'args': IFargs, 'class_func': DefaultCoupling},
         'square': {'args': SQargs, 'class_func': SquareCoupling},
         'phasic': {'args': PHIargs, 'class_func': PhasicCoupling}
         }
    return dNl.NestDict(d)


def Im0():
    from lib.model.modules.intermitter import Intermitter, BranchIntermitter
    from lib.model.modules.intermitter import NengoIntermitter
    from lib.registry.dist_dict import build_dist_dict, get_dist
    d0 = build_dist_dict()

    dist_args = {k: get_dist(k=k, d0=d0) for k in ['stridechain_dist', 'run_dist', 'pause_dist']}

    IMargs = {
        'run_mode': {'dtype': str, 'v0': 'stridechain', 'vs': ['stridechain', 'run'],
                     'h': 'The generation mode of run epochs.'},
        'EEB': {'v0': 0.0, 'lim': (0.0, 1.0), 'sym': 'EEB', 'k': 'EEB', 'disp': 'Exploitation:Exploration balance',
                'h': 'The baseline exploitation-exploration balance. 0 means only exploitation, 1 only exploration.'},
        'EEB_decay': {'v0': 1.0, 'lim': (0.0, 2.0), 'sym': sub('c', 'EEB'),
                      'k': 'c_EEB', 'disp': 'EEB decay coefficient',
                      'h': 'The exponential decay coefficient of the exploitation-exploration balance when no food is detected.'},
        'crawl_bouts': {**bT, 'disp': 'crawling bouts', 'k': 'epochs_C',
                        'h': 'Whether crawling bouts (runs/stridechains) are generated.'},
        'feed_bouts': {**bF, 'disp': 'feeding bouts', 'k': 'epochs_F',
                       'h': 'Whether feeding bouts (feedchains) are generated.'},
        'crawl_freq': {'v0': 1.43, 'lim': (0.5, 2.5), 'k': 'f_C', 'dv': 0.01, 'u': ureg.Hz,
                       'sym': sub('f', 'C'),
                       'disp': 'crawling frequency',
                       'h': 'The default frequency of the CRAWLER oscillator when simulating offline.'},
        'feed_freq': {'v0': 2.0, 'lim': (0.5, 4.0), 'dv': 0.01, 'k': 'f_F', 'u': ureg.Hz,
                      'sym': sub('f', 'F'),
                      'disp': 'feeding frequency',
                      'h': 'The default frequency of the FEEDER oscillator when simulating offline.'},
        'feeder_reoccurence_rate': {'lim': (0.0, 1.0), 'disp': 'feed reoccurence', 'sym': sub('r', 'F'),
                                    'h': 'The default reoccurence rate of the feeding motion.'},
        **dist_args

    }

    BRargs = {
        'c': {'v0': 0.7, 'lim': (0.0, 1.0),
              'disp': 'branch coefficient',
              'sym': subsup('c', 'Im', 'br'),
              'h': 'The ISING branching coef.'},
        'sigma': {'v0': 1.0, 'lim': (0.0, 10.0),
                  'disp': 'branch sigma',
                  'sym': subsup('s', 'Im', 'br'),
                  'h': 'The ISING branching coef.'},
        'beta': {'v0': 0.15, 'lim': (0.0, 10.0),
                 'disp': 'Exp beta',
                 'sym': subsup('b', 'Im', 'br'),
                 'h': 'The beta coef for the exponential bout generation.'},
        **IMargs

    }

    d = {'default': {'args': IMargs, 'class_func': Intermitter},
         'nengo': {'args': IMargs, 'class_func': NengoIntermitter},
         'branch': {'args': BRargs, 'class_func': BranchIntermitter},
         }
    return dNl.NestDict(d)


def sensor_kws(k0, l0):
    d = {
        'perception': {'dtype': str, 'v0': 'log', 'vs': ['log', 'linear', 'null'],
                       'disp': f'{l0} sensing transduction mode',
                       'k': f'mod_{k0}',
                       'sym': sub('mod', k0),
                       'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input.'},
        'decay_coef': {'v0': 0.1, 'lim': (0.0, 2.0), 'disp': f'{l0} decay coef',
                       'sym': sub('c', k0), 'k': f'c_{k0}',
                       'h': f'The exponential decay coefficient of the {l0} sensory activation.'},
        'brute_force': {**bF, 'disp': 'ability to interrupt locomotion', 'sym': sub('bf', k0),
                        'k': f'bf_{k0}',
                        'h': 'Whether to apply direct rule-based modulation on locomotion or not.'},
    }
    return d


def Olf0():
    from lib.model.modules.sensor import Olfactor
    args = sensor_kws(k0='O', l0='olfaction')
    d = {'default': {'args': args, 'class_func': Olfactor},
         # 'nengo': {'args': IMargs, 'class_func': NengoIntermitter},
         # 'branch': {'args': BRargs, 'class_func': BranchIntermitter},
         }
    return dNl.NestDict(d)


def Tou0():
    from lib.model.modules.sensor import Toucher
    args = {
        **sensor_kws(k0='T', l0='tactile'),
        'state_specific_best': {**bT, 'disp': 'state-specific or the global highest evaluated gain',
                                'sym': sub('state_specific', 'T'),
                                'k': 'state_specific',
                                'h': 'Whether to use the state-specific or the global highest evaluated gain after the end of the memory training period.'},
        'initial_gain': {'v0': 40.0, 'lim': (-100.0, 100.0),
                         'disp': 'tactile sensitivity coef', 'sym': sub('G', 'T'), 'k': 'G_T',
                         'h': 'The initial gain of the tactile sensor.'},
        'touch_sensors': {'dtype': List[int], 'lim': (0, 8), 'k': 'sens_touch',
                          'sym': sub('N', 'T'), 'disp': 'tactile sensor contour locations',
                          'h': 'The number of touch sensors existing on the larva body.'},
    }
    d = {'default': {'args': args, 'class_func': Toucher},
         # 'nengo': {'args': IMargs, 'class_func': NengoIntermitter},
         # 'branch': {'args': BRargs, 'class_func': BranchIntermitter},
         }
    return dNl.NestDict(d)


def W0():
    from lib.model.modules.sensor import WindSensor
    args = {
        'weights': {
            'hunch_lin': {'v0': 10.0, 'lim': (-100.0, 100.0), 'disp': 'HUNCH->CRAWLER',
                          'sym': sub('w', 'HC'), 'k': 'w_HC',
                          'h': 'The connection weight between the HUNCH neuron ensemble and the CRAWLER module.'},
            'hunch_ang': {'v0': 0.0, 'lim': (-100.0, 100.0), 'disp': 'HUNCH->TURNER',
                          'sym': sub('w', 'HT'), 'k': 'w_HT',
                          'h': 'The connection weight between the HUNCH neuron ensemble and the TURNER module.'},
            'bend_lin': {'v0': 0.0, 'lim': (-100.0, 100.0), 'disp': 'BEND->CRAWLER',
                         'sym': sub('w', 'BC'), 'k': 'w_BC',
                         'h': 'The connection weight between the BEND neuron ensemble and the CRAWLER module.'},
            'bend_ang': {'v0': -10.0, 'lim': (-100.0, 100.0), 'disp': 'BEND->TURNER',
                         'sym': sub('w', 'BT'), 'k': 'w_BT',
                         'h': 'The connection weight between the BEND neuron ensemble and the TURNER module.'},
        },

        **sensor_kws(k0='W', l0='windsensor'),
    }
    d = {'default': {'args': args, 'class_func': WindSensor},
         # 'nengo': {'args': IMargs, 'class_func': NengoIntermitter},
         # 'branch': {'args': BRargs, 'class_func': BranchIntermitter},
         }
    return dNl.NestDict(d)


def Th0():
    from lib.model.modules.sensor import Thermosensor
    args = {'cool_gain': {'v0': 0.0, 'lim': (-1000.0, 1000.0),
                          'disp': 'cool thermosensing gain', 'sym': sub('G', 'cool'), 'k': 'G_cool',
                          'h': 'The gain of the cool thermosensor.'},
            'warm_gain': {'v0': 0.0, 'lim': (-1000.0, 1000.0),
                          'disp': 'warm thermosensing gain', 'sym': sub('G', 'warm'), 'k': 'G_warm',
                          'h': 'The gain of the warm thermosensor.'},

            **sensor_kws(k0='Th', l0='thermosensor'),
            }
    d = {'default': {'args': args, 'class_func': Thermosensor},
         # 'nengo': {'args': IMargs, 'class_func': NengoIntermitter},
         # 'branch': {'args': BRargs, 'class_func': BranchIntermitter},
         }
    return dNl.NestDict(d)


def Fee0():
    from lib.model.modules.feeder import Feeder

    Fargs = {
        'initial_freq': {'v0': 2.0, 'lim': (0.0, 4.0), 'k': 'f_F0',
                         'disp': 'feeding frequency', 'sym': sub('f', 'F'), 'u': ureg.Hz,
                         'h': 'The initial default frequency of the repetitive feeding behavior'},
        'feed_radius': {'v0': 0.1, 'lim': (0.1, 10.0), 'sym': sub('rad', 'F'),
                        'disp': 'feeding radius', 'k': 'rad_F',
                        'h': 'The radius around the mouth in which food is consumable as a fraction of the body length.'},
        'V_bite': {'v0': 0.0005, 'lim': (0.0001, 0.01), 'dv': 0.0001,
                   'sym': sub('V', 'F'), 'disp': 'feeding volume ratio', 'k': 'V_F',
                   'h': 'The volume of food consumed on a single feeding motion as a fraction of the body volume.'}
    }

    d = {'default': {'args': Fargs, 'class_func': Feeder},
         # 'nengo': {'args': IMargs, 'class_func': NengoIntermitter},
         # 'branch': {'args': BRargs, 'class_func': BranchIntermitter},
         }
    return dNl.NestDict(d)


def Mem0():
    from lib.model.modules.memory import RLmemory, RLOlfMemory, RLTouchMemory

    RLargs = {
        # 'modality': {'dtype': str, 'v0': 'olfaction', 'vs': ['olfaction', 'touch'],
        #              'h': 'The modality for which the memory module is used.'},
        'Delta': {'v0': 0.1, 'lim': (0.0, 10.0), 'h': 'The input sensitivity of the memory.'},
        'state_spacePerSide': {'dtype': int, 'v0': 0, 'lim': (0, 20), 'disp': 'state space dim',
                               'h': 'The number of discrete states to parse the state space on either side of 0.'},
        'gain_space': {'dtype': List[float], 'v0': [-300.0, -50.0, 50.0, 300.0], 'lim': (-1000.0, 1000.0),
                       'dv': 1.0, 'h': 'The possible values for memory gain to choose from.'},
        'update_dt': {'v0': 1.0, 'lim': (0.0, 10.0), 'dv': 1.0,
                      'h': 'The interval duration between gain switches.'},
        'alpha': {'v0': 0.05, 'lim': (0.0, 2.0), 'dv': 0.01,
                  'h': 'The alpha parameter of reinforcement learning algorithm.'},
        'gamma': {'v0': 0.6, 'lim': (0.0, 2.0),
                  'h': 'The probability of sampling a random gain rather than exploiting the currently highest evaluated gain for the current state.'},
        'epsilon': {'v0': 0.3, 'lim': (0.0, 2.0),
                    'h': 'The epsilon parameter of reinforcement learning algorithm.'},
        'train_dur': {'v0': 20.0, 'lim': (0.0, 100.0),
                      'h': 'The duration of the training period after which no further learning will take place.'}
    }

    MBargs = {}

    # d = {'RL': {'args': RLargs, 'class_func': RLmemory},
    #      'MB': {'args': MBargs, 'class_func': RLmemory},
    #      # 'branch': {'args': BRargs, 'class_func': BranchIntermitter},
    #      }
    d = {'olfaction': {'args': RLargs, 'class_func': RLOlfMemory},
         'touch': {'args': MBargs, 'class_func': RLTouchMemory},
         # 'branch': {'args': BRargs, 'class_func': BranchIntermitter},
         }
    return dNl.NestDict(d)


def dict0():
    kws = {'kwargs': {'dt': 0.1}}
    d0 = {}
    d0['turner'] = {'mode': Tur0(), **kws}
    d0['crawler'] = {'mode': Cr0(), **kws}
    d0['intermitter'] = {'mode': Im0(), **kws}
    d0['interference'] = {'mode': If0(), 'kwargs': {}}
    d0['feeder'] = {'mode': Fee0(), **kws}
    d0['olfactor'] = {'mode': Olf0(), **kws}
    d0['toucher'] = {'mode': Tou0(), **kws}
    d0['thermosensor'] = {'mode': Th0(), **kws}
    d0['windsensor'] = {'mode': W0(), **kws}
    d0['memory'] = {'mode': Mem0(), **kws}
    return dNl.NestDict(d0)


def Phy0():
    args = {
        'torque_coef': {'v0': 0.5, 'lim': (0.1, 1.0), 'dv': 0.01, 'disp': 'torque coefficient',
                        'sym': sub('c', 'T'), 'u_name': sup('sec', -2), 'u': ureg.s ** -2,
                        'h': 'Conversion coefficient from TURNER output to torque-per-inertia-unit.'},
        'ang_vel_coef': {'v0': 1.0, 'lim': (0.0, 5.0), 'dv': 0.01, 'disp': 'angular velocity coefficient',
                         'h': 'Conversion coefficient from TURNER output to angular velocity.'},
        'ang_damping': {'v0': 1.0, 'lim': (0.1, 2.0), 'disp': 'angular damping', 'sym': 'z',
                        'u_name': sup('sec', -1), 'u': ureg.s ** -1,
                        'h': 'Angular damping exerted on angular velocity.'},
        'lin_damping': {'v0': 1.0, 'lim': (0.0, 10.0), 'disp': 'linear damping', 'sym': 'zl',
                        'u_name': sup('sec', -1), 'u': ureg.s ** -1,
                        'h': 'Linear damping exerted on forward velocity.'},
        'body_spring_k': {'v0': 1.0, 'lim': (0.0, 10.0), 'dv': 0.1, 'disp': 'body spring constant',
                          'sym': 'k', 'u_name': sup('sec', -2), 'u': ureg.s ** -2,
                          'h': 'Larva-body torsional spring constant reflecting deformation resistance.'},
        'bend_correction_coef': {'v0': 1.0, 'lim': (0.8, 1.5), 'disp': 'bend correction coefficient',
                                 'sym': sub('c', 'b'),
                                 'h': 'Correction coefficient of bending angle during forward motion.'},
        'ang_mode': {'dtype': str, 'v0': 'torque', 'vs': ['torque', 'velocity'], 'disp': 'angular mode',
                     'h': 'Whether the Turner module output is equivalent to torque or angular velocity.'},
    }
    d = {'args': args}
    return dNl.NestDict(d)


def Bod0():
    args = {
        'initial_length': {'v0': 0.004, 'lim': (0.0, 0.01), 'dv': 0.0001,
                           'disp': 'length', 'sym': '$l$', 'u': ureg.m, 'k': 'l0', 'h': 'The initial body length.'},
        'length_std': {'v0': 0.0, 'lim': (0.0, 0.001), 'dv': 0.0001, 'u': ureg.m, 'k': 'l_std',
                       'h': 'The standard deviation of the initial body length.'},
        'Nsegs': {'dtype': int, 'v0': 2, 'lim': (1, 12), 'disp': 'number of body segments', 'sym': sub('N', 'segs'),
                  'u_name': '# $segments$', 'k': 'Nsegs',
                  'h': 'The number of segments comprising the larva body.'},
        'seg_ratio': {'k': 'seg_r', 'lim': (0.0, 1.0),
                      'h': 'The length ratio of the body segments. If null, equal-length segments are generated.'},

        'shape': {'dtype': str, 'v0': 'drosophila_larva', 'vs': ['drosophila_larva', 'zebrafish_larva'],
                  'k': 'body_shape', 'h': 'The body shape.'},
    }
    d = {'args': args}
    return dNl.NestDict(d)


def DEB0():
    gut_args = {
        'M_gm': {'v0': 10 ** -2, 'lim': (0.0, 10.0), 'disp': 'gut scaled capacity',
                 'sym': 'M_gm',
                 'k': 'M_gm',
                 'h': 'Gut capacity in C-moles per unit of gut volume.'},
        'y_P_X': {'v0': 0.9, 'disp': 'food->product yield',
                  'sym': 'y_P_X', 'k': 'y_P_X',
                  'h': 'Yield of product per unit of food.'},
        'J_g_per_cm2': {'v0': 10 ** -2 / (24 * 60 * 60), 'lim': (0.0, 10.0), 'disp': 'digestion secretion rate',
                        'sym': 'J_g_per_cm2', 'k': 'J_g_per_cm2',
                        'h': 'Secretion rate of enzyme per unit of gut surface per second.'},
        'k_g': {'v0': 1.0, 'lim': (0.0, 10.0), 'disp': 'digestion decay rate', 'sym': 'k_g',
                'k': 'k_g',
                'h': 'Decay rate of digestive enzyme.'},
        'k_dig': {'v0': 1.0, 'lim': (0.0, 10.0), 'disp': 'digestion rate', 'sym': 'k_dig',
                  'k': 'k_dig',
                  'h': 'Rate constant for digestion : k_X * y_Xg.'},
        'f_dig': {'v0': 1.0, 'disp': 'digestion response',
                  'sym': 'f_dig', 'k': 'f_dig',
                  'h': 'Scaled functional response for digestion : M_X/(M_X+M_K_X)'},
        'M_c_per_cm2': {'v0': 5 * 10 ** -8, 'lim': (0.0, 10.0), 'disp': 'carrier density',
                        'sym': 'M_c_per_cm2', 'k': 'M_c_per_cm2',
                        'h': 'Area specific amount of carriers in the gut per unit of gut surface.'},
        'constant_M_c': {**bT, 'disp': 'constant carrier density', 'sym': 'constant_M_c',
                         'k': 'constant_M_c',
                         'h': 'Whether to assume a constant amount of carrier enzymes on the gut surface.'},
        'k_c': {'v0': 1.0, 'lim': (0.0, 10.0), 'disp': 'carrier release rate', 'sym': 'k_c',
                'k': 'gut_k_c',
                'h': 'Release rate of carrier enzymes.'},
        'k_abs': {'v0': 1.0, 'lim': (0.0, 10.0), 'disp': 'absorption rate', 'sym': 'k_abs',
                  'k': 'gut_k_abs',
                  'h': 'Rate constant for absorption : k_P * y_Pc.'},
        'f_abs': {'v0': 1.0, 'lim': (0.0, 1.0), 'disp': 'absorption response',
                  'sym': 'f_abs', 'k': 'f_abs',
                  'h': 'Scaled functional response for absorption : M_P/(M_P+M_K_P)'},
    }

    DEB_args = {'species': {'dtype': str, 'v0': 'default', 'vs': ['default', 'rover', 'sitter'], 'disp': 'phenotype',
                            'k': 'species',
                            'h': 'The phenotype/species-specific fitted DEB model to use.'},
                'f_decay': {'v0': 0.1, 'lim': (0.0, 1.0), 'dv': 0.1, 'sym': sub('c', 'DEB'), 'k': 'c_DEB',
                            'disp': 'DEB functional response decay coef',
                            'h': 'The exponential decay coefficient of the DEB functional response.'},
                'absorption': {'v0': 0.5, 'lim': (0.0, 1.0), 'sym': sub('c', 'abs'),
                               'k': 'c_abs',
                               'h': 'The absorption ration for consumed food.'},
                'V_bite': {'v0': 0.0005, 'lim': (0.0, 0.1), 'dv': 0.0001,
                           'sym': sub('V', 'bite'),
                           'k': 'V_bite',
                           'h': 'The volume of food consumed on a single feeding motion as a fraction of the body volume.'},
                'hunger_as_EEB': {**bT,
                                  'h': 'Whether the DEB-generated hunger drive informs the exploration-exploitation balance.',
                                  'sym': 'hunger_as_EEB', 'k': 'hunger_as_EEB'},
                'hunger_gain': {'v0': 0.0, 'lim': (0.0, 1.0), 'sym': sub('G', 'hunger'),
                                'k': 'G_hunger', 'disp': 'hunger sensitivity to reserve reduction',
                                'h': 'The sensitivy of the hunger drive in deviations of the DEB reserve density.'},
                'assimilation_mode': {'dtype': str, 'v0': 'gut', 'vs': ['sim', 'gut', 'deb'],
                                      'sym': sub('m', 'ass'), 'k': 'ass_mod',
                                      'h': 'The method used to calculate the DEB assimilation energy flow.'},
                'DEB_dt': {'lim': (0.0, 1000.0), 'disp': 'DEB timestep (sec)', 'v0': None,
                           'sym': sub('dt', 'DEB'),
                           'k': 'DEB_dt',
                           'h': 'The timestep of the DEB energetics module in seconds.'},
                # 'gut_params':d['gut_params']
                }

    args = {**gut_args, **DEB_args}
    d = {'args': args}
    return dNl.NestDict(d)


def dict_aux():
    d0 = {}
    d0['physics'] = Phy0()
    d0['body'] = Bod0()
    d0['energetics'] = DEB0()
    return dNl.NestDict(d0)


def build_aux_dict():
    from lib.registry.par import v_descriptor
    from lib.registry.par_dict import preparePar
    d0 = dict_aux()
    d00 = dNl.NestDict(copy.deepcopy(d0))
    pre_d00 = dNl.NestDict(copy.deepcopy(d0))
    for mkey in d0.keys():
        for arg, vs in d0[mkey].args.items():
            pre_p = preparePar(p=arg, **vs)
            p = v_descriptor(**pre_p)
            pre_d00[mkey].args[arg] = pre_p
            d00[mkey].args[arg] = p
    return d0, pre_d00, d00


def build_modConf_dict():
    from lib.registry.par import v_descriptor
    from lib.registry.par_dict import preparePar
    d0 = dict0()
    d00 = dNl.NestDict(copy.deepcopy(d0))
    pre_d00 = dNl.NestDict(copy.deepcopy(d0))
    for mkey in d0.keys():
        for m, mdic in d0[mkey].mode.items():
            for arg, vs in mdic.args.items():
                pre_p = preparePar(p=arg, **vs)
                p = v_descriptor(**pre_p)
                pre_d00[mkey].mode[m].args[arg] = pre_p
                d00[mkey].mode[m].args[arg] = p
    return d0, pre_d00, d00


if __name__ == '__main__':
    from lib.conf.stored.conf import kConfDict
    from lib.conf.stored.conf import loadConf,loadRef

    refID = 'None.150controls'
    d = loadRef(refID)
    d.load(step=False,contour=False)
    e, c = d.endpoint_data, d.config
    print(d.existing(key='end', return_shorts=False))
    raise


    m=d.average_modelConf(new_id='test99', turner_mode='neural', crawler_mode='constant', interference_mode='phasic')
    conf=m.brain.crawler_params

    # mID='thermo_navigator'
    # m=loadConf(mID,'Model')
    # print()
    # raise

    #
    # for mID in kConfDict('Model'):
    #     print(mID)
    #     B = dd.init_brain_mID(mID=mID)
    from lib.registry.pars import preg

    # from lib.registry.parConfs import init_loco

    # from lib.aux.sim_aux import get_sample_bout_distros0
    # from lib.conf.stored.conf import loadConf

    #
    # sample = loadConf(refID, 'Ref')
    dd = preg.larva_conf_dict

    M=dd.module2(mkey='crawler',mode=conf['waveform'], **conf)
    for i in range(1000):
        AA=M.step()
        print(AA)

    # f=dd.mdicts2.crawler.mode[conf['waveform']].class_func
    # print(f)
    raise

    d = dd.mdicts2aux.energetics
    for k, v in d.items():
        print(k, v.keys())
    # from lib.conf.stored.conf import kConfDict
    #
    # for mID in kConfDict('Model'):
    #     print(mID)
    #     B = dd.init_brain_mID(mID=mID)
    #     # print(B.locomotor.intermitter.stridechain_dist)
    #     for i in range(1000):
    #         AA = B.step()
    #
    #     # try :
    #     #     for i in range(1000) :
    #     #         AA=B.step()
    #     #
    #     # except:
    #     #     print('-----------', mID)
    #         # print()
    # raise
    # #
    # mkey = 'intermitter'
    # mm = 'default'
    # #
    # conf0 = dd.init_dicts2[mkey].mode[mm].args
    # preconf0 = dd.mpredicts2[mkey].mode[mm].args
    # mconf0 = dd.mdicts2[mkey].mode[mm].args
    #
    # mconf = dd.conf2(mkey=mkey, mode=mm, refID=refID)
    #
    # # print(conf0.stridechain_dist)
    # # print()
    # # print(preconf0.stridechain_dist)
    # # print()
    # # print(mconf0.stridechain_dist)
    # # print()
    # print(mconf['stridechain_dist'])
    #
    # # mcconf=get_sample_bout_distros0(Im=mconf, bout_distros=sample.bout_distros)
    # #
    # # print()
    # # print(mcconf['stridechain_dist'])
