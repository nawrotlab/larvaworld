from typing import List, Tuple
import copy
from typing import Tuple
import numpy as np
import pandas as pd
import param
from lib.aux.par_aux import sub, subsup, circle, bar, tilde, sup
from lib.registry import reg

from lib.registry.units import ureg
from lib.aux import dictsNlists as dNl, data_aux

bF, bT = {'dtype': bool, 'v0': False, 'v': False}, {'dtype': bool, 'v0': True, 'v': True}


def Tur0():
    from lib.model.modules.basic import StepEffector, StepOscillator
    from lib.model.modules.turner import NeuralOscillator

    NEUargs = {
        'base_activation': {'dtype': float, 'v0': 20.0, 'lim': (10.0, 40.0), 'dv': 0.1,
                            'disp': 'tonic input', 'sym': '$I_{T}^{0}$', 'k': 'I_T0',
                            'codename': 'turner_input_constant',
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

    Tamp = {'initial_amp': {'lim': (0.1, 200.0), 'dv': 0.1, 'v0': 20.0,
                            'k': 'A_T0', 'codename': 'pause_front_orientation_velocity_mean',
                            'disp': 'output amplitude', 'sym': '$A_{T}^{0}$',
                            'h': 'The initial activity amplitude of the TURNER module.'},
            }

    SINargs = {**Tamp,
               'initial_freq': {'v0': 0.58, 'lim': (0.01, 2.0), 'dv': 0.01,
                                'k': 'f_T0',
                                'disp': 'bending frequency', 'sym': sub('f', 'T'), 'u_name': '$Hz$',
                                'u': ureg.Hz, 'codename': 'front_orientation_velocity_freq',
                                'h': 'The initial frequency of the repetitive lateral bending behavior if this is hardcoded (e.g. sinusoidal mode).'},
               }

    d = {'neural': {'args': NEUargs, 'class_func': NeuralOscillator, 'variable': ['base_activation', 'tau', 'n']},
         'sinusoidal': {'args': SINargs, 'class_func': StepOscillator, 'variable': ['initial_amp', 'initial_freq']},
         'constant': {'args': Tamp, 'class_func': StepEffector, 'variable': ['initial_amp']}
         }
    return dNl.NestDict(d)


def Cr0():
    from lib.model.modules.basic import StepEffector
    from lib.model.modules.crawler import SquareOscillator, PhaseOscillator, GaussOscillator
    str_kws = {'stride_dst_mean': {'v0': 0.23, 'lim': (0.0, 1.0), 'dv': 0.01,
                                   'k': 'str_sd_mu',
                                   'disp': r'stride distance mean', 'sym': sub(bar(circle('d')), 'S'),
                                   'u_name': '$body-lengths$', 'codename': 'stride_scaled_dst_mean',
                                   'h': 'The mean displacement achieved in a single peristaltic stride as a fraction of the body length.'},
               'stride_dst_std': {'v0': 0.04, 'lim': (0.0, 1.0),
                                  'k': 'str_sd_std',
                                  'disp': 'stride distance std', 'sym': sub(tilde(circle('d')), 'S'),
                                  'u_name': '$body-lengths$', 'codename': 'stride_scaled_dst_std',
                                  'h': 'The standard deviation of the displacement achieved in a single peristaltic stride as a fraction of the body length.'}}

    Camp = {'initial_amp': {'lim': (0.0, 2.0), 'dv': 0.1, 'v0': 0.3,
                            'k': 'A_C0', 'codename': 'stride_scaled_velocity_mean',
                            'disp': 'output amplitude', 'sym': subsup('A', 'C', 0),
                            'h': 'The initial output amplitude of the CRAWLER module.'}}
    Cfr = {'initial_freq': {'v0': 1.42, 'lim': (0.5, 2.5), 'dv': 0.1,
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
    Rargs = {'max_scaled_vel': {'v0': 0.51, 'lim': (0.0, 1.5), 'disp': 'maximum scaled velocity',
                                'codename': 'stride_scaled_velocity_max', 'k': 'str_sv_max', 'dv': 0.1,
                                'sym': sub(circle('v'), 'max'), 'u': ureg.s ** -1,
                                'u_name': '$body-lengths/sec$',
                                'h': 'The maximum scaled forward velocity.'},

             'max_vel_phase': {'v0': 3.49, 'lim': (0.0, 2 * np.pi), 'disp': 'max velocity phase',
                               'k': 'phi_v_max', 'dv': 0.1,
                               'sym': subsup('$\phi$', 'C', 'v'), 'u_name': 'rad', 'u': ureg.rad,
                               'codename': 'phi_scaled_velocity_max',
                               'h': 'The phase of the crawling oscillation cycle where forward velocity is maximum.'},
             **Cfr
             }

    d = {

        'gaussian': {'args': {**GAUargs, **str_kws}, 'class_func': GaussOscillator,
                     'variable': ['stride_dst_mean', 'stride_dst_std', 'std', 'initial_amp', 'initial_freq']},
        'square': {'args': {**SQargs, **str_kws}, 'class_func': SquareOscillator,
                   'variable': ['stride_dst_mean', 'stride_dst_std', 'duty', 'initial_amp', 'initial_freq']},
        'realistic': {'args': {**Rargs, **str_kws}, 'class_func': PhaseOscillator,
                      'variable': ['stride_dst_mean', 'stride_dst_std', 'max_scaled_vel', 'max_vel_phase',
                                   'initial_freq']},
        'constant': {'args': Camp, 'class_func': StepEffector, 'variable': ['initial_amp']}
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
                        'sym': '$c_{CT}^{0}$', 'k': 'c_CT0', 'codename': 'attenuation_min',
                        'h': 'CRAWLER:TURNER baseline suppression coefficient'},
        'attenuation_max': {'v0': 0.0, 'lim': (0.0, 1.0),
                            'disp': 'suppression relief coefficient',
                            'sym': '$c_{CT}^{1}$', 'k': 'c_CT1', 'codename': 'attenuation_max',
                            'h': 'CRAWLER:TURNER suppression relief coefficient.'}
    }

    PHIargs = {
        'max_attenuation_phase': {'v0': 3.4, 'lim': (0.0, 2 * np.pi), 'disp': 'max relief phase',
                                  'codename': 'phi_attenuation_max',
                                  'sym': '$\phi_{C}^{\omega}$', 'u': ureg.rad, 'k': 'phi_fov_max',
                                  'h': 'CRAWLER phase of minimum TURNER suppression.'},
        **IFargs
    }
    SQargs = {
        'crawler_phi_range': {'dtype': Tuple[float], 'v0': (1.57, 3.14), 'lim': (0.0, 2 * np.pi),
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

    d = {'default': {'args': IFargs, 'class_func': DefaultCoupling, 'variable': ['attenuation']},
         'square': {'args': SQargs, 'class_func': SquareCoupling,
                    'variable': ['attenuation', 'attenuation_max', 'crawler_phi_range']},
         'phasic': {'args': PHIargs, 'class_func': PhasicCoupling,
                    'variable': ['attenuation', 'attenuation_max', 'max_attenuation_phase']}
         }
    return dNl.NestDict(d)


def Im0():
    from lib.model.modules.intermitter import Intermitter, BranchIntermitter, NengoIntermitter

    dist_args = {k: reg.get_dist(k=k) for k in ['stridechain_dist', 'run_dist', 'pause_dist']}

    IMargs = {
        'run_mode': {'dtype': str, 'v0': 'stridechain', 'vs': ['stridechain', 'exec'],
                     'h': 'The generation mode of exec epochs.'},
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

    d = {'default': {'args': IMargs, 'class_func': Intermitter,
                     'variable': ['stridechain_dist', 'run_dist', 'pause_dist']},
         'nengo': {'args': IMargs, 'class_func': NengoIntermitter,
                   'variable': ['stridechain_dist', 'run_dist', 'pause_dist']},
         'branch': {'args': BRargs, 'class_func': BranchIntermitter,
                    'variable': ['c', 'sigma', 'beta', 'stridechain_dist', 'run_dist', 'pause_dist']},
         }
    return dNl.NestDict(d)


def sensor_kws(k0, l0):
    d = {
        'perception': {'dtype': str, 'v0': 'log', 'vs': ['log', 'linear', 'null'],
                       'disp': f'{l0} sensory transduction mode',
                       'k': f'mod_{k0}',
                       'sym': sub('mod', k0),
                       'h': 'The method used to calculate the perceived sensory activation from the current and previous sensory input.'},
        'decay_coef': {'v0': 0.1, 'lim': (0.0, 2.0), 'disp': f'{l0} output decay coef',
                       'sym': sub('c', k0), 'k': f'c_{k0}',
                       'h': f'The exponential decay coefficient of the {l0} sensory activation.'},
        'brute_force': {**bF, 'disp': 'ability to interrupt locomotion', 'sym': sub('bf', k0),
                        'k': f'bf_{k0}',
                        'h': 'Whether to apply direct rule-based modulation on locomotion or not.'},
    }
    return d


def Olf0():
    from lib.model.modules.sensor import Olfactor
    args = {
        'odor_dict': {'dtype': dict, 'k': 'G_O', 'v0': {},
                      'sym': sub('G', 'O'), 'disp': 'gain per odor ID',
                      'h': 'The dictionary of the olfactory gains.'},
        **sensor_kws(k0='O', l0='olfaction')}
    d = {'default': {'args': args, 'class_func': Olfactor, 'variable': ['perception', 'decay_coef', 'brute_force', 'odor_dict']},
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
    d = {'default': {'args': args, 'class_func': Toucher,
                     'variable': ['perception', 'decay_coef', 'brute_force', 'initial_gain']},
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
    d = {'default': {'args': args, 'class_func': WindSensor, 'variable': ['perception', 'decay_coef', 'brute_force']},
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
    d = {'default': {'args': args, 'class_func': Thermosensor,
                     'variable': ['perception', 'decay_coef', 'brute_force', 'cool_gain', 'warm_gain']},
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

    d = {'default': {'args': Fargs, 'class_func': Feeder, 'variable': ['initial_freq']},
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
    d = {'olfaction': {'args': RLargs, 'class_func': RLOlfMemory,
                       'variable': ['Delta', 'update_dt', 'alpha', 'epsilon']},
         'touch': {'args': MBargs, 'class_func': RLTouchMemory, 'variable': []},
         # 'branch': {'args': BRargs, 'class_func': BranchIntermitter},
         }
    return dNl.NestDict(d)


def SM0():
    args = {
        'sensor_delta_direction': {'v0': 0.4, 'dv': 0.01, 'lim': (0.2, 1.2), 'k': 'sens_Dor',
                                   'h': 'Sensor delta_direction'},
        'sensor_saturation_value': {'dtype': int, 'v0': 40, 'lim': (0, 200), 'k': 'sens_c_sat',
                                    'h': 'Sensor saturation value'},
        'obstacle_sensor_error': {'v0': 0.35, 'dv': 0.01, 'lim': (0.0, 1.0), 'k': 'sens_err',
                                  'h': 'Proximity sensor error'},
        'sensor_max_distance': {'v0': 0.9, 'dv': 0.01, 'lim': (0.1, 1.5), 'k': 'sens_dmax',
                                'h': 'Sensor max_distance'},
        'motor_ctrl_coefficient': {'dtype': int, 'v0': 8770, 'lim': (0, 10000), 'k': 'c_mot',
                                   'h': 'Motor ctrl_coefficient'},
        'motor_ctrl_min_actuator_value': {'dtype': int, 'v0': 35, 'lim': (0, 50), 'k': 'mot_vmin',
                                          'h': 'Motor ctrl_min_actuator_value'},
    }
    d = {'default': {'args': args, 'class_func': None,
                     'variable': ['sensor_delta_direction', 'sensor_saturation_value', 'obstacle_sensor_error',
                                  'sensor_max_distance', 'motor_ctrl_coefficient', 'motor_ctrl_min_actuator_value']},
         # 'nengo': {'args': IMargs, 'class_func': NengoIntermitter},
         # 'branch': {'args': BRargs, 'class_func': BranchIntermitter},
         }
    return dNl.NestDict(d)


def init_brain_modules():
    kws = {'kwargs': {'dt': 0.1}}
    d0 = {}
    d0['turner'] = {'mode': Tur0(), 'pref': 'brain.turner_params.', **kws}
    d0['crawler'] = {'mode': Cr0(), 'pref': 'brain.crawler_params.', **kws}
    d0['intermitter'] = {'mode': Im0(), 'pref': 'brain.intermitter_params.', **kws}
    d0['interference'] = {'mode': If0(), 'pref': 'brain.interference_params.', 'kwargs': {}}
    d0['feeder'] = {'mode': Fee0(), 'pref': 'brain.feeder_params.', **kws}
    d0['olfactor'] = {'mode': Olf0(), 'pref': 'brain.olfactor_params.', **kws}
    d0['toucher'] = {'mode': Tou0(), 'pref': 'brain.toucher_params.', **kws}
    d0['thermosensor'] = {'mode': Th0(), 'pref': 'brain.thermosensor_params.', **kws}
    d0['windsensor'] = {'mode': W0(), 'pref': 'brain.windsensor_params.', **kws}
    d0['memory'] = {'mode': Mem0(), 'pref': 'brain.memory_params.', **kws}

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
    d = {'args': args,
         'variable': ['torque_coef', 'ang_damping', 'body_spring_k', 'bend_correction_coef']
         }
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
    d = {'args': args, 'variable': ['initial_length', 'Nsegs']}
    return dNl.NestDict(d)


def DEB0():
    # from lib.model.DEB import gut,deb
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

    # args = {'gut' : gut_args, 'DEB' : DEB_args}
    d = {'gut': {'args': gut_args, 'variable': ['k_abs', 'k_g']},
         'DEB': {'args': DEB_args, 'variable': ['DEB_dt', 'hunger_gain']},
         # 'branch': {'args': BRargs, 'class_func': BranchIntermitter},
         }
    return dNl.NestDict(d)


def init_aux_modules():
    d0 = {}
    d0['physics'] = Phy0()
    d0['body'] = Bod0()
    d0['energetics'] = {'mode': DEB0()}
    d0['sensorimotor'] = {'mode': SM0(), 'pref': 'sensorimotor.'}
    return dNl.NestDict(d0)


def build_aux_module_dict(d0):

    d00 = dNl.NestDict(copy.deepcopy(d0))
    pre_d00 = dNl.NestDict(copy.deepcopy(d0))
    for mkey in d0.keys():
        if mkey in ['energetics', 'sensorimotor']:
            continue

        for arg, vs in d0[mkey].args.items():
            pre_p = data_aux.preparePar(p=arg, **vs)
            p = data_aux.v_descriptor(**pre_p)
            pre_d00[mkey].args[arg] = pre_p
            d00[mkey].args[arg] = p
    for mkey in ['energetics', 'sensorimotor']:
        for m, mdic in d0[mkey].mode.items():
            for arg, vs in mdic.args.items():
                pre_p = data_aux.preparePar(p=arg, **vs)
                p = data_aux.v_descriptor(**pre_p)
                pre_d00[mkey].mode[m].args[arg] = pre_p
                d00[mkey].mode[m].args[arg] = p
    return pre_d00, d00


def build_brain_module_dict(d0):

    d00 = dNl.NestDict(copy.deepcopy(d0))
    pre_d00 = dNl.NestDict(copy.deepcopy(d0))
    for mkey in d0.keys():
        for m, mdic in d0[mkey].mode.items():
            for arg, vs in mdic.args.items():
                pre_p = data_aux.preparePar(p=arg, **vs)
                p = data_aux.v_descriptor(**pre_p)
                pre_d00[mkey].mode[m].args[arg] = pre_p
                d00[mkey].mode[m].args[arg] = p
    return pre_d00, d00


def build_confdicts0():
    bdic0 = init_brain_modules()
    adic0 = init_aux_modules()

    bks = list(bdic0.keys())
    aks = list(adic0.keys())

    dic0 = dNl.NestDict({**bdic0, **adic0})

    bd = {'init': bdic0, 'pre': {}, 'm': {}, 'keys': bks}
    ad = {'init': adic0, 'pre': {}, 'm': {}, 'keys': aks}
    d = {'init': dic0, 'pre': {}, 'm': {}, 'keys': bks + aks}

    dd = {'brain': bd, 'aux': ad, 'model': d}
    return dNl.NestDict(dd)






class ModelRegistry:
    def __init__(self, load=False, save=False):

        reg.vprint('started LarvaConfDict', 0)


        self.path = reg.Path.LarvaConfDict
        if not load:

            self.dict0 = build_confdicts0()
            if save:
                dNl.save_dict(self.dict0, self.path)
        else:
            self.dict0 = dNl.load_dict(self.path)

        self.dict = self.build()
        self.full_dict = self.build_full_dict(D=self.dict)

        self.mcolor = dNl.NestDict({
            'body': 'lightskyblue',
            'physics': 'lightsteelblue',
            'energetics': 'lightskyblue',
            'Box2D_params': 'lightcoral',
            'crawler': 'lightcoral',
            'turner': 'indianred',
            'interference': 'lightsalmon',
            'intermitter': '#a55af4',
            'olfactor': 'palegreen',
            'windsensor': 'plum',
            'toucher': 'pink',
            'feeder': 'pink',
            'memory': 'pink',
            # 'locomotor': locomotor.DefaultLocomotor,
        })

        reg.vprint('completed LarvaConfDict', 0)

        # print('Completed LarvaConfDict')

    def build(self):
        dd=self.dict0
        dd.brain.pre, dd.brain.m = build_brain_module_dict(dd.brain.init)
        dd.aux.pre, dd.aux.m = build_aux_module_dict(dd.aux.init)

        dd.model.pre = dNl.NestDict({**dd.brain.pre, **dd.aux.pre})
        dd.model.m = dNl.NestDict({**dd.brain.m, **dd.aux.m})
        # dic2 = dNl.NestDict({**bdic2, **adic2})
        # dic1 = dNl.NestDict({**bdic1, **adic1})
        # dic0 = dNl.NestDict({**bdic0, **adic0})
        #
        # bks = list(bdic0.keys())
        # aks = list(adic0.keys())
        #
        # bd = {'init': bdic0, 'pre': bdic1, 'm': bdic2, 'keys': bks}
        # ad = {'init': adic0, 'pre': adic1, 'm': adic2, 'keys': aks}
        # d = {'init': dic0, 'pre': dic1, 'm': dic2, 'keys': bks + aks}
        #
        # dd = {'brain': bd, 'aux': ad, 'model': d}
        return dNl.NestDict(dd)

    def get_mdict(self, mkey, mode='default'):
        if mkey is None or mkey not in self.dict.model.keys:
            raise ValueError('Module key must be one of larva-model configuration keys')
        else:
            if mkey in self.dict.brain.keys + ['energetics']:
                return self.dict.model.m[mkey].mode[mode].args
            elif mkey in self.dict.aux.keys:
                return self.dict.model.m[mkey].args

    def generate_configuration(self, mdict, **kwargs):
        conf = dNl.NestDict()
        for d, p in mdict.items():
            if isinstance(p, param.Parameterized):
                conf[d] = p.v
            else:
                conf[d] = self.generate_configuration(mdict=p)
        conf = dNl.update_existingdict(conf, kwargs)
        # conf.update(kwargs)
        return conf

    def conf(self, mdict=None, mkey=None, mode=None, refID=None, **kwargs):
        if mdict is None:
            mdict = self.get_mdict(mkey, mode)
        conf0 = self.generate_configuration(mdict, **kwargs)
        if refID is not None and mkey == 'intermitter':
            conf0 = self.adapt_intermitter(refID=refID, mode=mode, conf=conf0)
        return dNl.NestDict(conf0)

    def module(self, mkey, mode=None, refID=None, mkwargs={}, **kwargs):
        if mode is None:
            mdict = self.dict.brain.m[mkey]
        else:
            mdict = self.dict.brain.m[mkey].mode[mode]
        mkws = self.dict.brain.m[mkey].kwargs
        conf0 = self.conf(mdict=mdict.args, prefix=False, refID=refID, **kwargs)
        func = mdict.class_func
        mkws.update(mkwargs)
        m = func(**conf0, **mkws)
        return m

    def multibconf(self, mbConf):
        multiconf = dNl.NestDict()
        for mkey, mdict in mbConf.items():
            if mkey == 'modules':
                multiconf.modules = mbConf.modules
            elif mdict is None:
                multiconf[mkey] = None
            else:
                multiconf[mkey] = self.conf(mdict)
        return multiconf

    def multiconf(self, mConf):
        mc = dNl.NestDict()
        mc.brain = self.multibconf(mConf['brain'])
        for mkey, mdict in mConf.items():
            if mkey == 'brain':
                continue
            if mdict is None:
                mc[mkey] = None
            else:
                mc[mkey] = self.conf(mdict)
                # mc[mkey] = self.conf(mdict)
        return mc

    def mutate(self, mdict, Pmut, Cmut):
        for d, p in mdict.items():
            p.mutate(Pmut, Cmut)
        # return mdict

    def randomize(self, mdict):
        for d, p in mdict.items():
            p.randomize()

    def compile_pdict(self, dic):
        pdict = dNl.NestDict()
        for mkey, ds in dic.items():
            mdict = self.dict.brain.m[mkey]
            for d in ds:
                pdict[d] = mdict[d]
        return pdict

    def mIDbconf(self, mID=None, m=None):
        if m is None:
            from lib.conf.stored.conf import loadConf
            m = loadConf(mID, 'Model').brain
        mIDconf = dNl.NestDict()
        mIDconf.modules = m.modules
        for mkey, mdic in self.dict.brain.m.items():
            if m.modules[mkey]:
                mmdic = m[f'{mkey}_params']
                mdic = self.update_mdict(mdic, mmdic)
                mIDconf[f'{mkey}_params'] = mdic
            else:
                mIDconf[f'{mkey}_params'] = None
        return mIDconf

    def mIDconf(self, mID=None, m=None):

        if m is None:
            # from lib.conf.stored.conf import loadConf
            m = self.loadConf(mID)

        mc = dNl.NestDict()
        mc.brain = self.mIDbconf(self, m=m.brain)
        for mkey, mdic in self.dict.aux.m.items():
            mmdic = m[mkey]
            mc[mkey] = self.update_mdict(mdic, mmdic)
        return mc

    def mIDtable_data(self, m, columns):
        def gen_rows2(var_mdict, parent, columns, data):
            for k, p in var_mdict.items():
                if isinstance(p, param.Parameterized):
                    ddd = [getattr(p, pname) for pname in columns]
                    row = [parent] + ddd
                    data.append(row)


        mF = dNl.flatten_dict(m)
        data = []
        for mkey in self.dict.brain.keys:
            if m.brain.modules[mkey]:
                d0 = self.dict.model.init[mkey]
                if f'{d0.pref}mode' in mF.keys():
                    mod_v = mF[f'{d0.pref}mode']
                else:
                    mod_v = 'default'

                if mkey == 'intermitter':
                    run_mode = m.brain[f'{mkey}_params']['run_mode']
                    var_ks = d0.mode[mod_v].variable
                    for var_k in var_ks:
                        if var_k == 'run_dist' and run_mode == 'stridechain':
                            continue
                        if var_k == 'stridechain_dist' and run_mode == 'exec':
                            continue
                        v = m.brain[f'{mkey}_params'][var_k]
                        if v is not None:
                            if v.name is not None:
                                vs1, vs2 = reg.get_dist(k=var_k, k0=mkey, v=v, return_tabrows=True)
                                data.append(vs1)
                                data.append(vs2)
                else:
                    var_mdict = self.variable_mdict(mkey, mode=mod_v)
                    var_mdict = self.update_mdict(var_mdict, m.brain[f'{mkey}_params'])
                    gen_rows2(var_mdict, mkey, columns, data)
        for aux_key in self.dict.aux.keys:
            if aux_key not in ['energetics', 'sensorimotor']:
                var_ks = self.dict.aux.init[aux_key].variable
                var_mdict = dNl.NestDict({k: self.dict.aux.m[aux_key].args[k] for k in var_ks})
                var_mdict = self.update_mdict(var_mdict, m[aux_key])
                gen_rows2(var_mdict, aux_key, columns, data)
        if m['energetics']:
            for mod, dic in self.dict.aux.init['energetics'].mode.items():
                var_ks = dic.variable
                var_mdict = dNl.NestDict({k: self.dict.aux.m['energetics'].mode[mod].args[k] for k in var_ks})
                var_mdict = self.update_mdict(var_mdict, m['energetics'].mod)
                gen_rows2(var_mdict, f'energetics.{mod}', columns, data)
        if 'sensorimotor' in m.keys():
            for mod, dic in self.dict.aux.init['sensorimotor'].mode.items():
                var_ks = dic.variable
                var_mdict = dNl.NestDict({k: self.dict.aux.m['sensorimotor'].mode[mod].args[k] for k in var_ks})
                var_mdict = self.update_mdict(var_mdict, m['sensorimotor'])
                gen_rows2(var_mdict, 'sensorimotor', columns, data)
        df = pd.DataFrame(data, columns=['field'] + columns)
        df.set_index(['field'], inplace=True)
        return df

    def mIDtable_data2(self, mID, columns):

        mConf = self.mIDconf(mID)
        # m = self.loadConf(mID)
        m = self.multiconf(mConf)
        data = []

        def gen_rows(mdic, mConf_dic, parent, data0, suf_keys=False):

            for n in mdic:
                p = mConf_dic[n]
                if isinstance(p, param.Parameterized):
                    ddd = [getattr(p, pname) for pname in columns]
                    row = [parent] + ddd
                    data0.append(row)
                else:
                    if suf_keys:
                        new_parent = f'{parent}.{n}'
                    else:
                        new_parent = parent
                    # print(p)
                    # print(isinstance(p, dict))
                    # print(k, p == mConf_dic[n])
                    for ii in p:
                        iii = mConf_dic[n][ii]
                        print(ii == iii)
                    data0 = gen_rows(p, mConf_dic[n], new_parent, data0, suf_keys=suf_keys)
            return data0

        def mvalid(k, dic, data0):

            dvalid = dNl.NestDict({
                'interference': {
                    'square': ['crawler_phi_range', 'attenuation', 'attenuation_max'],
                    'phasic': ['max_attenuation_phase', 'attenuation', 'attenuation_max'],
                    'default': ['attenuation']
                },
                'turner': {
                    'neural': ['base_activation', 'activation_range', 'n', 'tau'],
                    'constant': ['initial_amp'],
                    'sinusoidal': ['initial_amp', 'initial_freq']
                },
                'crawler': {
                    'realistic': ['initial_freq', 'max_scaled_vel', 'max_vel_phase', 'stride_dst_mean',
                                  'stride_dst_std'],
                    'constant': ['initial_amp']
                },
                'physics': ['ang_damping', 'torque_coef', 'body_spring_k', 'bend_correction_coef'],
                'body': ['initial_length', 'Nsegs'],
                'energetics': ['DEB'],
                'Box2D_params': [],
                'olfactor': ['decay_coef', 'perception'],
                'windsensor': ['weights'],
                'toucher': ['touch_sensors', 'decay_coef', 'perception', 'initial_gain'],
                'feeder': ['initial_freq', 'feed_radius', 'V_bite'],
                'memory': [],
                # 'intermitter': ['stridechain_dist', 'pause_dist']
            })

            if k in ['crawler', 'turner', 'interference']:
                vals = dvalid[k][dic.mode]
            elif k == 'intermitter':
                for kkk in ['stridechain_dist', 'pause_dist', 'run_dist']:
                    if dic[kkk] is not None:
                        if dic[kkk].name is not None:
                            vs1, vs2 = reg.get_dist(k=kkk, k0=k, v=dic[kkk], return_tabrows=True)
                            data0.append(vs1)
                            data0.append(vs2)

                vals = []
            else:
                vals = dvalid[k]
            return vals, data0

        for k in self.dict.model.keys:
            if k in self.dict.brain.keys:
                dic = m['brain'][f'{k}_params']
                dic0 = mConf['brain'][f'{k}_params']





            elif k in self.dict.aux.keys:
                dic = m[k]
                dic0 = mConf[k]
            if dic is None:
                continue
            valid, data = mvalid(k, dic, data)

            if len(valid) > 0:
                data = gen_rows(valid, dic0, k, data)

        df = pd.DataFrame(data, columns=['field'] + columns)
        df.set_index(['field'], inplace=True)
        return df

    def mIDtable(self, mID,m=None, columns=['parameter', 'symbol', 'value', 'unit'],
                 colWidths=[0.35, 0.1, 0.25, 0.15],**kwargs):
        from lib.plot.table import conf_table
        if m is None :
            m = self.loadConf(mID)
        df = self.mIDtable_data(m, columns=columns)
        row_colors = [None] + [self.mcolor[ii] for ii in df.index.values]
        from lib.aux.data_aux import arrange_index_labels
        df.index = arrange_index_labels(df.index)
        return conf_table(df, row_colors, mID=mID,colWidths=colWidths, **kwargs)

    def init_loco(self, conf, L):
        D = self.dict.model.m
        for k in ['crawler', 'turner', 'interference', 'feeder', 'intermitter']:

            if conf.modules[k]:

                m = conf[f'{k}_params']
                if k == 'feeder':
                    mode = 'default'
                else:
                    mode = m.mode
                # print(k, D[k], mode)
                kws = {kw: getattr(L, kw) for kw in D[k].kwargs.keys()}
                func = D[k].mode[mode].class_func
                M = func(**m, **kws)
                if k == 'intermitter':
                    M.disinhibit_locomotion(L)
                if k == 'crawler':
                    M.mode = m.mode
            else:
                M = None
            setattr(L, k, M)
        return L

    def init_brain(self, conf, B):
        D = self.dict.model.m
        for k in ['olfactor', 'toucher', 'windsensor', 'thermosensor']:
            if conf.modules[k]:
                m = conf[f'{k}_params']
                if k == 'windsensor':
                    m.gain_dict = {'windsensor': 1.0}
                mode = 'default'
                kws = {kw: getattr(B, kw) for kw in D[k].kwargs.keys()}
                M = D[k].mode[mode].class_func(**m, **kws)
                if k == 'toucher':
                    M.init_sensors(brain=B)


            else:
                M = None
            setattr(B, k, M)
        B.touch_memory = None
        B.memory = None
        if conf.modules['memory']:
            mm = conf['memory_params']
            mode = mm['modality']
            kws = {kw: getattr(B, kw) for kw in D['memory'].kwargs.keys()}
            if mode == 'olfaction' and B.olfactor:
                mm.gain = B.olfactor.gain
                B.memory = D['memory'].mode[mode].class_func(**mm, **kws)
            elif mode == 'touch' and B.toucher:
                mm.gain = B.toucher.gain
                B.touch_memory = D['memory'].mode[mode].class_func(**mm, **kws)
        return B

    def locoConf(self, module_modes=None, modkws={}):

        if module_modes is None:
            module_modes = {'crawler': 'realistic',
                            'turner': 'neural',
                            'interference': 'default',
                            'intermitter': 'default'}

        conf = dNl.NestDict()
        modules = dNl.NestDict()

        for mkey, mode in module_modes.items():
            mlongkey = f'{mkey}_params'
            if mode is None:
                modules[mkey] = False
                conf[mlongkey] = None
            else:
                modules[mkey] = True
                mdict = self.dict.brain.m[mkey].mode[mode].args
                if mkey in modkws.keys():
                    mkws = modkws[mkey]
                else:
                    mkws = {}
                conf[mlongkey] = self.generate_configuration(mdict, **mkws)

        conf.modules = modules
        return conf

    def brainConf(self, modes=None, modkws={}, nengo=False):

        if modes is None:
            modes = {'crawler': 'realistic',
                     'turner': 'neural',
                     'interference': 'phasic',
                     'intermitter': 'default'}

        conf = dNl.NestDict()
        modules = dNl.NestDict()

        for mkey in self.dict.brain.keys:
            mlongkey = f'{mkey}_params'
            if mkey not in modes.keys():
                modules[mkey] = False
                conf[mlongkey] = None
            else:
                mode = modes[mkey]
                modules[mkey] = True
                mdict = self.dict.brain.m[mkey].mode[mode].args
                if mkey in modkws.keys():
                    mkws = modkws[mkey]
                else:
                    mkws = {}
                conf[mlongkey] = self.generate_configuration(mdict, **mkws)
                conf[mlongkey]['mode'] = mode

        conf.modules = modules
        conf.nengo = nengo
        return conf

    def larvaConf(self, modes=None, energetics=None, auxkws={}, modkws={}, nengo=False, mID=None):
        bconf = self.brainConf(modes, modkws, nengo=nengo)

        conf = dNl.NestDict()
        conf.brain = bconf
        # for mkey in self.dict.brain.keys:

        for auxkey in self.dict.aux.keys:
            if auxkey in auxkws.keys():
                mkws = auxkws[auxkey]
            else:
                mkws = {}
            if auxkey == 'energetics':
                if energetics is None:
                    conf[auxkey] = None
                    continue
                else:
                    for m, mdic in self.dict.aux.m[auxkey].mode.items():
                        mdict = mdic.args
                        conf[auxkey][m] = self.generate_configuration(mdict, **mkws[m])
            elif auxkey == 'sensorimotor':
                continue
            else:
                mdict = self.dict.aux.m[auxkey].args
                conf[auxkey] = self.generate_configuration(mdict, **mkws)

        #  TODO thsi
        null_Box2D_params = {
            'joint_types': {
                'friction': {'N': 0, 'args': {}},
                'revolute': {'N': 0, 'args': {}},
                'distance': {'N': 0, 'args': {}}
            }
        }
        conf.Box2D_params = null_Box2D_params

        #
        # T0=

        # self.saveConf(conf, mID)

        return {mID: conf}

    def newConf(self, m0=None, mID0=None, mID=None, kwargs={}):
        if m0 is None:
            m0 = self.loadConf(mID=mID0)
        T0 = dNl.copyDict(m0)
        conf = dNl.update_nestdict(T0, kwargs)
        if mID is not None:
            self.saveConf(conf=conf, mID=mID)
        return conf

    def baseConfs(self):
        mod_dict = {'realistic': 'RE', 'square': 'SQ', 'gaussian': 'GAU', 'constant': 'CON',
                    'default': 'DEF', 'neural': 'NEU', 'sinusoidal': 'SIN', 'nengo': 'NENGO', 'phasic': 'PHI',
                    'branch': 'BR'}
        kws = {'modkws': {'interference': {'attenuation': 0.1, 'attenuation_max': 0.6}}}
        entries = {}
        for Cmod in ['realistic', 'square', 'gaussian', 'constant']:
            for Tmod in ['neural', 'sinusoidal', 'constant']:
                for Ifmod in ['phasic', 'square', 'default']:
                    for IMmod in ['nengo', 'branch', 'default']:
                        kkws = {
                            'mID': f'{mod_dict[Cmod]}_{mod_dict[Tmod]}_{mod_dict[Ifmod]}_{mod_dict[IMmod]}',
                            'modes': {'crawler': Cmod, 'turner': Tmod, 'interference': Ifmod, 'intermitter': IMmod},
                            'nengo': True if IMmod == 'nengo' else False,

                        }
                        if Ifmod != 'default':
                            kkws.update(**kws)
                        entries.update(self.larvaConf(**kkws))
        e1 = self.larvaConf(mID='loco_default', **kws)
        kws2 = {'modkws': {'interference': {'attenuation': 0.0}}}
        e2 = self.larvaConf(mID='Levy', modes={'crawler': 'constant', 'turner': 'sinusoidal', 'interference': 'default',
                                               'intermitter': 'default'}, **kws2)
        e3 = self.larvaConf(mID='NEU_Levy', modes={'crawler': 'constant', 'turner': 'neural', 'interference': 'default',
                                                   'intermitter': 'default'}, **kws2)
        e4 = self.larvaConf(mID='NEU_Levy_continuous',
                            modes={'crawler': 'constant', 'turner': 'neural', 'interference': 'default'}, **kws2)

        e5 = self.larvaConf(mID='CON_SIN', modes={'crawler': 'constant', 'turner': 'sinusoidal'})
        entries.update(**e1, **e2, **e3, **e4, **e5)
        mID0dic = {}
        # m0s=[]
        for Tmod in ['NEU', 'SIN']:
            for Ifmod in ['PHI', 'SQ', 'DEF']:
                mID0 = f'RE_{Tmod}_{Ifmod}_DEF'
                mID0dic[mID0] = entries[mID0]
                # mID0s.append(mID0)
                # m0s.append(entries[mID0])
                for mm in [f'{mID0}_avg', f'{mID0}_var', f'{mID0}_var2']:
                    if mm in self.storedConf():
                        mID0dic[mm] = self.loadConf(mm)
                        # mID0s.append(mm)
                        # m0s.append(self.loadConf(mm))

        olf_pars1 = self.generate_configuration(self.dict.brain.m['olfactor'].mode['default'].args,
                                                odor_dict={'Odor': {'mean': 150.0, 'std': 0.0}})
        olf_pars2 = self.generate_configuration(self.dict.brain.m['olfactor'].mode['default'].args,
                                                odor_dict={'CS': {'mean': 150.0, 'std': 0.0},
                                                           'UCS': {'mean': 0.0, 'std': 0.0}})
        kwargs1 = {'brain.modules.olfactor': True, 'brain.olfactor_params': olf_pars1}
        kwargs2 = {'brain.modules.olfactor': True, 'brain.olfactor_params': olf_pars2}

        feed_pars = self.generate_configuration(self.dict.brain.m['feeder'].mode['default'].args)
        feed_kws = {'brain.modules.feeder': True, 'brain.feeder_params': feed_pars, 'brain.intermitter_params.EEB': 0.5}
        RvSkws={}
        for species, k_abs, EEB in zip(['rover', 'sitter'], [0.8, 0.4], [0.67, 0.37]):
            DEB_pars=self.generate_configuration(self.dict.aux.m['energetics'].mode['DEB'].args,species=species, hunger_gain=1.0,DEB_dt=10.0)
            gut_pars=self.generate_configuration(self.dict.aux.m['energetics'].mode['gut'].args,k_abs=k_abs)
            energy_pars=dNl.NestDict({'DEB' : DEB_pars, 'gut':gut_pars})
            RvSkws[species] = {'wF' : {'energetics': energy_pars, 'brain.intermitter_params.EEB': EEB}, 'woF' :{'energetics': energy_pars} }

        # for m0 in m0s:
        for mID0, m0 in mID0dic.items():
            mID1 = f'{mID0}_nav'
            entries[mID1] = self.newConf(m0=m0, kwargs=kwargs1)
            mID1br = f'{mID1}_brute'
            entries[mID1br] = self.newConf(m0=entries[mID1], kwargs={'brain.olfactor_params.brute_force': True})
            mID2 = f'{mID0}_nav_x2'
            entries[mID2] = self.newConf(m0=m0, kwargs=kwargs2)
            mID2br = f'{mID2}_brute'
            entries[mID2br] = self.newConf(m0=entries[mID2], kwargs={'brain.olfactor_params.brute_force': True})

            mID01 = f'{mID0}_feeder'
            entries[mID01] = self.newConf(m0=m0, kwargs=feed_kws)
            mID02 = f'{mID0}_max_feeder'
            entries[mID02] = self.newConf(m0=entries[mID01], kwargs={'brain.intermitter_params.EEB': 0.9})
            mID11 = f'{mID0}_forager'
            entries[mID11] = self.newConf(m0=entries[mID1], kwargs=feed_kws)
            mID12 = f'{mID0}_max_forager'
            entries[mID12] = self.newConf(m0=entries[mID11], kwargs={'brain.intermitter_params.EEB': 0.9})

        entries['explorer'] = self.newConf(m0=entries['loco_default'], kwargs={})
        entries['navigator'] = self.newConf(m0=entries['explorer'], kwargs=kwargs1)
        for mID0 in ['Levy', 'NEU_Levy', 'NEU_Levy_continuous', 'CON_SIN']:
            entries[f'{mID0}_nav'] = self.newConf(m0=entries[mID0], kwargs=kwargs1)
            entries[f'{mID0}_nav_x2'] = self.newConf(m0=entries[mID0], kwargs=kwargs2)

        sm_pars = self.generate_configuration(self.dict.aux.m['sensorimotor'].mode['default'].args)
        entries['obstacle_avoider'] = self.newConf(m0=entries['RE_NEU_PHI_DEF_nav'], kwargs={'sensorimotor': sm_pars})


        sample_ks = [
            'brain.crawler_params.stride_dst_mean',
            'brain.crawler_params.stride_dst_std',
            'brain.crawler_params.max_scaled_vel',
            'brain.crawler_params.max_vel_phase',
            'brain.crawler_params.initial_freq',
        ]
        for mID0,RvSsuf,Fexists in zip(['RE_NEU_PHI_DEF', 'RE_NEU_PHI_DEF_feeder', 'RE_NEU_PHI_DEF_nav','RE_NEU_PHI_DEF_forager'], ['_loco', '', '_nav', '_forager'], ['woF', 'wF', 'woF', 'wF']):
            entries[f'v{mID0}'] = self.newConf(m0=entries[mID0], kwargs={k: 'sample' for k in sample_ks})
            for species,kws in RvSkws.items():
                entries[f'{species}{RvSsuf}']=self.newConf(m0=entries[mID0], kwargs=kws[Fexists])




        return entries

    def saveConf(self, conf, mID=None):
        if mID is not None:
            from lib.conf.stored.conf import saveConf
            saveConf(conf, 'Model', mID)

    def loadConf(self, mID=None):
        if mID is not None:
            from lib.conf.stored.conf import loadConf
            return loadConf(mID, 'Model')

    def storedConf(self, **kwargs):
        from lib.conf.stored.conf import kConfDict
        return kConfDict('Model', **kwargs)

    def build_full_dict(self, D):

        def register(dic, k0, full_dic):
            for k, p in dic.items():
                kk = f'{k0}.{k}'
                if isinstance(p, param.Parameterized):
                    full_dic[kk] = p
                else:
                    print(kk)
                    register(p, kk, full_dic)

        full_dic = dNl.NestDict()
        for aux_key in D.aux.keys:
            if aux_key in ['energetics', 'sensorimotor']:
                continue
            aux_dic = D.aux.m[aux_key]
            register(aux_dic.args, aux_key, full_dic)
        for aux_key in ['energetics', 'sensorimotor']:
            for m, mdic in D.aux.m[aux_key].mode.items():
                k0 = f'{aux_key}.{m}'
                register(mdic.args, k0, full_dic)

        for bkey in D.brain.keys:
            bkey0 = f'brain.{bkey}_params'
            bdic = D.brain.m[bkey]
            for mod in bdic.mode.keys():
                mdic = bdic.mode[mod].args
                register(mdic, bkey0, full_dic)

        return full_dic

    def diff_df(self, mIDs, ms=None, dIDs=None):
        dic = {}
        if dIDs is None:
            dIDs = mIDs
        if ms is None:
            ms = [self.loadConf(mID) for mID in mIDs]
        ms = [dNl.flatten_dict(m) for m in ms]
        ks = dNl.unique_list(dNl.flatten_list([list(m.keys()) for m in ms]))

        for k in ks:
            entry = {dID: m[k] if k in m.keys() else None for dID, m in zip(dIDs, ms)}
            l = list(entry.values())
            if all([a == l[0] for a in l]):
                continue
            else:
                if k in self.full_dict.keys():
                    k0 = self.full_dict[k].disp
                else:
                    k0 = k.split('.')[-1]
                k00 = k.split('.')[0]
                if k00 == 'brain':
                    k01 = k.split('.')[1]
                    k00 = k01.split('_')[0]
                entry['field'] = k00
                dic[k0] = entry
        df = pd.DataFrame.from_dict(dic).T
        df.index = df.index.set_names(['parameter'])
        # df=df.reset_index().rename(columns={df.index.name: 'parameter'})
        df.reset_index(drop=False, inplace=True)
        df.set_index(['field'], inplace=True)
        df.sort_index(inplace=True)

        # print(df.index.values)
        # raise
        row_colors = [None] + [self.mcolor[ii] for ii in df.index.values]
        from lib.aux.data_aux import arrange_index_labels
        df.index = arrange_index_labels(df.index)
        # df.index = df.index.set_names(['field'])
        # df.reset_index(drop=False, inplace=True)
        # df.set_index(['field','parameter'], inplace=True)

        return df, row_colors

    def adapt_crawler(self, refID=None, e=None, mode='realistic', average=True):
        if e is None:
            d = reg.loadRef(refID)
            d.load(step=False)
            e = d.endpoint_data

        mdict = self.dict.model.m['crawler'].mode[mode].args
        crawler_conf = dNl.NestDict({'mode': mode})
        for d, p in mdict.items():
            # print(d, p.codename)
            if isinstance(p, param.Parameterized):
                try:
                    crawler_conf[d] = epar(e, par=p.codename, average=average)
                except:
                    pass
            else:
                raise
        return crawler_conf

    def adapt_intermitter(self, refID=None, e=None, c=None, mode='default', conf=None):
        if e is None or c is None:
            d = reg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config

        if conf is None:
            mdict = self.dict.model.m['intermitter'].mode[mode].args
            conf = self.generate_configuration(mdict)
        conf.stridechain_dist = c.bout_distros.run_count
        try:
            ll1, ll2 = conf.stridechain_dist.range
            conf.stridechain_dist.range = (int(ll1), int(ll2))
        except:
            pass

        conf.run_dist = c.bout_distros.run_dur
        try:
            ll1, ll2 = conf.run_dist.range
            conf.run_dist.range = (np.round(ll1, 2), np.round(ll2, 2))
        except:
            pass
        conf.pause_dist = c.bout_distros.pause_dur
        try:
            ll1, ll2 = conf.pause_dist.range
            conf.pause_dist.range = (np.round(ll1, 2), np.round(ll2, 2))
        except:
            pass
        conf.crawl_freq = epar(e, 'fsv', average=True)
        conf.mode = mode
        return conf

    def adapt_mID(self, refID, mID0, mID=None, space_mkeys=['turner', 'interference'], save_to=None, e=None, c=None,
                  **kwargs):
        if mID is None:
            mID = f'{mID0}_fitted'
        print(f'Adapting {mID0} on {refID} as {mID} fitting {space_mkeys} modules')
        if e is None or c is None:
            d = reg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config
        if save_to is None:
            save_to = reg.datapath('GAoptimization', c.dir)
        m0 = self.loadConf(mID0)
        if 'crawler' not in space_mkeys:
            m0.brain.crawler_params = self.adapt_crawler(e=e, mode=m0.brain.crawler_params.mode)
            # print(m0.brain.crawler_params)
        if 'intermitter' not in space_mkeys:
            m0.brain.intermitter_params = self.adapt_intermitter(e=e, c=c, mode=m0.brain.intermitter_params.mode,
                                                                 conf=m0.brain.intermitter_params)
        m0.body.initial_length = epar(e, 'l', average=True, Nround=5)

        self.saveConf(conf=m0, mID=mID)

        from lib.sim.eval.model_fit import optimize_mID
        entry = optimize_mID(mID0=mID, space_mkeys=space_mkeys, dt=c.dt, refID=refID,
                             sim_ID=mID, save_to=save_to, **kwargs)
        return entry

    def adapt_6mIDs(self, refID, e=None, c=None):
        if e is None or c is None:
            d = reg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config

        from lib.sim.ga.functions import GA_optimization
        fit_kws = {
            'eval_metrics': {
                'angular kinematics': ['b', 'fov', 'foa'],
                'spatial displacement': ['v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                         'dsp_0_40_max', 'dsp_0_60_max'],
                'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr'],
            },
            'cycle_curves': ['fov', 'foa', 'b']
        }

        fit_dict = GA_optimization(refID, fitness_target_kws=fit_kws)
        entries = {}
        mIDs = []
        for Tmod in ['NEU', 'SIN']:
            for Ifmod in ['PHI', 'SQ', 'DEF']:
                mID0 = f'RE_{Tmod}_{Ifmod}_DEF'
                mID = f'{Ifmod}on{Tmod}'
                entry = self.adapt_mID(refID=refID, mID0=mID0, mID=mID, e=e, c=c,
                                       space_mkeys=['turner', 'interference'],
                                       fit_dict=fit_dict)
                entries.update(entry)
                mIDs.append(mID)
        return entries, mIDs

    def adapt_3modules(self, refID, e=None, c=None):
        if e is None or c is None:
            d = reg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config

        from lib.sim.ga.functions import GA_optimization
        fit_kws = {
            'eval_metrics': {
                'angular kinematics': ['b', 'fov', 'foa'],
                'spatial displacement': ['v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                         'dsp_0_40_max', 'dsp_0_60_max'],
                'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr'],
            },
            'cycle_curves': ['fov', 'foa', 'b']
        }

        fit_dict = GA_optimization(refID, fitness_target_kws=fit_kws)
        entries = {}
        mIDs = []
        # for Cmod in ['GAU', 'CON']:
        for Cmod in ['RE', 'SQ', 'GAU', 'CON']:
            for Tmod in ['NEU', 'SIN', 'CON']:
                for Ifmod in ['PHI', 'SQ', 'DEF']:
                    mID0 = f'{Cmod}_{Tmod}_{Ifmod}_DEF'
                    mID = f'{mID0}_fit'
                    entry = self.adapt_mID(refID=refID, mID0=mID0, mID=mID, e=e, c=c,
                                           space_mkeys=['crawler', 'turner', 'interference'],
                                           fit_dict=fit_dict)
                    entries.update(entry)
                    mIDs.append(mID)
        return entries, mIDs

    def add_var_mIDs(self, refID, e=None, c=None, mID0s=None, mIDs=None, sample_ks=None):
        if e is None or c is None:
            d = reg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config

        if mID0s is None:
            mID0s = list(c.modelConfs.average.keys())
        if mIDs is None:
            mIDs = [f'{mID0}_var' for mID0 in mID0s]
        if sample_ks is None:
            sample_ks = [
                'brain.crawler_params.stride_dst_mean',
                'brain.crawler_params.stride_dst_std',
                'brain.crawler_params.max_scaled_vel',
                'brain.crawler_params.max_vel_phase',
                'brain.crawler_params.initial_freq',
            ]
        kwargs = {k: 'sample' for k in sample_ks}
        entries = {}
        for mID0, mID in zip(mID0s, mIDs):
            m0 = dNl.copyDict(self.loadConf(mID0))
            m = dNl.update_existingnestdict(m0, kwargs)
            self.saveConf(conf=m, mID=mID)
            entries[mID] = m
        return entries

    def update_mdict(self, mdict, mmdic):
        if mmdic is None:
            return None
        else:
            for d, p in mdict.items():
                new_v = mmdic[d] if d in mmdic.keys() else None
                if isinstance(p, param.Parameterized):
                    if type(new_v) == list:
                        if p.parclass in [param.Range, param.NumericTuple, param.Tuple]:
                            new_v = tuple(new_v)
                    p.v = new_v
                else:
                    mdict[d] = self.update_mdict(mdict=p, mmdic=new_v)
            return mdict

    def variable_keys(self, mkey, mode='default'):
        d0 = self.dict.model.init[mkey]
        var_ks = d0.mode[mode].variable
        return var_ks

    def variable_mdict(self, mkey, mode='default'):
        var_ks = self.variable_keys(mkey, mode=mode)
        d00 = self.dict.model.m[mkey].mode[mode].args
        mdict = dNl.NestDict({k: d00[k] for k in var_ks})
        return mdict

    def space_dict(self, mkeys, mConf0):
        mF = dNl.flatten_dict(mConf0)
        dic = {}
        for mkey in mkeys:
            d0 = self.dict.model.init[mkey]
            if f'{d0.pref}mode' in mF.keys():
                mod_v = mF[f'{d0.pref}mode']
            else:
                mod_v = 'default'
            var_mdict = self.variable_mdict(mkey, mode=mod_v)
            for k, p in var_mdict.items():

                k0 = f'{d0.pref}{k}'

                if k0 in mF.keys():
                    dic[k0] = p
                    if type(mF[k0]) == list:
                        if dic[k0].parclass == param.Range:
                            mF[k0] = tuple(mF[k0])
                    dic[k0].v = mF[k0]
        return dNl.NestDict(dic)

    def to_string(self, mdict):
        s = ''
        for k, p in mdict.items():
            s = s + f'{p.d} : {p.v}'
        return s


def epar(e, k=None, par=None, average=True, Nround=2):
    if par is None:
        par = reg.Dic.PI.dict[k].d
    vs = e[par]
    if average:
        return np.round(vs.median(), Nround)
    else:
        return vs



