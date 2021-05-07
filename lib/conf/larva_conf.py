'''
The larva model parameters
'''
import copy

import numpy as np
import lib.conf.dtype_dicts as dtypes

''' Default exploration model'''
default_physics = {
    'torque_coef': 0.41,
    'ang_damping': 2.5,
    'body_spring_k': 0.02,
    'bend_correction_coef': 1.4,
}
default_coupling = {
    'crawler_phi_range': (0.45, 1.0),  # np.pi * 0.55,  # 0.9, #,
    'feeder_phi_range': (0.0, 0.0),  # np.pi * 0.3, #np.pi * 4 / 8,
    'attenuation_ratio': 0.1
}


neural_turner = dtypes.get_dict('turner',
                                mode='neural',
                                base_activation=20.0,
                                activation_range=[10.0, 40.0],
                                noise=0.15,
                                activation_noise=0.5,
                                )

sinusoidal_turner = dtypes.get_dict('turner',
                                    mode='sinusoidal',
                                    initial_amp=15.0,
                                    amp_range=[0.0, 50.0],
                                    initial_freq=0.3,
                                    freq_range=[0.1, 1.0],
                                    noise=0.15,
                                    activation_noise=0.5,
                                    )

default_crawler = {'waveform': 'realistic',
                   'freq_range': [0.5, 2.5],
                   'initial_freq': 'sample',  # From D1 fit
                   'step_to_length_mu': 'sample',  # From D1 fit
                   'step_to_length_std': 'sample',  # From D1 fit
                   'initial_amp': None,
                   'crawler_noise': 0.1,
                   'max_vel_phase': 1
                   }

constant_crawler = {'waveform': 'constant',
                    'freq_range': [0.5, 2.5],
                    'initial_freq': 'sample',  # From D1 fit
                    'step_to_length_mu': 'sample',  # From D1 fit
                    'step_to_length_std': 'sample',  # From D1 fit
                    'initial_amp': 0.0012,
                    'crawler_noise': 0.1,
                    'max_vel_phase': 1
                    }

intermittent_crawler = {'pause_dist': 'fit',
                        'stridechain_dist': 'fit',
                        'intermittent_crawler': True,
                        'intermittent_feeder': False,
                        'EEB_decay_coef': 1,
                        'EEB': 0}

sample_l3_seg2 = {'initial_length': 'sample',
                  'length_std': 0.0,
                  'Nsegs': 2,
                  'seg_ratio': [0.5, 0.5]  # [5 / 11, 6 / 11]
                  }

sample_l3_seg11 = {'initial_length': 'sample',  # From D1 fit
                   'length_std': 0.0,  # From D1 fit
                   'Nsegs': 11,
                   # 'seg_ratio': [5 / 11, 6 / 11]
                   }

brain_locomotion = {'modules': dtypes.get_dict('modules',
                                               crawler=True,
                                               turner=True,
                                               interference=True,
                                               intermitter=True),
                    'turner_params': neural_turner,
                    'crawler_params': default_crawler,
                    'interference_params': default_coupling,
                    'intermitter_params': intermittent_crawler,
                    'olfactor_params': None,
                    'feeder_params': None,
                    'memory_params': None,
                    'nengo': False}

exploring_larva = {'energetics_params': None,
                   'neural_params': brain_locomotion,
                   'sensorimotor_params': default_physics,
                   'body_params': sample_l3_seg2,
                   'odor_params': dtypes.get_dict('odor'),
                   }

# --------------------------------------------TURNER MODES--------------------------------------------------------------
intermitter_rover = {'pause_dist': 'fit',
                     'stridechain_dist': 'fit',
                     'intermittent_crawler': True,
                     'intermittent_feeder': True,
                     'EEB_decay_coef': 1,
                     'EEB': 0.5  # 0.57
                     }

intermitter_sitter = {'pause_dist': 'fit',
                      'stridechain_dist': 'fit',
                      'intermittent_crawler': True,
                      'intermittent_feeder': True,
                      'EEB_decay_coef': 1,
                      'EEB': 0.65  # 0.75
                      }


# ----------------------------------------------OLFACTOR MODES----------------------------------------------------------


def olfactor_conf(ids=['Odor'], means=[150.0], stds=None, noise=0.0):
    def new_odor_dict(ids: list, means: list, stds=None) -> dict:
        if stds is None:
            stds = np.array([0.0] * len(means))
        odor_dict = {}
        for id, m, s in zip(ids, means, stds):
            odor_dict[id] = {'mean': m,
                             'std': s}
        return odor_dict

    odor_dict = {} if ids is None else new_odor_dict(ids, means, stds)
    return {
        'odor_dict': odor_dict,
        'perception': 'log',
        'olfactor_noise': noise,
        'decay_coef': 0.5}



def brain_olfactor_conf(ids, means, stds=None, noise=0.0):
    return {'modules': dtypes.get_dict('modules',
                                       crawler=True,
                                       turner=True,
                                       interference=True,
                                       intermitter=True,
                                       olfactor=True,
                                       feeder=True),
            'turner_params': neural_turner,
            'crawler_params': default_crawler,
            'interference_params': default_coupling,
            'intermitter_params': intermittent_crawler,
            'olfactor_params': olfactor_conf(ids, means, stds, noise),
            'feeder_params': default_feeder,
            'memory_params': None,
            'nengo': False}


def odor_larva_conf(ids, means, stds=None, noise=0.0,
                    odor_id=None, odor_intensity=0.0, odor_spread=0.0001
                    ):
    return copy.deepcopy({'energetics_params': None,
                          'neural_params': brain_olfactor_conf(ids, means, stds, noise),
                          'sensorimotor_params': default_physics,
                          'body_params': sample_l3_seg2,
                          'odor_params': dtypes.get_dict('odor', odor_id=odor_id,
                                                         odor_intensity=odor_intensity, odor_spread=odor_spread)
                          })


# -----------------------------------------------FEEDER MODES-----------------------------------------------------------
default_feeder = {'feeder_freq_range': [1.0, 3.0],
                  'feeder_initial_freq': 2.0,
                  'feed_radius': 0.1,
                  'max_feed_amount_ratio': 0.00001}  # relative to length**2

# ----------------------------------------------SENSORIMOTOR MODES--------------------------------------------------------

# ----------------------------------------------ENERGETICS MODES--------------------------------------------------------


# C-Glucose absorption from [1]:
# Rovers :0.5
# Sitters : 0.15
# [1] K. R. Kaun et al., “Natural variation in food acquisition mediated via a Drosophila cGMP-dependent protein kinase,” J. Exp. Biol., vol. 210, no. 20, pp. 3547–3558, 2007.
energetics_rover = {'f_decay_coef': 0.1,  # 0.1,  # 0.3
                    'absorption_c': 0.5,
                    'hunger_affects_balance': True,
                    'hunger_sensitivity': 12.0,
                    'deb_on': True}

energetics_sitter = {'f_decay_coef': 0.1,  # 0.5,
                     'absorption_c': 0.15,
                     'hunger_affects_balance': True,
                     'hunger_sensitivity': 12.0,
                     'deb_on': True,
                     }

l3_seg11 = {'initial_length': 0.00428,
            'length_std': 0.00053,
            'Nsegs': 11}

l1_seg2 = {'initial_length': 0.0013,
           'length_std': 0.0001,
           'Nsegs': 2,
           'seg_ratio': [0.5, 0.5]  # [5 / 11, 6 / 11]
           }

l3_seg2 = {'initial_length': 0.003,
           'length_std': 0.0,
           'Nsegs': 2,
           'seg_ratio': [0.5, 0.5]  # [5 / 11, 6 / 11]
           }
# -------------------------------------------WHOLE NEURAL MODES---------------------------------------------------------

RL_memory = {'DeltadCon': 0.1,
             'state_spacePerOdorSide': 0,
             'gain_space': [-300.0, -50.0, 50.0, 300.0],
             'update_dt': 1,
             'alpha': 0.05,
             'gamma': 0.6,
             'epsilon': 0.3,
             'train_dur': 20,
             }

brain_RLolfactor = {
    'modules': dtypes.get_dict('modules',
                               crawler=True,
                               turner=True,
                               interference=True,
                               intermitter=True,
                               olfactor=True,
                               feeder=True,
                               memory=True),
    'turner_params': neural_turner,
    'crawler_params': default_crawler,
    'interference_params': default_coupling,
    'intermitter_params': intermittent_crawler,
    'olfactor_params': olfactor_conf(ids=None),
    'feeder_params': default_feeder,
    'memory_params': RL_memory,
    'nengo': False}

brain_immobile_olfactor = {
    'modules': dtypes.get_dict('modules', turner=True, olfactor=True),
    'turner_params': neural_turner,
    'crawler_params': None,
    'interference_params': None,
    'intermitter_params': None,
    'olfactor_params': olfactor_conf(),
    'feeder_params': None,
    'memory_params': None,
    'nengo': False}

brain_olfactor = {'modules': dtypes.get_dict('modules',
                                             turner=True,
                                             crawler=True,
                                             interference=True,
                                             intermitter=True,
                                             olfactor=True),
                  'turner_params': neural_turner,
                  'crawler_params': default_crawler,
                  'interference_params': default_coupling,
                  'intermitter_params': intermittent_crawler,
                  'olfactor_params': olfactor_conf(),
                  'feeder_params': None,
                  'memory_params': None,
                  'nengo': False}

brain_olfactor_x2 = {'modules': dtypes.get_dict('modules',
                                                turner=True,
                                                crawler=True,
                                                interference=True,
                                                intermitter=True,
                                                olfactor=True),
                     'turner_params': neural_turner,
                     'crawler_params': default_crawler,
                     'interference_params': default_coupling,
                     'intermitter_params': intermittent_crawler,
                     'olfactor_params': olfactor_conf(ids=['CS', 'UCS'], means=[150.0, 0.0]),
                     'feeder_params': None,
                     'memory_params': None,
                     'nengo': False}

brain_feeder = {'modules': dtypes.get_dict('modules',
                                           turner=True,
                                           crawler=True,
                                           interference=True,
                                           intermitter=True,
                                           feeder=True),
                'turner_params': neural_turner,
                'crawler_params': default_crawler,
                'interference_params': default_coupling,
                'intermitter_params': intermitter_rover,
                'olfactor_params': None,
                'feeder_params': default_feeder,
                'memory_params': None,
                'nengo': False}

brain_feeder_olfactor = {'modules': dtypes.get_dict('modules',
                                                    crawler=True,
                                                    turner=True,
                                                    interference=True,
                                                    intermitter=True,
                                                    olfactor=True,
                                                    feeder=True),
                         'turner_params': neural_turner,
                         'crawler_params': default_crawler,
                         'interference_params': default_coupling,
                         'intermitter_params': intermitter_rover,
                         'olfactor_params': olfactor_conf(),
                         'feeder_params': default_feeder,
                         'memory_params': None,
                         'nengo': False}

brain_rover = {'modules': dtypes.get_dict('modules',
                                          crawler=True,
                                          turner=True,
                                          interference=True,
                                          intermitter=True,
                                          olfactor=False,
                                          feeder=True),
               'turner_params': neural_turner,
               'crawler_params': default_crawler,
               'interference_params': default_coupling,
               'intermitter_params': intermitter_rover,
               'olfactor_params': None,
               'feeder_params': default_feeder,
               'memory_params': None,
               'nengo': False}

brain_sitter = {'modules': dtypes.get_dict('modules',
                                           crawler=True,
                                           turner=True,
                                           interference=True,
                                           intermitter=True,
                                           olfactor=False,
                                           feeder=True),
                'turner_params': neural_turner,
                'crawler_params': default_crawler,
                'interference_params': default_coupling,
                'intermitter_params': intermitter_sitter,
                'olfactor_params': None,
                'feeder_params': default_feeder,
                'memory_params': None,
                'nengo': False}

# -------------------------------------------WHOLE LARVA MODES---------------------------------------------------------
immobile_odor_larva = {'energetics_params': None,
                       'neural_params': brain_immobile_olfactor,
                       'sensorimotor_params': default_physics,
                       'body_params': sample_l3_seg2,
                       'odor_params': dtypes.get_dict('odor')}

odor_larva = {'energetics_params': None,
              'neural_params': brain_olfactor,
              'sensorimotor_params': default_physics,
              'body_params': sample_l3_seg2,
              'odor_params': dtypes.get_dict('odor')}

odor_larva_x2 = {'energetics_params': None,
                 'neural_params': brain_olfactor_x2,
                 'sensorimotor_params': default_physics,
                 'body_params': sample_l3_seg2,
                 'odor_params': dtypes.get_dict('odor')}

feeding_larva = {'energetics_params': None,
                 'neural_params': brain_feeder,
                 'sensorimotor_params': default_physics,
                 'body_params': sample_l3_seg2,
                 'odor_params': dtypes.get_dict('odor')}

feeding_odor_larva = {'energetics_params': None,
                      'neural_params': brain_feeder_olfactor,
                      'sensorimotor_params': default_physics,
                      'body_params': sample_l3_seg2,
                      'odor_params': dtypes.get_dict('odor')}

growing_rover = {'energetics_params': energetics_rover,
                 'neural_params': brain_rover,
                 'sensorimotor_params': default_physics,
                 'body_params': l1_seg2,
                 'odor_params': dtypes.get_dict('odor')}

growing_sitter = {'energetics_params': energetics_sitter,
                  'neural_params': brain_sitter,
                  'sensorimotor_params': default_physics,
                  'body_params': l1_seg2,
                  'odor_params': dtypes.get_dict('odor')}

# A larva model for imitating experimental datasets (eg contours)

imitation_physics = {
    'torque_coef': 0.4,
    'ang_damping': 1.0,
    'body_spring_k': 1.0
}

imitation_larva = {'energetics_params': None,
                   'neural_params': brain_locomotion,
                   'sensorimotor_params': imitation_physics,
                   'body_params': l3_seg11,
                   'odor_params': dtypes.get_dict('odor')}

brain_nengo = {'modules': dtypes.get_dict('modules',
                                          crawler=True,
                                          turner=True,
                                          interference=True,
                                          intermitter=True,
                                          olfactor=True,
                                          feeder=True),
               'turner_params': {'initial_freq': 0.3,
                                 'initial_amp': 10.0,
                                 'noise': 0.0},
               'crawler_params': default_crawler,
               'interference_params': default_coupling,
               # 'intermitter_params': intermittent_crawler,
               'intermitter_params': intermitter_rover,
               'olfactor_params': {'noise': 0.0},
               'feeder_params': default_feeder,
               'nengo': True}

nengo_larva = {'energetics_params': None,
               'neural_params': brain_nengo,
               'sensorimotor_params': default_physics,
               'body_params': l3_seg2,
               'odor_params': dtypes.get_dict('odor')}

odors3 = [f'{source}_odor' for source in ['Flag', 'Left_base', 'Right_base']]
odors5 = [f'{source}_odor' for source in ['Flag', 'Left_base', 'Right_base', 'Left', 'Right']]
odors2 = [f'{source}_odor' for source in ['Left', 'Right']]

follower_R = {**odor_larva_conf(ids=odors2, means=[150.0, 0.0],
                                  odor_id='Right_odor', odor_intensity=300.0, odor_spread=0.02)}

follower_L = {**odor_larva_conf(ids=odors2, means=[0.0, 150.0],
                                  odor_id='Left_odor', odor_intensity=300.0, odor_spread=0.02)}

king_larva_R = {**odor_larva_conf(ids=odors5, means=[150.0, 0.0, 0.0, 0.0, 0.0],
                                  odor_id='Right_odor', odor_intensity=2.0, odor_spread=0.00005)}

king_larva_L = {**odor_larva_conf(ids=odors5, means=[150.0, 0.0, 0.0, 0.0, 0.0],
                                  odor_id='Left_odor', odor_intensity=2.0, odor_spread=0.00005)}

flag_larva = {**odor_larva_conf(ids=odors3, means=[150.0, 0.0, 0.0]),
              }

RL_odor_larva = {'energetics_params': None,
                 'neural_params': brain_RLolfactor,
                 'sensorimotor_params': default_physics,
                 'body_params': sample_l3_seg2,
                 'odor_params': dtypes.get_dict('odor')}

basic_brain = {'modules': dtypes.get_dict('modules',
                                          turner=True,
                                          crawler=True,
                                          interference=False,
                                          intermitter=False,
                                          olfactor=True),
               'turner_params': sinusoidal_turner,
               # 'turner_params': neural_turner,
               # 'crawler_params': default_crawler,
               'crawler_params': constant_crawler,
               'interference_params': default_coupling,
               'intermitter_params': intermittent_crawler,
               'olfactor_params': olfactor_conf(),
               'feeder_params': None,
               'memory_params': None,
               'nengo': False}

basic_larva = {'energetics_params': None,
               'neural_params': basic_brain,
               'sensorimotor_params': default_physics,
               'body_params': {'initial_length': 'sample',
                               'length_std': 0.0,
                               'Nsegs': 1,
                               'seg_ratio': None  # [5 / 11, 6 / 11]
                               },
               'odor_params': dtypes.get_dict('odor'),
               }
