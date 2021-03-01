'''
The larva model parameters
'''

import numpy as np

''' Default exploration model'''
default_physics = {
    'torque_coef': 0.41,
    'ang_damping': 2.5,
    'body_spring_k': 0.02,
    'bend_correction_coef': 1.4,
}
default_coupling = {
    'crawler_interference_free_window': np.pi * 0.55,  # np.pi * 0.55,  # 0.9, #,
    'feeder_interference_free_window': 0.0,
    'crawler_interference_start': np.pi * 0.45,  # np.pi * 0.3, #np.pi * 4 / 8,
    'feeder_interference_start': 0.0,
    'interference_ratio': 0.0
}
default_turner = {'neural': True,
                  'base_activation': 20.0,
                  'activation_range': [10.0, 40.0],
                  'noise': 0.15,
                  'activation_noise': 0.5
                  }
default_crawler = {'waveform': 'realistic',
                   'freq_range': [0.5, 2.5],
                   'initial_freq': 'sample',  # From D1 fit
                   'step_to_length_mu': 'sample',  # From D1 fit
                   'step_to_length_std': 'sample',  # From D1 fit
                   'initial_amp': None,
                   'random_phi': True,
                   'noise': 0.1,
                   'max_vel_phase': np.pi
                   }
locomotion = {'turner': True,
              'crawler': True,
              'interference': True,
              'intermitter': True,
              'olfactor': False,
              'feeder': False}

intermittent_crawler = {'pause_dist': 'fit',
                        'stridechain_dist': 'fit',
                        'intermittent_crawler': True,
                        'explore2exploit_bias': 1}
sample_l3_seg2 = {'initial_length': 'sample',
                  'Nsegs': 2,
                  'seg_ratio': [0.5,0.5]#[5 / 11, 6 / 11]
                  }

sample_l3_seg11 = {'initial_length': 'sample',  # From D1 fit
                   'length_std': 0.0,  # From D1 fit
                   'Nsegs': 11,
                   # 'seg_ratio': [5 / 11, 6 / 11]
                   }

brain_locomotion = {'modules': locomotion,
                    'turner_params': default_turner,
                    'crawler_params': default_crawler,
                    'interference_params': default_coupling,
                    'intermitter_params': intermittent_crawler,
                    'olfactor_params': None,
                    'feeder_params': None,
                    'nengo': False}
exploring_larva = {'energetics_params': None,
                   'neural_params': brain_locomotion,
                   'sensorimotor_params': default_physics,
                   # 'body_params': sample_l3_seg11
                   'body_params': sample_l3_seg2
                   }

# -------------------------------------------LARVA MODES----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

sole_turner = {'turner': True,
               'crawler': False,
               'interference': False,
               'intermitter': False,
               'olfactor': False,
               'feeder': False}

sole_crawler = {'turner': False,
                'crawler': True,
                'interference': False,
                'intermitter': False,
                'olfactor': False,
                'feeder': False}

locomotion_no_interference = {'turner': True,
                              'crawler': True,
                              'interference': False,
                              'intermitter': True,
                              'olfactor': False,
                              'feeder': False}

two_osc_interference = {'turner': True,
                        'crawler': True,
                        'interference': True,
                        'intermitter': False,
                        'olfactor': False,
                        'feeder': False}

two_osc = {'turner': True,
           'crawler': True,
           'interference': False,
           'intermitter': False,
           'olfactor': False,
           'feeder': False}

olfactor_turner = {'turner': True,
                   'crawler': False,
                   'interference': False,
                   'intermitter': False,
                   'olfactor': True,
                   'feeder': False}

olfactor_locomotion = {'turner': True,
                       'crawler': True,
                       'interference': True,
                       'intermitter': True,
                       'olfactor': True,
                       'feeder': False}

feed_locomotion = {'turner': True,
                   'crawler': True,
                   'interference': True,
                   'intermitter': True,
                   'olfactor': False,
                   'feeder': True}

full_brain = {'turner': True,
              'crawler': True,
              'interference': True,
              'intermitter': True,
              'olfactor': True,
              'feeder': True}

growth_locomotion = {'turner': True,
                     'crawler': True,
                     'interference': True,
                     'intermitter': True,
                     'olfactor': False,
                     'feeder': True}

# --------------------------------------------TURNER MODES--------------------------------------------------------------
intermitter_rover = {'pause_dist': 'fit',
                     'stridechain_dist': 'fit',
                     'intermittent_crawler': True,
                     'intermittent_feeder': True,
                     'feeder_reoccurence_decay_coef': 1,
                     'explore2exploit_bias': 0.6  # 0.74 # 0.52
                     }

intermitter_sitter = {'pause_dist': 'fit',
                      'stridechain_dist': 'fit',
                      'intermittent_crawler': True,
                      'intermittent_feeder': True,
                      'feeder_reoccurence_decay_coef': 1,
                      'explore2exploit_bias': 0.36  # 0.44 # 0.4
                      }

# ----------------------------------------------OLFACTOR MODES----------------------------------------------------------

default_olfactor = {'olfactor_gain_mean': np.array([200.0]),
                    'olfactor_gain_std': np.array([0.0]),
                    'noise': 0.0,
                    'decay_coef': 1.0}

default_olfactor_x2 = {'olfactor_gain_mean': [-100.0, 0.0],
                       'olfactor_gain_std': [0.0, 0.0],
                       'noise': 0.0,
                       'decay_coef': 1.0}
# -----------------------------------------------FEEDER MODES-----------------------------------------------------------
default_feeder = {'freq_range': [1.0, 5.0],
                  'initial_freq': 2.5,
                  'feed_radius': 0.1,
                  'max_feed_amount_ratio': 0.0001}  # relative to current mass which is proportional to length**2

# ----------------------------------------------SENSORIMOTOR MODES--------------------------------------------------------

# ----------------------------------------------ENERGETICS MODES--------------------------------------------------------
# energetics_rover_old = {'food_to_biomass_ratio': 0.1,
#                         'f_decay_coef': 0.5,  # 0.1,  # 0.3
#                         'f_increment': 1.0,
#                         'hunger_affects_balance': True,
#                         'deb': True}
#
# energetics_sitter_old = {'food_to_biomass_ratio': 0.1,
#                          'f_decay_coef': 0.7,  # 0.5,
#                          'f_increment': 1.0,
#                          'hunger_affects_balance': True,
#                          'deb': True,
#                          }

# C-Glucose absorption from [1]:
# Rovers :0.5
# Sitters : 0.15
# [1] K. R. Kaun et al., “Natural variation in food acquisition mediated via a Drosophila cGMP-dependent protein kinase,” J. Exp. Biol., vol. 210, no. 20, pp. 3547–3558, 2007.
energetics_rover = {'food_to_biomass_ratio': 0.1,
                    'f_decay_coef': 0.1,  # 0.1,  # 0.3
                    'f_increment': 0.5,
                    'hunger_affects_balance': True,
                    'deb': True}

energetics_sitter = {'food_to_biomass_ratio': 0.1,
                     'f_decay_coef': 0.1,  # 0.5,
                     'f_increment': 0.15,
                     'hunger_affects_balance': True,
                     'deb': True,
                     }

l3_seg11 = {'initial_length': 0.00428,
            'length_std': 0.00053,
            'Nsegs': 11,
            'joint_type': {'distance': 2, 'revolute': 1},
            'interval': 0.0}

l1_seg2 = {'initial_length': 0.0013,
           'length_std': 0.0001,
           'Nsegs': 2,
           'seg_ratio': [0.5,0.5]#[5 / 11, 6 / 11]
           }

l3_seg2 = {'initial_length': 0.003,
           'length_std': 0.0,
           'Nsegs': 2,
           'seg_ratio': [0.5,0.5]#[5 / 11, 6 / 11]
           }
# -------------------------------------------WHOLE NEURAL MODES---------------------------------------------------------

brain_olfactor = {'modules': olfactor_locomotion,
                  'turner_params': default_turner,
                  'crawler_params': default_crawler,
                  'interference_params': default_coupling,
                  'intermitter_params': intermittent_crawler,
                  'olfactor_params': default_olfactor,
                  'feeder_params': None,
                  'nengo': False}

brain_olfactor_x2 = {'modules': olfactor_locomotion,
                     'turner_params': default_turner,
                     'crawler_params': default_crawler,
                     'interference_params': default_coupling,
                     'intermitter_params': intermittent_crawler,
                     'olfactor_params': default_olfactor_x2,
                     'feeder_params': None,
                     'nengo': False}

brain_feeder = {'modules': feed_locomotion,
                'turner_params': default_turner,
                'crawler_params': default_crawler,
                'interference_params': default_coupling,
                'intermitter_params': intermitter_rover,
                'olfactor_params': None,
                'feeder_params': default_feeder,
                'nengo': False}

brain_feeder_olfactor = {'modules': full_brain,
                         'turner_params': default_turner,
                         'crawler_params': default_crawler,
                         'interference_params': default_coupling,
                         'intermitter_params': intermitter_rover,
                         'olfactor_params': default_olfactor,
                         'feeder_params': default_feeder,
                         'nengo': False}

brain_rover = {'modules': growth_locomotion,
               'turner_params': default_turner,
               'crawler_params': default_crawler,
               'interference_params': default_coupling,
               'intermitter_params': intermitter_rover,
               'olfactor_params': None,
               'feeder_params': default_feeder,
               'nengo': False}

brain_sitter = {'modules': growth_locomotion,
                'turner_params': default_turner,
                'crawler_params': default_crawler,
                'interference_params': default_coupling,
                'intermitter_params': intermitter_sitter,
                'olfactor_params': None,
                'feeder_params': default_feeder,
                'nengo': False}

# -------------------------------------------WHOLE LARVA MODES---------------------------------------------------------

odor_larva = {'energetics_params': None,
              'neural_params': brain_olfactor,
              'sensorimotor_params': default_physics,
              'body_params': sample_l3_seg2}

odor_larva_x2 = {'energetics_params': None,
                 'neural_params': brain_olfactor_x2,
                 'sensorimotor_params': default_physics,
                 'body_params': sample_l3_seg2}

feeding_larva = {'energetics_params': None,
                 'neural_params': brain_feeder,
                 'sensorimotor_params': default_physics,
                 'body_params': sample_l3_seg2}

feeding_odor_larva = {'energetics_params': None,
                      'neural_params': brain_feeder_olfactor,
                      'sensorimotor_params': default_physics,
                      'body_params': sample_l3_seg2}

growing_rover = {'energetics_params': energetics_rover,
                 'neural_params': brain_rover,
                 'sensorimotor_params': default_physics,
                 'body_params': l1_seg2,
                 'id_prefix': 'Rover'}

growing_sitter = {'energetics_params': energetics_sitter,
                  'neural_params': brain_sitter,
                  'sensorimotor_params': default_physics,
                  'body_params': l1_seg2,
                  'id_prefix': 'Sitter'}

# A larva model for imitating experimental datasets (eg contours)

imitation_physics = {
    'torque_coef': 0.4,
    'lin_damping': 1.0,
    'ang_damping': 1.0,
    'body_spring_k': 1.0
}

imitation_larva = {'energetics_params': None,
                   'neural_params': brain_locomotion,
                   'sensorimotor_params': imitation_physics,
                   'body_params': l3_seg11}

brain_nengo = {'modules': full_brain,
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
               'body_params': l3_seg2}
