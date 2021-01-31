'''
The larva model parameters
'''

import numpy as np

''' Default exploration model'''
vel_torque_transformer = {
    # 'lin_vel_coef': 1.0,
    # 'ang_vel_coef': None,
    # 'lin_force_coef': None,
    'torque_coef': 0.41,  # 0.45, #  1.4,
    # 'lin_mode': 'velocity',
    # 'ang_mode': 'torque',
    # 'lin_damping': 1.0,
    'ang_damping': 2.5,  # 1.0, # 15.0,  # 26.0,
    'body_spring_k': 0.02,  # 1.0,
    'bend_correction_coef': 1.4,
    'static_torque': 0.0,
    # 'density': 300.0,
    # 'friction_pars': {'maxForce': 10 ** 0, 'maxTorque': 10 ** -1}
}
default_interference = {
    'crawler_interference_free_window': np.pi * 0.55,  # np.pi * 0.55,  # 0.9, #,
    'feeder_interference_free_window': 0.0,
    'crawler_interference_start': np.pi * 0.45,  # np.pi * 0.3, #np.pi * 4 / 8,
    'feeder_interference_start': 0.0,
    'interference_ratio': 0.0
}
neural_turner = {'neural': True,
                 'base_activation': 20.0,
                 'activation_range': [10.0, 40.0],
                 'noise': 0.15,
                 'activation_noise': 0.5,
                 'continuous': True,
                 'rebound': False
                 }
sample_realistic_crawler = {'waveform': 'realistic',
                            'freq_range': [0.5, 2.5],
                            'initial_freq': 'sample',  # From D1 fit
                            'initial_freq_std': 0.0,  # From D1 fit
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
              'feeder': False,
              'energetics': False,
              'nengo_olfactor': False,
              'nengo_three_oscillators': False}

# locomotion = ['turner', 'crawler', 'interference', 'intermitter']

pause_distro_explore_powerlaw = {'pause_duration_range': (0.3, 12.0),
                                 'pause_dist': 'powerlaw',
                                 'pause_coef': 2.514}

pause_distro_explore_lognormal = {'pause_duration_range': (0.3, 11.44),
                                  'pause_dist': 'lognormal',
                                  'pause_mean': -0.552,
                                  'pause_std': 0.525}

stridechain_distro_explore = {'stridechain_length_range': (1.0, 146.0),
                              'stridechain_dist': 'lognormal',
                              'stridechain_mean': 1.497,
                              'stridechain_std': 1.13}
intermittent_crawler_explore = {**pause_distro_explore_lognormal, **stridechain_distro_explore,
                                'intermittent_crawler': True,
                                'intermittent_feeder': False,
                                'feeder_reoccurence_rate': 0.9,
                                'intermittent_turner': False,
                                'turner_prepost_lag': [0.0, 0.0],
                                'explore2exploit_bias': 1}
sample_l3_seg2 = {'initial_length': 'sample',  # From D1 fit
                  'length_std': 0.0,  # From D1 fit
                  'Nsegs': 2,
                  'seg_ratio': [5 / 11, 6 / 11]}
sample_neural_params_locomotion = {'component_params': locomotion,
                                   'turner_params': neural_turner,
                                   'crawler_params': sample_realistic_crawler,
                                   'interference_params': default_interference,
                                   'intermitter_params': intermittent_crawler_explore,
                                   'olfactor_params': None,
                                   'feeder_params': None}
sample_exploring_larva = {'energetics_params': None,
                          'homeostatic_drive_params': None,
                          'neural_params': sample_neural_params_locomotion,
                          'sensorimotor_params': vel_torque_transformer,
                          'body_params': sample_l3_seg2}

# -------------------------------------------LARVA MODES----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

sole_turner = {'turner': True,
               'crawler': False,
               'interference': False,
               'intermitter': False,
               'olfactor': False,
               'feeder': False,
               'energetics': False,
               'nengo_olfactor': False,
               'nengo_three_oscillators': False}

sole_crawler = {'turner': False,
                'crawler': True,
                'interference': False,
                'intermitter': False,
                'olfactor': False,
                'feeder': False,
                'energetics': False,
                'nengo_olfactor': False,
                'nengo_three_oscillators': False}

locomotion_no_interference = {'turner': True,
                              'crawler': True,
                              'interference': False,
                              'intermitter': True,
                              'olfactor': False,
                              'feeder': False,
                              'energetics': False,
                              'nengo_olfactor': False,
                              'nengo_three_oscillators': False}

two_osc_interference = {'turner': True,
                        'crawler': True,
                        'interference': True,
                        'intermitter': False,
                        'olfactor': False,
                        'feeder': False,
                        'energetics': False,
                        'nengo_olfactor': False,
                        'nengo_three_oscillators': False}

two_osc = {'turner': True,
           'crawler': True,
           'interference': False,
           'intermitter': False,
           'olfactor': False,
           'feeder': False,
           'energetics': False,
           'nengo_olfactor': False,
           'nengo_three_oscillators': False}

olfactor_turner = {'turner': True,
                   'crawler': False,
                   'interference': False,
                   'intermitter': False,
                   'olfactor': True,
                   'feeder': False,
                   'energetics': False,
                   'nengo_olfactor': False,
                   'nengo_three_oscillators': False}

olfactor_locomotion = {'turner': True,
                       'crawler': True,
                       'interference': True,
                       'intermitter': True,
                       'olfactor': True,
                       'feeder': False,
                       'energetics': False,
                       'nengo_olfactor': False,
                       'nengo_three_oscillators': False}

feed_locomotion = {'turner': True,
                   'crawler': True,
                   'interference': True,
                   'intermitter': True,
                   'olfactor': False,
                   'feeder': True,
                   'energetics': False,
                   'nengo_olfactor': False,
                   'nengo_three_oscillators': False}

olfactor_feed_locomotion = {'turner': True,
                            'crawler': True,
                            'interference': True,
                            'intermitter': True,
                            'olfactor': True,
                            'feeder': True,
                            'energetics': False,
                            'nengo_olfactor': False,
                            'nengo_three_oscillators': False}

growth_locomotion = {'turner': True,
                     'crawler': True,
                     'interference': True,
                     'intermitter': True,
                     'olfactor': False,
                     'feeder': True,
                     'energetics': True,
                     'nengo_olfactor': False,
                     'nengo_three_oscillators': False}

# --------------------------------------------TURNER MODES--------------------------------------------------------------


pause_distro_chemo = {'pause_duration_range': (0.3, 26.0),
                      'pause_dist': 'powerlaw',
                      'pause_coef': 2.0}

stridechain_distro_chemo = {'stridechain_length_range': (1.0, 101.0),
                            'stridechain_dist': 'lognormal',
                            'stridechain_mean': 0.8316,
                            'stridechain_std': 0.8683}

intermittent_crawler_feeder = {**pause_distro_chemo, **stridechain_distro_chemo,
                               'intermittent_crawler': True,
                               'intermittent_feeder': True,
                               'feeder_reoccurence_decay_coef': 1.0,
                               'feeder_reoccurence_rate_on_success': 0.95,
                               'intermittent_turner': False,
                               'turner_prepost_lag': [0.0, 0.0],
                               'explore2exploit_bias': 0.5}

# ----------------------------------------------OLFACTOR MODES----------------------------------------------------------

log_olfactor = {'olfactor_gain_mean': np.array([200.0]),
                'olfactor_gain_std': np.array([0.0]),
                'activation_range': [-1.0, 1.0],
                'noise': 0.0,
                'perception': 'log',
                'decay_coef': 1.0}

log_olfactor_x2 = {'olfactor_gain_mean': [-100.0, 0.0],
                   'olfactor_gain_std': [0.0, 0.0],
                   'activation_range': [-1.0, 1.0],
                   'noise': 0.0,
                   'perception': 'log',
                   'decay_coef': 1.0}
# -----------------------------------------------FEEDER MODES-----------------------------------------------------------
feeder_params = {'freq_range': [1.0, 5.0],
                 'initial_freq': 2.5,
                 'feed_radius': 0.1,
                 'max_feed_amount_ratio': 0.0001}  # relative to current mass which is proportional to length**2

# ----------------------------------------------SENSORIMOTOR MODES--------------------------------------------------------

# ----------------------------------------------ENERGETICS MODES--------------------------------------------------------
energetics_params = {'food_to_biomass_ratio': 0.1,
                     'deb': True}

l3_seg11 = {'initial_length': 0.00428,
            'length_std': 0.00053,
            'Nsegs': 11,
            'joint_type': {'distance': 2, 'revolute': 1},
            'interval': 0.0}

l1_seg2 = {'initial_length': 0.0013,
           'length_std': 0.0001,
           'Nsegs': 2,
           'seg_ratio': [5 / 11, 6 / 11]}

l3_seg2 = {'initial_length': 0.003,
           'length_std': 0.0,
           'Nsegs': 2,
           'seg_ratio': [5 / 11, 6 / 11]}
# -------------------------------------------WHOLE NEURAL MODES---------------------------------------------------------

sample_neural_params_olfactor = {'component_params': olfactor_locomotion,
                                 'turner_params': neural_turner,
                                 'crawler_params': sample_realistic_crawler,
                                 'interference_params': default_interference,
                                 'intermitter_params': intermittent_crawler_explore,
                                 'olfactor_params': log_olfactor,
                                 'feeder_params': None}

sample_neural_params_olfactor_x2 = {'component_params': olfactor_locomotion,
                                    'turner_params': neural_turner,
                                    'crawler_params': sample_realistic_crawler,
                                    'interference_params': default_interference,
                                    'intermitter_params': intermittent_crawler_explore,
                                    'olfactor_params': log_olfactor_x2,
                                    'feeder_params': None}

neural_params_feeder = {'component_params': feed_locomotion,
                        'turner_params': neural_turner,
                        'crawler_params': sample_realistic_crawler,
                        'interference_params': default_interference,
                        'intermitter_params': intermittent_crawler_feeder,
                        'olfactor_params': None,
                        'feeder_params': feeder_params}

neural_params_feeder_olfactor = {'component_params': olfactor_feed_locomotion,
                                 'turner_params': neural_turner,
                                 'crawler_params': sample_realistic_crawler,
                                 'interference_params': default_interference,
                                 'intermitter_params': intermittent_crawler_feeder,
                                 'olfactor_params': log_olfactor,
                                 'feeder_params': feeder_params}

neural_params_growth = {'component_params': growth_locomotion,
                        'turner_params': neural_turner,
                        'crawler_params': sample_realistic_crawler,
                        'interference_params': default_interference,
                        'intermitter_params': intermittent_crawler_feeder,
                        'olfactor_params': None,
                        'feeder_params': feeder_params}

# -------------------------------------------WHOLE LARVA MODES---------------------------------------------------------

sample_odor_larva = {'energetics_params': None,
                     'homeostatic_drive_params': None,
                     'neural_params': sample_neural_params_olfactor,
                     'sensorimotor_params': vel_torque_transformer,
                     'body_params': sample_l3_seg2}

sample_odor_larva_x2 = {'energetics_params': None,
                        'homeostatic_drive_params': None,
                        'neural_params': sample_neural_params_olfactor_x2,
                        'sensorimotor_params': vel_torque_transformer,
                        'body_params': sample_l3_seg2}

feeding_larva = {'energetics_params': None,
                 'homeostatic_drive_params': None,
                 'neural_params': neural_params_feeder,
                 'sensorimotor_params': vel_torque_transformer,
                 'body_params': sample_l3_seg2}

feeding_odor_larva = {'energetics_params': None,
                      'homeostatic_drive_params': None,
                      'neural_params': neural_params_feeder_olfactor,
                      'sensorimotor_params': vel_torque_transformer,
                      'body_params': sample_l3_seg2}

growing_larva = {'energetics_params': energetics_params,
                 'homeostatic_drive_params': None,
                 'neural_params': neural_params_growth,
                 'sensorimotor_params': vel_torque_transformer,
                 'body_params': l1_seg2}

# A larva model for imitating experimental datasets (eg contours)

imitation_transformer = {'lin_vel_coef': 1.0,
                         'ang_vel_coef': None,
                         'lin_force_coef': None,
                         'torque_coef': 0.4,
                         'lin_mode': 'velocity',
                         'ang_mode': 'torque',
                         'lin_damping': 1.0,
                         'ang_damping': 1.0,
                         'body_spring_k': 1.0
                         }

imitation_larva = {'energetics_params': None,
                   'homeostatic_drive_params': None,
                   'neural_params': sample_neural_params_locomotion,
                   'sensorimotor_params': imitation_transformer,
                   'body_params': l3_seg11}

nengo_locomotion = {'turner': False,
                    'crawler': False,
                    'interference': False,
                    'intermitter': False,
                    'olfactor': False,
                    'feeder': False,
                    'energetics': False,
                    'nengo_olfactor': False,
                    'nengo_three_oscillators': True}

nengo_params = {'turner_initial_freq': 0.3,
                'turner_noise': 0.0,
                # 'turner_amp_range': None,
                'turner_initial_amp': 10.0,
                # 'turner_freq_range': None,
                # 'crawler_freq_range': None,
                'crawler_initial_freq': 1.3,
                'crawler_initial_amp': None,
                'crawler_noise': 0.0,
                'scaled_stride_step': 0.2,
                # 'crawler_interference_free_window': 0.25,
                'feeder_initial_freq': 2.0,
                'interference_pars' : default_interference}

nengo_neural_params = {'component_params': nengo_locomotion,
                       'turner_params': None,
                       'crawler_params': None,
                       'interference_params': None,
                       'intermitter_params': intermittent_crawler_explore,
                       # 'intermitter_params': intermittent_crawler_feeder,
                       'olfactor_params': None,
                       'feeder_params': None,
                       'nengo_params': nengo_params}

nengo_larva = {'energetics_params': None,
               'homeostatic_drive_params': None,
               'neural_params': nengo_neural_params,
               'sensorimotor_params': vel_torque_transformer,
               'body_params': l3_seg2}
