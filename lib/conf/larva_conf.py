'''
The larva model parameters
'''
import copy

import numpy as np
import lib.conf.dtype_dicts as dtypes

''' Default exploration model'''

default_coupling = {
    'crawler_phi_range': (0.45, 1.0),  # np.pi * 0.55,  # 0.9, #,
    'feeder_phi_range': (0.0, 0.0),  # np.pi * 0.3, #np.pi * 4 / 8,
    'attenuation': 0.1
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

RL_memory = {'DeltadCon': 0.1,
             'state_spacePerOdorSide': 0,
             # 'gain_space': [150.0],
             'gain_space': np.arange(-200.0, 200.0, 50.0).tolist(),
             'decay_coef_space': None,
             # 'decay_coef_space': np.round(np.arange(0.01, 1.5, 0.2),1).tolist(),
             'update_dt': 1,
             'alpha': 0.05,
             'gamma': 0.6,
             'epsilon': 0.7,
             'train_dur': 30,
             }

# -------------------------------------------WHOLE NEURAL MODES---------------------------------------------------------


brain_locomotion = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter'],
                                     turner=neural_turner,
                                     interference=default_coupling)

brain_RLolfactor = dtypes.brain_dict(
    ['turner', 'crawler', 'interference', 'intermitter', 'olfactor', 'feeder', 'memory'],
    turner=neural_turner,
    interference=default_coupling,
    memory=RL_memory)

brain_RLolfactor_feeder = dtypes.brain_dict(
    ['turner', 'crawler', 'interference', 'intermitter', 'olfactor', 'feeder', 'memory'],
    turner=neural_turner,
    interference=default_coupling,
    memory=RL_memory,
    intermitter=dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.5))

brain_immobile_olfactor = dtypes.brain_dict(['turner', 'olfactor'],
                                            odor_dict={'Odor': {'mean': 150.0, 'std': 0.0}},
                                            turner=neural_turner)

basic_brain = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'olfactor'],
                                odor_dict={'Odor': {'mean': 150.0, 'std': 0.0}},
                                turner=sinusoidal_turner,
                                crawler=dtypes.get_dict('crawler', waveform='constant', initial_amp=0.0012),
                                interference=default_coupling)

brain_olfactor = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'olfactor'],
                                   odor_dict={'Odor': {'mean': 150.0, 'std': 0.0}},
                                   turner=neural_turner,
                                   interference=default_coupling)

brain_olfactor_x2 = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'olfactor'],
                                      odor_dict={'CS': {'mean': 150.0, 'std': 0.0},
                                                 'UCS': {'mean': 0.0, 'std': 0.0}},
                                      turner=neural_turner,
                                      interference=default_coupling,
                                      )

brain_feeder_olfactor = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'olfactor', 'feeder'],
                                          odor_dict={'Odor': {'mean': 150.0, 'std': 0.0}},
                                          turner=neural_turner,
                                          interference=default_coupling,
                                          intermitter=dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.5),
                                          )

brain_feeder_olfactor_x2 = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'olfactor', 'feeder'],
                                             odor_dict={'CS': {'mean': 150.0, 'std': 0.0},
                                                        'UCS': {'mean': 0.0, 'std': 0.0}},
                                             turner=neural_turner,
                                             interference=default_coupling,
                                             intermitter=dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.5),
                                             )

brain_rover = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'feeder'],
                                turner=neural_turner,
                                interference=default_coupling,
                                intermitter=dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.37))

brain_sitter = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'feeder'],
                                 turner=neural_turner,
                                 interference=default_coupling,
                                 intermitter=dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.67))

brain_nengo = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'olfactor', 'feeder'],
                                odor_dict={'Odor': {'mean': 150.0, 'std': 0.0}},
                                turner={'initial_freq': 0.3, 'initial_amp': 10.0, 'noise': 0.0},
                                interference=default_coupling,
                                intermitter=dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.5),
                                olfactor={'noise': 0.0},
                                nengo=True
                                )

# -------------------------------------------WHOLE LARVA MODES---------------------------------------------------------

growing_rover = dtypes.larva_dict(brain_rover, body=dtypes.get_dict('body', initial_length=0.001),
                                  energetics=dtypes.get_dict('energetics', absorption=0.5))
growing_sitter = dtypes.larva_dict(brain_sitter, body=dtypes.get_dict('body', initial_length=0.001),
                                   energetics=dtypes.get_dict('energetics', absorption=0.15))

mock_brain_sitter = dtypes.brain_dict(['intermitter', 'feeder'],
                                      intermitter=dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.67))
mock_growing_sitter = dtypes.larva_dict(mock_brain_sitter, body=dtypes.get_dict('body', initial_length=0.001, Nsegs=1),
                                        energetics=dtypes.get_dict('energetics', absorption=0.15))

mock_growing_rover =mock_growing_sitter

nengo_larva = dtypes.larva_dict(brain_nengo)
RL_odor_larva = dtypes.larva_dict(brain_RLolfactor, body=dtypes.get_dict('body', initial_length='sample'))
RL_feed_odor_larva = dtypes.larva_dict(brain_RLolfactor_feeder, body=dtypes.get_dict('body', initial_length='sample'))
imitation_larva = dtypes.larva_dict(brain_locomotion, body=dtypes.get_dict('body', Nsegs=11),
                                    physics=dtypes.get_dict('physics', ang_damping=1.0, body_spring_k=1.0))

basic_larva = dtypes.larva_dict(basic_brain, body=dtypes.get_dict('body', initial_length='sample', Nsegs=1))
feeding_odor_larva = dtypes.larva_dict(brain_feeder_olfactor, body=dtypes.get_dict('body', initial_length='sample'))
feeding_odor_larva_x2 = dtypes.larva_dict(brain_feeder_olfactor_x2,
                                          body=dtypes.get_dict('body', initial_length='sample'))
feeding_larva = dtypes.larva_dict(brain_rover, body=dtypes.get_dict('body', initial_length='sample'))
immobile_odor_larva = dtypes.larva_dict(brain_immobile_olfactor, body=dtypes.get_dict('body', initial_length='sample'))
odor_larva = dtypes.larva_dict(brain_olfactor, body=dtypes.get_dict('body', initial_length='sample'))
odor_larva_x2 = dtypes.larva_dict(brain_olfactor_x2, body=dtypes.get_dict('body', initial_length='sample'))
exploring_larva = dtypes.larva_dict(brain_locomotion, body=dtypes.get_dict('body', initial_length='sample'))


def brain_olfactor_conf(ids, means, stds=None):
    return dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'olfactor', 'feeder'],
                             odor_dict=dtypes.new_odor_dict(ids, means, stds),
                             turner=neural_turner,
                             interference=default_coupling)


odors3 = [f'{source}_odor' for source in ['Flag', 'Left_base', 'Right_base']]
odors5 = [f'{source}_odor' for source in ['Flag', 'Left_base', 'Right_base', 'Left', 'Right']]
odors2 = [f'{source}_odor' for source in ['Left', 'Right']]

flag_larva = dtypes.larva_dict(brain_olfactor_conf(ids=odors3, means=[150.0, 0.0, 0.0]),
                               body=dtypes.get_dict('body', initial_length='sample'))

follower_R = dtypes.larva_dict(brain_olfactor_conf(ids=odors2, means=[150.0, 0.0]),
                               body=dtypes.get_dict('body', initial_length='sample'),
                               odor=dtypes.get_dict('odor', odor_id='Right_odor', odor_intensity=300.0,
                                                    odor_spread=0.02))

follower_L = dtypes.larva_dict(brain_olfactor_conf(ids=odors2, means=[0.0, 150.0]),
                               body=dtypes.get_dict('body', initial_length='sample'),
                               odor=dtypes.get_dict('odor', odor_id='Left_odor', odor_intensity=300.0,
                                                    odor_spread=0.02))

king_larva_R = dtypes.larva_dict(brain_olfactor_conf(ids=odors5, means=[150.0, 0.0, 0.0, 0.0, 0.0]),
                                 body=dtypes.get_dict('body', initial_length='sample'),
                                 odor=dtypes.get_dict('odor', odor_id='Right_odor', odor_intensity=2.0,
                                                      odor_spread=0.00005))

king_larva_L = dtypes.larva_dict(brain_olfactor_conf(ids=odors5, means=[150.0, 0.0, 0.0, 0.0, 0.0]),
                                 body=dtypes.get_dict('body', initial_length='sample'),
                                 odor=dtypes.get_dict('odor', odor_id='Left_odor', odor_intensity=2.0,
                                                      odor_spread=0.00005))

body_3c = dtypes.get_dict('body', initial_length=3.85 / 1000, length_std=0.35 / 1000)
freq_Fed = np.random.normal(1.244, 0.13)
freq_Deprived = np.random.normal(1.4, 0.14)
freq_Starved = np.random.normal(1.35, 0.15)

crawler_3c = dtypes.get_dict('crawler', step_to_length_mu=0.18, step_to_length_std=0.055, initial_freq=1.35,
                             freq_std=0.14)
pause_dist_Fed = {'range': (0.22, 69.0),
                  'name': 'lognormal',
                  'mu': -0.488,
                  'sigma': 0.705}
pause_dist_Deprived = {'range': (0.22, 91.0),
                       'name': 'lognormal',
                       'mu': -0.431,
                       'sigma': 0.79}
pause_dist_Starved = {'range': (0.22, 22.0),
                      'name': 'lognormal',
                      'mu': -0.534,
                      'sigma': 0.733}

stridechain_dist_Fed = {'range': (1, 63),
                        'name': 'lognormal',
                        'mu': 0.987,
                        'sigma': 0.885}
stridechain_dist_Deprived = {'range': (1, 99),
                             'name': 'lognormal',
                             'mu': 1.052,
                             'sigma': 0.978}
stridechain_dist_Starved = {'range': (1, 191),
                            'name': 'lognormal',
                            'mu': 1.227,
                            'sigma': 1.052}

stridechain_dist_3c = dtypes.get_dict('logn_dist', range=(1, 120), mu=1.1, sigma=0.95)
pause_dist_3c = dtypes.get_dict('logn_dist', range=(0.22, 56.0), mu=-0.48, sigma=0.74)

intermitter_3c = dtypes.get_dict('intermitter', pause_dist=pause_dist_3c, stridechain_dist=stridechain_dist_3c)

brain_3c = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter'],
                             crawler=crawler_3c,
                             intermitter=intermitter_3c,
                             turner=neural_turner,
                             interference=default_coupling)

exploring_3c_larva = dtypes.larva_dict(brain_3c, body=body_3c)
