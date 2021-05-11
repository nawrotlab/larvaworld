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
             'gain_space': np.arange(-300.0, 300.0, 100.0).tolist(),
             'update_dt': 3,
             'alpha': 0.05,
             'gamma': 0.6,
             'epsilon': 0.5,
             'train_dur': 240,
             }



# -------------------------------------------WHOLE NEURAL MODES---------------------------------------------------------




brain_locomotion = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter'],
                                   turner=neural_turner,
                                   interference=default_coupling)

brain_RLolfactor = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'olfactor', 'feeder', 'memory'],
                                   turner=neural_turner,
                                   interference=default_coupling,
                                   memory=RL_memory)



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

brain_rover = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'feeder'],
                                turner=neural_turner,
                                interference=default_coupling,
                                intermitter=dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.5))

brain_sitter = dtypes.brain_dict(['turner', 'crawler', 'interference', 'intermitter', 'feeder'],
                                 turner=neural_turner,
                                 interference=default_coupling,
                                 intermitter=dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.65))

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

nengo_larva = dtypes.larva_dict(brain_nengo)
RL_odor_larva = dtypes.larva_dict(brain_RLolfactor, body=dtypes.get_dict('body', initial_length='sample'))
imitation_larva = dtypes.larva_dict(brain_locomotion, body=dtypes.get_dict('body', Nsegs=11),
                                    physics=dtypes.get_dict('physics', ang_damping=1.0, body_spring_k=1.0))

basic_larva = dtypes.larva_dict(basic_brain, body=dtypes.get_dict('body', initial_length='sample', Nsegs=1))
feeding_odor_larva = dtypes.larva_dict(brain_feeder_olfactor, body=dtypes.get_dict('body', initial_length='sample'))
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
