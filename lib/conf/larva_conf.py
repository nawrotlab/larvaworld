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

brain_locomotion = {'modules': dtypes.get_dict('modules',
                                               crawler=True,
                                               turner=True,
                                               interference=True,
                                               intermitter=True),
                    'turner_params': neural_turner,
                    'crawler_params': dtypes.get_dict('crawler'),
                    'interference_params': default_coupling,
                    'intermitter_params': dtypes.get_dict('intermitter'),
                    'olfactor_params': None,
                    'feeder_params': None,
                    'memory_params': None,
                    'nengo': False}

exploring_larva = {'energetics': None,
                   'brain': brain_locomotion,
                   'physics': dtypes.get_dict('physics'),
                   'body': dtypes.get_dict('body', initial_length='sample'),
                   'odor': dtypes.get_dict('odor'),
                   }



# ----------------------------------------------OLFACTOR MODES----------------------------------------------------------


def olfactor_conf(ids=['Odor'], means=[150.0], stds=None):
    def new_odor_dict(ids: list, means: list, stds=None) -> dict:
        if stds is None:
            stds = np.array([0.0] * len(means))
        odor_dict = {}
        for id, m, s in zip(ids, means, stds):
            odor_dict[id] = {'mean': m,
                             'std': s}
        return odor_dict

    odor_dict = {} if ids is None else new_odor_dict(ids, means, stds)
    return {'odor_dict': odor_dict,**dtypes.get_dict('olfactor')}



def brain_olfactor_conf(ids, means, stds=None):
    return {'modules': dtypes.get_dict('modules',
                                       crawler=True,
                                       turner=True,
                                       interference=True,
                                       intermitter=True,
                                       olfactor=True,
                                       feeder=True),
            'turner_params': neural_turner,
            'crawler_params': dtypes.get_dict('crawler'),
            'interference_params': default_coupling,
            'intermitter_params': dtypes.get_dict('intermitter'),
            'olfactor_params': olfactor_conf(ids, means, stds),
            'feeder_params': dtypes.get_dict('feeder'),
            'memory_params': None,
            'nengo': False}


def odor_larva_conf(ids, means, stds=None,
                    odor_id=None, odor_intensity=0.0, odor_spread=0.0001
                    ):
    return copy.deepcopy({'energetics': None,
                          'brain': brain_olfactor_conf(ids, means, stds),
                          'physics': dtypes.get_dict('physics'),
                          'body': dtypes.get_dict('body', initial_length='sample'),
                          'odor': dtypes.get_dict('odor', odor_id=odor_id,
                                                         odor_intensity=odor_intensity, odor_spread=odor_spread)
                          })


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
    'crawler_params': dtypes.get_dict('crawler'),
    'interference_params': default_coupling,
    'intermitter_params': dtypes.get_dict('intermitter'),
    'olfactor_params': olfactor_conf(ids=None),
    'feeder_params': dtypes.get_dict('feeder'),
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
                  'crawler_params': dtypes.get_dict('crawler'),
                  'interference_params': default_coupling,
                  'intermitter_params': dtypes.get_dict('intermitter'),
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
                     'crawler_params': dtypes.get_dict('crawler'),
                     'interference_params': default_coupling,
                     'intermitter_params': dtypes.get_dict('intermitter'),
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
                'crawler_params': dtypes.get_dict('crawler'),
                'interference_params': default_coupling,
                'intermitter_params': dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.5),
                'olfactor_params': None,
                'feeder_params': dtypes.get_dict('feeder'),
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
                         'crawler_params': dtypes.get_dict('crawler'),
                         'interference_params': default_coupling,
                         'intermitter_params': dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.5),
                         'olfactor_params': olfactor_conf(),
                         'feeder_params': dtypes.get_dict('feeder'),
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
               'crawler_params': dtypes.get_dict('crawler'),
               'interference_params': default_coupling,
               'intermitter_params': dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.5),
               'olfactor_params': None,
               'feeder_params': dtypes.get_dict('feeder'),
               'memory_params': None,
               'nengo': False}

brain_sitter = {'modules': dtypes.get_dict('modules',
                                           crawler=True,
                                           turner=True,
                                           interference=True,
                                           intermitter=True,
                                           feeder=True),
                'turner_params': neural_turner,
                'crawler_params': dtypes.get_dict('crawler'),
                'interference_params': default_coupling,
                'intermitter_params': dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.65),
                'olfactor_params': None,
                'feeder_params': dtypes.get_dict('feeder'),
                'memory_params': None,
                'nengo': False}

# -------------------------------------------WHOLE LARVA MODES---------------------------------------------------------
immobile_odor_larva = {'energetics': None,
                       'brain': brain_immobile_olfactor,
                       'physics': dtypes.get_dict('physics'),
                       'body': dtypes.get_dict('body', initial_length='sample'),
                       'odor': dtypes.get_dict('odor')}

odor_larva = {'energetics': None,
              'brain': brain_olfactor,
              'physics': dtypes.get_dict('physics'),
              'body': dtypes.get_dict('body', initial_length='sample'),
              'odor': dtypes.get_dict('odor')}

odor_larva_x2 = {'energetics': None,
                 'brain': brain_olfactor_x2,
                 'physics': dtypes.get_dict('physics'),
                 'body': dtypes.get_dict('body', initial_length='sample'),
                 'odor': dtypes.get_dict('odor')}

feeding_larva = {'energetics': None,
                 'brain': brain_feeder,
                 'physics': dtypes.get_dict('physics'),
                 'body': dtypes.get_dict('body', initial_length='sample'),
                 'odor': dtypes.get_dict('odor')}

feeding_odor_larva = {'energetics': None,
                      'brain': brain_feeder_olfactor,
                      'physics': dtypes.get_dict('physics'),
                      'body': dtypes.get_dict('body', initial_length='sample'),
                      'odor': dtypes.get_dict('odor')}

growing_rover = {'energetics': dtypes.get_dict('energetics', absorption=0.5),
                 'brain': brain_rover,
                 'physics': dtypes.get_dict('physics'),
                 'body': dtypes.get_dict('body', initial_length=0.001),
                 'odor': dtypes.get_dict('odor')}

growing_sitter = {'energetics': dtypes.get_dict('energetics', absorption=0.15),
                  'brain': brain_sitter,
                  'physics': dtypes.get_dict('physics'),
                  'body': dtypes.get_dict('body', initial_length=0.001),
                  'odor': dtypes.get_dict('odor')}

# A larva model for imitating experimental datasets (eg contours)

imitation_larva = {'energetics': None,
                   'brain': brain_locomotion,
                   'physics': dtypes.get_dict('physics', ang_damping=1.0, body_spring_k=1.0),
                   'body': dtypes.get_dict('body', Nsegs=11),
                   'odor': dtypes.get_dict('odor')}

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
               'crawler_params': dtypes.get_dict('crawler'),
               'interference_params': default_coupling,
               'intermitter_params': dtypes.get_dict('intermitter', feed_bouts=True, EEB=0.5),
               'olfactor_params': {'noise': 0.0},
               'feeder_params': dtypes.get_dict('feeder'),
               'nengo': True}

nengo_larva = {'energetics': None,
               'brain': brain_nengo,
               'physics': dtypes.get_dict('physics'),
               'body': dtypes.get_dict('body'),
               'odor': dtypes.get_dict('odor')}

odors3 = [f'{source}_odor' for source in ['Flag', 'Left_base', 'Right_base']]
odors5 = [f'{source}_odor' for source in ['Flag', 'Left_base', 'Right_base', 'Left', 'Right']]
odors2 = [f'{source}_odor' for source in ['Left', 'Right']]

follower_R = odor_larva_conf(ids=odors2, means=[150.0, 0.0],
                                  odor_id='Right_odor', odor_intensity=300.0, odor_spread=0.02)

follower_L = odor_larva_conf(ids=odors2, means=[0.0, 150.0],
                                  odor_id='Left_odor', odor_intensity=300.0, odor_spread=0.02)

king_larva_R = odor_larva_conf(ids=odors5, means=[150.0, 0.0, 0.0, 0.0, 0.0],
                                  odor_id='Right_odor', odor_intensity=2.0, odor_spread=0.00005)

king_larva_L = odor_larva_conf(ids=odors5, means=[150.0, 0.0, 0.0, 0.0, 0.0],
                                  odor_id='Left_odor', odor_intensity=2.0, odor_spread=0.00005)

flag_larva = odor_larva_conf(ids=odors3, means=[150.0, 0.0, 0.0])

RL_odor_larva = {'energetics': None,
                 'brain': brain_RLolfactor,
                 'physics': dtypes.get_dict('physics'),
                 'body': dtypes.get_dict('body', initial_length='sample'),
                 'odor': dtypes.get_dict('odor')}

basic_brain = {'modules': dtypes.get_dict('modules',
                                          turner=True,
                                          crawler=True,
                                          interference=False,
                                          intermitter=False,
                                          olfactor=True),
               'turner_params': sinusoidal_turner,
               'crawler_params': dtypes.get_dict('crawler', waveform='constant', initial_amp=0.0012),
               'interference_params': default_coupling,
               'intermitter_params': dtypes.get_dict('intermitter'),
               'olfactor_params': olfactor_conf(),
               'feeder_params': None,
               'memory_params': None,
               'nengo': False}

basic_larva = {'energetics': None,
               'brain': basic_brain,
               'physics': dtypes.get_dict('physics'),
               'body': dtypes.get_dict('body', initial_length='sample', Nsegs=1),
               'odor': dtypes.get_dict('odor'),
               }
