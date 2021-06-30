from lib.conf import dtype_dicts as dtypes


def optimization(fit_par, minimize=True, threshold=0.0001, max_Nsims=10, Nbest=4,
                 operations={'mean': True, 'std': False, 'abs': False}):
    return {
        'fit_par': fit_par,
        'operations': operations,
        'minimize': minimize,
        'threshold': threshold,
        'max_Nsims': max_Nsims,
        'Nbest': Nbest
    }


def batch_methods(run='default', post='default', final='null'):
    return {'run': run,
            'post': post,
            'final': final}


def batch(exp, **kwargs):
    return dtypes.get_dict('batch_conf', exp=exp, **kwargs)


batch_dict = {
    'navigation': batch('chemotaxis_approach',space_search={
                                     'pars': ['Odor.mean', 'decay_coef'],
                                     'ranges': [(300.0, 1300.0), (0.1, 0.5)],
                                     'Ngrid': [3, 3]
                                 }, optimization=optimization('final_dst_to_source')),
    'local-search': batch('chemotaxis_local',space_search={
                                  'pars': ['Odor.mean', 'decay_coef'],
                                  'ranges': [(300.0, 1300.0), (0.1, 0.5)],
                                  'Ngrid': [3, 3]
                              }, optimization=optimization('final_dst_to_center')),
    'odor-preference': batch('odor_pref_test', space_search={
        'pars': ['CS.mean', 'UCS.mean'],
        'ranges': [(-100.0, 100.0), (-100.0, 100.0)],
        'Ngrid': [3, 3]
    }, batch_methods=batch_methods(run='odor_preference', post='null',final='odor_preference')),
    'food_patchy': batch('patchy_food',space_search={
                             'pars': ['EEB', 'feeder_initial_freq'],
                             'ranges': [(0.0, 1.0), (1.5, 2.5)],
                             'Ngrid': [3, 3]
                         }, optimization=optimization('ingested food volume')),
    'food_grid': batch('food_grid',space_search={
                           'pars': ['EEB', 'EEB_decay'],
                           'ranges': [(0.0, 1.0), (0.1, 2.0)],
                           'Ngrid': [6, 6]
                       }, optimization=optimization('amount_eaten')),
    'growth': batch('growth',space_search={
                        'pars': ['EEB', 'hunger_gain'],
                        'ranges': [(0.5, 0.8), (0.0, 0.0)],
                        'Ngrid': [8, 1]
                    }, optimization=optimization('deb_f_deviation', max_Nsims=20,
                                                 operations={'mean': True, 'abs': True})),
    'roversVSsitters': batch('rovers_sitters',space_search={
                                'pars': ['substrate_quality', 'hours_as_larva'],
                                'ranges': [(0.5, 0.8), (0, 100)],
                                'Ngrid': [2, 2]
                            }, batch_methods=batch_methods(run='deb', post='null', final='deb')),
}
#
# odor_pref_batch = batch('odor_pref_test', space_search={
#     'pars': ['CS.mean', 'UCS.mean'],
#     'ranges': [(-100.0, 100.0), (-100.0, 100.0)],
#     'Ngrid': [3, 3]
# }, batch_methods=batch_methods(run='odor_preference', post='null',
#                                final='odor_preference'))
#
# # odor_pref_batch = dtypes.get_dict('batch_conf', exp='odor_pref_test',
# #                                   space_search={
# #                                       'pars': ['CS.mean', 'UCS.mean'],
# #                                       'ranges': [(-100.0, 100.0), (-100.0, 100.0)],
# #                                       'Ngrid': [3, 3]
# #                                   }, batch_methods=batch_methods(run='odor_preference', post='null',
# #                                                                  final='odor_preference'))
# #
# # odor_pref_batch = {
# #     'exp': 'odor_pref_test',
# #     'space_search': {
# #         'pars': ['CS.mean', 'UCS.mean'],
# #         'ranges': [(-100.0, 100.0), (-100.0, 100.0)],
# #         'Ngrid': [3, 3]
# #     },
# #     'batch_methods': batch_methods(run='odor_preference', post='null', final='odor_preference'),
# #     'optimization': None,
# #     'run_kwargs': {'save_data_flag': False}
# # }
#
# chemorbit_batch = batch('chemotaxis_local',
#                         space_search={
#                             'pars': ['Odor.mean', 'decay_coef'],
#                             'ranges': [(300.0, 1300.0), (0.1, 0.5)],
#                             'Ngrid': [3, 3]
#                         }, optimization=optimization('final_dst_to_center'))
# # chemorbit_batch = dtypes.get_dict('batch_conf', exp='chemotaxis_local',
# #                                   space_search={
# #                                       'pars': ['Odor.mean', 'decay_coef'],
# #                                       'ranges': [(300.0, 1300.0), (0.1, 0.5)],
# #                                       'Ngrid': [3, 3]
# #                                   }, batch_methods=batch_methods(), optimization=optimization('final_dst_to_center'))
#
# #
# # chemorbit_batch = {
# #     'exp': 'chemotaxis_local',
# #     'space_search': {
# #         'pars': ['Odor.mean', 'decay_coef'],
# #         'ranges': [(300.0, 1300.0), (0.1, 0.5)],
# #         'Ngrid': [3, 3]
# #     },
# #     'batch_methods': batch_methods(),
# #     'optimization': optimization('final_dst_to_center'),
# #     'run_kwargs': {'save_data_flag': False}
# # }
#
# food_grid_batch = batch('food_grid',
#                         space_search={
#                             'pars': ['EEB', 'EEB_decay'],
#                             'ranges': [(0.0, 1.0), (0.1, 2.0)],
#                             'Ngrid': [6, 6]
#                         }, optimization=optimization('amount_eaten'))
#
# # food_grid_batch = {
# #     'exp': 'food_grid',
# #     'space_search': {
# #         'pars': ['EEB', 'EEB_decay'],
# #         'ranges': [(0.0, 1.0), (0.1, 2.0)],
# #         'Ngrid': [6, 6]
# #     },
# #     'batch_methods': batch_methods(),
# #     'optimization': optimization('amount_eaten'),
# #     'run_kwargs': {'save_data_flag': False}
# # }
#
# chemotax_batch = batch('chemotaxis_approach',
#                        space_search={
#                            'pars': ['Odor.mean', 'decay_coef'],
#                            'ranges': [(300.0, 1300.0), (0.1, 0.5)],
#                            'Ngrid': [3, 3]
#                        }, optimization=optimization('final_dst_to_source'))
#
# # chemotax_batch = {
# #     'exp': 'chemotaxis_approach',
# #     'space_search': {
# #         'pars': ['Odor.mean', 'decay_coef'],
# #         'ranges': [(300.0, 1300.0), (0.1, 0.5)],
# #         'Ngrid': [3, 3]
# #     },
# #     'batch_methods': batch_methods(),
# #     # 'optimization': optimization('x'),
# #     'optimization': optimization('final_dst_to_source'),
# #     'run_kwargs': {'save_data_flag': False}
# # }
#
# patchy_food_batch = batch('patchy_food',
#                           space_search={
#                               'pars': ['EEB', 'feeder_initial_freq'],
#                               'ranges': [(0.0, 1.0), (1.5, 2.5)],
#                               'Ngrid': [3, 3]
#                           }, optimization=optimization('ingested food volume'))
#
# # patchy_food_batch = {
# #     'exp': 'patchy_food',
# #     'space_search': {
# #         'pars': ['EEB', 'feeder_initial_freq'],
# #         'ranges': [(0.0, 1.0), (1.5, 2.5)],
# #         'Ngrid': [3, 3]
# #     },
# #     'batch_methods': batch_methods(),
# #     'optimization': optimization('ingested food volume'),
# #     'run_kwargs': {'save_data_flag': False}
# # }
#
# rovers_sitters_batch = batch('rovers_sitters',
#                              space_search={
#                                  'pars': ['substrate_quality', 'hours_as_larva'],
#                                  'ranges': [(0.5, 0.8), (0, 100)],
#                                  'Ngrid': [2, 2]
#                              }, batch_methods=batch_methods(run='deb', post='null', final='deb'))
#
# # rovers_sitters_batch = {
# #     'exp': 'rovers_sitters',
# #     'space_search': {
# #         'pars': ['substrate_quality', 'hours_as_larva'],
# #         'ranges': [(0.5, 0.8), (0, 100)],
# #         'Ngrid': [2, 2]
# #     },
# #     'batch_methods': batch_methods(run='deb', post='null', final='deb'),
# #     'optimization': None,
# #     'run_kwargs': {'save_data_flag': True}
# # }
#
# growth_batch = batch('growth',
#                      space_search={
#                          'pars': ['EEB', 'hunger_gain'],
#                          'ranges': [(0.5, 0.8), (0.0, 0.0)],
#                          'Ngrid': [8, 1]
#                      }, optimization=optimization('deb_f_deviation', max_Nsims=20,
#                                                   operations={'mean': True, 'abs': True}))
#
# # growth_batch = {
# #     'exp': 'growth',
# #     'space_search': {
# #         'pars': ['EEB', 'hunger_gain'],
# #         'ranges': [(0.5, 0.8), (0.0, 0.0)],
# #         'Ngrid': [8, 1]
# #     },
# #     'batch_methods': batch_methods(),
# #     'optimization': optimization('deb_f_deviation', max_Nsims=20, operations={'mean': True, 'abs': True}),
# #     'run_kwargs': {'save_data_flag': False}
# # }
