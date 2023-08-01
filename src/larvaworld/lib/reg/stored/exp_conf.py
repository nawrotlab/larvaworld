import numpy as np


from larvaworld.lib import reg, aux

def grouped_exp_dic():
    from larvaworld.lib.reg import gen

    def oG(c=1, id='Odor'):
        return gen.Odor(id=id, intensity=2.0 * c, spread=0.0002 * np.sqrt(c)).nestedConf
        # return reg.get_null('odor', id=id, intensity=2.0 * c, spread=0.0002 * np.sqrt(c))


    def oD(c=1, id='Odor'):
        return gen.Odor(id=id, intensity=300.0 * c, spread=0.1 * np.sqrt(c)).nestedConf





    def exp(id, env=None, l={}, enrichment=reg.gen.EnrichConf().nestedConf, dur=10.0,
            c=[],c0=['pose'],  as_entry=False, **kwargs):
        if env is None :
            env=id
        sim = {'duration': dur}
        kw = {'kwdic': {'sim_params': sim},
            'larva_groups': l,
            'env_params': env,
            'experiment': id,
            'enrichment': enrichment,
            'collections': c0 + c,
        }


        kw.update(kwargs)

        if not as_entry:
            return reg.stored.conf.Exp.gConf(**kw)
        else:
            return reg.stored.conf.Exp.entry(id=id, **kw)



    def food_exp(id,  c=['feeder'], dur=10.0,enrichment=gen.EnrichConf(anot_keys=['bout_detection','bout_distribution', 'source_attraction'],
                                                 proc_keys=['spatial', 'angular', 'source']).nestedConf, **kwargs):
        return exp(id,  sim={'duration': dur}, c=c,enrichment=enrichment,
                **kwargs)


    def game_exp(id,  dur=20.0, **kwargs):
        return exp(id, sim={'duration': dur}, **kwargs)


    def deb_exp(id, dur=5.0, **kwargs):
        return exp(id,  sim={'duration': dur},c=['feeder', 'gut'],
                   enrichment=gen.EnrichConf(proc_keys=['spatial']).nestedConf, **kwargs)





    def thermo_exp(id, dur=10.0, **kwargs):
        return exp(id, sim={'duration': dur}, c=['thermo'], enrichment=None, **kwargs)




    def pref_exp(id, dur=5.0, c=[], **kwargs):
        return exp(id, sim={'duration': dur}, c=c, enrichment=gen.EnrichConf(proc_keys=['PI']).nestedConf, **kwargs)


    def game_groups(dim=0.1, N=10, x=0.4, y=0.0, mode='king'):
        x = np.round(x * dim, 3)
        y = np.round(y * dim, 3)
        if mode == 'king':
            l = {**reg.lg('Left', N=N, loc=(-x, y), mID='gamer-5x', c='darkblue', o=oG(id='Left_odor')),
                 **reg.lg('Right', N=N, loc=(+x, y), mID='gamer-5x', c='darkred', o=oG(id='Right_odor'))}
        elif mode == 'flag':
            l = {**reg.lg('Left', N=N, loc=(-x, y), mID='gamer', c='darkblue'),
                 **reg.lg('Right', N=N, loc=(+x, y), mID='gamer', c='darkred')}
        elif mode == 'catch_me':
            l = {**reg.lg('Left', N=1, loc=(-0.01, 0.0), mID='follower-L', c='darkblue', o=oD(id='Left_odor')),
                 **reg.lg('Right', N=1, loc=(+0.01, 0.0), mID='follower-R', c='darkred', o=oD(id='Right_odor'))}
        return l

    def lgs_x4(N=5):
        return reg.lgs(
            mIDs=['RE_NEU_PHI_DEF_max_forager', 'RE_NEU_PHI_DEF_max_feeder', 'RE_NEU_PHI_DEF_nav', 'RE_NEU_PHI_DEF'],
            ids=['forager', 'Orco', 'navigator', 'explorer'], N=N)


    d0={
        'tethered': {'env': 'focus', 'dur':30.0,
                     'l' : reg.lg(mID='immobile', N=1, ors=(90.0, 90.0))
                     },
        'focus': {
            'l' : reg.lg(mID='Levy', N=1, ors=(90.0, 90.0))
        },
        'dish': {
            'l' : reg.lg(mID='loco_default', N=25, s=0.02)
        },
        'dispersion': {'env': 'arena_200mm',
                       'l' : reg.lg(mID='loco_default', N=25)
                       },
        'dispersion_x4':{'env': 'arena_200mm', 'dur':3.0,
                         'l' : reg.lgs(mIDs=['loco_default', 'Levy', 'nengo_explorer'],ids=['CoupledOsc', 'Levy', 'nengo'], N=5)
                         }
    }
    d00={id:exp(id=id, **kws) for id, kws in d0.items()}

    d1={
            'chemotaxis': {'env':'odor_gradient','dur':5.0,
                                   'l' :reg.lg(mID='NEU_Levy_continuous_nav', N=8, loc=(-0.04, 0.0), s=(0.005, 0.02),
                                        ors=(-30.0, 30.0))},
            'chemorbit': {'env':'mid_odor_gaussian', 'dur':3.0, 'l' :reg.lg(mID='RE_NEU_PHI_DEF_nav', N=3)},
            'chemorbit_x3': {'env':'mid_odor_gaussian', 'dur':3.0,
                                     'l' :reg.lgs(mIDs=['RE_NEU_PHI_DEF_nav', 'RL_navigator'],
                                           ids=['CoupledOsc', 'RL'], N=10)},
            'chemorbit_x4': {'env':'mid_odor_gaussian_square', 'dur':3.0, 'l' :lgs_x4()},
            'chemotaxis_diffusion': {'env': 'mid_odor_diffusion', 'l' :reg.lg(mID='RE_NEU_PHI_DEF_nav', N=30)},
            'chemotaxis_RL': {'env': 'mid_odor_diffusion', 'c' : ['memory'],
                                      'l' : reg.lg(mID='RL_navigator', N=10, mode='periphery', s=0.04)},
            'reorientation': {'env': 'mid_odor_diffusion', 'l' :reg.lg(mID='immobile', N=200, s=0.05)},
            'food_at_bottom': {'dur':1.0,
                                       'l' :reg.lgs(mIDs=['RE_NEU_PHI_DEF_max_feeder', 'RE_NEU_PHI_DEF_max_forager'],
                                             ids=['Orco', 'control'], N=5, sh='oval', loc=(0.0, 0.04), s=(0.04, 0.01))}
        }

    d11 = {id: exp(id=id, c0=['olfactor', 'pose'],
                   enrichment=gen.EnrichConf(anot_keys=['bout_detection','bout_distribution', 'source_attraction'],
                                                 proc_keys=['spatial', 'angular', 'source']).nestedConf,
                   **kws) for id, kws in d1.items()}

    d2={
            'anemotaxis': {'env': 'windy_arena', 'dur':0.5, 'l' : reg.lg(mID='nengo_explorer', N=4)},
            'anemotaxis_bordered': {'env': 'windy_arena_bordered', 'dur':0.5, 'l' : reg.lg(mID='nengo_explorer', N=4)},
            'puff_anemotaxis_bordered': {'env': 'puff_arena_bordered', 'dur':0.5, 'l' : reg.lg(mID='nengo_explorer', N=4)},
        }

    d22 = {id: exp(id=id, c0=['wind', 'pose'],
                   enrichment=gen.EnrichConf(proc_keys=['spatial', 'angular', 'wind']).nestedConf,
                   **kws) for id, kws in d2.items()}

    d3 = {
        'single_puff': {'env': 'single_puff', 'dur': 2.5,
                        'l': reg.lg(mID='nengo_explorer', N=20, sample='Coaster.Starved')}
    }

    d33 = {id: exp(id=id, c0=['wind','olfactor', 'pose'],
                   enrichment=gen.EnrichConf(anot_keys=['bout_detection', 'bout_distribution', 'source_attraction'],
                                             proc_keys=['spatial', 'angular', 'source', 'wind']).nestedConf,
                   **kws) for id, kws in d2.items()}

    d={
        'exploration': d00,
        'chemotaxis': d11,
        'anemotaxis': d22,
        'chemanemotaxis': d33,
        # {
        #     'tethered': simple_exp('tethered','focus', dur=30.0, l=reg.lg(mID='immobile', N=1, ors=[90.0, 90.0])),
        #     'focus': simple_exp('focus','focus', l=reg.lg(mID='Levy', N=1, ors=[90.0, 90.0])),
        #     'dish': simple_exp('dish', 'dish', l=reg.lg(mID='loco_default', N=25, s=0.02)),
        #     'dispersion': simple_exp('dispersion', 'arena_200mm', l=reg.lg(mID='explorer', N=25)),
        #     'dispersion_x4': simple_exp('dispersion_x4','arena_200mm', dur=3.0,
        #                                 l=reg.lgs(mIDs=['explorer', 'Levy', 'nengo_explorer'],
        #                                       ids=['CoupledOsc', 'Levy', 'nengo'], N=5)),
        # },

        # 'chemotaxis': {
        #     'chemotaxis': chem_exp('chemotaxis',env='odor_gradient',
        #                            l=reg.lg(mID='NEU_Levy_continuous_nav', N=8, p=(-0.04, 0.0), s=(0.005, 0.02),
        #                                 ors=(-30.0, 30.0))),
        #     'chemorbit': chem_exp('chemorbit', env='mid_odor_gaussian', dur=3.0, l=reg.lg(mID='RE_NEU_PHI_DEF_nav', N=3)),
        #     'chemorbit_x3': chem_exp('chemorbit_x3', env='mid_odor_gaussian', dur=3.0,
        #                              l=reg.lgs(mIDs=['RE_NEU_PHI_DEF_nav', 'RL_navigator'],
        #                                    ids=['CoupledOsc', 'RL'], N=10)),
        #     'chemorbit_x4': chem_exp('chemorbit_x4', env='mid_odor_gaussian_square', dur=3.0, l=lgs_x4()),
        #     'chemotaxis_diffusion': chem_exp('chemotaxis_diffusion',env= 'mid_odor_diffusion', dur=10.0, l=reg.lg(mID='RE_NEU_PHI_DEF_nav', N=30)),
        #     'chemotaxis_RL': chem_exp('chemotaxis_RL',env= 'mid_odor_diffusion', dur=10.0, c=['olfactor', 'memory'],
        #                               l=reg.lg(mID='RL_navigator', N=10, mode='periphery', s=0.04)),
        #     'reorientation': chem_exp('reorientation',env= 'mid_odor_diffusion', l=reg.lg(mID='immobile', N=200, s=0.05)),
        #     'food_at_bottom': chem_exp('food_at_bottom',env= 'food_at_bottom', dur=1.0,
        #                                l=reg.lgs(mIDs=['RE_NEU_PHI_DEF_max_feeder', 'RE_NEU_PHI_DEF_max_forager'],
        #                                      ids=['Orco', 'control'], N=5, sh='oval', p=(0.0, 0.04), s=(0.04, 0.01)))
        # },
        # 'anemotaxis': {
        #     'anemotaxis': anemo_exp('anemotaxis',env= 'windy_arena', dur=0.5, l=reg.lg(mID='nengo_explorer', N=4)),
        #     'anemotaxis_bordered': anemo_exp('anemotaxis_bordered',env= 'windy_arena_bordered', dur=0.5, l=reg.lg(mID='nengo_explorer', N=4)),
        #     'puff_anemotaxis_bordered': anemo_exp('puff_anemotaxis_bordered',env= 'puff_arena_bordered', dur=0.5, l=reg.lg(mID='nengo_explorer', N=4)),
        #     'single_puff': chemanemo_exp('single_puff',env= 'single_puff', dur=2.5, l=reg.lg(mID='nengo_explorer', N=20, sample='Coaster.Starved')),
        # },
        'thermotaxis': {
            'thermotaxis': thermo_exp('thermotaxis', env='thermo_arena', l=reg.lg(mID='thermo_navigator', N=10)),

        },

        'odor_preference': {
            'PItest_off': pref_exp('PItest_off',env= 'CS_UCS_off_food', dur=3.0, l=reg.lg(N=25, s=(0.005, 0.02), mID='RE_NEU_PHI_DEF_nav_x2')),
            'PItest_on': pref_exp('PItest_on',env= 'CS_UCS_on_food', l=reg.lg(N=25, s=(0.005, 0.02), mID='forager_x2')),
            'PItrain_mini': pref_exp('PItrain_mini',env= 'CS_UCS_on_food_x2', dur=1.0, c=['olfactor', 'memory'],
                                     trials='odor_preference_short', l=reg.lg(N=25, s=(0.005, 0.02), mID='RL_forager')),
            'PItrain': pref_exp('PItrain',env= 'CS_UCS_on_food_x2', dur=41.0, c=['olfactor', 'memory'],
                                trials='odor_preference', l=reg.lg(N=25, s=(0.005, 0.02), mID='RL_forager')),
            'PItest_off_RL': pref_exp('PItest_off_RL',env= 'CS_UCS_off_food', dur=105.0, c=['olfactor', 'memory'],
                                      l=reg.lg(N=25, s=(0.005, 0.02), mID='RL_navigator'))},
        'foraging': {
            'patchy_food': food_exp('patchy_food',env= 'patchy_food', l=reg.lg(mID='forager', N=25)),
            'patch_grid': food_exp('patch_grid',env= 'patch_grid', l=lgs_x4()),
            'MB_patch_grid': food_exp('MB_patch_grid',env= 'patch_grid',c=['feeder', 'olfactor'], l=reg.lgs(mIDs=['MB_untrained', 'MB_trained'], N=3)),
            'noMB_patch_grid': food_exp('noMB_patch_grid', env='patch_grid',c=['feeder', 'olfactor'], l=reg.lgs(mIDs=['noMB_untrained', 'noMB_trained'], N=4)),
            'random_food': food_exp('random_food', env='random_food', c=['feeder', 'toucher'], l=reg.lgs(mIDs=['RE_NEU_PHI_DEF_feeder', 'RL_forager'],
                                                                                  ids=['Orco', 'RL'], N=5,
                                                                                  mode='uniform',
                                                                                  shape='rectangular', s=0.04),
                                    enrichment=gen.EnrichConf(proc_keys=['spatial']).nestedConf, en=False),
            'uniform_food': food_exp('uniform_food',env= 'uniform_food', l=reg.lg(mID='RE_NEU_PHI_DEF_feeder', N=5, s=0.005)),
            'food_grid': food_exp('food_grid',env= 'food_grid', l=reg.lg(mID='RE_NEU_PHI_DEF_feeder', N=25)),
            'single_odor_patch': food_exp('single_odor_patch',env= 'single_odor_patch',
                                          l=reg.lgs(mIDs=['RE_NEU_PHI_DEF_feeder', 'forager'],
                                                ids=['Orco', 'control'], N=5, mode='periphery', s=0.01)),
            'single_odor_patch_x4': food_exp('single_odor_patch_x4',env= 'single_odor_patch', l=lgs_x4()),
            'double_patch': food_exp('double_patch',env= 'double_patch', l=reg.GTRvsS(N=5),
                                     c=['toucher', 'feeder', 'olfactor'],
                                     enrichment=reg.gen.EnrichConf(anot_keys=['bout_detection','bout_distribution', 'interference', 'patch_residency'],
                                                 proc_keys=['spatial', 'angular', 'source']).nestedConf, en=False),
            'tactile_detection': food_exp('tactile_detection',env= 'single_patch', dur=5.0, c=['toucher'],
                                          l=reg.lg(mID='toucher', N=15, mode='periphery', s=0.03), en=False),
            'tactile_detection_x3': food_exp('tactile_detection_x3', env='single_patch', dur=600.0, c=['toucher'],
                                             # l=lgs(mIDs=['toucher', 'toucher_brute'],
                                             l=reg.lgs(mIDs=['RL_toucher_2', 'RL_toucher_0', 'toucher', 'toucher_brute',
                                                           'gRL_toucher_0'],
                                                   # ids=['control', 'brute'], N=10), en=False),
                                                   ids=['RL_3sensors', 'RL_1sensor', 'control', 'brute',
                                                        'RL global best'],
                                                   N=10), en=False),
            'tactile_detection_g': food_exp('tactile_detection_g', env='single_patch', dur=600.0, c=['toucher'],
                                            l=reg.lgs(mIDs=['RL_toucher_0', 'gRL_toucher_0'],
                                                  ids=['RL state-specific best', 'RL global best'], N=10), en=False),
            'multi_tactile_detection': food_exp('multi_tactile_detection',env= 'multi_patch', dur=600.0, c=['toucher'],
                                                l=reg.lgs(mIDs=['RL_toucher_2', 'RL_toucher_0', 'toucher'],
                                                      ids=['RL_3sensors', 'RL_1sensor', 'control'], N=4), en=False),
            '4corners': exp('4corners', env='4corners', c=['memory'], l=reg.lg(mID='RL_forager', N=10, s=0.04))
        },

        'growth': {'growth': deb_exp('growth',env= 'food_grid', dur=24 * 60.0, l=reg.GTRvsS(age=0.0)),
                   'RvsS': deb_exp('RvsS',env= 'food_grid', dur=180.0, l=reg.GTRvsS(age=0.0)),
                   'RvsS_on': deb_exp('RvsS_on', env='food_grid', dur=20.0, l=reg.GTRvsS()),
                   'RvsS_off': deb_exp('RvsS_off', env='arena_200mm', dur=20.0, l=reg.GTRvsS()),
                   'RvsS_on_q75': deb_exp('RvsS_on_q75', env='food_grid', l=reg.GTRvsS(q=0.75)),
                   'RvsS_on_q50': deb_exp('RvsS_on_q50', env='food_grid', l=reg.GTRvsS(q=0.50)),
                   'RvsS_on_q25': deb_exp('RvsS_on_q25', env='food_grid', l=reg.GTRvsS(q=0.25)),
                   'RvsS_on_q15': deb_exp('RvsS_on_q15',env= 'food_grid', l=reg.GTRvsS(q=0.15)),
                   'RvsS_on_1h_prestarved': deb_exp('RvsS_on_1h_prestarved',env= 'food_grid', l=reg.GTRvsS(h_starved=1.0)),
                   'RvsS_on_2h_prestarved': deb_exp('RvsS_on_2h_prestarved',env= 'food_grid', l=reg.GTRvsS(h_starved=2.0)),
                   'RvsS_on_3h_prestarved': deb_exp('RvsS_on_3h_prestarved',env= 'food_grid', l=reg.GTRvsS(h_starved=3.0)),
                   'RvsS_on_4h_prestarved': deb_exp('RvsS_on_4h_prestarved', env='food_grid', l=reg.GTRvsS(h_starved=4.0)),

                   },

        'games': {
            'maze': game_exp('maze',env= 'maze', c=['olfactor'],
                             l=reg.lg(N=5, loc=(-0.4 * 0.1, 0.0), ors=(-60.0, 60.0), mID='RE_NEU_PHI_DEF_nav')),
            'keep_the_flag': game_exp('keep_the_flag', env='game', l=game_groups(mode='king')),
            'capture_the_flag': game_exp('capture_the_flag', env='game', l=game_groups(mode='flag')),
            'catch_me': game_exp('catch_me',env= 'arena_50mm_diffusion', l=game_groups(mode='catch_me'))
        },

        'zebrafish': {
            'prey_detection': exp('prey_detection',env= 'windy_blob_arena', l=reg.lg(mID='zebrafish', N=4, s=(0.002, 0.005)),
                                  sim={'duration': 20.0})
        },

        'other': {
            'realistic_imitation': exp('realistic_imitation',env= 'dish', l=reg.lg(mID='imitator', N=25), sim={'Box2D': True}, c=['midline', 'contour'])
            # 'imitation': imitation_exp('None.150controls', model='explorer'),
        }
    }

    return d


@reg.funcs.stored_conf("Exp")
def Exp_dict() :
    exp_dict = aux.merge_dicts(list(grouped_exp_dic().values()))
    return exp_dict

@reg.funcs.stored_conf("ExpGroup")
def ExpGroup_dict() :
    exp_group_dict = aux.AttrDict({k: {'simulations': list(v.keys())} for k, v in grouped_exp_dic().items()})
    return exp_group_dict
