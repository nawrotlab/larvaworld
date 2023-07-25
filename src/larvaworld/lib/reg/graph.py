import os
from larvaworld.lib import reg, aux, plot





class GraphRegistry:
    def __init__(self):
        self.dict = reg.funcs.graphs
        self.required_data_dict=reg.funcs.graph_required_data
        self.graphgroups = self.build_graphgroups()

    @property
    def ks(self):
        return list(self.dict.keys())

    def exists(self, ID):
        if isinstance(ID, str) and ID in self.dict.keys():
            return True
        else:
            return False

    def group_exists(self, gID):
        if isinstance(gID, str) and gID in self.graphgroups.keys():
            return True
        else:
            return False


    def eval_graphgroups(self, graphgroups,save_to=None,**kws):
        kws.update({'subfolder' : None})
        ds = aux.AttrDict()
        for gID, entries in self.graphgrouplist_to_dict(graphgroups).items():
            kws0 = {'save_to': f'{save_to}/{gID}' if save_to is not None else None, **kws}
            ds[gID] = self.eval_entries(entries, **kws0)
        return ds

    def graphgrouplist_to_dict(self, graphgroups):
        if isinstance(graphgroups, list) :
            ds = aux.AttrDict()
            for gg in graphgroups:
                # print(gg, type(gg))
                if isinstance(gg, str) and gg in self.graphgroups.keys():
                    gg = {gg:self.graphgroups[gg]}

                assert isinstance(gg, dict)
                assert len(gg) == 1
                gID = list(gg)[0]
                ds[gID]=gg[gID]
            return ds
        elif isinstance(graphgroups, dict):
            return graphgroups

    # def run_group(self, gg,save_to=None, **kwargs):
    #     if isinstance(gg, str) and gg in self.graphgroups.keys():
    #         gg = self.graphgroups[gg]
    #     assert isinstance(gg, dict)
    #     assert len(gg)==1
    #     gID = list(gg)[0]
    #     entries=gg[gID]
    #     # dir = f'{save_to}/{gID}' if save_to is not None else None
    #     kws0={'save_to' : f'{save_to}/{gID}' if save_to is not None else None, **kwargs}
    #     return aux.AttrDict({gID: self.eval_entries(entries, **kws0)})



    def eval_entries(self, entries, **kwargs):
        return aux.AttrDict({e['key']: self.run(ID=e['plotID'], **e['args'], **kwargs) for e in entries})



    def run(self, ID, **kwargs):
        assert self.exists(ID)
        return self.dict[ID](**kwargs)
        # try:
        #     return self.dict[ID](**kwargs)
        # except :
        #     reg.vprint(f'Failed to run graph {ID}',2)
        #     return None


    def run_group(self, gID, **kwargs):
        assert self.group_exists(gID)
        return self.eval_entries(self.graphgroups[gID], **kwargs)

    def entry(self, ID, name=None, **kwargs):
        assert self.exists(ID)
        args=kwargs
        if name is not None:
            args['name']=name
            key=name
        else :
            key = ID
        return {'key': key, 'plotID': ID, 'args': args}


    def entry_list(self, dict, ID='timeplot', **kwargs):
        return [self.entry(ID=ID,name=name,  **args, **kwargs) for name, args in dict.items()]


    def model_tables(self, mIDs,dIDs=None, save_to=None, **kwargs):
        ds = {}
        ds['mdiff_table'] = self.dict['model diff'](mIDs,dIDs=dIDs, save_to=save_to, **kwargs)
        gfunc=self.dict['model table']
        for mID in mIDs:
            try:
                ds[f'{mID}_table'] = gfunc(mID, save_to=save_to, **kwargs)
            except:
                print('TABLE FAIL', mID)
        if save_to is not None and len(ds)>1 :
            aux.combine_pdfs(file_dir=save_to, save_as="_MODEL_TABLES_.pdf", deep=False)
        return aux.AttrDict(ds)

    def model_summaries(self, mIDs, save_to=None, **kwargs):
        ds = {}
        for mID in mIDs:
            try:
                ds[f'{mID}_summary'] = self.dict['model summary'](mID, save_to=save_to, **kwargs)
            except:
                print('SUMMARY FAIL', mID)
        if save_to is not None and len(ds)>0 :
            aux.combine_pdfs(file_dir=save_to, save_as="_MODEL_SUMMARIES_.pdf", deep=False)
        return ds

    def store_model_graphs(self, mIDs, dir):
        f1 = f'{dir}/plots/model_tables'
        f2 = f'{dir}/plots/model_summaries'
        os.makedirs(f1, exist_ok=True)
        os.makedirs(f2, exist_ok=True)

        graphs = aux.AttrDict({
            'tables': self.model_tables(mIDs, save_to=f1),
            'summaries': self.model_summaries(mIDs, Nids=10, refDataset=self, save_to=f2)
        })
        return graphs

    def source_graphgroup(self, source_ID, pos=None, **kwargs):
        ID=source_ID
        gID = f"locomotion relative to source {ID}"
        d0 = [
            self.entry('bearing/turn', name=f'bearing to {ID}',min_angle=5.0,ref_angle=None,source_ID=ID, **kwargs),
            self.entry('bearing/turn', name='bearing to 270deg',min_angle=5.0,ref_angle=270,source_ID=ID, **kwargs),
            *[self.entry('timeplot', name=p, pars=[p], **kwargs) for p in [aux.nam.bearing_to(ID), aux.nam.dst_to(ID), aux.nam.scal(aux.nam.dst_to(ID))]],

        ]

        for chunk in ['stride', 'pause', 'Lturn', 'Rturn']:
            for dur in [0.0, 0.5, 1.0]:
                d0.append(self.entry('bearing to source/epoch', name=f'{chunk}_bearing2_{ID}_min_{dur}_sec',
                               min_dur=dur,chunk=chunk,source_ID=ID, **kwargs))
        return aux.AttrDict({gID: d0})

    def multisource_graphgroup(self, sources, **kwargs):
        graphgroups = []
        for source_ID, pos in sources.items():
            graphgroups.append(self.source_graphgroup(source_ID, pos=pos, **kwargs))
        return graphgroups

    def get_analysis_graphgroups(self, exp, sources, **kwargs):
        groups = ["traj", "general"]
        groups += self.multisource_graphgroup(sources, **kwargs)

        if exp in ['random_food']:
            groups.append("survival")
        else:
            dic = {
                'patch': ["patch"],
                'tactile': ["tactile"],
                'thermo': ["thermo"],
                'RvsS': ["deb", "intake"],
                'growth': ["deb", "intake"],
                'anemo': ["anemotaxis"],
                'puff': ["puff"],
                'chemo': ["chemo"],
                'RL': ["RL"],
                # 'dispersion': ['comparative_analysis'],
                'dispersion': ["endpoint", "distro", "dsp"],

            }
            for k, v in dic.items():
                if k in exp:
                    groups += v

        return groups

    # @property
    def build_graphgroups(self) :
        d= aux.AttrDict({
        'tactile': [
            self.entry('endpoint hist','time ratio on food (final)',ks=['on_food_tr']),
            self.entry('timeplot', 'time ratio on food',ks=['on_food_tr'],  unit='min'),
            self.entry('timeplot', 'time on food',ks=['cum_f_det'],  unit='min'),
            self.entry('timeplot', 'turner input',ks=['A_tur'],  unit='min', show_first=True),
            self.entry('timeplot', 'turner output',ks=['Act_tur'],  unit='min', show_first=True),
            self.entry('timeplot', 'tactile activation',ks=['A_touch'],  unit='min', show_first=True),
            self.entry('ethogram'),
        ],
        'chemo': [
            # autotime(['sv', 'fov', 'b', 'a']),
            self.entry('timeplots','chemosensation', ks=['c_odor1', 'dc_odor1', 'A_olf', 'A_T', 'I_T'], individuals=True),
            self.entry('trajectories'),
            # self.entry('turn amplitude'),
            # self.entry('angular pars', Npars=5),

        ],
        'intake': [
            # 'deb_analysis',
            # *[time(p) for p in ['sf_faeces_M', 'f_faeces_M', 'sf_abs_M', 'f_abs_M', 'f_am']],
            self.entry('food intake (timeplot)', 'food intake (raw)'),
            self.entry('food intake (timeplot)', 'food intake (filtered)', filt_amount=True),
            self.entry('pathlength', scaled=False),
            self.entry('barplot', name='food intake (barplot)', ks=['f_am']),
            self.entry('ethogram')

        ],
        'anemotaxis': [
            *[self.entry('nengo', name=p, group=p, same_plot=True if p == 'anemotaxis' else False)for p in
              ['anemotaxis', 'frequency', 'interference', 'velocity', 'crawler', 'turner', 'wind_effect_on_V',
               'wind_effect_on_Fr']],
            # *[self.entry('timeplot', ks=[p]) for p in ['A_wind', 'anemotaxis']],
            self.entry('timeplots', 'anemotaxis', ks=['A_wind', 'anemotaxis']),
            # *[scat(p) for p in [['o_wind', 'A_wind'], ['anemotaxis', 'o_wind']]],
            self.entry('endpoint hist', name='final anemotaxis', ks=['anemotaxis'])

        ],
        'thermo': [
            self.entry('trajectories'),
            self.entry('timeplots', 'thermosensation', ks=['temp_W', 'dtemp_W', 'temp_C', 'dtemp_C', 'A_therm'], show_first=True)
        ],
        'puff': [

            # self.entry('trajectories'),
            # self.entry('ethogram', add_samples=False),
            self.entry('pathlength', scaled=False),
            # *[self.entry('timeplot', ks=[p], absolute=True) for p in ['fov', 'foa']],
            self.entry('timeplots', 'angular moments', ks=['fov', 'foa'], absolute=True),
            # *[time(p, abs=True) for p in ['fov', 'foa','b', 'bv', 'ba']],
            # *[self.entry('timeplot', ks=[p]) for p in ['sv', 'sa']],
            self.entry('timeplots', 'translational moments', ks=['sv', 'sa']),
            # *[time(p) for p in ['sv', 'sa', 'v', 'a']],
        ],
        'RL': [
            self.entry('timeplot', 'olfactor_decay_table', ks=['D_olf'], table='best_gains'),
            self.entry('timeplot', 'olfactor_decay_table_inds',ks=['D_olf'],  table='best_gains', individuals=True),
            self.entry('timeplot', 'reward_table', ks=['cum_reward'], table='best_gains'),
            self.entry('timeplot', 'best_gains_table',ks=['g_odor1'], table='best_gains'),
            self.entry('timeplot', 'best_gains_table_x2',ks=['g_odor1', 'g_odor2'],  table='best_gains'),
        ],
        'patch': [self.entry('timeplots', 'Y position', ks=['y'], legend_loc='lower left'),
                  self.entry('navigation index'),
                  self.entry('turn amplitude'),
                  self.entry('turn duration'),
                  self.entry('turn amplitude VS Y pos', 'turn angle VS Y pos (scatter)', mode='scatter'),
                  self.entry('turn amplitude VS Y pos', 'turn angle VS Y pos (hist)', mode='hist'),
                  self.entry('turn amplitude VS Y pos', 'bearing correction VS Y pos', mode='hist', ref_angle=270),
                  ],
        'survival': [
            # 'foraging_list',
            self.entry('timeplot', 'time ratio on food', ks=['on_food_tr'], unit='min'),
            self.entry('food intake (timeplot)', 'food intake (raw)'),
            self.entry('pathlength', scaled=False)

        ],
        'deb': [
            *[self.entry('deb',  name = f'DEB.{m} (hours)',sim_only = False, mode=m, save_as=f"{m}_in_hours.pdf") for m in ['energy', 'growth', 'full']],
            *[self.entry('deb',  name = f'FEED.{m} (hours)',sim_only = True, mode=m, save_as=f"{m}_in_hours.pdf") for m in
              ['feeding', 'reserve_density', 'assimilation', 'food_ratio_1', 'food_ratio_2', 'food_mass_1',
               'food_mass_2', 'hunger', 'EEB', 'fs']],
        ],
        'endpoint': [

            self.entry('endpoint box', ks=['l', 'str_N', 'dsp_0_40_max', 'run_tr', 'fv', 'ffov', 'v_mu', 'sv_mu', 'tor5_mu', 'tor5_std',
                    'tor20_mu', 'tor20_std']),
            self.entry('endpoint box', ks=['l', 'fv', 'v_mu', 'run_tr']),
            self.entry('crawl pars')
        ],

        'submission': [
                self.entry('endpoint box', mode='tiny', Ncols=4),
                self.entry('crawl pars'),
                self.entry('epochs', stridechain_duration=True),
                self.entry('dispersal', range=(20, 100)),
                self.entry('dispersal', range=(10, 70)),
                self.entry('dispersal', range=(20, 80)),
                self.entry('dispersal', range=(20, 100), scaled=True),
                self.entry('dispersal', range=(10, 70), scaled=True),
                self.entry('dispersal', range=(20, 80), scaled=True),
            ],

        'distro': [
            self.entry('distros', mode='box'),
            self.entry('distros', mode='hist'),
            self.entry('angular pars', Npars=5)
        ],

        'dsp': [
            self.entry('dispersal', range=(0, 40)),
            # self.entry('dispersal', range=(0, 60)),
            self.entry('dispersal summary', range=(0, 40)),
            # self.entry('dispersal summary', range=(0, 60)),
        ],
        'general': [
            self.entry('ethogram', add_samples=False),
            self.entry('pathlength', scaled=False),
            # self.entry('navigation index'),
            self.entry('epochs', stridechain_duration=True),

        ],
        'stride': [
            self.entry('stride cycle'),
            self.entry('stride cycle', individuals=True),
        ],
        'traj': [
            self.entry('trajectories', mode='default', unit='mm'),
            self.entry('trajectories', name='aligned2origin', mode='origin', unit='mm', single_color=True),
        ],
        'track': [
            self.entry('stride track'),
            self.entry('turn track'),
        ]
        })
        return d

graphs = GraphRegistry()

