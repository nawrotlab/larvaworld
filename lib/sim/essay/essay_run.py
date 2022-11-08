from lib.registry import reg
from lib.registry.base import BaseRun
from lib.registry.output import set_output
from lib.aux import naming as nam, dictsNlists as dNl


class EssayRun(BaseRun):
    def __init__(self, enrichment, collections=['pose'],N=5,Npoints=3,dt=0.1,dur=None,
                 video=False, **kwargs):


        super().__init__(runtype='essay', **kwargs)
        if video:
            self.vis_kwargs = reg.get_null('visualization', mode='video', video_speed=60)
        else:
            self.vis_kwargs = reg.get_null('visualization', mode=None)
        self.N = N
        self.dt = dt
        self.dur = dur
        self.enrichment = enrichment
        self.collections = collections
        self.output = set_output(collections=collections,Npoints=Npoints)

        self.exp_dict = {}
        self.datasets = {}



    def get_exp_conf(self,env, lgs,dur=None,exp=None,**kwargs):
        if exp is None :
            exp=self.experiment
        if dur is None :
            dur=self.dur
        return dNl.NestDict({
            'dt': self.dt,
            'dur': self.dur,
            # 'model_class': WorldSim,
            'output': self.output,
            # 'id': self.id,
            'experiment': exp,
            'save_to': self.data_dir,
            # 'Box2D': sim_params.Box2D,
            'env_params': env,
            'larva_groups': lgs,
            # 'vis_kwargs': self.vis_kwargs,
            **kwargs
        })



    def simulate(self):
        print(f'Running essay "{self.id}"')
        for exp, confs in self.exp_dict.items():
            self.datasets[exp]=[]
            print(f'Running {len(confs)} versions of experiment {exp}')
            for conf in confs :
                ds=self.model_class(**conf, vis_kwargs=self.vis_kwargs).simulate()
                if ds is not None :
                    if self.enrichment:
                        for d in ds:
                            d._enrich(**self.enrichment, is_last=False, store=self.store_data)
                    self.datasets[exp].append(ds)

        return self.datasets

    def manual_anal(self):
        for exp, ds in self.datasets.items():
            self.solo_anal(exp=exp, ds0=ds)
        #return self.figs, self.results

    def solo_anal(self, exp, ds0):
        pass
        #return {}, None





class DoublePatch_Essay(EssayRun):
    def __init__(self, substrates=['sucrose', 'standard', 'cornmeal'], N=10, dur=5.0, olfactor=True, feeder=True,
                 arena_dims=(0.24, 0.24), patch_x=0.06,patch_radius=0.025,
                 **kwargs):

        enrichment={'proc_keys' : ['spatial', 'angular', 'source'], 'interference':False, 'on_food':True, 'bout_annotation':True}
        self.mID0s = ['rover', 'sitter']
        if olfactor:
            if feeder:
                suf = '_forager'
                self.mode = 'foragers'
            else:
                suf = '_nav'
                self.mode = 'navigators'
        else:
            if feeder:
                suf = ''
                self.mode = 'feeders'
            else:
                suf = '_loco'
                self.mode = 'locomotors'
        self.mIDs = [f'{mID0}{suf}' for mID0 in self.mID0s]
        graph_entries = [
            reg.GD.entry('double-patch summary', f'{self.mode}_fig1', args={'ks': None}),
            reg.GD.entry('double-patch summary', f'{self.mode}_fig2',
                         args={'ks': ['tur_tr', 'tur_N_mu', 'pau_tr', 'cum_d', 'f_am', 'on_food_tr']})
        ]

        super().__init__(N=N,experiment='DoublePatch', enrichment=enrichment,graph_entries=graph_entries,
                         collections=['pose', 'toucher', 'feeder', 'olfactor'], **kwargs)
        self.arena_dims = arena_dims
        self.patch_x = patch_x
        self.patch_radius = patch_radius
        self.substrates = substrates
        self.dur = dur



        self.ms=[reg.MD.loadConf(mID) for mID in self.mIDs]
        self.exp_dict = self.time_ratio_exp()

        self.mdiff_df, row_colors = reg.MD.diff_df(mIDs=self.mID0s,ms=self.ms)
        self.analysis_kws= {
            # 'datasets': self.datasets,
            # 'save_to': self.plot_dir,
            # 'show': self.show,
            'title': f"DOUBLE PATCH ESSAY (N={self.N}, duration={self.dur}')",
            'mdiff_df': self.mdiff_df
        }




    def get_larvagroups(self,age=120.0):


        kws0 = {
            'kwdic': {
                'distribution': {'N': self.N, 'scale': (0.005, 0.005)},
                'life_history': {'age': age, 'epochs': reg.GT.dict.epoch.entry(0, start=0.0, stop=age)},
                'odor': {}
            },
            'sample': 'None.150controls',
        }
        return reg.lgs(ids=['rover', 'sitter'],mIDs=self.mIDs, cs=['blue', 'red'],expand=True, **kws0)



    def get_sources(self, type='standard', q=1.0, Cpeak=2.0, Cscale=0.0002):

        kws0 = {'radius': self.patch_radius, 'default_color': 'green', 'amount': 0.1,
                'type': type, 'quality': q, 'group': 'Patch',
                'odor': {'odor_id': 'Odor', 'odor_intensity': Cpeak, 'odor_spread': Cscale}

                }

        return dNl.NestDict({
            'Left_patch': reg.CT.dict.Source.gConf(pos=(-self.patch_x, 0.0), **kws0),
            'Right_patch': reg.CT.dict.Source.gConf(pos=(self.patch_x, 0.0), **kws0),

        })



    def patch_env(self, type='standard', q=1.0, o='G'):
        if o == 'G':
            odorscape = {'odorscape': 'Gaussian'}
            Cpeak, Cscale = 2.0, 0.0002
        else:
            raise

        kws = {'kwdic': {
            'arena': {'arena_dims': self.arena_dims, 'arena_shape': 'rectangular'},
            'food_params': {'source_units': self.get_sources(type=type, q=q, Cpeak=Cpeak, Cscale=Cscale),
                            'source_groups': {}, 'food_grid': None},
        }, 'odorscape': odorscape, 'border_list': {}, 'windscape': None, 'thermoscape': None,

        }

        return reg.CT.dict.Env.gConf(**kws)



    def time_ratio_exp(self):
        confs = {}
        for n in self.substrates:




            conf=self.get_exp_conf(env=self.patch_env(type=n), lgs=self.get_larvagroups(),
                              id=f'{self.experiment}_{n}_{self.dur}min')

            confs[n] = [conf]
        return dNl.NestDict(confs)




    #
    # def global_anal(self):
    #     kwargs = {
    #         'datasets': self.datasets,
    #         'save_to': self.plot_dir,
    #         'show': self.show,
    #         'title': f"DOUBLE PATCH ESSAY (N={self.N}, duration={self.dur}')",
    #         'mdiff_df': self.mdiff_df
    #     }
    #     # entry = self.G.entry('double-patch summary', args={})
    #     self.figs.update(self.G.eval0(entry=self.G.entry('double-patch summary',title='fig1',
    #                                                      args={'name':f'{self.mode}_fig1','ks':None}), **kwargs))
    #     self.figs.update(self.G.eval0(entry=self.G.entry('double-patch summary',title='fig2',args={'name':f'{self.mode}_fig2',
    #                                                            'ks':['tur_tr', 'tur_N_mu', 'pau_tr','cum_d', 'f_am', 'on_food_tr']}), **kwargs))



def RvsSx4():

    # from lib.sim.essay.essay_run import DoublePatch_Essay
    sufs=['foragers', 'navigators','feeders', 'locomotors']
    i=0
    for o in [True,False]:
        for f in [True,False]:
            E = DoublePatch_Essay(video=False, N=5, dur=3, olfactor=o,feeder=f,
                         id=f'Essay_DoublePatch_{sufs[i]}')
        #     # print(E.patch_env())
            ds = E.run()
            #figs, results = E.anal()
            i+=1