import numpy as np





class ParRegistry:
    def __init__(self,verbose=1):
        self.verbose = verbose

    def vprint(self, text, verbose=0):
        if verbose >= self.verbose:
            print(text)

    @property
    def grouptype_dict(self):
        from lib.registry.order import GT
        return GT

    @property
    def conftype_dict(self):
        from lib.registry.order import CT
        return CT

    @property
    def path_dict(self):
        from lib.registry.paths import PathD
        return PathD

    @property
    def paths(self):
        from lib.registry.paths import PathD
        return PathD

    @property
    def larva_conf_dict(self):
        from lib.registry.order import MD
        return MD

    @property
    def init_dict(self):
        from lib.registry.order import PI
        return PI

    @property
    def output_dict(self):
        from lib.registry.output import output_dict
        return output_dict

    @property
    def dist_dict(self):
        from lib.registry.order import DD
        return DD

    @property
    def graph_dict(self):
        from lib.registry.order import GD
        return GD

    @property
    def parser_dict(self):
        from lib.registry.parser_dict import ParsD
        return ParsD

    @property
    def par_dict(self):
        from lib.registry.order import PD
        return PD

    @property
    def proc_func_dict(self):
        from lib.registry.order import ProcF
        return ProcF

    @property
    def dict(self):
        return self.par_dict.kdict

    def getPar(self, k=None, p=None, d=None, to_return='d'):
        return self.par_dict.getPar(k=k, d=d, p=p, to_return=to_return)

    def get_null(self, name, **kwargs):
        # print(self.get_null('intermitter'))
        # # print(kwargs)
        # print()
        # raise
        return self.init_dict.get_null(name=name, **kwargs)

    def oG(self, c=1, id='Odor'):
        return self.init_dict.get_null('odor', odor_id=id, odor_intensity=2.0 * c, odor_spread=0.0002 * np.sqrt(c))

    def oD(self, c=1, id='Odor'):
        return self.init_dict.get_null('odor', odor_id=id, odor_intensity=300.0 * c, odor_spread=0.1 * np.sqrt(c))
        # return self.odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)

    def arena(self, x, y=None):
        if y is None:
            return self.init_dict.get_null('arena', arena_shape='circular', arena_dims=(x, x))
        else:
            return self.init_dict.get_null('arena', arena_shape='rectangular', arena_dims=(x, y))



    def enr_dict(self, proc=[], bouts=[], to_keep=[], pre_kws={}, fits=True, on_food=False, interference=True,
                 def_kws={}, metric_definition=None, **kwargs):
        to_drop_keys = ['midline', 'contour', 'stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn', 'turn',
                        'unused']
        proc_type_keys = ['angular', 'spatial', 'source', 'dispersion', 'tortuosity', 'PI', 'wind']

        kw_dic0={
            'preprocessing' : pre_kws,
            'processing' : {k: True if k in proc else False for k in proc_type_keys},
            'annotation' : {**{k: True if k in bouts else False for k in ['stride', 'pause', 'turn']},
                               **{'fits': fits, 'on_food': on_food,'interference': interference}},
            'to_drop' : {k: True if k not in to_keep else False for k in to_drop_keys},
                }
        kws={k:self.init_dict.get_null(k,**v) for k,v in kw_dic0.items()}

        if metric_definition is None:
            metric_definition = self.init_dict.metric_def(**def_kws)
        dic = self.init_dict.get_null('enrichment',
                                      metric_definition=metric_definition, **kws, **kwargs)
        return dic

    def base_enrich(self, **kwargs):
        return self.enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
                             bouts=['stride', 'pause', 'turn'],
                             to_keep=['midline', 'contour'], **kwargs)


    def loadConf(self, conftype, id=None):
        return self.conftype_dict.dict[conftype].loadConf(id=id)

    def saveConf(self, conftype, id, conf):
        return self.conftype_dict.dict[conftype].saveConf(id=id, conf=conf)

    def deleteConf(self, conftype, id=None):
        return self.conftype_dict.dict[conftype].deleteConf(id=id)

    def expandConf(self, conftype, id=None):
        return self.conftype_dict.dict[conftype].expandConf(id=id)

    def loadRef(self, id, verbose=1):
        self.paths['Ref']


        from lib.stor.larva_dataset import LarvaDataset
        d = LarvaDataset(loadConf(id, 'Ref')['dir'], load_data=False)
        if verbose >= 1:
            print(f'Loaded stored reference dataset : {id}')
        return d


    # def loadRef(self, id=None):
    #     from lib.conf.stored
    #     return self.conftype_dict.loadRef(id=id)


    def storedConf(self, conftype):
        return self.conftype_dict.dict[conftype].ConfIDs


    def next_idx(self, id, conftype='Exp'):
        return self.conftype_dict.next_idx(conftype=conftype, id=id)



preg = ParRegistry()



# enrichment=preg.enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
#                                                           bouts=['stride', 'pause', 'turn'])

if __name__ == '__main__':
    group_id = 'AttP240-Fed'
    datagroup_id = 'Jovanic lab'
    idx = 1
    g = preg.loadConf(id=datagroup_id, conftype='Group')
    save_to = f'{g.path}/plots/{group_id}/trial{idx}'


    refIDs0 = ['Coaster.AttP240.Fed', 'Rehydration/AttP240.Fed', 'AttP240.Fed']
    dIDs = ['Coaster', 'Rehydration', 'Control']

    load_kws = {
        # 'step':True,
        'step': False,
        # 'end':False,
        'end': True,
        'h5_ks': ['epochs', 'angular', 'dspNtor'],
        #
    }

    ds = preg.conftype_dict.loadRefDs(refIDs0, **load_kws)

    ggs = ['traj']
    # ggs=['traj','endpoint']
    # ggs = ['track', 'general', 'dsp', 'traj']
    gd = preg.graph_dict.eval_graphgroups(graphgroups=ggs, datasets=ds, labels=dIDs,save_to=save_to)

