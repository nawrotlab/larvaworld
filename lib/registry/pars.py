import os
import shutil
import time

import numpy as np
import lib.aux.dictsNlists as dNl

from lib.aux.stor_aux import read
from lib.registry import reg


class ParRegistry:
    def __init__(self,verbose=1):
        self.verbose = verbose

    def vprint(self, text, verbose=0):
        if verbose >= self.verbose:
            print(text)



    @property
    def conftype_dict(self):
        return reg.CT

    @property
    def path_dict(self):
        return reg.Path

    @property
    def paths(self):
        return reg.Path

    @property
    def datapath(self):
        return reg.datapath

    @property
    def larva_conf_dict(self):
        return reg.MD

    @property
    def init_dict(self):
        return reg.PI

    @property
    def output_dict(self):
        from lib.registry.output import output_dict
        return output_dict

    @property
    def dist_dict(self):
        return reg.DD

    @property
    def graph_dict(self):
        return reg.GD

    @property
    def parser_dict(self):

        return reg.ParsD

    @property
    def par_dict(self):
        return reg.PD

    @property
    def proc_func_dict(self):
        return reg.PF

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
        return reg.get_null(name=name, **kwargs)

    def oG(self, c=1, id='Odor'):
        return reg.get_null('odor', odor_id=id, odor_intensity=2.0 * c, odor_spread=0.0002 * np.sqrt(c))

    def oD(self, c=1, id='Odor'):
        return reg.get_null('odor', odor_id=id, odor_intensity=300.0 * c, odor_spread=0.1 * np.sqrt(c))
        # return self.odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)

    def arena(self, x, y=None):
        if y is None:
            return reg.get_null('arena', arena_shape='circular', arena_dims=(x, x))
        else:
            return reg.get_null('arena', arena_shape='rectangular', arena_dims=(x, y))



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
        kws={k:reg.get_null(k,**v) for k,v in kw_dic0.items()}

        if metric_definition is None:
            metric_definition = self.init_dict.metric_def(**def_kws)
        dic = reg.get_null('enrichment',
                                      metric_definition=metric_definition, **kws, **kwargs)
        return dic

    def base_enrich(self, **kwargs):
        return self.enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
                             bouts=['stride', 'pause', 'turn'],
                             to_keep=['midline', 'contour'], **kwargs)


    def loadConf(self, conftype, id=None):
        # if id in self.conftype_dict.dict[conftype].keys():
        #     return self.conftype_dict.dict[conftype].loadConf(id=id)
        #     return reg.conf[conftype][id]
        # else :
        #     return None
        return self.conftype_dict.dict[conftype].loadConf(id=id)

    def saveConf(self, conftype, id, conf):
       # reg.conf[conftype][id]=conf
        return self.conftype_dict.dict[conftype].saveConf(id=id, conf=conf)

    def deleteConf(self, conftype, id=None):
        return self.conftype_dict.dict[conftype].deleteConf(id=id)

    def expandConf(self, conftype, id=None):
        return self.conftype_dict.dict[conftype].expandConf(id=id)



    def simRef(self,id, mID, **kwargs):
        from lib.aux.sample_aux import sim_model
        return sim_model(mID,  refID=id, **kwargs)



    def loadRefD(self, id, **kwargs):
        return self.loadRef(id, load=True, **kwargs)


    def loadRefDs(self, ids, **kwargs):
        ds = [self.loadRefD(id, **kwargs) for id in ids]
        return ds

    def deleteRef(self,id):
        path = self.paths['Ref']
        dic = dNl.load_dict(path, use_pickle=False)
        if id in dic.keys():
            shutil.rmtree(dic[id].dir,ignore_errors=True)
            dic.pop(id,None)
            self.vprint(f'Deleted Ref Configuration {id}')
            dNl.save_dict(dic, path, use_pickle=False)
    def loadRef(self, id, load=False, **kwargs):
        config = self.retrieveRef(id)
        if config is not None :
            from lib.stor.larva_dataset import LarvaDataset
            d = LarvaDataset(config['dir'], load_data=False)
            if not load:
                self.vprint(f'Loaded stored reference configuration : {id}')
                return d
            else:
                d.load(**kwargs)
                self.vprint(f'Loaded stored reference dataset : {id}')
                return d

        else:
            # self.vprint(f'Ref Configuration {id} does not exist. Returning None')
            return None
    def retrieveRef(self,id):
        dic = dNl.load_dict(self.paths['Ref'], use_pickle=False)
        if id in dic.keys():
            return dic[id]
        else:
            self.vprint(f'Ref Configuration {id} does not exist. Returning None')
            return None


    def storedRefIDs(self):
        dic = dNl.load_dict(self.paths['Ref'], use_pickle=False)
        return list(dic.keys())




    #@ load_timer
    def readRef(self, id, key='end',**kwargs):
        config = self.retrieveRef(id)
        if config is not None:
            D=config.dir_dict
            return read(D[key], key=key,**kwargs)
        else:
            return None

    def testRef(self, id):
        config = self.retrieveRef(id)
        if config is not None:
            D = config.dir_dict
            dic={}
            for k, d in D.items():
                if d.endswith('.h5') and os.path.exists(d):
                    try :
                        t0=time.time()
                        read(d, key=k)
                        dic[k]=np.round(time.time()-t0,2)
                    except :
                        dic[k]='FAIL'
            # if k not in D1.keys() :
            print(f'------- Loading times for {id}---------------------')
            print(dic)
            print()

    def lg(self, **kwargs):
        return reg.GT.dict.LarvaGroup.lg_entry(**kwargs)

    # def loadRef(self, id=None):
    #     from lib.conf.stored
    #     return self.conftype_dict.loadRef(id=id)


    def storedConf(self, conftype):
        return self.conftype_dict.dict[conftype].ConfIDs





preg = ParRegistry()


