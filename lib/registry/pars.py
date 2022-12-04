import numpy as np
import lib.aux.dictsNlists as dNl

from lib.registry import reg


class ParRegistry:
    def __init__(self,verbose=1):
        self.verbose = verbose



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
    def dict(self):
        return self.par_dict.kdict

    def getPar(self, k=None, p=None, d=None, to_return='d'):
        return self.par_dict.getPar(k=k, d=d, p=p, to_return=to_return)

    def get_null(self, name, **kwargs):
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


    def loadConf(self, conftype, id=None):
        return self.conftype_dict.dict[conftype].loadConf(id=id)

    def saveConf(self, conftype, id, conf):
       # reg.conf[conftype][id]=conf
        return self.conftype_dict.dict[conftype].saveConf(id=id, conf=conf)

    def deleteConf(self, conftype, id=None):
        return self.conftype_dict.dict[conftype].deleteConf(id=id)

    def expandConf(self, conftype, id=None):
        return self.conftype_dict.dict[conftype].expandConf(id=id)




    def storedRefIDs(self):
        dic = dNl.load_dict(self.paths['Ref'], use_pickle=False)
        return list(dic.keys())




    def lg(self, **kwargs):
        return reg.GT.dict.LarvaGroup.lg_entry(**kwargs)




    def storedConf(self, conftype):
        return self.conftype_dict.dict[conftype].ConfIDs

preg = ParRegistry()




