import copy
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import param

from lib.registry.par import v_descriptor
from lib.aux import dictsNlists as dNl








class ParRegistry:
    def __init__(self):
        pass

    @property
    def conftype_dict(self):
        from lib.registry.conftypes import conftype_dict
        return conftype_dict

    @property
    def path_dict(self):
        from lib.registry.paths import path_dict
        return path_dict

    @property
    def larva_conf_dict(self):
        from lib.registry.parConfs import larva_conf_dict
        return larva_conf_dict

    @property
    def init_dict(self):
        from lib.registry.init_pars import init_dict
        return init_dict

    @property
    def output_dict(self):
        from lib.registry.output import output_dict
        return output_dict

    @property
    def dist_dict(self):
        from lib.registry.dist_dict import dist_dict
        return dist_dict

    @property
    def graph_dict(self):
        from lib.plot.dict import graph_dict
        return graph_dict

    @property
    def parser_dict(self):
        from lib.registry.parser_dict import parser_dict
        return parser_dict

    @property
    def par_dict(self):
        from lib.registry.par_dict import basepar_dict
        return basepar_dict

    @property
    def dict(self):
        return self.par_dict.kdict

    def getPar(self, k=None, p=None, d=None, to_return='d'):
        return self.par_dict.getPar(k=k, d=d, p=p, to_return=to_return)

    def get_null(self, name, **kwargs):
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

        if metric_definition is None:
            from lib.conf.stored.data_conf import metric_def
            metric_definition = metric_def(**def_kws)
        pre = self.init_dict.get_null('preprocessing', **pre_kws)
        proc = self.init_dict.get_null('processing', **{k: True if k in proc else False for k in proc_type_keys})
        annot = self.init_dict.get_null('annotation', **{k: True if k in bouts else False for k in ['stride', 'pause', 'turn']},
                              fits=fits,
                              on_food=on_food, interference=interference)
        to_drop = self.init_dict.get_null('to_drop', **{k: True if k not in to_keep else False for k in to_drop_keys})
        dic = self.init_dict.get_null('enrichment', metric_definition=metric_definition, preprocessing=pre, processing=proc,
                            annotation=annot,
                            to_drop=to_drop, **kwargs)
        return dic

    def base_enrich(self, **kwargs):
        return self.enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
                             bouts=['stride', 'pause', 'turn'],
                             to_keep=['midline', 'contour'], **kwargs)


    def loadConf(self, conftype, id=None):
        return self.conftype_dict.loadConf(conftype=conftype,id=id)



    def expandConf(self, conftype, id=None):
        return self.conftype_dict.expandConf(conftype=conftype, id=id)


    def loadRef(self, id=None):
        return self.conftype_dict.loadRef(id=id)


    def storedConf(self, conftype):
        return self.conftype_dict.storedConf(conftype=conftype)


    def next_idx(self, id, conftype='Exp'):
        F0 = self.path_dict["SimIdx"]
        try:
            with open(F0) as f:
                d = json.load(f)
        except:
            ksExp = self.storedConf('Exp')
            ksBatch = self.storedConf('Batch')
            ksEssay = self.storedConf('Essay')
            ksGA = self.storedConf('Ga')
            ksEval = self.storedConf('Exp')
            dExp = dict(zip(ksExp, [0] * len(ksExp)))
            dBatch = dict(zip(ksBatch, [0] * len(ksBatch)))
            dEssay = dict(zip(ksEssay, [0] * len(ksEssay)))
            dGA = dict(zip(ksGA, [0] * len(ksGA)))
            dEval = dict(zip(ksEval, [0] * len(ksEval)))
            # batch_idx_dict.update(loadConfDict('Batch'))
            d = {'Exp': dExp,
                 'Batch': dBatch,
                 'Essay': dEssay,
                 'Eval': dEval,
                 'Ga': dGA}
        if not conftype in d.keys():
            d[conftype] = {}
        if not id in d[conftype].keys():
            d[conftype][id] = 0
        d[conftype][id] += 1
        with open(F0, "w") as fp:
            json.dump(d, fp)
        return d[conftype][id]


preg = ParRegistry()
