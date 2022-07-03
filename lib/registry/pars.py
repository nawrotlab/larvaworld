import copy

import numpy as np
import pandas as pd

from lib.registry.par import v_descriptor
from lib.aux import dictsNlists as dNl


class ParRegistry:
    def __init__(self, mode='build', object=None, save=True, load_funcs=False):
        from lib.registry import paths,init_pars,par_funcs,parser_dict,dist_dict,parConfs,par_dict

        self.path_dict = paths.build_path_dict()
        self.init_dict = init_pars.ParInitDict().dict
        # self.mfunc = par_funcs.module_func_dict()
        self.parser_dict = parser_dict.ParserDict(init_dict=self.init_dict).dict
        self.dist_dict0 = dist_dict.DistDict()
        self.dist_dict = self.dist_dict0.dict
        self.larva_conf_dict = parConfs.LarvaConfDict(init_dict=self.init_dict, dist_dict0=self.dist_dict0)
        self.larva_conf_dict2 = parConfs.LarvaConfDict2(dist_dict0=self.dist_dict0)





        if load_funcs:
            self.func_dict = dNl.load_dict(self.path_dict['ParFuncDict'])
        else:
            self.func_dict = par_funcs.build_func_dict()
            dNl.save_dict(self.func_dict, self.path_dict['ParFuncDict'])

        if mode == 'load':
            self.dict = self.load()
        elif mode == 'build':
            self.dict_entries = par_dict.BaseParDict(func_dict=self.func_dict).dict_entries
            self.dict = self.finalize_dict(self.dict_entries)
            self.ddict = dNl.NestDict({p.d: p for k, p in self.dict.items()})
            self.pdict = dNl.NestDict({p.p: p for k, p in self.dict.items()})
            if save:
                self.save()

    @ property
    def graph_dict(self):
        from lib.plot.dict import graph_dict
        return graph_dict

    def finalize_dict(self, entries):
        dic = dNl.NestDict()
        for prepar in entries:
            p = v_descriptor(**prepar)
            dic[p.k] = p
        return dic

    def save(self):

        df = pd.DataFrame.from_records(self.dict_entries, index='k')
        df.to_csv(self.path_dict['ParDf'])

    def load(self):
        # FIXME Not working
        df = pd.read_csv(self.path_dict['ParDf'], index_col=0)
        entries = df.to_dict(orient='records')
        dict = self.finalize_dict(entries)
        return dict

    def get(self, k, d, compute=True):
        p = self.dict[k]
        res = p.exists(d)

        if res['step']:
            if hasattr(d, 'step_data'):
                return d.step_data[p.d]
            else:
                return d.read(key='step')[p.d]
        elif res['end']:
            if hasattr(d, 'endpoint_data'):
                return d.endpoint_data[p.d]
            else:
                return d.read(key='end', file='endpoint_h5')[p.d]
        else:
            for key in res.keys():
                if key not in ['step', 'end'] and res[key]:
                    return d.read(key=f'{key}.{p.d}', file='aux_h5')

        if compute:
            self.compute(k, d)
            return self.get(k, d, compute=False)
        else:
            print(f'Parameter {p.disp} not found')

    def compute(self, k, d):
        p = self.dict[k]
        res = p.exists(d)
        if not any(list(res.values())):
            k0s = p.required_ks
            for k0 in k0s:
                self.compute(k0, d)
            p.compute(d)

    def runtime_pars(self):
        return [v.d for k, v in self.dict.items()]

    def auto_load(self, ks, datasets):
        dic = {}
        for k in ks:
            dic[k] = {}
            for d in datasets:
                vs = self.get(k=k, d=d, compute=True)
                dic[k][d.id] = vs
        return dNl.NestDict(dic)

    def getPar(self, k=None, p=None, d=None, to_return='d'):
        if k is not None:
            d0 = self.dict
            k0 = k
        elif d is not None:
            d0 = self.ddict
            k0 = d
        elif p is not None:
            d0 = self.pdict
            k0 = p

        if type(k0) == str:
            par = d0[k0]
            if type(to_return) == list:
                return [getattr(par, i) for i in to_return]
            elif type(to_return) == str:
                return getattr(par, to_return)
        elif type(k0) == list:
            pars = [d0[i] for i in k0]
            if type(to_return) == list:
                return [[getattr(par, i) for par in pars] for i in to_return]
            elif type(to_return) == str:
                return [getattr(par, to_return) for par in pars]

    def get_null(self, name, key='v', **kwargs):
        def v0(d):
            if d is None:
                return None

            null = dNl.NestDict()
            for k, v in d.items():
                if not isinstance(v, dict):
                    null[k] = v
                elif 'k' in v.keys() or 'h' in v.keys() or 't' in v.keys():
                    null[k] = None if key not in v.keys() else v[key]
                else:
                    null[k] = v0(v)
            return null

        if key != 'v':
            raise
        dic2 = v0(self.init_dict[name])
        if name not in ['visualization', 'enrichment']:
            dic2.update(kwargs)
            return dNl.NestDict(dic2)
        else:
            for k, v in dic2.items():
                if k in list(kwargs.keys()):
                    dic2[k] = kwargs[k]
                elif isinstance(v, dict):
                    for k0, v0 in v.items():
                        if k0 in list(kwargs.keys()):
                            dic2[k][k0] = kwargs[k0]
            return dNl.NestDict(dic2)

    def odor(self, i, s, id='Odor'):
        return self.get_null('odor', odor_id=id, odor_intensity=i, odor_spread=s)

    def oG(self, c=1, id='Odor'):
        return self.odor(i=2.0 * c, s=0.0002 * np.sqrt(c), id=id)

    def oD(self, c=1, id='Odor'):
        return self.odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)

    def arena(self, x, y=None):
        if y is None:
            return self.get_null('arena', arena_shape='circular', arena_dims=(x, x))
        else:
            return self.get_null('arena', arena_shape='rectangular', arena_dims=(x, y))

    def enr_dict(self, proc=[], bouts=[], to_keep=[], pre_kws={}, fits=True, on_food=False, def_kws={},
                 metric_definition=None,**kwargs):
        from lib.registry.init_pars import proc_type_keys, bout_keys, to_drop_keys

        if metric_definition is None:
            from lib.conf.stored.data_conf import metric_def
            metric_definition = metric_def(**def_kws)
        pre = self.get_null('preprocessing', **pre_kws)
        proc = self.get_null('processing', **{k: True if k in proc else False for k in proc_type_keys})
        annot = self.get_null('annotation', **{k: True if k in bouts else False for k in bout_keys}, fits=fits,
                              on_food=on_food)
        to_drop = self.get_null('to_drop', **{k: True if k not in to_keep else False for k in to_drop_keys})
        dic = self.get_null('enrichment', metric_definition=metric_definition, preprocessing=pre, processing=proc,
                            annotation=annot,
                            to_drop=to_drop, **kwargs)
        return dic

    def base_enrich(self, **kwargs):
        return self.enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
                        bouts=['stride', 'pause', 'turn'],
                        to_keep=['midline', 'contour'], **kwargs)

    def newConf(self, conftype, id=None,kwargs={}, id0=None):
        d={
            'Tracker' : 'tracker',
            'Model' : 'larva_conf',
            'Exp' : 'exp_conf',
            'Env' : 'env_conf',
            'Essay' : 'essay_conf',
            'Ga' : 'ga_conf',
        }
        k0=d[conftype]


        if id0 is None :
            k0 = f'{conftype.lower()}_conf'
            T0 = preg.get_null(k0)
        else :
            from lib.conf.stored.conf import expandConf
            T0 = dNl.NestDict(copy.deepcopy(expandConf(id0, conftype)))
        T = dNl.update_nestdict(T0, kwargs)
        if id is not None:
            self.saveConf(T, conftype, id)
        return T

    def saveConf(self, conf, conftype, id=None,**kwargs):
        if id is not None :
            from lib.conf.stored.conf import saveConf
            saveConf(conf, conftype, id,**kwargs)

    def loadConf(self, conftype, id=None,**kwargs):
        if id is not None :
            from lib.conf.stored.conf import loadConf
            return loadConf(id, conftype,**kwargs)

    def loadRef(self, id=None):
        if id is not None :
            from lib.conf.stored.conf import loadRef
            return loadRef(id)

    def deleteConf(self, conftype, id=None,**kwargs):
        if id is not None:
            from lib.conf.stored.conf import deleteConf
            return deleteConf(id, conftype,**kwargs)

    def storedConf(self, conftype,**kwargs):
        from lib.conf.stored.conf import kConfDict
        return kConfDict(conftype,**kwargs)

    # def new_tracker_format(self, id=None, kwargs={}):
    #     T0 = preg.get_null('tracker')
    #     T=dNl.update_nestdict(T0, kwargs)
    #     if id is not None:
    #         self.saveConf(T,'Tracker', id)


preg = ParRegistry()




