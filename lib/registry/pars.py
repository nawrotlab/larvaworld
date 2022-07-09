import copy
import json
import pickle
import shutil

import numpy as np
import pandas as pd
import param

from lib.registry.par import v_descriptor
from lib.aux import dictsNlists as dNl


def store_reference_data_confs():
    from lib.stor.larva_dataset import LarvaDataset
    DATA = preg.path_dict["DATA"]
    dds = [
        [f'{DATA}/JovanicGroup/processed/3_conditions/AttP{g}@UAS_TNT/{c}' for g
         in ['2', '240']] for c in ['Fed', 'Deprived', 'Starved']]
    dds = dNl.flatten_list(dds)
    dds.append(f'{DATA}/SchleyerGroup/processed/FRUvsQUI/Naive->PUR/EM/exploration')
    dds.append(f'{DATA}/SchleyerGroup/processed/no_odor/200_controls')
    dds.append(f'{DATA}/SchleyerGroup/processed/no_odor/10_controls')
    for dr in dds:
        try:
            d = LarvaDataset(dr, load_data=False)
            d.save_config(add_reference=True)
        except:
            pass


class ParRegistry:
    def __init__(self, mode='build', object=None, save=True, load_funcs=False):
        from lib.registry import paths, init_pars, par_funcs, parser_dict, dist_dict, parConfs, par_dict, output
        self.conftypes = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial',
                          'Life', 'Body']
        self.path_dict = paths.build_path_dict()
        self.confID_dict = self.build_confID_dict()
        self.init_dict = init_pars.ParInitDict().dict
        # self.mfunc = par_funcs.module_func_dict()
        self.parser_dict = parser_dict.ParserDict(init_dict=self.init_dict).dict
        self.dist_dict0 = dist_dict.DistDict()
        self.dist_dict = self.dist_dict0.dict
        self.output_dict = output.output_dict
        self.larva_conf_dict = parConfs.LarvaConfDict(dist_dict0=self.dist_dict0)

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

    @property
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

    def oG(self, c=1, id='Odor'):
        return self.get_null('odor', odor_id=id, odor_intensity=2.0 * c, odor_spread=0.0002 * np.sqrt(c))
        # return self.odor(i=2.0 * c, s=0.0002 * np.sqrt(c), id=id)

    def oD(self, c=1, id='Odor'):
        return self.get_null('odor', odor_id=id, odor_intensity=300.0 * c, odor_spread=0.1 * np.sqrt(c))
        # return self.odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)

    def arena(self, x, y=None):
        if y is None:
            return self.get_null('arena', arena_shape='circular', arena_dims=(x, x))
        else:
            return self.get_null('arena', arena_shape='rectangular', arena_dims=(x, y))

    def enr_dict(self, proc=[], bouts=[], to_keep=[], pre_kws={}, fits=True, on_food=False,interference=True,  def_kws={},
                 metric_definition=None, **kwargs):
        to_drop_keys = ['midline', 'contour', 'stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn', 'turn',
                        'unused']
        proc_type_keys = ['angular', 'spatial', 'source', 'dispersion', 'tortuosity', 'PI', 'wind']

        if metric_definition is None:
            from lib.conf.stored.data_conf import metric_def
            metric_definition = metric_def(**def_kws)
        pre = self.get_null('preprocessing', **pre_kws)
        proc = self.get_null('processing', **{k: True if k in proc else False for k in proc_type_keys})
        annot = self.get_null('annotation', **{k: True if k in bouts else False for k in ['stride', 'pause', 'turn']}, fits=fits,
                              on_food=on_food,interference=interference)
        to_drop = self.get_null('to_drop', **{k: True if k not in to_keep else False for k in to_drop_keys})
        dic = self.get_null('enrichment', metric_definition=metric_definition, preprocessing=pre, processing=proc,
                            annotation=annot,
                            to_drop=to_drop, **kwargs)
        return dic

    def base_enrich(self, **kwargs):
        return self.enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
                             bouts=['stride', 'pause', 'turn'],
                             to_keep=['midline', 'contour'], **kwargs)

    def newConf(self, conftype, id=None, kwargs={}, id0=None):
        d = {
            'Tracker': 'tracker',
            'Model': 'larva_conf',
            'Exp': 'exp_conf',
            'Env': 'env_conf',
            'Essay': 'essay_conf',
            'Ga': 'ga_conf',
        }
        k0 = d[conftype]

        if id0 is None:
            k0 = f'{conftype.lower()}_conf'
            T0 = preg.get_null(k0)
        else:
            from lib.conf.stored.conf import expandConf
            T0 = dNl.NestDict(copy.deepcopy(expandConf(id0, conftype)))
        T = dNl.update_nestdict(T0, kwargs)
        if id is not None:
            self.saveConf(conf=T, conftype=conftype, id=id)
        return T

    def saveConf(self, conf, conftype, id=None, mode='overwrite', verbose=1):
        if id is not None:
            try:
                d = self.loadConfDict(conftype)
            except:
                d = {}
            if id is None:
                id = conf['id']

            if id in list(d.keys()):
                for k, v in conf.items():
                    if type(k) == dict and k in list(d[id].keys()) and mode == 'update':
                        d[id][k].update(conf[k])
                    else:
                        d[id][k] = v
            else:
                d[id] = conf
            self.saveConfDict(d, conftype=conftype)
            if verbose >= 1:
                print(f'{conftype} Configuration saved under the id : {id}')

    def loadConf(self, conftype, id=None):
        if id is not None:
            try:
                conf_dict = self.loadConfDict(conftype)
                conf = conf_dict[id]
                return dNl.NestDict(conf)
            except:
                raise ValueError(f'{conftype} Configuration {id} does not exist')

    def saveConfDict(self, ConfDict, conftype, use_pickle=False):
        path = self.path_dict[conftype]
        if conftype == 'Ga':
            use_pickle = True
        if use_pickle:
            with open(path, 'wb') as fp:
                pickle.dump(ConfDict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, "w") as f:
                json.dump(ConfDict, f)

    def loadConfDict(self, conftype, use_pickle=False):
        path = self.path_dict[conftype]
        if conftype == 'Ga':
            use_pickle = True
        try:
            if use_pickle:
                with open(path, 'rb') as tfp:
                    d = pickle.load(tfp)
            else:
                with open(path) as f:
                    d = json.load(f)
        except:
            d = {}
        return dNl.NestDict(d)

    def expandConf(self, conftype, id=None):
        if id is not None:
            conf = self.loadConf(id=id, conftype=conftype)
            try:
                if conftype == 'Batch':
                    conf.exp = self.expandConf(id=conf.exp, conftype='Exp')
                elif conftype == 'Exp':
                    conf.experiment = id
                    conf.env_params = self.expandConf(id=conf.env_params, conftype='Env')
                    conf.trials = self.loadConf(id=conf.trials, conftype='Trial')
                    for k, v in conf.larva_groups.items():
                        if type(v.model) == str:
                            v.model = self.loadConf(id=v.model, conftype='Model')
                elif conftype == 'Ga':
                    conf.experiment = id
                    conf.env_params = self.expandConf(id=conf.env_params, conftype='Env')
            except:
                pass
            return conf

    def loadRef(self, id=None):
        if id is not None:
            from lib.stor.larva_dataset import LarvaDataset
            return LarvaDataset(self.loadConf(id=id, conftype='Ref')['dir'], load_data=False)

    def deleteConf(self, conftype, id=None):
        if id is not None:
            if conftype == 'Data':
                DataGroup = self.loadConf(id=id, conftype=conftype)
                path = DataGroup['path']
                try:
                    shutil.rmtree(path)
                except:
                    pass
            d = self.loadConfDict(conftype=conftype)
            try:
                d.pop(id, None)
                self.saveConfDict(d, conftype=conftype)
                print(f'Deleted {conftype} configuration under the id : {id}')
            except:
                pass

    def storedConf(self, conftype):
        return list(self.loadConfDict(conftype=conftype).keys())

    def conf_selector_func(self, conftype, default=None, single_choice=True, **kwargs):
        def func():

            kws = {
                'objects': self.storedConf(conftype),
                'default': default,
                'allow_None': True,
                'empty_default': True,
            }
            if single_choice:
                func0 = param.Selector
            else:
                func0 = param.ListSelector
            return func0(**kws, **kwargs)

        return func

    def build_confID_dict(self):
        from lib.aux.par_aux import sub
        from lib.registry.par_dict import preparePar
        d = {}
        for conftype in self.conftypes:
            low = conftype.lower()
            k = f'{low}ID'
            dic = {'dtype': str, 'vparfunc': self.conf_selector_func(conftype), 'vs': self.storedConf(conftype),
                   'sym': sub('ID', low), 'k': k, 'h': f'The {conftype} configuration ID', 'p': conftype,
                   'disp': f'{conftype} ID'}
            prepar = preparePar(**dic)
            p = v_descriptor(**prepar)
            d[p.p] = p
        return dNl.NestDict(d)

    def storeConfs(self, conftypes=None):
        if conftypes is None:
            conftypes = self.conftypes

        for conftype in conftypes:
            if conftype == 'Ref':
                store_reference_data_confs()
                continue
            elif conftype == 'Trial':
                from lib.conf.stored.aux_conf import trial_dict as d
            elif conftype == 'Life':
                from lib.conf.stored.aux_conf import life_dict as d
            elif conftype == 'Body':
                from lib.conf.stored.aux_conf import body_dict as d
            elif conftype == 'Tracker':
                from lib.conf.stored.data_conf import tracker_formats as d
            elif conftype == 'Group':
                from lib.conf.stored.data_conf import importformats as d
            elif conftype == 'Model':
                self.larva_conf_dict.baseConfs()
                from lib.conf.stored.larva_conf import mod_dict as d
            elif conftype == 'ModelGroup':
                from lib.conf.stored.larva_conf import mod_group_dict as d
            elif conftype == 'Env':
                from lib.conf.stored.env_conf import env_dict as d
            elif conftype == 'Exp':
                from lib.conf.stored.exp_conf import exp_dict as d
            elif conftype == 'ExpGroup':
                from lib.conf.stored.exp_conf import exp_group_dict as d
            elif conftype == 'Essay':
                from lib.conf.stored.essay_conf import essay_dict as d
            elif conftype == 'Batch':
                from lib.conf.stored.batch_conf import batch_dict as d

            elif conftype == 'Ga':
                from lib.conf.stored.ga_conf import ga_dic as d
            else:
                continue
            for id, conf in d.items():
                self.saveConf(conf=conf, conftype=conftype, id=id)

        self.confID_dict = self.build_confID_dict()

    def next_idx(self, id, conftype='Exp'):
        # from lib.conf.pars.pars import ParDict
        # F0 = ParDict.path_dict["SimIdx"]
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

