import copy
import param
from lib.registry import reg, base

from lib.aux import naming as nam, dictsNlists as dNl, data_aux





def confReset_funcs(k):
    from lib.conf.stored import aux_conf, data_conf, batch_conf, exp_conf, env_conf, essay_conf, ga_conf, larva_conf

    d = dNl.NestDict({
        'Ref': data_conf.Ref_dict,
        'Model': larva_conf.Model_dict,
        'ModelGroup': larva_conf.ModelGroup_dict,
        'Env': env_conf.Env_dict,
        'Exp': exp_conf.Exp_dict,
        'ExpGroup': exp_conf.ExpGroup_dict,
        'Essay': essay_conf.Essay_dict,
        'Batch': batch_conf.Batch_dict,
        'Ga': ga_conf.Ga_dict,
        'Tracker': data_conf.Tracker_dict,
        'Group': data_conf.Group_dict,
        'Trial': aux_conf.Trial_dict,
        'Life': aux_conf.Life_dict,
        'Body': aux_conf.Body_dict
    })
    return d[k]


class ConfType(base.BaseType):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.path = reg.Path[self.k]
        self.use_pickle = False if self.k != 'Ga' else True

    # @property
    def loadDict(self):
        # print(self.k, self.use_pickle)
        try:

            return dNl.load_dict(self.path, self.use_pickle)
        except:
            return dNl.NestDict()

    def loadConf(self, id):
        d = self.loadDict()
        if id in d.keys():
            return dNl.NestDict(d[id])
        else:
            print(f'{self.k} Configuration {id} does not exist')
            raise ValueError()

    def expand_mdict(self, m0=None):
        if m0 is  None:
            if self.mdict is None:
                return None
            else :
                m0=self.mdict

        def retrieve(p,ct):
            conf = None
            if p in ct.ConfIDs:
                # print(v)
                conf = ct.loadConf(p)
            elif isinstance(p, param.Parameterized):
                if p.v in ct.ConfIDs:
                    conf = ct.loadConf(p.v)
            if conf is not None:
                mm = copy.deepcopy(ct.mdict)
                mm = data_aux.update_existing_mdict(mm, conf)
                return mm
            else :
                return ct.mdict





        if len(self.subks) > 0:
            for subID, subk in self.subks.items():
                ct = self.CT.dict[subk]
                if ct.mdict is None :
                    continue

                if subID == 'larva_groups' and subk == 'Model':

                    for k,dic in m0[subID].v.items():
                        if 'model' in dic.keys():
                            p=dic.model
                            mm=retrieve(p,ct)
                            dic.model=mm

                else:
                    m0[subID] = retrieve(m0[subID],ct)
                    # m0[subID]=mm
            return m0






    def expandConf(self, id=None,conf=None):
        if conf is None:
            if id in self.ConfIDs:

            # from lib.registry.pars import preg
                conf = self.loadConf(id)
            else :
                return None
        if len(self.subks) > 0:
            for subID, subk in self.subks.items():
                ct = self.parent.dict[subk]
                if subID == 'larva_groups' and subk == 'Model':
                    for k, v in conf['larva_groups'].items():
                        if v.model in ct.ConfIDs:
                            v.model = ct.loadConf(id=v.model)
                else:
                    if conf[subID] in ct.ConfIDs:
                        conf[subID] = ct.loadConf(id=conf[subID])

            return conf

    def saveConf(self, id, conf, mode='overwrite'):
        d = self.loadDict()

        if id in d.keys() and mode == 'update':
            d[id] = dNl.update_nestdict(d[id], dNl.flatten_dict(conf))
        else:
            d[id] = dNl.NestDict(conf)
        self.saveDict(d)
        reg.vprint(f'{self.k} Configuration saved under the id : {id}')

    def saveDict(self, d):
        dNl.save_dict(d, self.path, self.use_pickle)

    def reset_func(self):
        return confReset_funcs(self.k)()

    def resetDict(self):
        dd = self.reset_func()
        d = self.loadDict()

        N0, N1 = len(d), len(dd)

        d.update(dd)

        Ncur = len(d)
        Nnew = Ncur - N0
        Nup = N1 - Nnew
        self.saveDict(d)

        reg.vprint(f'{self.k}  configurations : {Nnew} added , {Nup} updated,{Ncur} now existing',2)



    def deleteConf(self, id=None):
        if id is not None:
            d = self.loadDict()
            if id in d.keys():
                d.pop(id, None)
                self.saveDict(d)
                reg.vprint(f'Deleted {self.k} configuration under the id : {id}', 2)

    @property
    def ConfIDs(self):
        return list(self.loadDict().keys())

    def ConfSelector(self, **kwargs):
        from lib.registry.par import selector_func
        def func():
            return selector_func(objects=self.ConfIDs, **kwargs)

        return func

    def ConfParsarg(self):
        return {'dest': f'{self.k}_experiment', 'choices': self.ConfIDs, 'help': f'The {self.k} mode'}

    def ConfID_entry(self, default=None, k=None, symbol=None, single_choice=True):
        from typing import List
        from lib.aux.par_aux import sub
        low = self.k.lower()
        if single_choice:
            t = str
            IDstr = 'ID'
        else:
            t = List[str]
            IDstr = 'IDs'
        if k is None:
            k = f'{low}{IDstr}'
        if symbol is None:
            symbol = sub(IDstr, low)
        d = {'dtype': t, 'vparfunc': self.ConfSelector(default=default, single_choice=single_choice),
             'vs': self.ConfIDs, 'v': default,
             'symbol': symbol, 'k': k, 'h': f'The {self.k} configuration {IDstr}',
             'disp': f'{self.k} {IDstr}'}
        return dNl.NestDict(d)




    def checkDict(self):
        d = self.loadDict()
        eval = {}
        for id, conf in d.items():
            try:
                eval[id] =data_aux.update_mdict(self.mdict, conf)
            except:
                eval[id] = None
        return eval


class ConfTypeDict:
    def __init__(self,load=False, save=False):


        reg.vprint('started ConfTypes',0)
        self.conftypes = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body', 'Source', 'Tree']

        self.dict = self.build(self.conftypes)

        reg.vprint('completed ConfTypes',0)

    def build_subk_dict(self, ks):
        d0 = dNl.NestDict({k: {} for k in ks})
        d1 = dNl.NestDict({
            'Batch': {'exp': 'Exp'},
            'Ga': {'env_params': 'Env'},
            'Exp': {'env_params': 'Env',
                    'trials': 'Trial',
                    'larva_groups': 'Model',
                    }
        })
        d0.update(d1)
        return d0



    def build(self, ks):

        self.subk_dict = self.build_subk_dict(ks)

        d = dNl.NestDict({k: ConfType(k=k, subks=subks, parent=self) for k, subks in self.subk_dict.items()})

        return d

    def saveConf(self, conf, conftype, id=None, **kwargs):
        self.dict[conftype].saveConf(id=id, conf=conf, **kwargs)

    #
    def loadConf(self, conftype, id=None):
        return self.dict[conftype].loadConf(id=id)

    def loadRef(self, id=None):
        if id is not None:
            conf = self.dict.Ref.loadConf(id)
            from lib.stor.larva_dataset import LarvaDataset
            d = LarvaDataset(conf.dir, load_data=False)
            reg.vprint(f'Loaded stored reference configuration : {id}')
            return d

    def loadRefD(self, id=None, **kwargs):
        if id is not None:
            d = self.loadRef(id)
            d.load(**kwargs)
            reg.vprint(f'Loaded stored reference dataset : {id}',2)
            return d

    def loadRefDs(self, ids, **kwargs):
        ds = [self.loadRefD(id, **kwargs) for id in ids]
        return ds

    # def confDict_funcs(self,k):
    #     from lib.conf.stored import aux_conf, data_conf, batch_conf, exp_conf, env_conf, essay_conf, ga_conf, larva_conf
    #     # raise
    #     d = dNl.NestDict({
    #         'Ref': data_conf.Ref_dict,
    #         'Model': larva_conf.Model_dict,
    #         'ModelGroup': larva_conf.ModelGroup_dict,
    #         'Env': env_conf.Env_dict,
    #         'Exp': exp_conf.Exp_dict,
    #         'ExpGroup': exp_conf.ExpGroup_dict,
    #         'Essay': essay_conf.Essay_dict,
    #         'Batch': batch_conf.Batch_dict,
    #         'Ga': ga_conf.Ga_dict,
    #         'Tracker': data_conf.Tracker_dict,
    #         'Group': data_conf.Group_dict,
    #         'Trial': aux_conf.Trial_dict,
    #         'Life': aux_conf.Life_dict,
    #         'Body': aux_conf.Body_dict
    #     })
    #     return d[k]

    def resetConfs(self, ks=None):
        if ks is None:
            ks = self.conftypes

        for k in ks:
            self.dict[k].resetDict()

