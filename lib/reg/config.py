import copy

import param
from lib import reg, aux


def confInit_ks(k):

    d = aux.NestDict({
        'Ref': None,
        'Eval': 'eval_conf',
        'Replay': 'replay',
        'Model': 'larva_conf',
        'Source': 'food',
        'LarvaGroup': 'LarvaGroup',
        'ModelGroup': 'ModelGroup',
        'Env': 'env_conf',
        'Exp': 'exp_conf',
        'ExpGroup': 'ExpGroup',
        # 'essay': 'essay_params',
        'sim': 'sim_params',
        'Essay': 'essay_params',
        'Batch': 'batch_conf',
        'Ga': 'GAconf',
        'Tracker': 'tracker',
        'Group': 'DataGroup',
        'Trial': 'trials',
        'Life': 'life_history',
        'Tree': None,
        'Body': 'body_shape'
    })
    return d[k]


def update_mdict(mdict, mmdic):
    if mmdic is None or mdict is None:
        return None
    elif not isinstance(mmdic, dict) or not isinstance(mdict, dict):
        return mdict
    else:
        for d, p in mdict.items():
            new_v = mmdic[d] if d in mmdic.keys() else None
            if isinstance(p, param.Parameterized):
                if type(new_v) == list:
                    if p.parclass in [param.Range, param.NumericTuple, param.Tuple]:
                        new_v = tuple(new_v)
                p.v = new_v
            else:
                mdict[d] = update_mdict(mdict=p, mmdic=new_v)
        return mdict


def update_existing_mdict(mdict, mmdic):
    if mmdic is None:
        return mdict
    else:
        for d, v in mmdic.items():
            p = mdict[d]
            if isinstance(p, param.Parameterized):
                if type(v) == list:
                    if p.parclass in [param.Range, param.NumericTuple, param.Tuple]:
                        v = tuple(v)

                p.v = v
            elif isinstance(p, dict) and isinstance(v, dict):
                mdict[d] = update_existing_mdict(mdict=p, mmdic=v)
        return mdict


class ConfType(reg.base.BaseType):
    def __init__(self,path, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        # self.path = reg.Path[self.k]
        self.use_pickle = False if self.k != 'Ga' else True

    # @property
    def loadDict(self):
        # print(self.k, self.use_pickle)
        try:

            return aux.load_dict(self.path, self.use_pickle)
        except:
            return aux.NestDict()

    def loadConf(self, id):
        d = self.loadDict()
        if id in d.keys():
            return aux.NestDict(d[id])
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
                mm = update_existing_mdict(mm, conf)
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
                conf = self.loadConf(id=id)
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
            d[id] = aux.update_nestdict(d[id], aux.flatten_dict(conf))
        else:
            d[id] = aux.NestDict(conf)
        self.saveDict(d)
        reg.vprint(f'{self.k} Configuration saved under the id : {id}')

    def saveDict(self, d):
        aux.save_dict(d, self.path, self.use_pickle)


    def resetDict(self):
        # from lib.conf.stored.facade import stored_confs
        from lib.reg import funcs
        dd = funcs.stored_confs[self.k]()
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

    # def ConfSelector(self, **kwargs):
    #     def func():
    #         return selector_func(objects=self.ConfIDs, **kwargs)
    #
    #     return func

    def ConfParsarg(self):
        return {'dest': f'{self.k}_experiment', 'choices': self.ConfIDs, 'help': f'The {self.k} mode'}

    # def ConfID_entry(self, default=None, k=None, symbol=None, single_choice=True):
    #     from typing import List
    #     from lib.aux.par_aux import sub
    #     low = self.k.lower()
    #     if single_choice:
    #         t = str
    #         IDstr = 'ID'
    #     else:
    #         t = List[str]
    #         IDstr = 'IDs'
    #     if k is None:
    #         k = f'{low}{IDstr}'
    #     if symbol is None:
    #         symbol = sub(IDstr, low)
    #     d = {'dtype': t, 'vparfunc': self.ConfSelector(default=default, single_choice=single_choice),
    #          'vs': self.ConfIDs, 'v': default,
    #          'symbol': symbol, 'k': k, 'h': f'The {self.k} configuration {IDstr}',
    #          'disp': f'{self.k} {IDstr}'}
    #     return dNl.NestDict(d)




    def checkDict(self):
        d = self.loadDict()
        eval = {}
        for id, conf in d.items():
            try:
                eval[id] =update_mdict(self.mdict, conf)
            except:
                eval[id] = None
        return eval


class ConfTypeDict:
    def __init__(self, init_dic, Path):
        reg.vprint('started ConfTypes',0)
        self.conftypes = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body', 'Tree', 'Source']

        subk_dict = self.build_subk_dict()
        self.dict = aux.NestDict({k: ConfType(k=k, subks=subks, parent=self, path=Path[k]) for k, subks in subk_dict.items()})
        # self.dict = self.build(self.conftypes)
        self.build_mDicts(init_dic)
        reg.vprint('completed ConfTypes',0)

    def build_subk_dict(self):
        d0 = {k: {} for k in self.conftypes}
        d1 = {
            'Batch': {'exp': 'Exp'},
            'Ga': {'env_params': 'Env'},
            'Exp': {'env_params': 'Env',
                    'trials': 'Trial',
                    'larva_groups': 'Model',
                    }
        }
        d0.update(d1)
        return d0


    def build_mDicts(self, init_dic):
        for k, ct in self.dict.items():
            k0 = confInit_ks(k)
            if k0 is not None and k0 in init_dic.keys():
                dict0 = init_dic[k0]
            else:
                dict0 = None

            ct.set_dict0(dict0)
    # def build(self, ks):
    #
    #     self.subk_dict = self.build_subk_dict(ks)
    #
    #     d = dNl.NestDict({k: ConfType(k=k, subks=subks, parent=self) for k, subks in self.subk_dict.items()})
    #
    #     return d

    def resetConfs(self, ks=None):
        if ks is None:
            ks = self.conftypes

        for k in ks:

            self.dict[k].resetDict()
            # try :
            #     self.dict[k].resetDict()
            # except :
                # print(f'{k} configuration reset failed')
                # pass


class GroupType(reg.base.BaseType):
    def __init__(self, dict0, **kwargs):
        super().__init__(**kwargs)
        self.set_dict0(dict0)



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
                # print(m, conf)

                mm = update_existing_mdict(mm, conf)

                return mm
            else :
                return ct.mdict


        for subID, subk in self.subks.items():
            if self.parent.dict[subk].mdict is None :
                continue

            if subID == 'larva_groups' and subk == 'Model':

                for k,dic in m0[subID].v.items():
                    if 'model' in dic.keys():
                        p=dic.model
                        mm=retrieve(p,self.parent.dict['Model'])
                        dic.model=mm

            else:
                # ct=CT[subk]
                # mm=copy.deepcopy(ct.mdict)
                # p = m0[subID]
                m0[subID] = retrieve(m0[subID], self.parent.dict[subk])
                # m0[subID]=mm
        return m0







    #
    # def ConfID_entry(self, default=None, k=None, symbol=None, single_choice=True):
    #     from typing import List
    #     from lib.aux.par_aux import sub
    #     low = self.k.lower()
    #     if single_choice:
    #         t = str
    #         IDstr = 'ID'
    #     else:
    #         t = List[str]
    #         IDstr = 'IDs'
    #     if k is None:
    #         k = f'{low}{IDstr}'
    #     if symbol is None:
    #         symbol = sub(IDstr, low)
    #     d = {'dtype': t, 'vparfunc': self.ConfSelector(default=default, single_choice=single_choice),
    #          'vs': self.ConfIDs, 'v': default,
    #          'symbol': symbol, 'k': k, 'h': f'The {self.k} configuration {IDstr}',
    #          'disp': f'{self.k} {IDstr}'}
    #     return dNl.NestDict(d)






    #
    #
    #
    # def checkDict(self):
    #     d = self.loadDict()
    #     eval = {}
    #     for id, conf in d.items():
    #         try:
    #             eval[id] =update_mdict(self.mdict, conf)
    #         except:
    #             eval[id] = None
    #     return eval


    def RvsS_groups(self,N=1, age=72.0, q=1.0, h_starved=0.0, sample='None.150controls', substrate_type='standard',pref='',
                    navigator=False, expand=False, **kwargs):
        if age==0.0 :
            epochs={}
        else :
            if h_starved==0:
                eps={
                    0 : {'start': 0.0, 'stop' : age, 'substate':{'type': substrate_type, 'quality': q}}
                }
            else :
                eps = {
                    0: {'start': 0.0, 'stop': age-h_starved, 'substate': {'type': substrate_type, 'quality': q}},
                    1: {'start': age-h_starved, 'stop': age, 'substate': {'type': substrate_type, 'quality': 0}},
                }
            epochs={}
            for id,kws in eps.items():
                epochs.update(reg.group.dict.epoch.entry(id=id,**kws))




        kws0 = {
            'kwdic': {
                'distribution': {'N': N, 'scale': (0.005, 0.005)},
                'life_history': {'age': age,
                                 'epochs': epochs,
                                 },
            'odor':{}
            },
            'sample': sample,
        }

        mcols = ['blue', 'red']
        mID0s = ['rover', 'sitter']
        lgs = {}
        for mID0, mcol in zip(mID0s, mcols):
            id=f'{pref}{mID0.capitalize()}'



            if navigator :
                mID0=f'navigator_{mID0}'
            if expand:
                mID0=reg.loadConf(conftype="Model", id=mID0)



            kws = {
                'default_color': mcol,
                'model': mID0,
                **kws0
            }

            lgs.update(self.entry(id, **kws))
        return aux.NestDict(lgs)

    def lg_entry(self, id=None, c='black', N=1, mode='uniform', sh='circle', loc=(0.0, 0.0), ors=(0.0, 360.0),
           s=(0.0, 0.0), mID='explorer',age=0.0, epochs={},  o=None,sample = None, expand=False, **kwargs):
        if id is None :
            id=mID
        m=mID if not expand else reg.loadConf(conftype="Model", id=mID)
        if type(s) == float:
            s = (s, s)
        kws = {'kwdic': {
            'distribution': {'N': N, 'scale': s, 'orientation_range': ors, 'loc': loc, 'shape': sh, 'mode': mode},
            'life_history': {'age': age,'epochs': epochs}
        },
               'default_color': c, 'model': m,'sample':sample,  **kwargs}
        if o is not None:
            kws['odor'] = o

        return self.entry(id=id, **kws)

    def lgs(self, mIDs, ids=None, cs=None,**kwargs):

        if ids is None:
            ids = mIDs
        N = len(mIDs)
        if cs is None :
            cs = aux.N_colors(N)
        return aux.NestDict(aux.merge_dicts([self.lg_entry(id, c=c, mID=mID, **kwargs) for mID, c, id in zip(mIDs, cs, ids)]))



class GroupTypeDict:
    def __init__(self,init_dic):



        reg.vprint('started GroupTypes',0)
        self.grouptypes = ['LarvaGroup', 'SourceGroup', 'epoch']
        subk_dict = self.build_subk_dict(self.grouptypes)

        self.dict = aux.NestDict({k: GroupType(k=k, subks=subks, parent=self, dict0=init_dic[k]) for k, subks in subk_dict.items()})
        # self.dict = self.build(self.grouptypes, init_dic)

        reg.vprint('completed GroupTypes',0)

    def build_subk_dict(self, ks):
        d0 = aux.NestDict({k: {} for k in ks})
        d1 = aux.NestDict({
            'LarvaGroup': {'Model'},
            # 'Ga': {'env_params': 'Env'},
            # 'Exp': {'env_params': 'Env',
            #         'trials': 'Trial',
            #         'larva_groups': 'Model',
            #         }
        })
        d0.update(d1)
        return d0


# # from lib.registry.conf import ConfTypeDict, GroupTypeDict
conf0 =ConfTypeDict(reg.par.PI, reg.Path)
group =GroupTypeDict(reg.par.PI)