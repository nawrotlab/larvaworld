
import numpy as np

from lib.aux import naming as nam, dictsNlists as dNl
from lib.registry.base import BaseConfDict
from lib.registry import reg

def confInit_ks(k):

    d = dNl.NestDict({
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


def get_default(d,key='v') :
    if d is None:
        return None
    null = dNl.NestDict()
    for k, v in d.items():
        if not isinstance(v, dict):
            null[k] = v
        elif 'k' in v.keys() or 'h' in v.keys() or 'dtype' in v.keys():
            null[k] = None if key not in v.keys() else v[key]
        else:
            null[k] = get_default(v,key)
    return null

def update_default(name, dic, **kwargs):
    if name not in ['visualization', 'enrichment']:
        dic.update(kwargs)
        return dNl.NestDict(dic)
    else:
        for k, v in dic.items():
            if k in list(kwargs.keys()):
                dic[k] = kwargs[k]
            elif isinstance(v, dict):
                for k0, v0 in v.items():
                    if k0 in list(kwargs.keys()):
                        dic[k][k0] = kwargs[k0]
        return dNl.NestDict(dic)


class ParInitDict(BaseConfDict):
    # def __init__(self):
    #
    #
    #
    #
    #     super().__init__()


    def build(self):

        CTs=reg.CT.dict

        from lib.registry.initDicts import buildInitDict
        d = buildInitDict(CTs)

        d=dNl.NestDict(d)

        # self.default_dict = self.build_default_dict(d)

        self.build_mDicts(CTs, d)
        return d

    def build_mDicts(self, CTs, d):
        for k, ct in CTs.items():
            k0 = confInit_ks(k)
            if k0 is not None and k0 in d.keys():
                dict0 = d[k0]
            else:
                dict0 = None

            ct.set_dict0(dict0)







    def i2m(self, k):
        from lib.aux.data_aux import init2mdict
        return init2mdict({k:self.dict[k]})

class ParDefaultDict(BaseConfDict):
    # def __init__(self):
    #     super().__init__()


    def build(self):
        dic = {}
        for name, d in reg.PI.dict.items():
            dic[name] = get_default(d, key='v')
        return dNl.NestDict(dic)

    def null(self,name, kws={},key='v'):
        if key != 'v':
            raise
        d0=self.dict[name]
        return dNl.update_nestdict(d0,kws)

    def get_null(self, name, key='v', **kwargs):
        if key != 'v':
            raise
        return update_default(name, self.dict[name], **kwargs)

    def metric_def(self, ang={}, sp={}, **kwargs):
        def ang_def(fv=(1, 2), rv=(-2, -1), **kwargs):
            return self.get_null('ang_definition',front_vector=fv, rear_vector=rv, **kwargs)

        return self.get_null('metric_definition',
                             angular=ang_def(**ang),
                             spatial=self.get_null('spatial_definition', **sp),
                             **kwargs)

    def oG(self, c=1, id='Odor'):
        return self.get_null('odor', odor_id=id, odor_intensity=2.0 * c, odor_spread=0.0002 * np.sqrt(c))

    def oD(self, c=1, id='Odor'):
        return self.get_null('odor', odor_id=id, odor_intensity=300.0 * c, odor_spread=0.1 * np.sqrt(c))
        # return self.odor(i=300.0 * c, s=0.1 * np.sqrt(c), id=id)

    def arena(self, x, y=None):
        if y is None:
            return self.get_null('arena', arena_shape='circular', arena_dims=(x, x))
        else:
            return self.get_null('arena', arena_shape='rectangular', arena_dims=(x, y))



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
        kws={k:self.get_null(k,**v) for k,v in kw_dic0.items()}

        if metric_definition is None:
            metric_definition = self.metric_def(**def_kws)
        dic = self.get_null('enrichment',
                                      metric_definition=metric_definition, **kws, **kwargs)
        return dic

    def base_enrich(self, **kwargs):
        return self.enr_dict(proc=['angular', 'spatial', 'dispersion', 'tortuosity'],
                             bouts=['stride', 'pause', 'turn'],
                             to_keep=['midline', 'contour'], **kwargs)