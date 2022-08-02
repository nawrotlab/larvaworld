
import numpy as np

from lib.aux import dictsNlists as dNl



class ParInitDict:
    def __init__(self, load=False, save=False,verbose=None):
        if verbose is None:
            from lib.registry.units import base_verbose
            verbose = base_verbose
        self.verbose = verbose

        if not load:
            from lib.registry.initDicts import I0
            self.dict = I0
            self.default_dict = self.build_default_dict(self.dict)

            from lib.aux.stdout import vprint
            vprint('started InitParDict', self.verbose)

            # print('started InitParDict')
            # if save:
            #     dNl.save_dict(self.default_dict, self.default_dict_path)
            #     dNl.save_dict(self.dict, self.dict_path)
        else:
            from lib.registry.pars import preg
            self.dict_path = preg.path_dict['ParInitDict']
            self.default_dict_path = preg.path_dict['ParDefaultDict']
            self.dict = dNl.load_dict(self.dict_path)
            self.default_dict = dNl.load_dict(self.default_dict_path)





    def build_default_dict(self, d0):
        dic ={}
        for name, d in d0.items() :
            dic[name] = get_default(d,key='v')
        return dNl.NestDict(dic)


    def get_null(self, name, key='v', **kwargs):
        if key != 'v':
            raise
        return update_default(name, self.default_dict[name], **kwargs)

    def null(self,name, kws={},key='v'):
        if key != 'v':
            raise
        d0=self.default_dict[name]
        return dNl.update_nestdict(d0,kws)


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

    def i2m(self, k):
        from lib.aux.data_aux import init2mdict
        return init2mdict({k:self.dict[k]})


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


init_dict=ParInitDict()
#
# t0=time.time()
# # dd = init_dict.build_default_dict(init_dict.dict)
# dd = get_default(init_dict.dict['env_conf'],key='v')
# t1=time.time()
# # dd = dNl.load_dict(init_dict.default_dict_path)
# dd =init_dict.default_dict['env_conf']
# t2=time.time()
#
# I=init_dict
# # print(t1-t0)
# # print(t2-t1)
# # from lib.registry.pars import preg
# # print(init_dict.get_null('processing'))
# # print(preg.init_dict.get_null('processing'))
# # print(init_dict.get_null('enrichment').processing)
# print(I.null('processing'))
# print(I.null('enrichment').processing)
# # print(get_default(init_dict.dict['enrichment'].processing))
# # # print(get_default(init_dict.dict['enrichment'].processing))
# # print(init_dict.dict['processing'])
# print(I.default_dict['processing'])
