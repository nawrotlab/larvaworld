import pandas as pd

from lib.registry.par import v_descriptor
from lib.aux import dictsNlists as dNl


class ParRegistry:
    def __init__(self, mode='build', object=None, save=True, load_funcs=False):
        from lib.registry.par_dict import BaseParDict
        from lib.registry.par_funcs import build_func_dict
        from lib.registry.paths import build_path_dict
        self.path_dict = build_path_dict()

        from lib.registry.init_pars import ParInitDict
        self.init_dict = ParInitDict().dict

        from lib.registry.par_funcs import module_func_dict
        self.mfunc = module_func_dict()

        from lib.registry.parser_dict import ParserDict
        self.parser_dict = ParserDict(init_dict=self.init_dict).dict

        from lib.registry.dist_dict import DistDict
        self.dist_dict = DistDict().dict
        from lib.registry.parConfs import LarvaConfDict
        self.larva_conf_dict = LarvaConfDict(init_dict=self.init_dict, mfunc=self.mfunc, dist_dict=self.dist_dict)



        if load_funcs:
            self.func_dict = dNl.load_dict(self.path_dict['ParFuncDict'])
        else:
            self.func_dict = build_func_dict()
            dNl.save_dict(self.func_dict, self.path_dict['ParFuncDict'])

        if mode == 'load':
            self.dict = self.load()
        elif mode == 'build':
            self.dict_entries = BaseParDict(func_dict=self.func_dict).dict_entries

            self.dict = self.finalize_dict(self.dict_entries)

            self.ddict = dNl.NestDict({p.d: p for k, p in self.dict.items()})
            self.pdict = dNl.NestDict({p.p: p for k, p in self.dict.items()})
            if save:
                self.save()

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

# ParDict = ParRegistry()
preg = ParRegistry()

# def getPar(k=None, p=None, d=None, to_return='d', PF=ParDict):
#     if k is not None:
#         d0 = PF.dict
#         k0 = k
#     elif d is not None:
#         d0 = PF.ddict
#         k0 = d
#     elif p is not None:
#         d0 = PF.pdict
#         k0 = p
#
#     if type(k0) == str:
#         par = d0[k0]
#         if type(to_return) == list:
#             return [getattr(par, i) for i in to_return]
#         elif type(to_return) == str:
#             return getattr(par, to_return)
#     elif type(k0) == list:
#         pars = [d0[i] for i in k0]
#         if type(to_return) == list:
#             return [[getattr(par, i) for par in pars] for i in to_return]
#         elif type(to_return) == str:
#             return [getattr(par, to_return) for par in pars]





