import numpy as np
import pandas as pd

from lib.registry.pars import preg


class FuncParHelper:

    def __init__(self) :

        self.func_df=self.inspect_funcs()

    def get_func(self, func):
        module=self.func_df['module'].loc[func]
        return getattr(module, func)

    def apply_func(self,func,s,**kwargs):
        f=self.get_func(func)
        kws={k:kwargs[k] for k in kwargs.keys() if k in self.func_df['args'].loc[func]}
        f(s=s,**kws)
        return s

    def assemble_func_df(self,arg='s'):
        from lib.process import angular, spatial, basic,aux
        arg_dicts = {}
        for module in [angular, spatial, aux, basic]:
            dic = self.get_arg_dict(module, arg)
            arg_dicts.update(dic)
        df = pd.DataFrame.from_dict(arg_dicts,orient='index')

        return df

    def get_arg_dict(self, module, arg):
        from inspect import getmembers, isfunction, signature


        # funcnames = []
        arg_dict={}
        funcs = getmembers(module, isfunction)
        for k, f in funcs:
            args = signature(f)
            args = list(args.parameters.keys())
            if arg in args:
                if k!='store_aux_dataset' :
                    # funcnames.append(k)
                    arg_dict[k]= {'args' : args, 'module':module}
        return arg_dict

    def inspect_funcs(self, arg='s'):
        df=self.assemble_func_df(arg)
        new_cols=['requires', 'depends', 'computes']
        for col in new_cols :
            df[col]=np.nan

        df[new_cols]=self.manual_fill(df[new_cols])
        return df

    def manual_fill(self,df):
        df.loc['comp_ang_from_xy'] = ['x', 'y'], ['ang_from_xy'], ['fov', 'foa']
        df.loc['angular_processing'] = [], ['comp_orientations', 'comp_bend', 'comp_ang_from_xy', 'comp_angular',
                                            'comp_extrema', 'compute_LR_bias', 'store_aux_dataset'], []
        df.loc['comp_angular'] = ['fo', 'ro', 'b'], ['unwrap_orientations'], ['fov', 'foa', 'rov', 'roa', 'bv', 'ba']
        df.loc['unwrap_orientations'] = ['fo', 'ro'], [], ['fou', 'rou']
        df.loc['comp_orientation_1point'] = ['x', 'y'], [], ['fov']
        df.loc['compute_LR_bias'] = ['b', 'bv', 'fov'], [], []
        df.loc['comp_orientations'] = ['xys'], ['comp_orientation_1point'], ['fo', 'ro']
        df.loc['comp_bend'] = ['fo', 'ro'], ['comp_angles'], ['b']
        df.loc['comp_angles'] = ['xys'], [], ['angles']
        return df

    def is_computed_by(self, short):
        return [k for k in self.func_df.index if short in self.func_df['computes'].loc[k]]

    def requires(self, func):
        return self.func_df['requires'].loc[func]

    def depends(self,func):
        return self.func_df['depends'].loc[func]

    def requires_all(self, func):
        import lib.aux.dictsNlists as dNl
        shorts=[]
        shorts.append(self.requires(func))
        for f in self.depends(func) :
            shorts.append(self.requires_all(func))
        shorts=dNl.unique_list(shorts)
        return shorts

    def get_options(self, short):
        options={}
        for func in self.is_computed_by(short):
            options[func]=self.requires(func)
        return options

    def how_to_compute(self, s, par=None, short=None, **kwargs):
        if par is None :
            par = preg.getPar(short)
        elif short is None :
            short=preg.getPar(d=par, to_return='k')
        if par in s.columns :
            return True
        else :
            options=self.get_options(short)
            available= []
            for i,(func, shorts) in enumerate(options.items()) :
                pars = preg.getPar(shorts)
                if all([p in s.columns for p in pars]):

                    available.append(func)
            if len(available)==0 :
                return False
            else :
                return available

    def compute(self,s,**kwargs):
        res=self.how_to_compute(s=s,**kwargs)
        if res in [True, False]:
            return res
        else:
            self.apply_func(res[0],s=s, **kwargs)
            return self.compute(s=s,**kwargs)

if __name__ == '__main__':
    # preg.storeConfs()
    preg.storeConfs(conftypes=['Ga'])