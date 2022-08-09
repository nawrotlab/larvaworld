from lib.aux import naming as nam, dictsNlists as dNl
from lib.aux.data_aux import get_ks
import timeit

from lib.decorators.timer1 import timing


class BaseType():
    def __init__(self, k, subks={}):
        self.k = k
        self.subks = subks
        self.mdict=None
        self.dict0 =None
        self.ks = None

    def build_mdict(self):
        from lib.aux.data_aux import init2mdict
        self.mdict = init2mdict(self.dict0)
        self.ks = get_ks(self.mdict)


    def set_dict0(self, dict0):
        self.dict0=dict0
        if self.dict0 is not None and self.mdict is None :
            self.build_mdict()

    def gConf_kws(self,dic):
        kws0={}
        for k,kws in dic.items():
            m0=self.mdict[k]
            kws0[k]=self.gConf(m0,**kws)
        return dNl.NestDict(kws0)


    def gConf(self,m0=None,kwdic=None,**kwargs):
        if m0 is  None:
            if self.mdict is None:
                return None
            else :
                m0=self.mdict
        if kwdic is not None:
            kws0=self.gConf_kws(kwdic)
            kwargs.update(kws0)
        from lib.aux.data_aux import gConf
        return dNl.NestDict(gConf(m0,**kwargs))

    def entry(self, id, **kwargs):
        return dNl.NestDict({id: self.gConf(**kwargs)})

class BaseConfDict :
    def __init__(self, mode='build',verbose=1):
        with timing() as timer:
            self.name=self.__class__.__name__
            self.verbose = verbose
            self.vprint(f'Initializing {self.name} in mode {mode}')


            from lib.registry.pars import preg
            self.path = preg.paths[self.name]
            if mode=='build':
                self.dict = self.build()

            elif mode=='load':
                self.dict = self.load()
            else:
                pass

            self.vprint(f'Completed {self.name} successfully')
        self.vprint(f'Class {self.name} Initialization time {timer.time_sec} in mode {mode}', 2)



    def vprint(self, text, verbose=2):
        if verbose >= self.verbose:
            print(text)

    def build(self):
        return dNl.NestDict()
        # pass

    def load(self):
        return dNl.load_dict(self.path)

    def save(self, d=None):
        if d is None :
            d=self.dict
        dNl.save_dict(d, self.path)


ppp=BaseConfDict()


