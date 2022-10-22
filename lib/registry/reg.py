import os
from lib.aux import dictsNlists as dNl
from functools import lru_cache
VERBOSE = 0

def init0():
    from lib.registry.paths import AllConfDict, ExpandedConfDict
    global conf
    conf = AllConfDict()
    global conF
    conF = ExpandedConfDict()


def init(ks=None):
    # global VERBOSE
    # VERBOSE = 0

    from lib.registry.paths import build_path_dict, buildSampleDic, build_datapath_structure, build_datapath_dict, \
        build_datafunc_dict
    global Path
    global SampleDic

    global datafunc_dict
    global datapath_dict

    Path = build_path_dict()
    SampleDic = buildSampleDic()
    datapath_dict, datafunc_dict = build_datapath_structure()


    from lib.registry import ConfTypeDict, LarvaConfDict, ParInitDict, GraphDict, ProcFuncDict, ParserDict, \
        BaseParDict, GroupTypeDict, DistDict

    reg_dict=dNl.NestDict({
        'CT': ConfTypeDict.ConfTypeDict,
        'DD': DistDict.DistDict,
        'MD': LarvaConfDict.LarvaConfDict,
        'PI': ParInitDict.ParInitDict,
        'DEF': ParInitDict.ParDefaultDict,
        'GT': GroupTypeDict.GroupTypeDict,
        'GD': GraphDict.GraphDict,

        'ParsD':ParserDict.ParserDict,
        'PF':ProcFuncDict.ProcFuncDict,
        'PD': BaseParDict.BaseParDict,
    })

    load_mode= {'DEF' : {'mode' : 'load'}}

    # import random
    # PLUGINS = dict()

    # def register(func):
    #     """Register a function as a plug-in"""
    #     PLUGINS[func.__name__] = func
    #     return func

    class Foo(object):

        #@property
        @lru_cache()
        def prop(self, func, kwargs):
            # print("once")
            return func(**kwargs)
    #         else :
    #             print("exists")
    #             return reg_dict[key]()
    #
    # def eee(k,v):
    #     if k not in globals():
    #         globals()[k]=None
    #     else :
    #         return foo.prop(v)
    foo=Foo()
    if ks is None :
        ks=list(reg_dict.keys())
    for k in ks:

        # kws=dNl.NestDict()
        # print(k)
        if k in load_mode.keys() :
            kws=load_mode[k]
        else :
            kws = {}
            # kws.mode='load'
        globals()[k] = reg_dict[k](**kws)
        # globals()[k] = foo.prop(reg_dict[k], kwargs=kws)
        # eee(k,v)
    #     globals()[k]




def init2():
    pass
    #

    #
    # # CT.build_mDicts(PI=PI, MD=MD)
    #



def getPar(k=None, p=None, d=None, to_return='d'):
    return PD.getPar(k=k, d=d, p=p, to_return=to_return)


def next_idx(id, conftype='Exp'):
    f = Path.SimIdx
    if not os.path.isfile(f):
        d = dNl.NestDict({k: dNl.NestDict() for k in ['Exp', 'Batch', 'Essay', 'Eval', 'Ga']})
    else:
        d = dNl.load_dict(f, use_pickle=False)

    if not conftype in d.keys():
        d[conftype] = {}
    if not id in d[conftype].keys():
        d[conftype][id] = 0
    d[conftype][id] += 1
    dNl.save_dict(d, f, use_pickle=False)
    return d[conftype][id]


def vprint(text='', verbose=0):
    if verbose >= VERBOSE:
        print(text)


def datafunc(filepath_key, mode='load'):
    DD = datafunc_dict
    if filepath_key in DD.keys():
        return DD[filepath_key][mode]
    else:
        return None


def datapath(filepath_key, dir=None):
    DD = datapath_dict
    if dir is not None and filepath_key in DD.keys():
        return f'{dir}{DD[filepath_key]}'
    else:
        return None


def lgs(**kwargs):
    return GT.dict.LarvaGroup.lgs(**kwargs)

def get_null(name, **kwargs):
    return DEF.get_null(name=name, **kwargs)


def loadRef(id, load=False, **kwargs):
    c = retrieveRef(id)
    if c is not None :
        from lib.stor.larva_dataset import LarvaDataset
        d = LarvaDataset(c.dir, load_data=False)
        if not load:
            vprint(f'Loaded stored reference configuration : {id}')
            return d
        else:
            d.load(**kwargs)
            vprint(f'Loaded stored reference dataset : {id}')
            return d

    else:
        # self.vprint(f'Ref Configuration {id} does not exist. Returning None')
        return None

def retrieveRef(id):
    dic = dNl.load_dict(Path.Ref, use_pickle=False)
    if id in dic.keys():
        return dic[id]
    else:
        vprint(f'Ref Configuration {id} does not exist. Returning None',1)
        return None

def saveRef(id, conf):
    path=Path.Ref
    dic = dNl.load_dict(path, use_pickle=False)
    dic[id] = conf
    dNl.save_dict(dic, path, use_pickle=False)
