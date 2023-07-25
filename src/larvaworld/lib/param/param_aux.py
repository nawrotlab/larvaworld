import random
from types import FunctionType
import typing
import param
import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

import numpy as np
from param import Parameterized, Number,NumericTuple,Integer,Selector,String, ListSelector, Range, Magnitude, Boolean,ClassSelector,Parameter, List, Dict

from larvaworld.lib import aux


def param_dtype(parclass):
    func_dic = {

        Number : float,
        Integer : int,
        String : str,
        Boolean : bool,
        Dict : dict,
        List : list,
        ClassSelector : type,

        # typing.List[int]: List,
        # typing.List[str]: List,
        # typing.List[float]: List,
        # typing.List[typing.Tuple[float]]: List,
        # FunctionType: param.Callable,
        # typing.Tuple[float]: Range,
        # typing.Tuple[int]: NumericTuple,
        # TypedDict: Dict
    }

    return func_dic[parclass]

def get_vfunc(dtype, lim, vs):
    func_dic = {
        float: Number,
        int: Integer,
        str: String,
        bool: Boolean,
        dict: Dict,
        list: List,
        type: ClassSelector,
        typing.List[int]: List,
        typing.List[str]: List,
        typing.List[float]: List,
        typing.List[typing.Tuple[float]]: List,
        FunctionType: param.Callable,
        typing.Tuple[float]: Range,
        typing.Tuple[int]: NumericTuple,
        TypedDict: Dict
    }
    if dtype == float and lim == (0.0, 1.0):
        return Magnitude
    if type(vs) == list and dtype in [str, int]:
        return Selector
    elif dtype in func_dic.keys():
        return func_dic[dtype]
    else:
        return Parameter

def vpar(vfunc, v0, h, lab, lim, dv, vs):
    f_kws = {
        'default': v0,
        'doc': h,
        'label': lab,
        'allow_None': True
    }
    if vfunc in [List, Number, Range]:
        if lim is not None:
            f_kws['bounds'] = lim
    if vfunc in [Range, Number]:
        if dv is not None:
            f_kws['step'] = dv
    if vfunc in [Selector]:
        f_kws['objects'] = vs
    func = vfunc(**f_kws, instantiate=True)
    return func

def param_to_arg(k, p):
    c = p.__class__
    # dtype=aux.param_dtype(c)
    v = p.default
    d = aux.AttrDict({
        'key': k,
        'short': k,
        'help': p.doc,
    })
    if v is not None:
        d.default = v
    if c == param.Boolean:
        d.action = 'store_true' if not v else 'store_false'
    elif c == param.String:
        d.type = str
    elif c in param.Integer.__subclasses__():
        d.type = int
    elif c in param.Number.__subclasses__():
        d.type = float
    elif c in param.Tuple.__subclasses__():
        d.type = tuple

    if hasattr(p, 'objects'):
        d.choices = p.objects
        if c in param.List.__subclasses__():
            d.nargs = '+'
        if hasattr(p, 'item_type'):
            d.type = p.item_type
    return d


