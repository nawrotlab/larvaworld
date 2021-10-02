import subprocess
from ast import literal_eval
from pydoc import locate
from typing import List, Tuple, Union, Type
import PySimpleGUI as sg

SYMBOL_UP = '▲'
SYMBOL_DOWN = '▼'
window_size = (2000, 1200)


def col_size(x_frac=1.0, y_frac=1.0, win_size=None):
    if win_size is None:
        win_size = window_size
    return int(win_size[0] * x_frac), int(win_size[1] * y_frac)


w_kws = {
    'finalize': True,
    'resizable': True,
    'default_button_element_size': (6, 1),
    'default_element_size': (14, 1),
    'font': ('Helvetica', 8, 'normal'),
    'auto_size_text': False,
    'auto_size_buttons': False,
    'text_justification': 'left',
}
col_kws = {'vertical_alignment': 't', 'expand_x': False, 'expand_y': False}
b_kws = {'font': ('size', 6)}
b3_kws = {'font': ('size', 6),
          'size': (3, 1)}
b6_kws = {'font': ('size', 6),
          'size': (6, 1)}
b12_kws = {'font': ('size', 6),
           'size': (12, 1)}
spin_size={'size' :  (4, 1)}


def t_kws(w, h=1):
    return {'size': (w, h)}


def get_disp_name(name) -> str:
    n = "%s%s" % (name[0].upper(), name[1:])
    n = n.replace('_', ' ')
    return n


def retrieve_value(v, t):
    # print(v, type(v))

    if v in ['', 'None', None, ('', ''), ('', '')]:
        vv = None
    elif v in ['sample', 'fit']:
        vv = v
    elif t in ['bool', bool]:
        if v in ['False', False, 0, '0']:
            vv = False
        elif v in ['True', True, 1, '1']:
            vv = True
    elif t in ['float', float]:
        vv = float(v)
    elif t in ['str', str]:
        vv = str(v)
    elif t in ['int', int]:
        vv = int(v)
    elif type(v) == t:
        vv = v
    elif t in [List[tuple], List[float], List[int]]:
        # elif t in [List[tuple], List[float], List[int]]:
        if type(v) == str:
            v = v.replace('{', ' ')
            v = v.replace('}', ' ')
            v = v.replace('[', ' ')
            v = v.replace(']', ' ')
            v = v.replace('(', ' ')
            v = v.replace(')', ' ')

            if t == List[tuple]:
                v = v.replace(',', ' ')
                vv = [tuple([float(x) for x in t.split()]) for t in v.split('   ')]
            elif t == List[float]:
                vv = [float(x) for x in v.split(',')]
            elif t == List[int]:
                vv = [int(x) for x in v.split(',')]
        elif type(v) == list:
            vv = v
    elif t in [Tuple[float, float], Tuple[int, int], Union[Tuple[float, float], Tuple[int, int]]] and type(v) == str:
        v = v.replace('{', '')
        v = v.replace('}', '')
        v = v.replace('[', '')
        v = v.replace(']', '')
        v = v.replace('(', '')
        v = v.replace(')', '')
        v = v.replace("'", '')
        v = v.replace(",", ' ')
        if t == Tuple[float, float]:
            vv = tuple([float(x) for x in v.split()])
        elif t == Tuple[int, int]:
            vv = tuple([int(x) for x in v.split()])
        elif t == Union[Tuple[float, float], Tuple[int, int]]:
            vv = tuple([float(x) if '.' in x else int(x) for x in v.split()])
    elif t == Type and type(v) == str:
        if 'str' in v:
            vv = str
        elif 'float' in v:
            vv = float
        elif 'bool' in v:
            vv = bool
        elif 'int' in v:
            vv = int
        else:
            vv = locate(v)
    elif t == tuple or t == list:
        try:
            vv = literal_eval(v)
        except:
            vv = [float(x) for x in v.split()]
            if t == tuple:
                vv = tuple(vv)
    elif type(t) == list:
        vv = retrieve_value(v, type(t[0]))
        # if vv not in t:
        #     print(vv)
        #     raise ValueError(f'Retrieved value {vv} not in list {t}')
    else:
        vv = v
    return vv


def retrieve_dict(dic, type_dic):
    return {k: retrieve_value(v, type_dic[k]) for k, v in dic.items()}


def gui_col(element_list, x_frac=0.25, y_frac=1.0, **kwargs):
    l = []
    for e in element_list:
        l += e.get_layout(as_col=False)
    c = sg.Col(l, **col_kws, size=col_size(x_frac=x_frac, y_frac=y_frac), **kwargs)
    return c

def gui_cols(cols, x_fracs=None, y_fracs=None, **kwargs) :
    N=len(cols)
    if x_fracs is None :
        x_fracs=[1.0/N]*N
    elif type(x_fracs)==float :
        x_fracs = [x_fracs] * N
    if y_fracs is None :
        y_fracs=[1.0]*N
    ls=[]
    for col,x,y in zip(cols, x_fracs, y_fracs) :
        l=gui_col(col, x_frac=x, y_frac=y, **kwargs)
        ls.append(l)
    return [ls]



def gui_row(element_list, x_frac=1.0, y_frac=0.5,x_fracs=None, **kwargs):
    N=len(element_list)
    if x_fracs is None :
        x_fracs=[x_frac/N]*N
    l = [sg.Col(e, **col_kws, size=col_size(x_frac=x, y_frac=y_frac)) for e, x in zip(element_list, x_fracs)]
    # r = sg.Col(*l, **col_kws, size=col_size(x_frac=x_frac, y_frac=y_frac), **kwargs)
    return l


def collapse(layout, key, visible=True):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this seciton visible / invisible
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Col(layout, key=key, visible=visible))

