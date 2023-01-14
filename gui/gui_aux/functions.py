import PySimpleGUI as sg

window_size = (2400, 1200)  #(1512,982)
# window_size = sg.Window.get_screen_size() #(2400, 1200)


def col_size(x_frac=1.0, y_frac=1.0, win_size=None):
    if win_size is None:
        win_size = window_size
    return int(win_size[0] * x_frac), int(win_size[1] * y_frac)


w_list = 25
w_kws = {
    'finalize': True,
    'resizable': True,
    'default_button_element_size': (6, 1),
    'default_element_size': (14, 1),
    'font': ('Helvetica', 10, 'normal'),
    'auto_size_text': False,
    'auto_size_buttons': False,
    'text_justification': 'left',
    'element_justification': 'center',
    'debugger_enabled': True,
    # 'border_depth': 4,
}
col_kws = {'vertical_alignment': 't', 'expand_x': False, 'expand_y': False}
# b_kws = {'font': ('size', 6)}
# b3_kws = {'font': ('size', 6),
#           'size': (3, 1)}
b6_kws = {'font': ('size', 6),
          'size': (6, 1)}
# b12_kws = {'font': ('size', 6),
#            'size': (12, 1)}
# spin_size = {'size': (4, 1)}
tab_kws = {'font': ("Helvetica", 14, "normal"), 'selected_title_color': 'darkblue', 'title_color': 'grey',
           'tab_background_color': 'lightgrey'}




def t_kws(w, h=1):
    return {'size': (w, h)}


def get_disp_name(name) -> str:
    n = "%s%s" % (name[0].upper(), name[1:])
    n = n.replace('_', ' ')
    return n


def retrieve_value(v, t):
    from typing import List, Tuple, Union, Type

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
            from pydoc import locate
            vv = locate(v)
    elif t == tuple or t == list:
        try:
            from ast import literal_eval
            vv = literal_eval(v)
        except:
            vv = [float(x) for x in v.split()]
            if t == tuple:
                vv = tuple(vv)
    elif type(t) == list:
        vv = retrieve_value(v, type(t[0]))
    else:
        vv = v
    return vv


# def retrieve_dict(dic, type_dic):
#     return {k: retrieve_value(v, type_dic[k]) for k, v in dic.items()}


def gui_col(element_list, x_frac=0.25, y_frac=1.0, as_pane=False, pad=None, add_to_bottom=[], **kwargs):
    l = []
    for e in element_list:
        if not as_pane:
            l += e.get_layout(as_col=False)
        else:
            l += e.get_layout(as_pane=True, pad=pad)
    l += add_to_bottom
    c = sg.Col(l, **col_kws, size=col_size(x_frac=x_frac, y_frac=y_frac), **kwargs)
    return c


def gui_cols(cols, x_fracs=None, y_fracs=None, **kwargs):
    N = len(cols)
    if x_fracs is None:
        x_fracs = [1.0 / N] * N
    elif type(x_fracs) == float:
        x_fracs = [x_fracs] * N
    if y_fracs is None:
        y_fracs = [1.0] * N
    ls = []
    for col, x, y in zip(cols, x_fracs, y_fracs):
        l = gui_col(col, x_frac=x, y_frac=y, **kwargs)
        # if as_pane:
        #     l=sg.Pane([l])
        ls.append(l)
    return [ls]


def gui_row(element_list, x_frac=1.0, y_frac=0.5, x_fracs=None, as_pane=False, pad=None, **kwargs):
    N = len(element_list)
    if x_fracs is None:
        x_fracs = [x_frac / N] * N
    l = []
    for e, x in zip(element_list, x_fracs):
        if not as_pane:
            ll = sg.Col(e, **col_kws, size=col_size(x_frac=x, y_frac=y_frac), **kwargs)
        else:
            try:
                ll = sg.Col(e.get_layout(as_pane=True), **col_kws, size=col_size(x_frac=x, y_frac=y_frac), **kwargs)
            except:
                ll = sg.Col([[sg.Pane([sg.Col(e)], pad=pad, border_width=8)]], **col_kws,
                            size=col_size(x_frac=x, y_frac=y_frac), **kwargs)
        l.append(ll)
    return l


def gui_rowNcol(element_list, x_fracs, y_fracs, as_pane=False):
    l = []
    for i, e in enumerate(element_list):
        if type(e) == list:
            if all([type(ee) != list for ee in e]):
                e = gui_col(e, x_frac=x_fracs[i], y_frac=y_fracs[i], as_pane=as_pane)
            else:
                e = gui_rowNcol(e, x_fracs=x_fracs[i], y_fracs=y_fracs[i], as_pane=as_pane)
            l.append(e)
        else:
            e = gui_row([e], x_frac=x_fracs[i], y_frac=y_fracs[i], as_pane=as_pane)
            l.append(e)
    return l







