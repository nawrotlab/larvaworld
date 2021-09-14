import copy
import inspect
import os
from tkinter import PhotoImage
from typing import Tuple, List
import numpy as np
import PySimpleGUI as sg
import operator

from PySimpleGUI import Pane, \
    LISTBOX_SELECT_MODE_EXTENDED
from matplotlib import ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import lib.conf.conf
import lib.conf.init_dtypes
from lib.conf.conf import loadConfDict, saveConf, deleteConf, loadConf, expandConf
import lib.aux.functions as fun
from lib.conf.par import runtime_pars, getPar
from lib.gui.aux import SYMBOL_UP, SYMBOL_DOWN, col_size, w_kws, b6_kws, t_kws, get_disp_name, retrieve_value
from lib.gui.buttons import graphic_button, button_dict, named_bool_button, BoolButton
from lib.stor import paths as paths
import lib.conf.dtype_dicts as dtypes


def get_table(v, pars_dict, Nagents):
    data = []
    for i in range(Nagents):
        dic = {}
        for j, (p, t) in enumerate(pars_dict.items()):
            dic[p] = retrieve_value(v[(i, p)], t)
        data.append(dic)
    return data


def set_agent_dict(dic, type_dic, header='unique_id', title='Agent list'):
    t0 = fun.agent_dict2list(dic, header=header)
    t1 = gui_table(t0, type_dic, title=title)
    dic = fun.agent_list2dict(t1, header=header)
    return dic


def table_window(data, pars_dict, title, return_layout=False):
    d = pars_dict
    t12_kws_c = {**t_kws(12),
                 'justification': 'center'}

    ps = list(d.keys())
    Nagents, Npars = len(data), len(ps)

    l0 = [sg.T(' ', **t_kws(2))] + [sg.T(p, k=p, enable_events=True, **t12_kws_c) for p in ps]
    l1 = [[sg.T(i + 1, **t_kws(2))] + [sg.In(data[i][p], k=(i, p), **t12_kws_c) if type(d[p]) != list else sg.Combo(
        d[p], default_value=data[i][p], k=(i, p), enable_events=True, readonly=True,
        **t_kws(12)) for p in ps] for i in range(Nagents)]
    l2 = [sg.B(ii, **b6_kws) for ii in ['Add', 'Remove', 'Ok', 'Cancel']]

    l = [l0, *l1, l2]

    if return_layout:
        return l

    w = sg.Window(title, l, default_element_size=(20, 1), element_padding=(1, 1),
                  return_keyboard_events=True, finalize=True, force_toplevel=True)
    w.close_destroys_window = True
    return Nagents, Npars, ps, w


def gui_table(data, pars_dict, title='Agent list', sortable=False):
    data0 = copy.deepcopy(data)
    """
        Another simple table created from Input Text Elements.  This demo adds the ability to "navigate" around the drawing using
        the arrow keys. The tab key works automatically, but the arrow keys are done in the code below.
    """

    sg.change_look_and_feel('Dark Brown 2')  # No excuse for gray windows
    # Show a "splash" mode message so the user doesn't give up waiting
    sg.popup_quick_message('Hang on for a moment, this will take a bit to create....', auto_close=True,
                           non_blocking=True)

    Nrows, Npars, ps, w = table_window(data, pars_dict, title)
    # cell = (0, 0)
    while True:
        e, v = w.read()
        if e in (None, 'Cancel'):
            w.close()
            return data0
        if e == 'Ok':
            data = get_table(v, pars_dict, Nrows)
            w.close()
            return data
        elem = w.find_element_with_focus()
        cell = elem.Key if elem and type(elem.Key) is tuple else (0, 0)
        r, c = cell
        try:
            if e.startswith('Down'):
                r = r + 1 * (r < Nrows - 1)
            elif e.startswith('Left'):
                c = c - 1 * (c > 0)
            elif e.startswith('Right'):
                c = c + 1 * (c < Npars - 1)
            elif e.startswith('Up'):
                r = r - 1 * (r > 0)
        except:
            pass
        if sortable and e in ps:  # Perform a sort if a column heading was clicked
            try:
                table = [[int(v[(r, c)]) for c in range(Npars)] for r in range(Nrows)]
                new_table = sorted(table, key=operator.itemgetter(ps.index(e)))
            except:
                sg.popup_error('Error in table', 'Your table must contain only ints if you wish to sort by column')
            else:
                for i in range(Nrows):
                    for j in range(Npars):
                        w[(i, j)].update(new_table[i][j])
                [w[c].update(font='Any 14') for c in ps]  # make all column headings be normal fonts
                w[e].update(font='Any 14 bold')  # bold the font that was clicked
        # if the current cell changed, set focus on new cell
        if cell != (r, c):
            cell = r, c
            w[cell].set_focus()  # set the focus on the element moved to
            w[cell].update(
                select=True)  # when setting focus, also highlight the data in the element so typing overwrites
        if e == 'Add':
            data = get_table(v, pars_dict, Nrows)
            try:
                new_row = data[r]
            except:
                new_row = {k: None for k in pars_dict.keys()}
            data.append(new_row)
            w.close()
            Nrows, Npars, ps, w = table_window(data, pars_dict, title)
        elif e == 'Remove':
            data = get_table(v, pars_dict, Nrows)
            data = [d for i, d in enumerate(data) if i != r]
            w.close()
            Nrows, Npars, ps, w = table_window(data, pars_dict, title)


def update_window_from_dict(w, dic, prefix=None):
    if dic is not None:
        for k, v in dic.items():
            if prefix is not None:
                k = f'{prefix}_{k}'
            if type(v) == bool:
                b = w[f'TOGGLE_{k}']
                if isinstance(b, BoolButton):
                    b.set_state(v)
            elif type(v) == dict:
                new_prefix = k if prefix is not None else None
                update_window_from_dict(w, v, prefix=new_prefix)
            elif isinstance(w[k], TupleSpin) or isinstance(w[k], MultiSpin):
                w[k].update(w, v)
            elif v is None:
                w.Element(k).Update(value='')
            else:
                w.Element(k).Update(value=v)


def save_conf_window(conf, conftype, disp=None):
    if disp is None:
        disp = conftype
    l = [
        named_list(f'Store new {disp}', f'{disp}_ID', list(loadConfDict(conftype).keys()),
                   readonly=False, enable_events=False),
        [sg.Ok(), sg.Cancel()]]
    e, v = sg.Window(f'{disp} configuration', l).read(close=True)
    if e == 'Ok':
        id = v[f'{disp}_ID']
        saveConf(conf, conftype, id)
        return id
    elif e == 'Cancel':
        return None


class SectionDict:
    def __init__(self, name, dict, type_dict=None, toggled_subsections=True):
        self.init_dict = dict
        self.type_dict = type_dict
        self.toggled_subsections = toggled_subsections
        self.name = name
        self.subdicts = {}

    def init_section(self, text_kws={}, value_kws={}):
        d = self.type_dict
        l = []
        for k, v in self.init_dict.items():
            k_disp = get_disp_name(k)
            k0 = f'{self.name}_{k}'
            if type(v) == bool:
                ii = named_bool_button(k_disp, v, k0)
            elif type(v) == dict:
                type_dict = d[k] if d is not None else None
                self.subdicts[k0] = CollapsibleDict(k0, True, disp_name=k, dict=v, type_dict=type_dict,
                                                    toggle=self.toggled_subsections,
                                                    toggled_subsections=self.toggled_subsections)
                ii = self.subdicts[k0].get_layout()
            else:
                t0 = d[k]
                t = type(t0)
                if t == list:
                    if all([type(i) in [int, float] for i in t0 if i not in ['', None]]):
                        temp = sg.Spin(values=t0, initial_value=v, key=k0, **value_kws)
                    else:
                        temp = sg.Combo(t0, default_value=v, key=k0, enable_events=True,
                                        readonly=True, **value_kws)
                elif t in [tuple, Tuple[float, float], Tuple[int, int]]:
                    temp = TupleSpin(range=t0, initial_value=v, key=k0, **value_kws)
                elif t == dict and list(t0.keys()) == ['type', 'value_list']:
                    if t0['type'] == list:
                        tuples = False
                        temp = MultiSpin(value_list=t0['value_list'], initial_value=v, key=k0, tuples=tuples,
                                         **value_kws)
                    elif t0['type'] == List[tuple]:
                        tuples = True
                        temp = MultiSpin(value_list=t0['value_list'], initial_value=v, key=k0, tuples=tuples,
                                         **value_kws)
                    elif t0['type'] == tuple:
                        temp = TupleSpin(value_list=t0['value_list'], initial_value=v, key=k0, **value_kws)
                else:
                    temp = sg.In(v, key=k0, **value_kws)
                ii = [sg.Text(f'{k_disp}:', **text_kws), temp]
            l.append(ii)
        return l

    def get_type(self, k, v):
        try:
            return self.type_dict[k]
        except:
            return type(v)
            # return type(v) if self.type_dict is None else self.type_dict[k]

    def get_dict(self, v, w):
        d = copy.deepcopy(self.init_dict)
        if d is None:
            return d
        for i, (k, v0) in enumerate(d.items()):

            k0 = f'{self.name}_{k}'
            t = self.get_type(k, v0)
            if t == bool or type(v0)==bool:
                d[k] = w[f'TOGGLE_{k0}'].get_state()
            elif type(t) == tuple or (type(t) == dict and list(t.keys()) == ['type', 'value_list']):
                d[k] = w[k0].get()
            elif t == dict or type(t) == dict:
                d[k] = self.subdicts[k0].get_dict(v, w)
            else:
                d[k] = retrieve_value(v[k0], t)
        return d

    def get_subdicts(self):
        subdicts = {}
        for s in list(self.subdicts.values()):
            subdicts.update(s.get_subdicts())
        return subdicts


class TupleSpin(Pane):
    def __init__(self, initial_value, key, range=None, value_list=None, steps=1000, decimals=3, **value_kws):
        w, h = w_kws['default_button_element_size']
        # size=(int(w/2), h)
        value_kws.update({'size': (w - 3, h)})
        self.steps = steps
        self.initial_value = initial_value
        v0, v1 = initial_value if type(initial_value) == tuple else ('', '')
        self.integer = True if all([type(v0) == int, type(v1) == int]) else False
        if range is not None:
            r0, r1 = range
            arange = fun.value_list(r0, r1, self.integer, steps, decimals)
            arange = [''] + arange
        else:
            arange = value_list
        self.key = key
        self.k0, self.k1 = [f'{key}_{i}' for i in [0, 1]]
        self.s0 = sg.Spin(values=arange, initial_value=v0, key=self.k0, **value_kws)
        self.s1 = sg.Spin(values=arange, initial_value=v1, key=self.k1, **value_kws)
        pane_list = [sg.Col([[self.s0, self.s1]])]
        super().__init__(pane_list=pane_list, key=self.key)

    def get(self):
        t0, t1 = self.s0.get(), self.s1.get()
        res = (t0, t1) if all([t not in ['', None, np.nan] for t in [t0, t1]]) else None
        return res

    def update(self, window, value):
        if value not in [None, '', (None, None), [None, None]]:
            v0, v1 = value
        else:
            v0, v1 = ['', '']
        window.Element(self.k0).Update(value=v0)
        window.Element(self.k1).Update(value=v1)

        # return window


class MultiSpin(Pane):
    def __init__(self, initial_value, value_list, key, steps=100, decimals=2, Nspins=4, tuples=False, **value_kws):
        w, h = w_kws['default_button_element_size']
        value_kws.update({'size': (w - 3, h)})
        self.value_kws = value_kws
        # b_kws={'size' : (1,1)}
        self.Nspins = Nspins
        self.steps = steps
        self.initial_value = initial_value
        self.value_list = [''] + value_list
        if initial_value is None:
            self.v_spins = [''] * Nspins
            self.N = 0
        elif type(initial_value) in [list, tuple]:
            self.N = len(initial_value)
            self.v_spins = [vv for vv in initial_value] + [''] * (Nspins - self.N)
        self.key = key
        self.tuples = tuples
        self.add_key, self.remove_key = f'SPIN+ {key}', f'SPIN- {key}'
        self.k_spins = [f'{key}_{i}' for i in range(Nspins)]
        self.visibles = [True] * (self.N + 1) + [True] * (Nspins - self.N - 1)
        self.spins = self.build_spins()
        self.layout = self.build_layout()

        add_button = graphic_button('Button_Add', self.add_key, tooltip='Add another item in the list.')
        remove_button = graphic_button('Button_Remove', self.remove_key, tooltip='Remove last item in the list.')
        self.buttons = sg.Col([[add_button], [remove_button]])
        pane_list = [sg.Col([self.layout])]
        super().__init__(pane_list=pane_list, key=self.key)

    def build_spins(self):
        if not self.tuples:
            spins = [sg.Spin(values=self.value_list, initial_value=vv, key=kk, visible=vis,
                             **self.value_kws) for vv, kk, vis in zip(self.v_spins, self.k_spins, self.visibles)]
        else:
            spins = [TupleSpin(value_list=self.value_list, initial_value=vv, key=kk,
                               **self.value_kws) for vv, kk, vis in zip(self.v_spins, self.k_spins, self.visibles)]
        return spins

    def get(self):
        vs = [s.get() for s in self.spins if s.get() not in [None, '']]
        vs = vs if len(vs) > 0 else None
        return vs

    def update(self, window, value):
        if value in [None, '', (None, None), [None, None]]:
            self.N = 0
            if not self.tuples:
                for i, k in enumerate(self.k_spins):
                    window.Element(k).Update(value='')
            else:
                for i, spin in enumerate(self.spins):
                    spin.update(window, '')

        elif type(value) in [list, tuple]:
            self.N = len(value)
            if not self.tuples:
                for i, k in enumerate(self.k_spins):
                    vv = value[i] if i < self.N else ''
                    window.Element(k).Update(value=vv, visible=True)
            else:
                for i, spin in enumerate(self.spins):
                    vv = value[i] if i < self.N else ''
                    spin.update(window, vv)

    def add_spin(self, window):
        # vs0=self.get()
        # if vs0 is not None :
        #
        #     self.update(window, vs0+[2])
        if self.N < self.Nspins:
            self.N += 1
            # window[self.k_spins[self.N]].update(value=None, visible=True)
            # window.Element(self.k_spins[1]).Update(visible=True)
            # window.Element(self.k_spins[self.N]).Update(visible=True)
            window.Element(self.k_spins[self.N]).Update(value='', visible=True)
            # window.Element(self.k_spins[self.N]).Update(visible=True)
        #     # window.refresh()
        #     vs0 = self.get()

    def remove_spin(self, window):
        # vs0 = self.get()
        # if vs0 is not None and len(vs0)>1:
        #     self.update(window, vs0[:-1])
        if self.N > 0:
            # window[self.k_spins[self.N]].update(value=None, visible=False)
            # window.Element(self.k_spins[self.N]).Update(visible=False)
            # window.Element(self.k_spins[self.N]).Update(value=None)
            window.Element(self.k_spins[self.N]).Update(value='', visible=False)
            # window.refresh()
            self.N -= 1

    def build_layout(self):
        if not self.tuples:
            return self.spins
        else:
            if self.Nspins > 3:
                spins = fun.group_list_by_n(self.spins, 2)
                spins = [sg.Col(spins)]
                return spins
            else:
                return self.spins


def import_window(datagroup_id, raw_folder, raw_dic, dirs_as_ids=True):
    M, E = 'Merge', 'Enumerate'
    E0 = f'{E}_id'
    proc_dir = {}
    N = len(raw_dic)
    raw_ids = list(raw_dic.keys())
    raw_dirs = list(raw_dic.values())
    raw_dirs = [fun.remove_prefix(dr, f'{raw_folder}/') for dr in raw_dirs]
    raw_dirs = [fun.remove_suffix(dr, f'/{id}') for dr, id in zip(raw_dirs, raw_ids)]
    raw_dirs = fun.unique_list(raw_dirs)
    groupID0 = raw_dirs[0] if len(raw_dirs) == 1 else ''
    # groupID0= else ''
    if N == 0:
        return proc_dir
    w_size = (1200, 800)
    h_kws = {
        'font': ('Helvetica', 8, 'bold'),
        'justification': 'center',
    }
    b_merged = named_bool_button(name=M, state=False, toggle_name=None)
    b_num = named_bool_button(name=E, state=False, toggle_name=None, disabled=False)
    # b_num = named_bool_button(name=E, state=False, toggle_name=None, input_key=E0, input_text='dish')
    group_id = [sg.T('Group ID :', **t_kws(8)), sg.In(default_text=groupID0, k='import_group_id', **t_kws(14))]
    l00 = sg.Col([[*group_id, *b_num, *b_merged],
                  [sg.T('RAW DATASETS', **h_kws, **t_kws(30)), sg.T('NEW DATASETS', **h_kws, **t_kws(30))]])
    l01 = sg.Col([
        [sg.T(id, **t_kws(30)), sg.T('  -->  ', **t_kws(8)), sg.In(default_text=id, k=f'new_{id}', **t_kws(30))] for id
        in
        list(raw_dic.keys())],
        vertical_scroll_only=True, scrollable=True, expand_y=True, vertical_alignment='top',
        size=col_size(y_frac=0.4, win_size=w_size))

    s1 = CollapsibleDict('build_conf', True, default=True, disp_name='Configuration', text_kws=t_kws(24),
                         value_kws=t_kws(8))
    c = {}
    for s in [s1]:
        c.update(**s.get_subdicts())
    l = [[sg.Col([
        [l00],
        [l01],
        s1.get_layout(),
        [sg.Col([[sg.Ok(), sg.Cancel()]], size=col_size(y_frac=0.2, win_size=w_size))],
    ])]]
    w = sg.Window('Build new datasets from raw files', l, size=w_size)
    while True:
        e, v = w.read()
        gID = v['import_group_id']

        if e in (None, 'Exit', 'Cancel'):
            w.close()
            break
        else:
            toggled = check_togglesNcollapsibles(w, e, v, c)
            merge = w[f'TOGGLE_{M}'].get_state()
            for i, (id, dir) in enumerate(raw_dic.items()):
                if i != 0:
                    w.Element(f'new_{id}').Update(visible=not merge)
            enum = w[f'TOGGLE_{E}'].get_state()
            if E in toggled:
                if not enum:
                    for i, (id, dir) in enumerate(raw_dic.items()):
                        w.Element(f'new_{id}').Update(value=id)
                else:
                    # v_enum = v['import_group_id']
                    # v_enum = v[E0]
                    for i, (id, dir) in enumerate(raw_dic.items()):
                        w.Element(f'new_{id}').Update(value=f'{gID}_{i}')
                        # w.Element(f'new_{id}').Update(value=f'{v_enum}_{i}')
            if e == 'Ok':
                conf = s1.get_dict(values=v, window=w)
                kws = {
                    'datagroup_id': datagroup_id,
                    'folders': None,
                    'group_ids': gID,
                    **conf}
                w.close()
                from lib.stor.managing import build_datasets
                if not merge:
                    print(f'------ Building {N} discrete datasets ------')
                    for id, dir in raw_dic.items():
                        new_id = v[f'new_{id}']
                        fdir = fun.remove_prefix(dir, f'{raw_folder}/')
                        if dirs_as_ids:
                            temp = fun.remove_suffix(fdir, f'{id}')
                            names = [f'{temp}{new_id}']
                        else:
                            names = [fdir]
                        dd = build_datasets(ids=[new_id], names=names, raw_folders=[fdir], **kws)[0]
                        if dd is not None:
                            proc_dir[new_id] = dd
                        else:
                            del proc_dir[new_id]

                else:
                    print(f'------ Building a single merged dataset ------')
                    id0 = f'{list(raw_dic.keys())[0]}'
                    fdir = [fun.remove_prefix(dir, f'{raw_folder}/') for dir in raw_dic.values()]
                    new_id = v[f'new_{id0}']
                    temp = fun.remove_suffix(fdir[0], id0)
                    names = [f'{temp}{new_id}']
                    dd = build_datasets(ids=[new_id], names=names, raw_folders=[fdir], **kws)[0]
                    proc_dir[dd.id] = dd
                break
    return proc_dir


def change_dataset_id(w, v, dic, k0):
    v0 = v[k0]
    k = 'NEW_ID'
    if len(v0) > 0:
        for i in range(len(v0)):
            old_id = v0[i]
            l = [[sg.Text('Enter new dataset ID', size=(20, 1)), sg.In(default_text=old_id, k=k, size=(10, 1))],
                 [sg.Button('Store'), sg.Ok(), sg.Cancel()]]
            e1, v1 = sg.Window('Change dataset ID', l).read(close=True)
            new_id = v1[k]
            if e1 == 'Ok':
                dic[new_id] = dic.pop(old_id)
                w.Element(k0).Update(values=list(dic.keys()))
            elif e1 == 'Store':
                d = dic[old_id]
                d.set_id(new_id)
                dic[new_id] = dic.pop(old_id)
                w.Element(k0).Update(values=list(dic.keys()))
    return dic


def named_list(text, key, choices, default_value=None, drop_down=True, list_width=20,
               readonly=True, enable_events=True, single_line=True, next_to_header=None, as_col=True,
               list_kws={}, list_height=None, header_text_kws=None):
    if list_height is None:
        list_height = len(choices)
    if header_text_kws is None:
        header_text_kws = {'size': (len(text), 1)}
    t = [sg.Text(text, **header_text_kws)]
    if next_to_header is not None:
        t += next_to_header
    if drop_down:
        l = [sg.Combo(choices, key=key, default_value=default_value,
                      size=(list_width, 1), enable_events=enable_events, readonly=readonly, **list_kws)]
    else:
        l = [sg.Listbox(choices, key=key, default_values=[default_value],
                        size=(list_width, list_height), enable_events=enable_events, **list_kws)]
    if single_line:
        return t + l
    else:
        if as_col:
            return sg.Col([t, l])
        else:
            return [t, l]


class GuiElement:
    def __init__(self, name, layout=None, layout_col_kwargs={}):
        self.name = name
        self.layout = layout
        self.layout_col_kwargs = layout_col_kwargs

    # def build_layout(self, **kwargs):
    #     self.layout=None
    # return l

    def get_layout(self, as_col=True, **kwargs):
        self.layout_col_kwargs.update(kwargs)
        return [sg.Col(self.layout, **self.layout_col_kwargs)] if as_col else self.layout


class DataList(GuiElement):
    def __init__(self, name, tab, dict={}, buttons=['select_all', 'remove', 'changeID', 'browse'], button_args={},
                 raw=False, named_list_kws={'list_kws': {'select_mode': LISTBOX_SELECT_MODE_EXTENDED}}, **kwargs):
        super().__init__(name=name)
        self.tab = tab
        self.dict = dict
        self.buttons = buttons
        self.button_args = button_args
        self.raw = raw
        self.named_list_kws = named_list_kws
        self.list_key = f'{self.name}_IDS'
        self.browse_key = f'BROWSE {self.name}'
        self.layout = self.build_layout(**kwargs)
        self.tab.datalists[self.name] = self

    def build_buttons(self):
        bl = []
        for n in self.buttons:
            if n in list(self.button_args.keys()):
                kws = self.button_args[n]
            else:
                kws = {}
            l = button_dict[n](self.name, **kws)
            bl.append(l)
        return bl

    def build_layout(self, **kwargs):
        bl = self.build_buttons()
        l = named_list(get_disp_name(self.name), self.list_key, list(self.dict.keys()),
                       drop_down=False, list_width=25, list_height=5,
                       single_line=False, next_to_header=bl, as_col=False,
                       **self.named_list_kws,
                       # list_kws={'select_mode': LISTBOX_SELECT_MODE_EXTENDED},
                       **kwargs)
        return l

    def update_window(self, w):
        w.Element(self.list_key).Update(values=list(self.dict.keys()))

    def eval(self, e, v, w, c, d, g):
        from lib.stor.managing import detect_dataset, enrich_datasets
        n = self.name
        k = self.list_key
        d0 = self.dict
        v0 = v[k]
        datagroup_id = self.tab.current_ID(v) if self.raw else None
        if e == self.browse_key:
            d0.update(detect_dataset(datagroup_id, v[self.browse_key], raw=self.raw))
            self.update_window(w)
        elif e == f'SELECT_ALL {n}':
            w.Element(k).Update(set_to_index=np.arange(len(d0)).tolist())
        elif e == f'REMOVE {n}':
            for i in range(len(v0)):
                d0.pop(v0[i], None)
            self.update_window(w)
        elif e == f'CHANGE_ID {n}':
            d0 = change_dataset_id(w, v, d0, k0=k)
        elif e == f'REPLAY {n}':
            if len(v0) > 0:
                dd = d0[v0[0]]
                dd.visualize(vis_kwargs=self.tab.gui.get_vis_kwargs(v, mode='video'),
                             **self.tab.gui.get_replay_kwargs(v))
        elif e == f'ADD REF {n}':
            from lib.stor.larva_dataset import LarvaDataset
            dd = LarvaDataset(dir=f'{paths.RefFolder}/reference')
            d0[dd.id] = dd
            self.update_window(w)
        elif e == f'BUILD {n}':
            dl1 = self.tab.datalists[self.tab.proc_key]
            d1 = dl1.dict
            k1 = dl1.list_key
            raw_dic = {id: dir for id, dir in d0.items() if id in v[k]}
            proc_dir = import_window(datagroup_id=datagroup_id, raw_folder=self.tab.raw_folder, raw_dic=raw_dic)
            d1.update(proc_dir)
            dl1.update_window(w)
        elif e == f'ENRICH {n}':
            dds = [dd for id, dd in d0.items() if id in v[k]]
            if len(dds) > 0:
                enrich_conf = c['enrichment'].get_dict(v, w)
                enrich_datasets(datagroup_id=datagroup_id, datasets=dds, enrich_conf=enrich_conf)
        self.dict = d0


class Collapsible(GuiElement):
    def __init__(self, name, state, content, disp_name=None, toggle=None, disabled=False, next_to_header=None,
                 auto_open=False, header_dict=None, header_value=None, header_list_width=10, header_list_kws={},
                 header_text_kws=t_kws(12), header_key=None, **kwargs):
        super().__init__(name=name)
        # self.name = name
        if disp_name is None:
            disp_name = get_disp_name(name)
        self.disp_name = disp_name
        self.state = state
        self.toggle = toggle
        self.auto_open = auto_open
        self.sec_key = f'SEC {self.name}'
        self.toggle_key = f'TOGGLE_{self.name}'
        self.sec_symbol = self.get_symbol()
        self.header_dict = header_dict
        self.header_value = header_value
        if header_key is None:
            header_key = f'SELECT LIST {name}'
        self.header_key = header_key
        if header_dict is None:
            header_disp = [sg.T(disp_name, enable_events=True, text_color='black', **header_text_kws)]
        else:
            header_disp = named_list(text=f'{disp_name}:', choices=list(header_dict.keys()),
                                     default_value=header_value, key=self.header_key,
                                     list_width=header_list_width, list_kws=header_list_kws,
                                     header_text_kws=header_text_kws)

        header = [self.sec_symbol] + header_disp
        if toggle is not None:
            header.append(BoolButton(name, toggle, disabled))
        if next_to_header is not None:
            header += next_to_header
        temp = collapse(content, self.sec_key, self.state)
        self.layout = [header, [temp]]

    def get_symbol(self):
        return sg.T(SYMBOL_DOWN if self.state else SYMBOL_UP, k=f'OPEN {self.sec_key}',
                    enable_events=True, text_color='black', **t_kws(2))

    def update(self, window, dict, use_prefix=True):
        if dict is None:
            self.disable(window)
        elif self.toggle == False:
            self.disable(window)
        else:
            self.enable(window)
            prefix = self.name if use_prefix else None
            update_window_from_dict(window, dict, prefix=prefix)
        return window

    def update_header(self, w, id):
        self.header_value = id
        w.Element(self.header_key).Update(value=id)
        self.update(w, self.header_dict[id])

    def click(self, w):
        if self.state is not None:
            self.state = not self.state
            self.sec_symbol.update(SYMBOL_DOWN if self.state else SYMBOL_UP)
            # self.content.update(visible=self.state)
            w[self.sec_key].update(visible=self.state)

    def disable(self, w):
        if self.toggle is not None:
            w[self.toggle_key].set_state(state=False, disabled=True)
        self.close(w)
        self.state = None

    def enable(self, w):
        if self.toggle is not None:
            w[self.toggle_key].set_state(state=True, disabled=False)
        if self.auto_open:
            self.open(w)
        elif self.state is None:
            self.state = False

    def open(self, w):
        self.state = True
        self.sec_symbol.update(SYMBOL_DOWN)
        w[self.sec_key].update(visible=self.state)

    def close(self, w):
        self.state = False
        self.sec_symbol.update(SYMBOL_UP)
        w[self.sec_key].update(visible=False)

    def get_subdicts(self):
        subdicts = {}
        subdicts[self.name] = self
        return subdicts


class CollapsibleTable(Collapsible):
    def __init__(self, name, state, type_dict, headings, dict={}, **kwargs):
        self.dict = dict
        self.type_dict = type_dict
        if 'unique_id' in list(type_dict.keys()):
            self.header = 'unique_id'
        elif 'group' in list(type_dict.keys()):
            self.header = 'group'
        else:
            self.header = None
        self.headings = headings
        self.Ncols = len(headings)
        self.col_widths = []
        self.col_visible = [True] * self.Ncols
        self.color_idx = None
        for i, p in enumerate(self.headings):
            if p in ['id', 'group']:
                self.col_widths.append(10)
            elif p in ['color']:
                self.col_widths.append(8)
                self.color_idx = i
                self.col_visible[i] = False
            elif p in ['model']:
                self.col_widths.append(14)
            elif type_dict[p] in [int, float]:
                self.col_widths.append(np.max([len(p), 6]))
            else:
                self.col_widths.append(10)

        self.data = self.set_data(dict)
        self.key = f'TABLE {name}'
        content = self.get_content()
        self.edit_key = f'EDIT_TABLE {name}'
        b = [graphic_button('Document_2_Edit', self.edit_key, tooltip=f'Create new {name}')]
        super().__init__(name, state, content=content, next_to_header=b, **kwargs)

    def set_data(self, dic):
        if dic is not None and len(dic) != 0:
            if self.header is not None:
                data = []
                for id, pars in dic.items():
                    row = [id]
                    for j, p in enumerate(self.headings[1:]):
                        for k, v in pars.items():
                            if k == 'default_color' and p == 'color':
                                row.append(v)
                            elif k == p:
                                row.append(v)
                    data.append(row)
            else:
                dic2 = {k: dic[k] for k in self.headings}
                l = list(dic2.values())
                N = len(l[0])
                data = [[j[i] for j in l] for i in range(N)]
        else:
            data = [[''] * self.Ncols]
        return data

    def get_content(self):
        content = [[sg.Table(values=self.data[:][:], headings=self.headings, col_widths=self.col_widths,
                             max_col_width=30, background_color='lightblue', header_font=('Helvetica', 8, 'bold'),
                             auto_size_columns=False,
                             visible_column_map=self.col_visible,
                             # display_row_numbers=True,
                             justification='center',
                             font=w_kws['font'],
                             num_rows=len(self.data),
                             alternating_row_color='lightyellow',
                             key=self.key
                             )]]
        return content

    def update(self, window, dic, use_prefix=True):
        self.dict = dic
        self.data = self.set_data(dic)
        if self.color_idx is not None:
            row_cols = []
            for i in range(len(self.data)):
                c0 = self.data[i][self.color_idx]
                if c0 == '':
                    c2, c1 = ['lightblue', 'black']
                else:
                    try:
                        c2, c1 = fun.invert_color(c0, return_self=True)
                    except:
                        c2, c1 = ['lightblue', 'black']
                        # c2, c1 = [c0, 'black']
                row_cols.append((i, c1, c2))
        else:
            row_cols = None
        window[self.key].update(values=self.data, num_rows=len(self.data), row_colors=row_cols)
        self.open(window) if self.data[0][0] != '' else self.close(window)

    def edit_table(self, window):
        if self.header is not None:
            dic = set_agent_dict(self.dict, self.type_dict, header=self.header, title=self.disp_name)
            self.update(window, dic)
        else:
            t0 = [dict(zip(self.headings, l)) for l in self.data] if self.data != [[''] * self.Ncols] else []
            t1 = gui_table(t0, self.type_dict, title='Parameter space')
            if t1 != t0:
                dic = {k: [l[k] for l in t1] for k in self.headings}
                self.update(window, dic)

    def get_dict(self, *args, **kwargs):
        return self.dict


class CollapsibleDict(Collapsible):
    def __init__(self, name, state, dict=None, dict_name=None, type_dict=None, toggled_subsections=True, default=False,
                 text_kws={}, value_kws={}, **kwargs):
        if dict_name is None:
            dict_name = name
        self.dict_name = dict_name
        if default and dict is None and type_dict is None:
            dict = dtypes.get_dict(self.dict_name)
            type_dict = dtypes.get_dict_dtypes(self.dict_name)
        self.sectiondict = SectionDict(name=name, dict=dict, type_dict=type_dict,
                                       toggled_subsections=toggled_subsections)
        content = self.sectiondict.init_section(text_kws=text_kws, value_kws=value_kws)
        super().__init__(name, state, content, **kwargs)

    def get_dict(self, values, window, check_toggle=True):
        if self.state is None:
            return None
        elif check_toggle and self.toggle == False:
            return None
        else:
            return self.sectiondict.get_dict(values, window)

    def get_subdicts(self):
        subdicts = {}
        # subdicts[self.dict_name] = self
        subdicts[self.name] = self
        all_subdicts = {**subdicts, **self.sectiondict.get_subdicts()}
        return all_subdicts


class Table(sg.Table):
    def add_row(self, window, row, sort_idx=None):
        vs = self.get()
        vs.append(row)
        if sort_idx is not None:
            vs.sort(key=lambda x: x[sort_idx])
        # self.epochs.append([t1, t2, q])
        # self.epochs.sort(key=lambda x: x[0])
        window.Element(self.Key).Update(values=vs, num_rows=len(vs))

    def remove_row(self, window, idx):
        vs = self.get()
        vs.remove(vs[idx])
        window.Element(self.Key).Update(values=vs, num_rows=len(vs))


def collapse(layout, key, visible=True):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this seciton visible / invisible
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Col(layout, key=key, visible=visible))


def set_kwargs(dic, title='Arguments', type_dict=None, **kwargs):
    sec_dict = SectionDict(name=title, dict=dic, type_dict=type_dict)
    l = sec_dict.init_section()
    l.append([sg.Ok(), sg.Cancel()])
    w = sg.Window(title, l, **w_kws, **kwargs)
    while True:
        e, v = w.read()
        if e == 'Ok':
            new_dic = sec_dict.get_dict(v, w)
            break
        elif e in ['Cancel', None]:
            new_dic = dic
            break
        check_toggles(w, e)
    w.close()
    del sec_dict
    return new_dic

    # if kwargs != {}:
    #     layout = []
    #     for i, (k, v) in enumerate(kwargs.items()):
    #         if not mode(v) == dict and not mode(v) == np.ndarray:
    #             layout.append([sg.Text(k, size=(20, 1)), sg.Input(default_text=str(v), k=f'KW_{i}', size=(20, 1))])
    #     layout.append([sg.Ok(), sg.Cancel()])
    #     event, values = sg.Window(title, layout).read(close=True)
    #     if event == 'Ok':
    #         for i, (k, v) in enumerate(kwargs.items()):
    #             if mode(v) == np.ndarray:
    #                 continue
    #             if not mode(v) == dict:
    #                 vv = values[f'KW_{i}']
    #                 if mode(v) == bool:
    #                     if vv == 'False':
    #                         vv = False
    #                     elif vv == 'True':
    #                         vv = True
    #                 elif mode(v) == list or mode(v) == tuple:
    #                     vv = literal_eval(vv)
    #
    #                 elif v is None:
    #                     if vv == 'None':
    #                         vv = None
    #                     else:
    #                         vv = vv
    #                 else:
    #                     vv = mode(v)(vv)
    #                 kwargs[k] = vv
    #             else:
    #                 kwargs[k] = set_kwargs(v, title=k)
    #
    # return kwargs


def set_agent_kwargs(agent, **kwargs):
    class_name = type(agent).__name__
    type_dict = dtypes.get_dict_dtypes(class_name)
    title = f'{class_name} args'
    dic = {}
    for p in list(type_dict.keys()):
        dic[p] = getattr(agent, p)
    new_dic = set_kwargs(dic, title, type_dict=type_dict, **kwargs)
    for p, v in new_dic.items():
        if p == 'unique_id':
            agent.set_id(v)
        else:
            setattr(agent, p, v)
    return agent


def object_menu(selected, **kwargs):
    object_list = ['', 'Larva', 'Food', 'Border']
    title = 'Select object_class mode'
    layout = [
        [sg.Text(title)],
        [sg.Listbox(default_values=[selected], values=object_list, change_submits=False, size=(20, len(object_list)),
                    key='SELECTED_OBJECT',
                    enable_events=True)],
        [sg.Ok(), sg.Cancel()]]
    window = sg.Window(title, layout, **kwargs)
    while True:
        event, values = window.read()
        if event == 'Ok':
            sel = values['SELECTED_OBJECT'][0]
            break
        elif event in (None, 'Cancel'):
            sel = selected
            break
    window.close()
    return sel


class GraphList(GuiElement):
    def __init__(self, name, fig_dict={}, next_to_header=None, default_values=None, canvas_size=(1000, 800),
                 list_size=None, list_header='Graphs', auto_eval=True, canvas_kws={'background_color': 'Lightblue'},
                 graph=False, subsample=1,
                 canvas_col_kws={'scrollable': False, 'vertical_scroll_only': False, 'expand_y': True,
                                 'expand_x': True}):
        super().__init__(name=name)
        self.subsample = subsample
        self.auto_eval = auto_eval
        self.list_size = list_size
        self.list_header = list_header
        self.canvas_size = canvas_size
        # self.name = name
        self.next_to_header = next_to_header
        self.fig_dict = fig_dict
        self.layout, self.list_key = self.init_layout(name, fig_dict, default_values)
        self.canvas, self.canvas_key, self.canvas_element = self.init_canvas(name, canvas_kws, canvas_col_kws, graph)
        self.fig_agg = None
        self.draw_key = 'unreachable'

    def init_layout(self, name, fig_dict, default_values):
        list_key = f'{name}_GRAPH_LIST'
        values = list(fig_dict.keys())
        h = int(np.max([len(values), 10]))

        header = [sg.T(self.list_header, **t_kws(10))]
        if self.next_to_header is not None:
            header += self.next_to_header
        if self.list_size is None:
            h = int(np.max([len(values), 10]))
            self.list_size = (25, h)
        l = [header, [sg.Listbox(values=values, default_values=default_values, change_submits=True, size=self.list_size,
                                 key=list_key, auto_size_text=True)]]
        return l, list_key

    def init_canvas(self, name, canvas_kws, canvas_col_kws, graph=False):
        canvas_key = f'{name}_CANVAS'
        kws = {
            # 'size': self.canvas_size,
            'key': canvas_key,
            **canvas_kws}
        if graph:
            g = sg.Graph(canvas_size=self.canvas_size, **kws)
        else:
            g = sg.Canvas(size=self.canvas_size, **kws)
        canvas = GuiElement(name=canvas_key, layout=[[g]], layout_col_kwargs=canvas_col_kws)
        return canvas, canvas_key, g

    def draw_fig(self, w, fig):
        if self.fig_agg:
            delete_figure_agg(self.fig_agg)
        self.fig_agg = draw_canvas(w[self.canvas_key].TKCanvas, fig)

    def update(self, w, fig_dict):
        self.fig_dict = fig_dict
        w.Element(self.list_key).Update(values=list(fig_dict.keys()))

    def evaluate(self, w, list_values):
        if len(list_values) > 0 and self.auto_eval:
            choice = list_values[0]
            fig = self.fig_dict[choice]
            if type(fig) == str and os.path.isfile(fig):
                self.show_fig(w, fig)
            else:
                self.draw_fig(w, fig)

    def show_fig(self, w, fig):
        c = w[self.canvas_key].TKCanvas
        c.pack()
        img = PhotoImage(file=fig)
        img = img.subsample(self.subsample)
        W, H = self.canvas_size
        c.create_image(int(W / 2), int(H / 2), image=img)
        # c.create_image(250, 250, image=img)
        self.fig_agg = img


class ButtonGraphList(GraphList):
    def __init__(self, name, **kwargs):
        self.draw_key = f'{name}_DRAW_FIG'
        l = [
            graphic_button('Button_Load', f'{name}_REFRESH_FIGS', tooltip='Detect available graphs.'),
            graphic_button('System_Equalizer', f'{name}_FIG_ARGS', tooltip='Configure the graph arguments.'),
            # graphic_button('preferences', f'{self.name}_SAVEd_FIG'),
            graphic_button('Chart', self.draw_key, tooltip='Draw the graph.'),
            graphic_button('File_Add', f'{name}_SAVE_FIG', tooltip='Save the graph to a file.')
        ]
        super().__init__(name=name, next_to_header=l, **kwargs)

        self.fig, self.save_to, self.save_as = None, '', ''
        self.func, self.func_kws = None, {}

    def evaluate(self, w, list_values):
        if len(list_values) > 0:
            choice = list_values[0]
            if self.fig_dict[choice] != self.func:
                self.func = self.fig_dict[choice]
                self.func_kws = self.get_graph_kws(self.func)

    def get_graph_kws(self, func):
        signature = inspect.getfullargspec(func)
        vs = signature.defaults
        if vs is None:
            return {}
        kws = dict(zip(signature.args[-len(vs):], vs))
        for k in ['datasets', 'labels', 'save_to', 'save_as', 'return_fig', 'deb_dicts']:
            if k in kws.keys():
                del kws[k]
        return kws

    def generate(self, w, data):
        if self.func is not None and len(list(data.keys())) > 0:
            try:
                self.fig, self.save_to, self.save_as = self.func(datasets=list(data.values()), labels=list(data.keys()),
                                                                 return_fig=True, **self.func_kws)
                self.draw_fig(w, self.fig)
            except:
                print('Plot not available')

    def refresh_figs(self, w, data):
        k = self.list_key
        w.Element(k).Update(values=list(self.fig_dict.keys()))
        if len(data) > 0:
            valid = []
            for i, (name, func) in enumerate(self.fig_dict.items()):
                w.Element(k).Update(set_to_index=i)
                w.refresh()
                try:
                    fig, save_to, save_as = func(datasets=list(data.values()), labels=list(data.keys()),
                                                 return_fig=True, **self.get_graph_kws(func))
                    valid.append(name)
                except:
                    pass
            w.Element(k).Update(values=valid, set_to_index=None)

    def save_fig(self):
        kDir, kFil = 'SAVE_AS', 'SAVE_TO'
        if self.fig is not None:
            l = [
                [sg.T('Filename', **t_kws(10)), sg.In(default_text=self.save_as, k=kDir, **t_kws(80))],
                [sg.T('Directory', **t_kws(10)), sg.In(self.save_to, k=kFil, **t_kws(80)),
                 sg.FolderBrowse(initial_folder=paths.get_parent_dir(), key=kFil, change_submits=True)],
                [sg.Ok(), sg.Cancel()]]

            e, v = sg.Window('Save figure', l).read(close=True)
            if e == 'Ok':
                path = os.path.join(v[kFil], v[kDir])
                self.fig.savefig(path, dpi=300)
                print(f'Plot saved as {v[kDir]}')

    def set_fig_args(self):
        self.func_kws = set_kwargs(self.func_kws, title='Graph arguments')


def delete_objects_window(selected):
    ids = [sel.unique_id for sel in selected]
    title = 'Delete objects?'
    l = [
        [sg.T(title)],
        [sg.Listbox(default_values=ids, values=ids, size=(20, len(ids)), k='DELETE_OBJECTS', enable_events=True)],
        [sg.Ok(), sg.Cancel()]]
    w = sg.Window(title, l)
    while True:
        e, v = w.read()
        if e == 'Ok':
            res = True
            break
        elif e == 'Cancel':
            res = False
            break
    w.close()
    return res


def draw_canvas(canvas, figure, side='top', fill='both', expand=True):
    agg = FigureCanvasTkAgg(figure, canvas)
    agg.draw()
    agg.get_tk_widget().pack(side=side, fill=fill, expand=expand)
    return agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


class DynamicGraph:
    def __init__(self, agent, pars=[], available_pars=None):
        sg.theme('DarkBlue15')
        self.agent = agent
        if available_pars is None:
            available_pars = runtime_pars
        self.available_pars = available_pars
        self.pars = pars
        self.dt = self.agent.model.dt
        self.init_dur = 20
        W, H = (1550, 1000)
        Wc, Hc = self.canvas_size = (W - 50, H - 200)
        self.my_dpi = 96
        self.figsize = (int(Wc / self.my_dpi), int(Hc / self.my_dpi))
        self.layout = self.build_layout()

        self.window = sg.Window(f'{self.agent.unique_id} Dynamic Graph', self.layout, finalize=True, location=(0, 0),
                                size=(W, H))
        self.canvas_elem = self.window.FindElement('-CANVAS-')
        self.canvas = self.canvas_elem.TKCanvas
        self.fig_agg = None

        self.cur_layout = 1

    def build_layout(self):
        Ncols = 4
        par_lists = [list(a) for a in np.array_split(self.available_pars, Ncols)]
        l0 = [[sg.T('Choose parameters')],
              [sg.Col([*[[sg.CB(p, k=f'k_{p}', **t_kws(24))] for p in par_lists[i]]]) for i in range(Ncols)],
              [sg.B('Ok', **t_kws(8)), sg.B('Cancel', **t_kws(8))]]

        l1 = [
            [sg.Canvas(size=(1280, 1200), k='-CANVAS-')],
            [sg.T('Time in seconds to display on screen')],
            [sg.Slider(range=(0.1, 60), default_value=self.init_dur, size=(40, 10), orientation='h',
                       k='-SLIDER-TIME-')],
            [sg.B('Choose', **t_kws(8))]]
        l = [[sg.Col(l0, k='-COL1-'), sg.Col(l1, visible=False, k='-COL2-')]]
        return l

    def evaluate(self):
        e, v = self.window.read(timeout=0)
        if e is None:
            self.window.close()
            return False
        elif e == 'Choose':
            self.window[f'-COL2-'].update(visible=False)
            self.window[f'-COL1-'].update(visible=True)
            self.cur_layout = 1
        elif e == 'Ok':
            self.window[f'-COL1-'].update(visible=False)
            self.window[f'-COL2-'].update(visible=True)
            self.pars = [p for p in self.available_pars if v[f'k_{p}']]
            self.update_pars()
            self.cur_layout = 2
        elif e == 'Cancel':
            self.window[f'-COL1-'].update(visible=False)
            self.window[f'-COL2-'].update(visible=True)
            self.cur_layout = 2

        if self.cur_layout == 2 and self.Npars > 0:
            secs = v['-SLIDER-TIME-']
            Nticks = int(secs / self.dt)  # draw this many data points (on next line)
            t = self.agent.model.Nticks * self.dt
            trange = np.linspace(t - secs, t, Nticks)
            ys = self.update(Nticks)
            for ax, y in zip(self.axs, ys):
                ax.lines.pop(0)
                ax.plot(trange, y, color='black')
            self.axs[-1].set_xlim(np.min(trange), np.max(trange))
            self.fig_agg.draw()
        return True

    def update(self, Nticks):
        y_nan = np.ones(Nticks) * np.nan
        ys = []
        for p, v in self.yranges.items():
            self.yranges[p] = np.append(v, getattr(self.agent, p))
            dif = self.yranges[p].shape[0] - Nticks
            if dif >= 0:
                y = self.yranges[p][-Nticks:]
            else:
                y = y_nan
                y[-dif:] = self.yranges[p]
            ys.append(y)
        return ys

    def update_pars(self):
        self.pars, syms, us, lims, pcs = getPar(d=self.pars, to_return=['d', 's', 'l', 'lim', 'p'])
        self.Npars = len(self.pars)
        self.yranges = {}

        self.fig, axs = plt.subplots(self.Npars, 1, figsize=self.figsize, dpi=self.my_dpi, sharex=True)
        self.axs = axs.ravel() if self.Npars > 1 else [axs]
        Nticks = int(self.init_dur / self.dt)
        for i, (ax, p, l, u, lim, p_col) in enumerate(zip(self.axs, self.pars, syms, us, lims, pcs)):
            p0 = p_col if hasattr(self.agent, p_col) else p
            self.yranges[p0] = np.ones(Nticks) * np.nan
            ax.grid()
            ax.plot(range(Nticks), self.yranges[p0], color='black', label=l)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.legend(loc='upper right')
            ax.set_ylabel(u, fontsize=10)
            if lim is not None:
                ax.set_ylim(lim)
            ax.tick_params(axis='y', which='major', labelsize=10)
            if i == self.Npars - 1:
                ax.set_xlabel('time, $sec$')
                ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        self.fig.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.99, wspace=0.01, hspace=0.05)
        if self.fig_agg:
            delete_figure_agg(self.fig_agg)
        self.fig_agg = draw_canvas(self.canvas, self.fig)


def check_multispins(w, e):
    if e.startswith('SPIN+'):
        k = e.split()[-1]
        # w[k].add_spin(w)
        w.Element(k).add_spin(w)
    elif e.startswith('SPIN-'):
        k = e.split()[-1]
        # w[k].remove_spin(w)
        w.Element(k).remove_spin(w)


def check_collapsibles(w, e, v, c):
    if e.startswith('OPEN SEC'):
        sec = e.split()[-1]
        c[sec].click(w)
    elif e.startswith('SELECT LIST'):
        sec = e.split()[-1]
        c[sec].update_header(w, v[e])


def check_toggles(w, e):
    if 'TOGGLE' in e:
        w[e].toggle()


def check_togglesNcollapsibles(w, e, v, c):
    toggled = []
    check_collapsibles(w, e, v, c)
    if 'TOGGLE' in e:
        w[e].toggle()
        name = e[7:]
        if name in list(c.keys()):
            c[name].toggle = not c[name].toggle
        toggled.append(name)
    return toggled


def default_run_window(w, e, v, c={}, g={}):
    check_togglesNcollapsibles(w, e, v, c)
    check_multispins(w, e)
    for name, graph_list in g.items():
        if e == graph_list.list_key:
            graph_list.evaluate(w, v[graph_list.list_key])

    if e.startswith('EDIT_TABLE'):
        c[e.split()[-1]].edit_table(w)


def load_shortcuts():
    try:
        conf = loadConfDict('Settings')
    except:
        conf = {'keys': {}, 'pygame_keys': {}}
        for title, dic in lib.conf.init_dtypes.keyboard_controls.items():
            conf['keys'].update(dic)
        conf['pygame_keys'] = {k: dtypes.get_pygame_key(v) for k, v in conf['keys'].items()}
    return conf


def gui_terminal(size=col_size(y_frac=0.3)):
    return sg.Output(size=size, key='Terminal', background_color='black', text_color='white',
                     echo_stdout_stderr=True, font=('Helvetica', 8, 'normal'),
                     tooltip='Terminal output')


class SelectionList(GuiElement):
    def __init__(self, tab, conftype=None, disp=None, actions=[], sublists={}, idx=None, progress=False, width=24,
                 with_dict=False, name=None, **kwargs):
        self.conftype = conftype if conftype is not None else tab.conftype
        if name is None:
            name = self.conftype
        super().__init__(name=name)
        self.with_dict = with_dict
        self.width = width
        self.tab = tab
        self.actions = actions

        if disp is None:
            disps = [k for k, v in self.tab.gui.tab_dict.items() if v[1] == self.conftype]
            if len(disps) == 1:
                disp = disps[0]
            elif len(disps) > 1:
                raise ValueError('Each selectionList is associated with a single configuration type')
        self.disp = disp

        if not progress:
            self.progressbar = None
        else:
            self.progressbar = ProgressBarLayout(self)
        self.k0 = f'{self.conftype}_CONF'
        if idx is not None:
            self.k = f'{self.k0}{idx}'
        else:
            self.k = self.get_next(self.k0)

        self.layout = self.build(**kwargs)
        self.sublists = sublists
        self.tab.selectionlists[self.conftype] = self

    def c(self):
        return self.tab.gui.collapsibles

    def d(self):
        return self.tab.gui.dicts

    def g(self):
        return self.tab.gui.graph_lists

    def set_g(self, g):
        self.tab.gui.graph_lists = g

    def set_d(self, d):
        self.tab.gui.dicts = d

    def build(self, append=[], **kwargs):

        acts = self.actions
        n = self.disp
        bs = []
        if self.progressbar is not None:
            append += self.progressbar.l

        if 'load' in acts:
            bs.append(graphic_button('Button_Load', f'LOAD_{n}', tooltip=f'Load the configuration for a {n}.'))
        if 'edit' in acts:
            bs.append(
                graphic_button('Document_2_Edit', f'EDIT_{n}', tooltip=f'Configure an existing or create a new {n}.')),
        if 'save' in acts:
            bs.append(graphic_button('Document_2_Add', f'SAVE_{n}', tooltip=f'Save a new {n} configuration.'))
        if 'delete' in acts:
            bs.append(graphic_button('Document_2_Remove', f'DELETE_{n}',
                                     tooltip=f'Delete an existing {n} configuration.'))
        if 'run' in acts:
            bs.append(graphic_button('Button_Play', f'RUN_{n}', tooltip=f'Run the selected {n}.'))
        if 'search' in acts:
            bs.append(graphic_button('Search_Add', f'SEARCH_{n}', initial_folder=paths.DataFolder, change_submits=True,
                                     enable_events=True, target=(0, -1), button_type=sg.BUTTON_TYPE_BROWSE_FOLDER,
                                     tooltip='Browse to add datasets to the list.\n Either directly select a dataset directory or a parent directory containing multiple datasets.'))

        if self.with_dict:
            nn = self.tab.gui.tab_dict[n][2]
            self.collapsible = CollapsibleDict(n, True, dict=dtypes.get_dict(nn), type_dict=dtypes.get_dict_dtypes(nn),
                                               header_list_width=self.width, header_dict=loadConfDict(self.conftype),
                                               next_to_header=bs, header_key=self.k,
                                               header_list_kws={'tooltip': f'The currently loaded {n}.'}, **kwargs)

            temp = self.collapsible.get_layout(as_col=False)

        else:
            self.collapsible = None
            temp = named_list(text=n.capitalize(), key=self.k, choices=self.confs, default_value=None,
                              drop_down=True, list_width=self.width, single_line=False, next_to_header=bs, as_col=False,
                              list_kws={'tooltip': f'The currently loaded {n}.'})

        if self.progressbar is not None:
            temp.append(self.progressbar.l)
        return temp
        # l = [sg.Col(temp)]
        # return l

    def eval(self, e, v, w, c, d, g):
        n = self.disp
        id = v[self.k]
        k0 = self.conftype

        if e == f'LOAD_{n}' and id != '':
            conf = loadConf(id, k0)
            self.tab.update(w, c, conf, id)
            if self.progressbar is not None:
                self.progressbar.reset(w)
            for kk, vv in self.sublists.items():
                vv.update(w, conf[kk])

        elif e == f'SAVE_{n}':
            conf = self.tab.get(w, v, c, as_entry=True)
            for kk, vv in self.sublists.items():
                conf[kk] = v[vv.k]
            id = self.save(conf)
            if id is not None:
                self.update(w, id)
        elif e == f'DELETE_{n}' and id != '':
            deleteConf(id, k0)
            self.update(w)
        elif e == f'RUN_{n}' and id != '':
            conf = self.tab.get(w, v, c, as_entry=False)
            for kk, vv in self.sublists.items():
                if not vv.with_dict :
                    conf[kk] = expandConf(id=v[vv.k], conf_type=vv.conftype)
                else :
                    conf[kk] = vv.collapsible.get_dict(v,w)
            d, g = self.tab.run(v, w, c, d, g, conf, id)
            self.set_d(d)
            self.set_g(g)
        elif e == f'EDIT_{n}':
            conf = self.tab.get(w, v, c, as_entry=False)
            new_conf = self.tab.edit(conf)
            self.tab.update(w, c, new_conf, id=None)
        elif self.collapsible is not None and e == self.collapsible.header_key:
            self.collapsible.update_header(w, id)

    def update(self, w, id='', all=False):
        w.Element(self.k).Update(values=self.confs, value=id, size=(self.width, self.Nconfs))
        if self.collapsible is not None:
            self.collapsible.update_header(w, id)
        if all:
            for i in range(5):
                k = f'{self.k0}{i}'
                if k in w.AllKeysDict.keys():
                    w[k].update(values=self.confs, value=id)

    def save(self, conf):
        return save_conf_window(conf, self.conftype, disp=self.disp)

        # for i in range(3):
        #     k = f'{self.conf_k}{i}'
        #     w.Element(k, silent_on_error=True).Update(values=list(loadConfDict(k).keys()),value=id)

    def get_next(self, k0):
        w = self.tab.gui.window if hasattr(self.tab.gui, 'window') else None
        idx = int(np.min([i for i in range(5) if f'{k0}{i}' not in w.AllKeysDict.keys()])) if w is not None else 0
        return f'{k0}{idx}'

    def get_subdicts(self):
        if self.collapsible is not None:
            return self.collapsible.get_subdicts()
        else:
            return {}

    @property
    def confs(self):
        return list(loadConfDict(self.conftype).keys())

    @property
    def Nconfs(self):
        return len(self.confs)


class ProgressBarLayout:
    def __init__(self, list):
        self.list = list
        n = self.list.disp
        self.k = f'{n}_PROGRESSBAR'
        self.k_complete = f'{n}_COMPLETE'
        self.l = [sg.Text('Progress :', **t_kws(8)),
                  sg.ProgressBar(100, orientation='h', size=(8.8, 20), key=self.k,
                                 bar_color=('green', 'lightgrey'), border_width=3),
                  graphic_button('Button_Check', self.k_complete, visible=False,
                                 tooltip='Whether the current {n} was completed.')]

    def reset(self, w):
        w[self.k].update(0)
        w[self.k_complete].update(visible=False)

    def run(self, w, min=0, max=100):
        w[self.k_complete].update(visible=False)
        w[self.k].update(0, max=max)
