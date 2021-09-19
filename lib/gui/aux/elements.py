import copy
import inspect
import os
from tkinter import PhotoImage
from typing import Tuple, List
import numpy as np
import PySimpleGUI as sg

from PySimpleGUI import Pane, LISTBOX_SELECT_MODE_EXTENDED
from matplotlib import ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from lib.conf.conf import loadConfDict, deleteConf, loadConf, expandConf
import lib.aux.functions as fun
from lib.conf.par import runtime_pars, getPar
from lib.gui.aux.functions import SYMBOL_UP, SYMBOL_DOWN, w_kws, t_kws, get_disp_name, retrieve_value, collapse
from lib.gui.aux.buttons import named_bool_button, BoolButton, GraphButton, button_row
from lib.gui.aux.windows import gui_table, set_kwargs, save_conf_window, import_window, change_dataset_id

from lib.stor import paths as paths
import lib.conf.dtype_dicts as dtypes


class SectionDict:
    def __init__(self, name, dict, type_dict=None, toggled_subsections=True):
        self.init_dict = dict
        self.type_dict = type_dict
        self.toggled_subsections = toggled_subsections
        self.name = name
        self.subdicts = {}

    def init_section(self, value_kws={}):
        d = self.type_dict
        l = []
        for k, v in self.init_dict.items():
            k_disp = get_disp_name(k)
            k0 = f'{self.name}_{k}'
            if type(v) == bool:
                ii = named_bool_button(k_disp, v, k0)
            elif type(v) == dict:
                type_dict = d[k] if d is not None else None
                self.subdicts[k0] = CollapsibleDict(k0, disp_name=k, dict=v, type_dict=type_dict,
                                                    toggle=self.toggled_subsections,
                                                    toggled_subsections=self.toggled_subsections)
                ii = self.subdicts[k0].get_layout()
            else :
                temp = sg.In(v, key=k0, **value_kws)
                if d is not None:
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
                    # else:

                ii = [sg.Text(f'{k_disp}:'), temp]
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
        for k, v0 in d.items():
            k0 = f'{self.name}_{k}'
            t = self.get_type(k, v0)
            if t == bool or type(v0) == bool:
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

        add_button = GraphButton('Button_Add', self.add_key, tooltip='Add another item in the list.')
        remove_button = GraphButton('Button_Remove', self.remove_key, tooltip='Remove last item in the list.')
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


class GuiElement:
    def __init__(self, name, layout=None, layout_col_kwargs={}):
        self.name = name
        self.layout = layout
        self.layout_col_kwargs = layout_col_kwargs

    def get_layout(self, as_col=True, **kwargs):
        self.layout_col_kwargs.update(kwargs)
        return [sg.Col(self.layout, **self.layout_col_kwargs)] if as_col else self.layout




class ProgressBarLayout:
    def __init__(self, list):
        self.list = list
        n = self.list.disp
        self.k = f'{n}_PROGRESSBAR'
        self.k_complete = f'{n}_COMPLETE'
        self.l = [sg.Text('Progress :', **t_kws(8)),
                  sg.ProgressBar(100, orientation='h', size=(8.8, 20), key=self.k,
                                 bar_color=('green', 'lightgrey'), border_width=3),
                  GraphButton('Button_Check', self.k_complete, visible=False,
                                 tooltip='Whether the current {n} was completed.')]

    def reset(self, w):
        w[self.k].update(0)
        w[self.k_complete].update(visible=False)

    def run(self, w, min=0, max=100):
        w[self.k_complete].update(visible=False)
        w[self.k].update(0, max=max)

class HeadedElement(GuiElement):
    def __init__(self, name, header, content=[], single_line=True):
        layout = [header + content] if single_line else [header, content]
        super().__init__(name=name, layout=layout)


class SelectionList(GuiElement):
    def __init__(self, tab, conftype=None, disp=None, buttons=[], sublists={}, idx=None, progress=False, width=24,
                 with_dict=False, name=None, **kwargs):
        self.conftype = conftype if conftype is not None else tab.conftype
        if name is None:
            name = self.conftype
        super().__init__(name=name)
        self.with_dict = with_dict
        self.width = width
        self.tab = tab

        if disp is None:
            disps = [k for k, v in self.tab.gui.tab_dict.items() if v[1] == self.conftype]
            if len(disps) == 1:
                disp = disps[0]
            elif len(disps) > 1:
                raise ValueError('Each selectionList is associated with a single configuration type')
        self.disp = disp

        self.progressbar = ProgressBarLayout(self) if progress else None
        self.k0 = f'{self.conftype}_CONF'
        if idx is not None:
            self.k = f'{self.k0}{idx}'
        else:
            self.k = self.get_next(self.k0)
        self.sublists = sublists
        self.tab.selectionlists[self.conftype] = self

        bs = button_row(self.disp, buttons)

        self.layout = self.build(bs=bs, **kwargs)

    def build(self, bs, **kwargs):
        n = self.disp
        if self.with_dict:
            nn = self.tab.gui.tab_dict[n][2]
            self.collapsible = CollapsibleDict(nn, default=True,
                                               header_list_width=self.width, header_dict=loadConfDict(self.conftype),
                                               next_to_header=bs, header_key=self.k,disp_name=get_disp_name(n),
                                               header_list_kws={'tooltip': f'The currently loaded {n}.'}, **kwargs)

            l = self.collapsible.get_layout(as_col=False)

        else:
            self.collapsible = None
            l = NamedList(self.name, key=self.k, choices=self.confs, default_value=None,
                             drop_down=True, size=(self.width, None),
                             list_kws={'tooltip': f'The currently loaded {n}.'},
                             header_kws={'text': n.capitalize(), 'after_header': bs, 'single_line': False}).layout
        if self.progressbar is not None:
            l.append(self.progressbar.l)
        return l

    def eval(self, e, v, w, c, d, g):
        n = self.disp
        id = v[self.k]
        k0 = self.conftype

        if e==self.k :
            conf = loadConf(id, k0)
            for kk, vv in self.sublists.items():
                if type(conf[kk])==str :
                    vv.update(w, conf[kk])
                if type(conf[kk])==list :
                    vv.update(w, values=conf[kk])

        if e == f'LOAD {n}' and id != '':
            conf = loadConf(id, k0)
            self.tab.update(w, c, conf, id)
            if self.progressbar is not None:
                self.progressbar.reset(w)
            for kk, vv in self.sublists.items():
                vv.update(w, conf[kk])

        elif e == f'SAVE {n}':
            conf = self.tab.get(w, v, c, as_entry=True)
            for kk, vv in self.sublists.items():
                conf[kk] = v[vv.k]
            id = self.save(conf)
            if id is not None:
                self.update(w, id)
        elif e == f'DELETE {n}' and id != '':
            deleteConf(id, k0)
            self.update(w)
        elif e == f'RUN {n}' and id != '':
            try:
                conf = self.tab.get(w, v, c, as_entry=False)
                for kk, vv in self.sublists.items():
                    if not vv.with_dict:
                        conf[kk] = expandConf(id=v[vv.k], conf_type=vv.conftype)
                    else:
                        conf[kk] = vv.collapsible.get_dict(v, w)
            except:
                return
            d, g = self.tab.run(v, w, c, d, g, conf, id)
            self.tab.gui.dicts = d
            self.tab.gui.graph_lists = g

        elif e == f'EDIT {n}':
            conf = self.tab.get(w, v, c, as_entry=False)
            new_conf = self.tab.edit(conf)
            self.tab.update(w, c, new_conf, id=None)
        elif self.collapsible is not None and e == self.collapsible.header_key:
            self.collapsible.update_header(w, id)

    def update(self, w, id='', all=False, values=None):
        if values is None :
            values=self.confs
        w.Element(self.k).Update(values=values, value=id, size=(self.width, self.Nconfs))
        if self.collapsible is not None:
            self.collapsible.update_header(w, id)
        if all:
            for i in range(5):
                k = f'{self.k0}{i}'
                if k in w.AllKeysDict.keys():
                    w[k].update(values=values, value=id)

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


class Header(HeadedElement):
    def __init__(self, name, content=[], text=None, before_header=None, after_header=None, header_text_kws=None,
                 single_line=True):
        if text is None:
            text = name
        header = self.build_header(text, before_header, after_header, header_text_kws)
        super().__init__(name=name, header=header, content=content, single_line=single_line)

    def build_header(self, text, before_header, after_header, header_text_kws):
        if header_text_kws is None:
            header_text_kws = {'size': (len(text), 1)}
        header = [sg.Text(text, **header_text_kws)]
        if after_header is not None:
            header += after_header
        if before_header is not None:
            header = before_header + header
        return header


class NamedList(Header):
    def __init__(self, name, key, choices, default_value=None, drop_down=True, size=(25, None),
                 readonly=True, enable_events=True, list_kws={}, aux_cols=None, select_mode=None, header_kws={}, **kwargs):

        self.aux_cols = aux_cols
        self.key = key
        self.W, self.H = size
        if self.H is None:
            self.H = len(choices)
        content = self.build_list(choices, default_value, drop_down, readonly, enable_events, select_mode, list_kws, **kwargs)
        super().__init__(name=name, content=content, **header_kws)
    def build_list(self, choices, default_value, drop_down, readonly, enable_events, select_mode, list_kws, **kwargs):
        kws = {'key': self.key, 'enable_events': enable_events, **list_kws, **kwargs}
        if drop_down:
            l = [sg.Combo(choices, default_value=default_value, size=(self.W, 1), readonly=readonly, **kws)]
        else:
            if self.aux_cols is None:
                l = [sg.Listbox(choices, default_values=[default_value], size=(self.W, self.H),select_mode= select_mode, **kws)]
            else:
                N = len(self.aux_cols)
                vs = [[''] * (N + 1)] * len(choices)
                w00 = 0.35
                w0 = int(self.W * w00)
                col_widths = [w0] + [int(self.W * (1 - w00) / N)] * N
                l = [Table(values=vs, headings=['Dataset ID'] + self.aux_cols, col_widths=col_widths,
                           display_row_numbers=True,select_mode= select_mode, **kws)]
        return l


class DataList(NamedList):
    def __init__(self, name, tab, dict={}, buttons=['select_all', 'remove', 'changeID', 'browse'], button_args={},
                 raw=False, select_mode= LISTBOX_SELECT_MODE_EXTENDED,drop_down=False, **kwargs):

        self.tab = tab
        self.dict = dict
        # self.buttons = buttons
        # self.button_args = button_args
        self.raw = raw
        self.list_key = f'{name}_IDS'
        self.browse_key = f'BROWSE {name}'
        self.tab.datalists[name] = self
        after_header=button_row(name, buttons, button_args)
        header_kws = {'text': get_disp_name(name), 'single_line': False, 'after_header': after_header}
        super().__init__(name=name, header_kws=header_kws, key=self.list_key, choices=list(self.dict.keys()),
                      drop_down=drop_down, select_mode=select_mode, **kwargs)


    # def build_buttons(self, name):
    #     bl = []
    #     for n in self.buttons:
    #         if n in list(self.button_args.keys()):
    #             kws = self.button_args[n]
    #         else:
    #             kws = {}
    #         l = button_dict[n](name, **kws)
    #         bl.append(l)
    #     return bl


    def update_window(self, w):
        ks = list(self.dict.keys())
        if self.aux_cols is None:
            w.Element(self.list_key).Update(values=ks)
        else:
            vs = self.get_aux_cols(ks)
            w.Element(self.list_key).Update(values=vs)

    def get_aux_cols(self, ks):
        # df=np.zeros((len(ks), len(self.aux_cols)+1))
        ls = []
        for k in ks:
            # for i,k in enumerate(ks):
            l = [k]
            d = self.dict[k]
            for c in self.aux_cols:
                # for j,c in enumerate(self.aux_cols):
                try:
                    a = getattr(d, c)
                except:
                    a = ''
                l.append(a)
            ls.append(l)
        return ls

    def eval(self, e, v, w, c, d, g):
        from lib.stor.managing import detect_dataset
        n = self.name
        k = self.list_key
        d0 = self.dict
        v0 = v[k]
        kks = [v0[i] if self.aux_cols is None else list(d0.keys())[v0[i]] for i in range(len(v0))]
        datagroup_id = self.tab.current_ID(v) if self.raw else None
        if e == self.browse_key:
            d0.update(detect_dataset(datagroup_id, v[self.browse_key], raw=self.raw))
            self.update_window(w)
        elif e == f'SELECT_ALL {n}':
            ks = np.arange(len(d0)).tolist()
            if self.aux_cols is None:
                w.Element(k).Update(set_to_index=ks)
            else:
                w.Element(k).Update(select_rows=ks)
        elif e == f'REMOVE {n}':
            for kk in kks:
                d0.pop(kk, None)
            self.update_window(w)
        elif e == f'CHANGE_ID {n}':
            d0 = change_dataset_id(d0, old_ids=kks)
            self.update_window(w)
        elif e == f'REPLAY {n}':
            if len(v0) > 0:
                dd = d0[kks[0]]
                dd.visualize(vis_kwargs=self.tab.gui.get_vis_kwargs(v, mode='video'),
                             **self.tab.gui.get_replay_kwargs(v))
        elif e == f'ADD_REF {n}':
            from lib.stor.larva_dataset import LarvaDataset
            dd = LarvaDataset(dir=f'{paths.RefFolder}/reference')
            d0[dd.id] = dd
            self.update_window(w)
        elif e == f'IMPORT {n}':
            dl1 = self.tab.datalists[self.tab.proc_key]
            d1 = dl1.dict
            k1 = dl1.list_key
            raw_dic = {id: dir for id, dir in d0.items() if id in v[k]}
            proc_dic = import_window(datagroup_id=datagroup_id, raw_dic=raw_dic)
            d1.update(proc_dic)
            dl1.update_window(w)
        elif e == f'ENRICH {n}':
            dds = [dd for i,(id, dd) in enumerate(d0.items()) if i in v[k]]
            if len(dds) > 0:
                enrich_conf = c['enrichment'].get_dict(v, w)
                for dd in dds :
                    dd.enrich(**enrich_conf)
        self.dict = d0


class Collapsible(HeadedElement):
    def __init__(self, name, state=False, content=[], disp_name=None, toggle=None, disabled=False, next_to_header=None,
                 auto_open=False, header_dict=None, header_value=None, header_list_width=10, header_list_kws={},
                 header_text_kws=t_kws(12), header_key=None, **kwargs):
        if disp_name is None:
            disp_name = get_disp_name(name)
        self.disp_name = disp_name
        self.state = state
        self.toggle = toggle
        self.auto_open = auto_open
        self.sec_key = f'SEC {name}'

        self.toggle_key = f'TOGGLE_{name}'
        self.sec_symbol = self.get_symbol()
        self.header_dict = header_dict
        self.header_value = header_value
        if header_key is None:
            header_key = f'SELECT LIST {name}'
        self.header_key = header_key
        after_header=[BoolButton(name, toggle, disabled)] if toggle is not None else []
        if next_to_header is not None:
            after_header += next_to_header
        header_kws = {'text': disp_name, 'header_text_kws': header_text_kws,
                      'before_header' : [self.sec_symbol], 'after_header' : after_header}
        if header_dict is None:
            header_l = Header(name, **header_kws)
        else:
            header_l = NamedList(name, choices=list(header_dict.keys()),
                                 default_value=header_value, key=self.header_key,
                                 size=(header_list_width, None), list_kws=header_list_kws,
                                 header_kws=header_kws)
        header = header_l.get_layout()
        super().__init__(name=name, header=header, content = [collapse(content, self.sec_key, self.state)], single_line=False)

    def get_symbol(self):
        return sg.T(SYMBOL_DOWN if self.state else SYMBOL_UP, k=f'OPEN {self.sec_key}',
                    enable_events=True, text_color='black', **t_kws(2))

    def update(self, w, dict, use_prefix=True):
        if dict is None:
            self.disable(w)
        elif self.toggle == False:
            self.disable(w)
        else:
            self.enable(w)
            prefix = self.name if use_prefix else None
            self.update_window(w, dict, prefix=prefix)
        return w

    def update_window(self, w, dic, prefix=None):
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
                    self.update_window(w, v, prefix=new_prefix)
                elif isinstance(w[k], TupleSpin) or isinstance(w[k], MultiSpin):
                    w[k].update(w, v)
                elif v is None:
                    w.Element(k).Update(value='')
                else:
                    w.Element(k).Update(value=v)

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
    def __init__(self, name, type_dict={}, headings=[], dict={}, **kwargs):
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
        b = [GraphButton('Document_2_Edit', self.edit_key, tooltip=f'Create new {name}')]
        super().__init__(name, content=content, next_to_header=b, **kwargs)

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
        # content = [[sg.Table(values=self.data[:][:], headings=self.headings, col_widths=self.col_widths,
        #                      max_col_width=30, background_color='lightblue', header_font=('Helvetica', 8, 'bold'),
        #                      auto_size_columns=False,
        #                      visible_column_map=self.col_visible,
        #                      # display_row_numbers=True,
        #                      justification='center',
        #                      font=w_kws['font'],
        #                      num_rows=len(self.data),
        #                      alternating_row_color='lightyellow',
        #                      key=self.key
        #                      )]]
        content = [[Table(values=self.data[:][:], headings=self.headings, col_widths=self.col_widths,
                             visible_column_map=self.col_visible,
                             # display_row_numbers=True,
                             num_rows=len(self.data),
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
            dic = self.set_agent_dict()
            self.update(window, dic)
        else:
            t0 = [dict(zip(self.headings, l)) for l in self.data] if self.data != [[''] * self.Ncols] else []
            t1 = gui_table(t0, self.type_dict, title='Parameter space')
            if t1 != t0:
                dic = {k: [l[k] for l in t1] for k in self.headings}
                self.update(window, dic)

    def set_agent_dict(self):
        t0 = fun.agent_dict2list(self.dict, header=self.header)
        t1 = gui_table(t0, self.type_dict, title=self.disp_name)
        return fun.agent_list2dict(t1, header=self.header)

    def get_dict(self, *args, **kwargs):
        return self.dict


class CollapsibleDict(Collapsible):
    def __init__(self, name, dict=None, dict_name=None, type_dict=None, toggled_subsections=True, default=False,
                 text_kws={}, value_kws={}, **kwargs):
        if dict_name is None:
            dict_name = name
        self.dict_name = dict_name
        if default and dict is None and type_dict is None:
            dict = dtypes.get_dict(self.dict_name)
            type_dict = dtypes.get_dict_dtypes(self.dict_name)
        self.sectiondict = SectionDict(name=name, dict=dict, type_dict=type_dict,
                                       toggled_subsections=toggled_subsections)
        content = self.sectiondict.init_section(value_kws=value_kws)
        super().__init__(name, content=content, **kwargs)

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
    def __init__(self, values=[], background_color='lightblue', header_background_color='lightgrey',
                 alternating_row_color='lightyellow',
                 auto_size_columns=False, text_color='black', header_font=('Helvetica', 8, 'bold'),
                 justification='center', **kwargs):
        super().__init__(values, auto_size_columns=auto_size_columns, justification=justification,
                         background_color=background_color, header_background_color=header_background_color,
                         alternating_row_color=alternating_row_color, text_color=text_color, header_font=header_font,
                         **kwargs)

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


class GraphList(NamedList):
    def __init__(self, name, tab,fig_dict={}, next_to_header=None, default_values=None, canvas_size=(1000, 800),
                 list_size=None, list_header='Graphs', auto_eval=True, canvas_kws={'background_color': 'Lightblue'},
                 graph=False, subsample=1):

        self.tab = tab
        self.tab.graphlists[name] = self
        self.fig_dict = fig_dict
        self.subsample = subsample
        self.auto_eval = auto_eval
        self.list_key = f'{name}_GRAPH_LIST'

        values = list(fig_dict.keys())
        if list_size is None:
            h = int(np.max([len(values), 10]))
            list_size = (25, h)

        header_kws = {'text': list_header, 'after_header': next_to_header,
                      'header_text_kws': t_kws(10), 'single_line': False}
        default_value = default_values[0] if default_values is not None else None
        super().__init__(name=name, key=self.list_key, choices=values, default_value=default_value, drop_down=False,
                         size=list_size, header_kws=header_kws, auto_size_text=True)

        self.canvas_size = canvas_size
        self.canvas_key = f'{name}_CANVAS'
        self.canvas, self.canvas_element = self.init_canvas(canvas_size, canvas_kws, graph)
        self.fig_agg = None

    def init_canvas(self, size, canvas_kws, graph=False):
        k=self.canvas_key
        if graph:
            g = sg.Graph(canvas_size=size,k=k, **canvas_kws)
        else:
            g = sg.Canvas(size=size,k=k, **canvas_kws)
        canvas = GuiElement(name=k, layout=[[g]])
        return canvas, g

    def draw_fig(self, w, fig):
        if self.fig_agg:
            delete_figure_agg(self.fig_agg)
        self.fig_agg = draw_canvas(w[self.canvas_key].TKCanvas, fig)

    def update(self, w, fig_dict=None):
        if fig_dict is None :
            fig_dict=self.fig_dict
        else :
            self.fig_dict = fig_dict
        w.Element(self.list_key).Update(values=list(fig_dict.keys()))

    def eval(self, e, v, w, c, d, g):
        if e == self.list_key and self.auto_eval:
            v0=v[self.list_key]
            if len(v0) > 0:
                choice = v0[0]
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
    def __init__(self, name, buttons=['refresh_figs', 'conf_fig', 'draw_fig', 'save_fig'],
                 button_args={},**kwargs):

        after_header = button_row(name, buttons, button_args)
        super().__init__(name=name, next_to_header=after_header, **kwargs)

        self.fig, self.save_to, self.save_as = None, '', ''
        self.func, self.func_kws = None, {}


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
        self.update(w)
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

    def eval(self, e, v, w, c, d, g):
        n = self.name
        k = self.list_key
        d0 = self.fig_dict

        if e == f'BROWSE_FIGS {n}':
            v0 = v[k]
            v1=v[f'BROWSE_FIGS {n}']
            id=v0.split('/')[-1].split('.')[-2]
            d0[id]=v1
            self.update(w,d0)
        elif e == f'REMOVE_FIGS {n}':
            v0 = v[k]
            for kk in v0:
                d0.pop(kk, None)
            self.update(w,d0)
        elif e == k:
            v0 = v[k]
            if len(v0) > 0:
                fig = d0[v0[0]]
                try:
                    if fig != self.func:
                        self.func = fig
                        self.func_kws = self.get_graph_kws(self.func)
                except :
                    if type(fig) == str and os.path.isfile(fig):
                        self.show_fig(w, fig)
                    else:
                        self.draw_fig(w, fig)
        elif e == f'REFRESH_FIGS {n}':
            self.refresh_figs(w, self.tab.base_dict)
        elif e == f'SAVE_FIG {n}':
            self.save_fig()
        elif e == f'CONF_FIG {n}':
            if self.func_kws != {}:
                self.func_kws = set_kwargs(self.func_kws, title='Graph arguments')
        elif e == f'DRAW_FIG {n}':
            self.generate(w, self.tab.base_dict)




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
        w=self.window
        e, v = w.read(timeout=0)
        if e is None:
            w.close()
            return False
        elif e == 'Choose':
            w[f'-COL2-'].update(visible=False)
            w[f'-COL1-'].update(visible=True)
            self.cur_layout = 1
        elif e == 'Ok':
            w[f'-COL1-'].update(visible=False)
            w[f'-COL2-'].update(visible=True)
            self.pars = [p for p in self.available_pars if v[f'k_{p}']]
            self.update_pars()
            self.cur_layout = 2
        elif e == 'Cancel':
            w[f'-COL1-'].update(visible=False)
            w[f'-COL2-'].update(visible=True)
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


