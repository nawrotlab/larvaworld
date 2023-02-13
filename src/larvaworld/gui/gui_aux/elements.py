import copy
import inspect
import os
import pandas as pd
from typing import Tuple, List, TypedDict
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt


from larvaworld.lib import reg, aux
from larvaworld.gui import gui_aux


SYMBOL_UP = '▲'
SYMBOL_DOWN = '▼'

col_idx_dict = {
    'LarvaGroup': [[0, 1, 2, 3, 6], [4], [5]],
    'enrichment': [[0], [5, 1, 3], [2, 4]],
    'metric_definition': [[0, 1, 4], [2, 3, 5]],
}


class SectionDict:
    def __init__(self, name, dict, type_dict=None, toggled_subsections=True, value_kws={}):
        self.init_dict = dict
        self.type_dict = type_dict
        self.toggled_subsections = toggled_subsections
        self.name = name
        self.subdicts = {}
        self.layout = self.init_section(type_dict, value_kws)

    def init_section(self, type_dict, value_kws={}):
        d = type_dict
        items = []
        for k, v in self.init_dict.items():
            k_disp = gui_aux.get_disp_name(k)
            k0 = f'{self.name}_{k}'
            if type(v) == bool:
                ii = gui_aux.named_bool_button(k_disp, v, k0)
            else:
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

                    elif t == dict and list(t0.keys()) == ['type', 'values']:
                        spin_kws = {
                            'values': t0['values'],
                            'initial_value': v,
                            'key': k0,
                            **value_kws
                        }
                        if t0['type'] == list:
                            temp = MultiSpin(tuples=False, **spin_kws)
                        elif t0['type'] == List[tuple]:
                            temp = MultiSpin(tuples=True, **spin_kws)
                        elif t0['type'] == tuple:
                            temp = MultiSpin(**spin_kws)
                        elif t0['type'] in [float, int]:
                            temp = sg.Spin(**spin_kws)

                ii = [sg.Text(f'{k_disp}:'), temp]
            items.append(ii)
        return items

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
            elif type(t) == tuple or (type(t) == dict and list(t.keys()) == ['type', 'values']):
                d[k] = w[k0].get()
            elif t == dict or type(t) == dict:
                d[k] = self.subdicts[k0].get_dict(v, w)
            else:
                d[k] = gui_aux.retrieve_value(v[k0], t)
        return d

    def get_subdicts(self):
        subdicts = {}
        for s in list(self.subdicts.values()):
            subdicts.update(s.get_subdicts())
        return subdicts


class SingleSpin(sg.Spin):
    def __init__(self, values, initial_value, dtype=float, value_kws={}, **kwargs):
        if values is None:
            values=[]
        spin_kws = {
            'values': [''] + values,
            'initial_value': initial_value if initial_value is not None else '',
            **value_kws,
            **kwargs,
        }
        super().__init__(**spin_kws)
        self.dtype = dtype

    def get(self):
        v = super().get()
        if v in ['', None]:
            return None
        if self.dtype == int:
            return int(v)
        elif self.dtype == float:
            return float(v)


class MultiSpin(sg.Pane):
    def __init__(self, initial_value, values, key, steps=100, Nspins=2, group_by_N=None,
                 tuples=False, dtype=float, value_kws={}, indexing=False, **kwargs):
        self.indexing = indexing
        self.value_kws = value_kws
        self.group_by_N = group_by_N
        self.Nspins = Nspins
        self.steps = steps
        self.dtype = dtype
        self.initial_value = initial_value
        self.values = values
        if initial_value is None:
            self.v_spins = [''] * Nspins
            self.N = 0
        elif self.Nspins == 1:
            self.N = 1
            self.v_spins = [initial_value]
        else:
            self.N = len(initial_value)
            self.v_spins = [vv for vv in initial_value] + [''] * (Nspins - self.N)
        # self.key = key
        self.tuples = tuples
        self.add_key, self.remove_key = f'SPIN+ {key}', f'SPIN- {key}'
        self.k_spins = [f'{key}_{i}' for i in range(Nspins)]
        self.visibles = [True] * (self.N + 1) + [True] * (Nspins - self.N - 1)
        self.spins = self.build_spins()
        self.layout = self.build_layout()

        add_button = gui_aux.GraphButton('Button_Add', self.add_key, tooltip='Add another item in the list.')
        remove_button = gui_aux.GraphButton('Button_Remove', self.remove_key, tooltip='Remove last item in the list.')
        self.buttons = sg.Col([[add_button], [remove_button]])
        pane_list = [sg.Col([self.layout])]
        super().__init__(pane_list=pane_list, key=key)

    def build_spins(self):
        spin_kws = {
            'values': self.values,
            'dtype': self.dtype,
            'value_kws': self.value_kws,
        }
        func = MultiSpin if self.tuples else SingleSpin
        spins = [func(initial_value=vv, key=kk, visible=vis, **spin_kws) for vv, kk, vis in
                 zip(self.v_spins, self.k_spins, self.visibles)]
        return spins

    def get(self):
        vs = [s.get() for s in self.spins if s.get() not in [None, '']]
        return vs if len(vs) > 0 else None

    def update(self, value):
        if value in [None, '', (None, None), [None, None]]:
            self.N = 0
            for spin in self.spins:
                spin.update('')

        else:
            self.N = len(value)
            for i, spin in enumerate(self.spins):
                vv = value[i] if i < self.N else ''
                spin.update(vv)

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
        Ng = self.group_by_N
        ss = self.spins
        if self.indexing:
            ss = [sg.Col([[sg.T(i, **gui_aux.t_kws(1)), s]]) for i, s in enumerate(ss)]

        if Ng is not None and self.Nspins >= Ng:
            spins = aux.group_list_by_n(ss, Ng)
            spins = [sg.Col(spins)]
            return spins
        else:
            return ss


class GuiElement:
    """ The base class for all Gui Elements. Holds the basic description of an Element like size and colors """

    def __init__(self, name, layout=None, layout_col_kwargs={}):
        self.name = name
        self.layout = layout
        self.layout_col_kwargs = layout_col_kwargs

    def get_layout(self, as_col=True, as_pane=False, **kwargs):
        if not as_col:
            return self.layout
        elif not as_pane:
            self.layout_col_kwargs.update(kwargs)
            return [sg.Col(self.layout, **self.layout_col_kwargs)]
        else:
            return [[sg.Pane([sg.Col(self.layout)], border_width=8, **kwargs)]]


class ProgressBarLayout:
    def __init__(self, list, size=(8.8, 20)):
        self.list = list
        n = self.list.disp
        self.k = f'{n}_PROGRESSBAR'
        self.k_complete = f'{n}_COMPLETE'
        self.k_incomplete = f'{n}_INCOMPLETE'
        self.l = [sg.Text('Progress :', **gui_aux.t_kws(8)),
                  sg.ProgressBar(100, orientation='h', size=size, key=self.k,
                                 bar_color=('green', 'lightgrey'), border_width=3),
                  gui_aux.GraphButton('Button_Check', self.k_complete, visible=False,
                              tooltip=f'Whether the current {n} was completed.'),
                  gui_aux.GraphButton('Button_stop', self.k_incomplete, visible=False,
                              tooltip=f'Whether the current {n} is running.'),
                  ]

    def reset(self, w):
        w[self.k].update(0)
        w[self.k_complete].update(visible=False)
        w[self.k_incomplete].update(visible=False)

    def done(self, w):
        w[self.k_complete].update(visible=True)
        w[self.k_incomplete].update(visible=False)

    def run(self, w, min=0, max=100):
        w[self.k_complete].update(visible=False)
        w[self.k_incomplete].update(visible=True)
        w[self.k].update(0, max=max)

    # def start(self):
    #     # To make compaible with terminal exec
    #     return self


class HeadedElement(GuiElement):
    def __init__(self, name, header, content=[], single_line=True):
        layout = [header + content] if single_line else [header, content]
        super().__init__(name=name, layout=layout)


class SelectionList(GuiElement):
    def __init__(self, tab, conftype=None, disp=None, buttons=[], button_kws={}, sublists={}, idx=None, progress=False,
                 width=24, with_dict=False, name=None, single_line=False, root_key=None, **kwargs):
        self.conftype = conftype if conftype is not None else tab.conftype

        if name is None:
            name = self.conftype
        super().__init__(name=name)
        # print(name, with_dict)
        self.single_line = single_line
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
        # print(disp)

        self.progressbar = ProgressBarLayout(self, size=(self.width - 16, 20)) if progress else None
        self.k0 = f'{self.conftype}_CONF'
        if idx is not None:
            self.k = f'{self.k0}{idx}'
        else:
            self.k = self.get_next(self.k0)
        self.sublists = sublists
        self.tab.selectionlists[self.conftype] = self
        self.root_key = root_key
        self.tree = GuiTreeData(self.root_key) if self.root_key is not None else None

        bs = gui_aux.button_row(self.disp, buttons, button_kws)

        self.layout = self.build(bs=bs, **kwargs)

    def build(self, bs, **kwargs):
        n = self.disp
        if self.with_dict:
            # print(self.tab.gui.tab_dict[n][2])
            self.collapsible = CollapsibleDict(name= self.tab.gui.tab_dict[n][2], default=True,
                                               header_list_width=self.width, header_dict=aux.load_dict(reg.Path[self.conftype]),
                                               # header_list_width=self.width, header_dict=reg.conf0.dict[self.conftype].loadDict(),
                                               next_to_header=bs, header_key=self.k, disp_name=gui_aux.get_disp_name(n),
                                               header_list_kws={'tooltip': f'The currently loaded {n}.'}, **kwargs)

            l = self.collapsible.get_layout(as_col=False)

        else:
            self.collapsible = None
            l = NamedList(self.name, key=self.k, choices=self.confs, default_value=None,
                          drop_down=True, size=(self.width, None),
                          list_kws={'tooltip': f'The currently loaded {n}.'},
                          header_kws={'text': n.capitalize(), 'after_header': bs,
                                      'single_line': self.single_line, **kwargs}).layout
        if self.progressbar is not None:
            l.append(self.progressbar.l)
        return l

    def eval(self, e, v, w, c, d, g):
        n = self.disp
        id = v[self.k]
        k0 = self.conftype
        if e == self.k:
            conf = reg.loadConf(id=id, conftype=k0)
            for kk, vv in self.sublists.items():
                if type(conf[kk]) == str:
                    vv.update(w, conf[kk])
                if type(conf[kk]) == list:
                    vv.update(w, values=conf[kk])

        if e == f'LOAD {n}' and id != '':
            self.load(w, c, id)

        elif e == f'SAVE {n}':
            conf = self.get(w, v, c, as_entry=True)
            id = self.save(conf)
            if id is not None:
                self.update(w, id)
        elif e == f'DELETE {n}' and id != '':
            if self.delete(id, k0):
                self.update(w)
        elif e == f'EXEC {n}' and id != '':
        # FIXME Seems lie EXEC is used and not RUN
        # elif e == f'RUN {n}' and id != '':
            try:
                conf = self.get(w, v, c, as_entry=False)
                d, g = self.tab.run(v, w, c, d, g, conf, id)
                self.tab.gui.dicts = d
                self.tab.gui.graph_lists = g
            except:
                pass

        elif e == f'EDIT {n}':
            conf = self.tab.get(w, v, c, as_entry=False)
            new_conf = self.tab.edit(conf)
            self.tab.update(w, c, new_conf, id=None)
        elif e == f'TREE {n}' and self.tree is not None:
            self.tree.test()
        elif e == f'CONF_TREE {n}' and id != '':
            conf = self.get(w, v, c, as_entry=False)
            entries = gui_aux.tree_dict(d=conf, parent_key=id, sep='.')
            tree = GuiTreeData(entries=entries, headings=['value'], col_widths=[40, 20])
            tree.test()

        elif self.collapsible is not None and e == self.collapsible.header_key:
            self.collapsible.update_header(w, id)

            # try:
            #     self.tab.DL0.dict[self.tab.active_id].terminate()
            # except :
            #     pass

    def update(self, w, id='', all=False, values=None):
        if values is None:
            values = self.confs
        w.Element(self.k).Update(values=values, value=id, size=(self.width, self.Nconfs))
        if self.collapsible is not None:
            self.collapsible.update_header(w, id)
        if all:
            for i in range(5):
                k = f'{self.k0}{i}'
                if k in w.AllKeysDict.keys():
                    w[k].update(values=values, value=id)

    def save(self, conf):
        return gui_aux.save_conf_window(conf, self.conftype, disp=self.disp)

    def delete(self, id, k0):
        return gui_aux.delete_conf_window(id, conftype=k0, disp=self.disp)

    def load(self, w, c, id):
        conf = reg.loadConf(id=id, conftype=self.conftype)
        self.tab.update(w, c, conf, id)
        if self.progressbar is not None:
            self.progressbar.reset(w)
        for kk, vv in self.sublists.items():
            vv.update(w, conf[kk])
            try:
                vv.load(w, c, id=conf[kk])
            except:
                pass

    def get(self, w, v, c, as_entry=True):
        conf = self.tab.get(w, v, c, as_entry=as_entry)
        if as_entry:
            for kk, vv in self.sublists.items():
                try:
                    conf[kk] = v[vv.k]
                except:
                    conf[kk] = vv.get_dict()

        else:
            for kk, vv in self.sublists.items():
                if isinstance(vv, SelectionList):
                    if not vv.with_dict:
                        conf[kk] = reg.expandConf(id=v[vv.k], conftype=vv.conftype)
                    else:
                        conf[kk] = vv.collapsible.get_dict(v, w)
                else:
                    conf[kk] = vv.get_dict()
                    if kk == 'larva_groups':
                        for n, gr in conf[kk].items():
                            if type(gr['model']) == str:
                                gr['model'] = reg.loadConf(id=gr['model'], conftype= 'Model')
        return conf

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
        return reg.storedConf(self.conftype)
        # return kConfDict(self.conftype)

    @property
    def Nconfs(self):
        return len(self.confs)


class Header(HeadedElement):
    def __init__(self, name, content=[], text=None, single_line=True, **kwargs):
        if text is None:
            text = name
        header = self.build_header(text, **kwargs)
        super().__init__(name=name, header=header, content=content, single_line=single_line)

    def build_header(self, text, before_header=None, after_header=None, text_kws=None):
        if text_kws is None:
            text_kws = {'size': (len(text), 1)}
        header = [sg.Text(text, **text_kws)]
        if after_header is not None:
            header += after_header
        if before_header is not None:
            header = before_header + header
        return header


class NamedList(Header):
    def __init__(self, name, key, choices, default_value=None, drop_down=True, size=(gui_aux.w_list, None),
                 readonly=True,
                 enable_events=True, list_kws={}, aux_cols=None, select_mode=None, header_kws={}, **kwargs):

        self.aux_cols = aux_cols
        self.key = key
        self.W, self.H = size
        if self.H is None:
            self.H = len(choices)
        content = self.build_list(choices, default_value, drop_down, readonly, enable_events, select_mode, list_kws,
                                  **kwargs)
        super().__init__(name=name, content=content, **header_kws)

    def build_list(self, choices, default_value, drop_down, readonly, enable_events, select_mode, list_kws, **kwargs):
        kws = {'key': self.key, 'enable_events': enable_events, **list_kws, **kwargs}
        if drop_down:
            l = [sg.Combo(choices, default_value=default_value, size=(self.W, 1), readonly=readonly, **kws)]
        else:
            if self.aux_cols is None:
                l = [sg.Listbox(choices, default_values=[default_value], size=(self.W, self.H), select_mode=select_mode,
                                **kws)]
            else:
                N = len(self.aux_cols)
                vs = [[''] * (N + 1)] * len(choices)
                w00 = 0.35
                w0 = int(self.W * w00)
                col_widths = [w0] + [int(self.W * (1 - w00) / N)] * N
                l = [Table(values=vs, headings=['Dataset ID'] + self.aux_cols, col_widths=col_widths,
                           display_row_numbers=True, select_mode=select_mode, size=(self.W, self.H), **kws)]
        return l


class DataList(NamedList):
    def __init__(self, name, tab, dict={}, buttons=['select_all', 'remove', 'changeID', 'browse'], button_args={},
                 raw=False, select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, drop_down=False, disp=None, **kwargs):
        if disp is None:
            disp = gui_aux.get_disp_name(name)
        self.tab = tab
        self.dict = dict
        self.raw = raw
        self.browse_key = f'BROWSE {name}'
        self.tab.datalists[name] = self
        header_kws = {'text': disp, 'single_line': False, 'after_header': gui_aux.button_row(name, buttons, button_args)}
        super().__init__(name=name, header_kws=header_kws, key=f'{name}_IDS', choices=list(self.dict.keys()),
                         drop_down=drop_down, select_mode=select_mode, **kwargs)

    def update_window(self, w):

        ks = list(self.dict.keys())
        # print(self.name, ks, 0)
        if self.aux_cols is None:
            w.Element(self.key).Update(values=ks)
        else:
            vs = self.get_aux_cols(ks)
            w.Element(self.key).Update(values=vs)
        # print(self.name, ks, 1)

    def get_aux_cols(self, ks):
        ls = []
        for k in ks:
            l = [k]
            d = self.dict[k]
            for c in self.aux_cols:
                try:
                    a = getattr(d, c)
                except:
                    a = ''
                l.append(a)
            ls.append(l)
        return ls

    def add(self, w, entries, replace=False):
        if replace:
            self.dict = entries
        else:
            self.dict.update(entries)
        self.update_window(w)

    def remove(self, w, ids):
        # print(self.dict.keys())
        for kk in ids:
            self.dict.pop(kk, None)
        # print(self.dict.keys())
        # print()
        self.update_window(w)

    def eval(self, e, v, w, c, d, g):

        n = self.name
        k = self.key
        d0 = self.dict
        v0 = v[k]
        kks = [v0[i] if self.aux_cols is None else list(d0.keys())[v0[i]] for i in range(len(v0))]
        # print(kks)
        datagroup_id = self.tab.current_ID(v) if self.raw else None
        if e == self.browse_key:
            new = detect_dataset(datagroup_id, v[self.browse_key], raw=self.raw)
            self.add(w, new)
        elif e == f'SELECT_ALL {n}':
            ks = np.arange(len(d0)).tolist()
            if self.aux_cols is None:
                w.Element(k).Update(set_to_index=ks)
            else:
                w.Element(k).Update(select_rows=ks)
        elif e == f'REMOVE {n}':
            self.remove(w, kks)
        elif e == f'CHANGE_ID {n}':
            self.dict = gui_aux.change_dataset_id(d0, old_ids=kks)
            self.update_window(w)
        elif e == f'SAVE_REF {n}':
            gui_aux.save_ref_window(d0[kks[0]])
        elif e == f'REPLAY {n}':
            if len(v0) > 0:
                d0[kks[0]].visualize(vis_kwargs=self.tab.gui.get_vis_kwargs(v, mode='video'),
                                     **self.tab.gui.get_replay_kwargs(v))
        elif e == f'IMITATE {n}':
            if len(v0) > 0:
                if d0[kks[0]].config['refID'] is None:
                    gui_aux.save_ref_window(d0[kks[0]])
                from larvaworld.lib.reg.config import imitation_exp
                exp_conf = imitation_exp(d0[kks[0]].config['refID'])
                exp_conf.screen_kws['vis_kwargs'] = self.tab.gui.get_vis_kwargs(v)
                self.tab.imitate(exp_conf)
        elif e == f'ADD_REF {n}':
            dds = gui_aux.add_ref_window()
            if dds is not None:
                self.add(w, dds)
        elif e == f'IMPORT {n}':
            dl1 = self.tab.datalists[self.tab.proc_key]
            d1 = dl1.dict
            k1 = dl1.key
            raw_dic = {id: dir for id, dir in d0.items() if id in v[k]}
            if len(raw_dic) > 0:
                proc_dic = gui_aux.import_window(datagroup_id=datagroup_id, raw_dic=raw_dic)
                d1.update(proc_dic)
                dl1.update_window(w)
        elif e == f'ENRICH {n}':
            dds = [dd for i, (id, dd) in enumerate(d0.items()) if i in v[k]]
            if len(dds) > 0:
                enrich_conf = c['enrichment'].get_dict(v, w)
                for dd in dds:
                    dd.enrich(**enrich_conf)


class Collapsible(HeadedElement, GuiElement):
    def __init__(self, name, state=False, content=[], disp_name=None, toggle=None, disabled=False, next_to_header=None,
                 auto_open=False, header_dict=None, header_value=None, header_list_width=10, header_list_kws={},
                 header_text_kws=gui_aux.t_kws(12), header_key=None, use_header=True, Ncols=1, col_idx=None, **kwargs):
        if disp_name is None:
            disp_name = gui_aux.get_disp_name(name)
        self.disp_name = disp_name
        self.state = state
        self.toggle = toggle
        content = self.arrange(content, Ncols=Ncols, col_idx=col_idx)
        if use_header:

            self.auto_open = auto_open
            self.sec_key = f'SEC {name}'

            self.toggle_key = f'TOGGLE_{name}'
            self.sec_symbol = self.get_symbol()
            self.header_dict = header_dict
            self.header_value = header_value
            if header_key is None:
                header_key = f'SELECT LIST {name}'
            self.header_key = header_key
            after_header = [gui_aux.BoolButton(name, toggle, disabled)] if toggle is not None else []
            if next_to_header is not None:
                after_header += next_to_header
            header_kws = {'text': disp_name, 'text_kws': header_text_kws,
                          'before_header': [self.sec_symbol], 'after_header': after_header}

            if header_dict is None:
                header_l = Header(name, **header_kws)
            else:
                header_l = NamedList(name, choices=list(header_dict.keys()),
                                     default_value=header_value, key=self.header_key,
                                     size=(header_list_width, None), list_kws=header_list_kws,
                                     header_kws=header_kws)
            header = header_l.get_layout()
            HeadedElement.__init__(self, name=name, header=header,
                                   content=[collapse(content, self.sec_key, self.state)],
                                   single_line=False)
        else:
            GuiElement.__init__(self, name=name, layout=content)

    def get_symbol(self):
        return sg.T(SYMBOL_DOWN if self.state else SYMBOL_UP, k=f'OPEN {self.sec_key}',
                    enable_events=True, text_color='black', **gui_aux.t_kws(2))

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
                    if isinstance(b, gui_aux.BoolButton):
                        b.set_state(v)
                elif type(v) == dict:
                    self.update_window(w, v, prefix=k if prefix is not None else None)
                elif isinstance(w[k], MultiSpin) or isinstance(w[k], SingleSpin):
                    w[k].update(v)
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
            self.sec_symbol.update(SYMBOL_DOWN if self.state else gui_aux.YMBOL_UP)
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

    def arrange(self, content, Ncols=1, col_idx=None):
        if col_idx is not None:
            content = [[content[i] for i in idx] for idx in col_idx]
            content = [[sg.Col(ii, **gui_aux.col_kws) for ii in content]]
        elif Ncols > 1:
            content = aux.group_list_by_n([*content], int(np.ceil(len(content) / Ncols)))
            content = [[sg.Col(ii, **gui_aux.col_kws) for ii in content]]
        return content


class CollapsibleTable(Collapsible):
    def __init__(self, name, index=None, dict_name=None, heading_dict=None, dict={},
                 buttons=[], button_args={}, col_widths=None, num_rows=1, **kwargs):
        if dict_name is None:
            dict_name = name
        if index is None:
            index = name
        self.index = index
        self.dict_name = dict_name
        self.key = f'TABLE {name}'
        # from lib.registry.dtypes import null_dict

        self.null_dict = reg.get_null(dict_name)
        if heading_dict is None:
            heading_dict = {k: k for k in self.null_dict.keys()}
        self.heading_dict = heading_dict
        self.heading_dict_inv = {v: k for k, v in heading_dict.items()}
        self.headings = list(heading_dict.keys())
        self.dict = dict
        self.data = self.dict2data()
        self.Ncols = len(self.headings) + 1

        col_visible = [True] * self.Ncols
        self.color_idx = None
        for i, p in enumerate(self.headings):
            if p in ['color']:
                self.color_idx = i + 1
                col_visible[i + 1] = False
        if col_widths is None:
            col_widths = [10]
            for i, p in enumerate(self.headings):
                if p in ['id', 'group']:
                    col_widths.append(10)
                elif p in ['color']:
                    col_widths.append(8)
                elif p in ['model']:
                    col_widths.append(14)
                else:
                    col_widths.append(10)
        after_header = gui_aux.button_row(name, buttons, button_args)
        content = [[Table(values=self.data, headings=[index] + self.headings,
                          def_col_width=7, key=self.key, num_rows=max([num_rows, len(self.data)]),
                          col_widths=col_widths, visible_column_map=col_visible)]]
        super().__init__(name, content=content, next_to_header=after_header, **kwargs)

    def update(self, w, dic=None, use_prefix=True):
        if dic is not None:
            self.dict = dic
        self.data = self.dict2data()
        if self.color_idx is not None:
            row_cols = []
            for i in range(len(self.data)):
                c0 = self.data[i][self.color_idx]
                if c0 == '':
                    c2, c1 = ['lightblue', 'black']
                else:
                    try:
                        c2, c1 = aux.invert_color(c0, return_self=True)
                    except:
                        c2, c1 = ['lightblue', 'black']
                row_cols.append((i, c1, c2))
        else:
            row_cols = None
        w[self.key].update(values=self.data, num_rows=len(self.data), row_colors=row_cols)
        if self.auto_open:
            self.open(w) if len(self.dict) > 0 else self.close(w)

    def get_dict(self, *args, **kwargs):
        return self.dict

    def dict2data(self):
        dH = self.heading_dict
        d = self.dict
        data = []
        for id in d.keys():
            dF = d[id].flatten()
            row = [id] + [dF[dH[h]] for h in self.headings]
            data.append(row)
        return data

    def eval(self, e, v, w, c, d, g):
        K = self.key
        Ks = v[K]
        if e == f'ADD {self.name}':
            id = self.data[Ks[0]][0] if len(Ks) > 0 else None
            entry = gui_aux.entry_window(id=id, base_dict=self.dict)
            self.dict.update(**entry)
            self.update(w)
        elif e == f'REMOVE {self.name}':
            for k in Ks:
                id = self.data[k][0]
                self.dict.pop(id, None)
            self.update(w)


def v_layout(k0, args, value_kws0={}, **kwargs):
    t = args['dtype']
    v = args['initial_value']
    vs = args['values']
    Ndig = args['Ndigits']
    value_kws = copy.deepcopy(value_kws0)
    if 'size' not in value_kws0.keys() and Ndig is not None:
        value_kws['size'] = (Ndig, 1)
    if t == bool:
        temp = gui_aux.BoolButton(k0, v)
    elif t == str:
        if vs is None:
            temp = sg.In(v, key=k0, **value_kws)
        else:
            temp = sg.Combo(vs, default_value=v, key=k0, enable_events=True, readonly=True, **value_kws)
    elif t == List[str]:
        temp = sg.In(v, key=k0, **value_kws)
    else:
        if Ndig is not None:
            value_kws['size'] = (Ndig, 1)
        spin_kws = {
            'values': vs,
            'initial_value': v,
            'key': k0,
            'dtype': aux.base_dtype(t),
            'value_kws': value_kws,
            **kwargs
        }
        if t in [List[float], List[int]]:
            temp = MultiSpin(tuples=False, **spin_kws)
        elif t in [List[Tuple[float]], List[Tuple[int]]]:
            # print(k0)
            temp = MultiSpin(tuples=True, **spin_kws)
        elif t in [Tuple[float], Tuple[int]]:
            temp = MultiSpin(**spin_kws, Nspins=2)
        elif t in [float, int]:
            temp = SingleSpin(**spin_kws)
    return temp


def combo_layout(name, title, dic, **kwargs):
    d = {p: [] for p in ['mu', 'std', 'r', 'noise']}
    for k, args in dic.items():
        kws = {
            'values': args['values'],

            'key': f'{name}_{k}',
            **kwargs
        }
        spin_kws = {
            'initial_value': args['initial_value'],
            'dtype': aux.base_dtype(args['dtype']),
            'value_kws': gui_aux.t_kws(5),
            **kws
        }
        disp = args['disp']
        if disp in ['initial', 'mean']:
            d['mu'] = [sg.T(f'{disp}:', **gui_aux.t_kws(5)), SingleSpin(**spin_kws)]
        elif disp in ['noise']:
            d['noise'] = [sg.T(f'{disp}:', **gui_aux.t_kws(5)), SingleSpin(**spin_kws)]
        elif disp in ['std']:
            d['std'] = [sg.T(f'{disp}:', **gui_aux.t_kws(3)), SingleSpin(**spin_kws)]
        elif disp in ['range']:
            d['r'] = [sg.T(f'{disp}:', **gui_aux.t_kws(5)), MultiSpin(**spin_kws, Nspins=2)]
        elif disp in ['name']:
            d['name'] = [sg.T(f'{title}:', **gui_aux.t_kws(6)),
                         sg.Combo(**kws, default_value=args['initial_value'], enable_events=True, readonly=True,
                                  **gui_aux.t_kws(10))]
        elif disp in ['fit']:
            d['fit'] = gui_aux.BoolButton(kws['key'], args['initial_value'])
    if 'name' in d.keys():
        ii = d['name']
    else:
        ii = [sg.T(f'{title}', **gui_aux.t_kws(20), justification='center', font=('Helvetica', 10, 'bold'))]
    if 'fit' in d.keys():
        ii.append(d['fit'])
    l = [sg.Col([ii, d['mu'] + d['std'] + d['noise'], d['r']], vertical_alignment=True)]
    return [sg.Pane(l, border_width=4)]

def collapse(layout, key, visible=True):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this seciton visible / invisible
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Col(layout, key=key, visible=visible))


class CollapsibleDict(Collapsible):
    def __init__(self, name, dict_name=None, type_dict=None, value_kws={}, text_kws={}, as_entry=None,
                 subdict_state=False, **kwargs):
        if type_dict is None:
            from larvaworld.gui.gui_aux.dtypes import par,par_dict
            entry = par(name=as_entry, dtype=str, v='Unnamed') if as_entry is not None else {}
            if dict_name is None:
                dict_name = name
            D=reg.par.PI
            if dict_name in D.keys():
                dic = par_dict(d0=D[dict_name])
                type_dict = {**entry, **dic}
        self.as_entry = as_entry
        self.subdict_state = subdict_state

        content, self.subdicts = self.build(name, type_dict, value_kws, text_kws)
        self.dtypes = {k: type_dict[k]['dtype'] for k in type_dict.keys()}
        del type_dict
        super().__init__(name, content=content, **kwargs)

    def get_subdicts(self):
        subdicts = {}
        for s in list(self.subdicts.values()):
            subdicts.update(s.get_subdicts())
        return {self.name: self, **subdicts}

    def get_dict(self, v, w, check_toggle=True):
        # from lib.registry.dtypes import base_dtype
        if self.state is None or (check_toggle and self.toggle == False):
            return None
        else:
            d = {}
            for k, t in self.dtypes.items():
                k0 = f'{self.name}_{k}'
                if t == bool:
                    d[k] = w[f'TOGGLE_{k0}'].get_state()
                elif aux.base_dtype(t) in [int, float]:
                    d[k] = w[k0].get()
                elif t == dict or type(t) == dict:
                    d[k] = self.subdicts[k0].get_dict(v, w)
                else:
                    d[k] = gui_aux.retrieve_value(v[k0], t)
            if self.as_entry is None:
                return d
            else:
                id = d[self.as_entry]
                d.pop(self.as_entry, None)
                return {id: d}

    def set_element_size(self, text_kws, Ncols):
        if 'size' not in text_kws.keys():
            text_kws['size'] = gui_aux.w_kws['default_element_size']
        text_kws['size'] = int(text_kws['size'][0] / Ncols), text_kws['size'][1]
        return text_kws

    def build(self, name, type_dict, value_kws={}, text_kws={}):
        subdicts = {}
        content = []
        for k, args in type_dict.items():
            k0 = f'{name}_{k}'
            if args['dtype'] == dict:
                subdicts[k0] = CollapsibleDict(k0, disp_name=k, type_dict=args['content'],
                                               text_kws=text_kws, value_kws=value_kws, state=self.subdict_state)
                ii = subdicts[k0].get_layout()
            else:
                temp = v_layout(k0, args, value_kws)
                ii = [sg.T(f'{gui_aux.get_disp_name(k)}:', **text_kws), temp]
            content.append(ii)
        return content, subdicts


class PadElement:
    def __init__(self, name, dict_name=None, disp_name=None, toggle=None, disabled=False, text_kws={}, value_kws={},
                 layout_pane_kwargs={'border_width': 8}, header_width=None, background_color=None, after_header=None,
                 **kwargs):
        self.name = name
        self.background_color = background_color
        self.layout_pane_kwargs = layout_pane_kwargs
        self.header_width = header_width
        self.toggle = toggle
        self.disabled = disabled
        self.text_kws = text_kws
        self.value_kws = value_kws
        self.after_header = after_header
        self.toggle_key = f'TOGGLE_{name}'
        self.subdicts = {}
        if disp_name is None:
            disp_name = gui_aux.get_disp_name(self.name)
        self.disp_name = disp_name

        if dict_name is None:
            dict_name = name
        self.dict_name = dict_name

    def build_header(self, header_width):
        header = [
            [sg.T(self.disp_name.upper(), justification='center', background_color=self.background_color,
                  border_width=3,
                  **gui_aux.t_kws(header_width)
                  )]]
        if self.after_header is not None:
            header[0] += self.after_header
        if self.toggle is not None:
            header[0] += [gui_aux.BoolButton(self.name, self.toggle, self.disabled)]
        return header

    def get_layout(self, as_col=True, as_pane=True, **kwargs):
        kws = copy.deepcopy(self.layout_pane_kwargs)
        kws.update(kwargs)
        return [[sg.Pane([sg.Col(self.layout)], **kws)]]

    def get_subdicts(self):
        subdicts = {}
        for s in list(self.subdicts.values()):
            subdicts.update(s.get_subdicts())
        return {self.name: self, **subdicts}

    def disable(self, w):
        if self.toggle is not None:
            w[self.toggle_key].set_state(disabled=True, state=False)

    def enable(self, w):
        if self.toggle is not None:
            w[self.toggle_key].set_state(disabled=False, state=True)


class PadDict(PadElement):
    def __init__(self, name, Ncols=1, subconfs={}, col_idx=None, header_width=None, row_idx=None,
                 type_dict=None, content=None, **kwargs):
        self.subconfs = subconfs
        super().__init__(name=name, **kwargs)
        if col_idx is None:
            col_idx = col_idx_dict.get(self.dict_name, None)
        if col_idx is not None:
            Ncols = len(col_idx)
        if type_dict is None :
            from larvaworld.gui.gui_aux.dtypes import par_dict
            D = reg.par.PI
            if self.dict_name in D.keys() :
                type_dict = par_dict(d0=D[self.dict_name])
        self.type_dict = type_dict
        if content is None:
            content = self.build(name)
        self.content = self.arrange_content(content, col_idx=col_idx, row_idx=row_idx, Ncols=Ncols)
        self.dtypes = {k: type_dict[k]['dtype'] for k in type_dict.keys()} if type_dict is not None else None

        self.header = self.build_header(self.get_header_width(header_width, Ncols))
        self.layout = self.header + self.content

    def get_header_width(self, header_width, Ncols):
        if header_width is not None:
            return header_width
        else:
            if 'size' in self.text_kws.keys():
                s1 = self.text_kws['size'][0]
            else:
                s1 = gui_aux.w_kws['default_element_size'][0]
            if 'size' in self.value_kws.keys():
                s2 = self.value_kws['size'][0]
            else:
                s2 = gui_aux.w_kws['default_button_element_size'][0]
            s = s1 + s2
            return (s + 1) * Ncols

    def arrange_content(self, content, col_idx=None, row_idx=None, Ncols=1):
        # print(len(content), col_idx, self.name, self.dict_name)
        if col_idx is not None:
            content = [[content[i] for i in idx] for idx in col_idx]
            content = [[sg.Col(ii, **gui_aux.col_kws) for ii in content]]
        elif row_idx is not None:
            content = [[content[i] for i in idx] for idx in row_idx]
        elif Ncols > 1:
            content = aux.group_list_by_n([*content], int(np.ceil(len(content) / Ncols)))
            content = [[sg.Col(ii, **gui_aux.col_kws) for ii in content]]
        return content

    def get_dict(self, v, w):
        # from lib.registry.dtypes import base_dtype
        if self.toggle is not None:
            if not w[self.toggle_key].get_state():
                return None

        d = {}
        for k, t in self.dtypes.items():
            k0 = f'{self.name}_{k}'
            if t == bool:
                d[k] = w[f'TOGGLE_{k0}'].get_state()
            elif aux.base_dtype(t) in [int, float]:
                d[k] = w[k0].get()
            elif t in [dict, TypedDict] or type(t) == dict:
                d[k] = self.subdicts[k0].get_dict(v, w)
            else:
                d[k] = gui_aux.retrieve_value(v[k0], t)
        return d

    def build(self, name, **kwargs):
        combos = {}
        l = []
        for k, args in self.type_dict.items():
            subconfkws = self.subconfs[k] if k in self.subconfs.keys() else {}
            if args['dtype'] in [dict, TypedDict]:
                k0 = f'{name}_{k}'
                subkws = {
                    'text_kws': self.text_kws,
                    'value_kws': self.value_kws,
                    'background_color': self.background_color,
                    **kwargs
                }
                subkws.update(subconfkws)
                if args['dtype'] == dict:
                    self.subdicts[k0] = PadDict(k0, disp_name=k, dict_name=k, type_dict=args['content'], **subkws)
                else:
                    self.subdicts[k0] = PadTable(k0, dict_name=args['entry'], disp_name=args['disp'],
                                                 index=f'ID', **subkws)
                ii = self.subdicts[k0].get_layout()[0]
            elif args['combo'] is not None:
                if args['combo'] not in combos.keys():
                    combos[args['combo']] = {}
                combos[args['combo']].update({k: args})
                continue
            else:
                text_kws = {**self.text_kws}
                if 'text_kws' in subconfkws.keys():
                    text_kws.update(subconfkws['text_kws'])
                # # FIXME temporary solution
                # if 'header_width' in subconfkws.keys():
                #     subconfkws.pop('header_width' , None)
                disp = args['disp']
                ii = [sg.T(f'{gui_aux.get_disp_name(disp)}:', tooltip=args['tooltip'], **text_kws),
                      v_layout(f'{name}_{k}', args, self.value_kws, **subconfkws)]
            l.append(ii)
        for title, dic in combos.items():
            l.append(combo_layout(name, title, dic))
        return l

    def update(self, w, d):
        if d is not None:
            for k, t in self.dtypes.items():
                k0 = f'{self.name}_{k}'
                if t == bool:
                    w[f'TOGGLE_{k0}'].set_state(d[k])
                elif aux.base_dtype(t) in [int, float]:
                    w[k0].update(d[k])
                elif t in [dict, TypedDict] or type(t) == dict:
                    self.subdicts[k0].update(w, d[k])
                elif d[k] is None:
                    w.Element(k0).Update(value='')
                else:
                    w.Element(k0).Update(value=d[k])
            try:
                self.enable(w)
            except:
                pass
        else:
            try:
                self.disable(w)
            except:
                pass
        return w


class PadTable(PadElement):
    def __init__(self, name, index=None, heading_dict=None, dict={}, header_width=28,
                 buttons=[], button_args={}, col_widths=None, num_rows=5, **kwargs):
        after_header = gui_aux.button_row(name, buttons, button_args)
        super().__init__(name=name, after_header=after_header, **kwargs)
        if index is None:
            index = self.name
        self.index = index
        self.key = f'TABLE {self.name}'
        if heading_dict is None:
            # from lib.registry.dtypes import null_dict

            heading_dict = {k: k for k in reg.get_null(self.dict_name).keys()}
        self.heading_dict = heading_dict
        self.headings = list(heading_dict.keys())
        self.dict = dict
        self.data = self.dict2data()
        col_visible = [True] * (len(self.headings) + 1)
        self.color_idx = None
        for i, p in enumerate(self.headings):
            if p in ['color']:
                self.color_idx = i + 1
                col_visible[i + 1] = False
        self.header_width = header_width - 2 * len(buttons)
        self.content = self.build(col_widths=col_widths, num_rows=num_rows, col_visible=col_visible)
        self.header = self.build_header(self.header_width)
        self.layout = self.header + self.content

    def build(self, col_widths, num_rows, col_visible):
        if col_widths is None:
            w = int(self.header_width / (len(self.headings) + 1))
            col_widths = [w]
            for i, p in enumerate(self.headings):
                if p in ['id', 'group']:
                    col_widths.append(w)
                elif p in ['color']:
                    col_widths.append(8)
                elif p in ['model']:
                    col_widths.append(w + 4)
                elif p in ['N']:
                    col_widths.append(4)
                else:
                    col_widths.append(w)
        return [[Table(values=self.data, headings=[self.index] + self.headings,
                       def_col_width=7, key=self.key, num_rows=max([num_rows, len(self.data)]),
                       col_widths=col_widths, visible_column_map=col_visible)]]

    def update(self, w, d=None):
        if d is not None:
            self.dict = d
        self.data = self.dict2data()
        if self.color_idx is not None:
            row_cols = []
            for i in range(len(self.data)):
                c0 = self.data[i][self.color_idx]
                if c0 == '':
                    c2, c1 = ['lightblue', 'black']
                else:
                    try:
                        c2, c1 = aux.invert_color(c0, return_self=True)
                    except:
                        c2, c1 = ['lightblue', 'black']
                row_cols.append((i, c1, c2))
        else:
            row_cols = None
        w[self.key].update(values=self.data, num_rows=len(self.data), row_colors=row_cols)

    def get_dict(self, *args, **kwargs):
        return self.dict

    def dict2data(self):
        dH = self.heading_dict
        d = self.dict
        data = []
        for id in d.keys():
            dF = d[id].flatten()
            row = [id] + [dF[dH[h]] for h in self.headings]
            data.append(row)
        return data

    def eval(self, e, v, w, c, d, g):
        K = self.key
        Ks = v[K]
        if e == f'ADD {self.name}':
            if len(Ks) > 0:
                id = self.data[Ks[0]][0]
            else:
                id = None
            entry = gui_aux.entry_window(id=id, base_dict=self.dict, index=self.index, dict_name=self.dict_name)
            self.dict.update(**entry)
            self.update(w)
        elif e == f'REMOVE {self.name}':
            for k in Ks:
                id = self.data[k][0]
                self.dict.pop(id, None)
            self.update(w)
        elif e == f'CONF_TREE {self.name}':
            ids = [ff['model'] for ff in list(self.dict.values())]
            entries = gui_aux.multiconf_to_tree(ids, 'Model')
            tree = GuiTreeData(entries=entries, headings=[ids], col_widths=[40] + [20] * len(ids))

            tree.test()


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
        window.Element(self.Key).Update(values=vs, num_rows=len(vs))

    def remove_row(self, window, idx):
        vs = self.get()
        vs.remove(vs[idx])
        window.Element(self.Key).Update(values=vs, num_rows=len(vs))


class GraphList(NamedList):
    def __init__(self, name, tab, fig_dict={}, next_to_header=None, default_values=None, canvas_size=(1000, 800),
                 list_size=None, list_header='Graphs', auto_eval=True, canvas_kws={'background_color': 'Lightblue'},
                 graph=False, subsample=1, **kwargs):
        self.tab = tab
        self.tab.graphlists[name] = self
        self.fig_dict = fig_dict
        self.subsample = subsample
        self.auto_eval = auto_eval
        self.list_key = f'{name}_GRAPH_LIST'
        values = list(fig_dict.keys())
        if list_size is None:
            h = int(np.max([len(values), 10]))
            list_size = (gui_aux.w_list, h)
        header_kws = {'text': list_header, 'after_header': next_to_header,
                      'text_kws': gui_aux.t_kws(14), 'single_line': False}
        default_value = default_values[0] if default_values is not None else None
        super().__init__(name=name, key=self.list_key, choices=values, default_value=default_value, drop_down=False,
                         size=list_size, header_kws=header_kws, auto_size_text=True, **kwargs)
        self.canvas_size = canvas_size
        self.canvas_key = f'{name}_CANVAS'
        self.canvas, self.canvas_element = self.init_canvas(canvas_size, canvas_kws, graph)
        self.fig_agg = None

    def init_canvas(self, size, canvas_kws, graph=False):
        k = self.canvas_key
        if graph:
            g = sg.Graph(canvas_size=size, k=k, **canvas_kws)
        else:
            g = sg.Canvas(size=size, k=k, **canvas_kws)
        canvas = GuiElement(name=k, layout=[[g]])
        return canvas, g

    def draw_fig(self, w, fig):
        if self.fig_agg:
            delete_figure_agg(self.fig_agg)
        self.fig_agg = draw_canvas(w[self.canvas_key].TKCanvas, fig)

    def update(self, w, fig_dict=None):
        if fig_dict is None:
            fig_dict = self.fig_dict
        else:
            self.fig_dict = fig_dict
        w.Element(self.list_key).Update(values=list(fig_dict.keys()))

    def eval(self, e, v, w, c, d, g):
        if e == self.list_key and self.auto_eval:
            v0 = v[self.list_key]
            if len(v0) > 0:
                choice = v0[0]
                fig = self.fig_dict[choice]
                if type(fig) == str and os.path.isfile(fig):
                    self.show_fig(w, fig)
                else:
                    self.draw_fig(w, fig)

    def show_fig(self, w, fig):
        from tkinter import PhotoImage

        c = w[self.canvas_key].TKCanvas
        c.pack()
        img = PhotoImage(file=fig).subsample(self.subsample)
        W, H = self.canvas_size
        c.create_image(int(W / 2), int(H / 2), image=img)
        self.fig_agg = img


class ButtonGraphList(GraphList):
    def __init__(self, name, buttons=['refresh_figs', 'conf_fig', 'draw_fig', 'save_fig'], button_args={}, **kwargs):
        super().__init__(name=name, next_to_header=gui_aux.button_row(name, buttons, button_args), **kwargs)
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
                self.fig, self.save_to, self.save_as = self.func(datasets=list(data.values()),
                                                                 return_fig=True, **self.func_kws)
                fig = resize_fig(self.fig, self.canvas_size)
                self.draw_fig(w, fig)
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
                [sg.T('Filename', **gui_aux.t_kws(10)), sg.In(default_text=self.save_as, k=kDir, **gui_aux.t_kws(80))],
                [sg.T('Directory', **gui_aux.t_kws(10)), sg.In(self.save_to, k=kFil, **gui_aux.t_kws(80)),
                 sg.FolderBrowse(initial_folder=reg.ROOT_DIR, key=kFil, change_submits=True)],
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
            v1 = v[f'BROWSE_FIGS {n}']
            id = v0.split('/')[-1].split('.')[-2]
            d0[id] = v1
            self.update(w, d0)
        elif e == f'REMOVE_FIGS {n}':
            v0 = v[k]
            for kk in v0:
                d0.pop(kk, None)
            self.update(w, d0)
        elif e == k:
            v0 = v[k]
            if len(v0) > 0:
                fig = d0[v0[0]]
                try:
                    if fig != self.func:
                        self.func = fig
                        self.func_kws = self.get_graph_kws(self.func)
                except:
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
                self.func_kws = gui_aux.set_kwargs(self.func_kws, title='Graph arguments')
        elif e == f'DRAW_FIG {n}':
            self.generate(w, self.tab.base_dict)


def draw_canvas(canvas, figure, side='top', fill='both', expand=True):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    agg = FigureCanvasTkAgg(figure, canvas)
    agg.draw()
    agg.get_tk_widget().pack(side=side, fill=fill, expand=expand)
    return agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


def resize_fig(fig, canvas_size, margin=1.0):
    x, y = fig.get_size_inches() * fig.dpi  # size in pixels
    x0, y0 = canvas_size
    x0 *= margin
    y0 *= margin
    if x > x0:
        y *= (x0 / x)
        x = x0
    if y > y0:
        x *= (y0 / y)
        y = y0
    fig.set_size_inches(x / fig.dpi, y / fig.dpi)
    fig.tight_layout()
    return fig


class DynamicGraph:
    def __init__(self, agent, pars=[], available_pars=None):
        sg.theme('DarkBlue15')
        self.agent = agent
        if available_pars is None:
            available_pars = reg.par.runtime_pars()
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
              [sg.Col([*[[sg.CB(p, k=f'k_{p}', **gui_aux.t_kws(24))] for p in par_lists[i]]]) for i in range(Ncols)],
              [sg.B('Ok', **gui_aux.t_kws(8)), sg.B('Cancel', **gui_aux.t_kws(8))]]
        l1 = [
            [sg.Canvas(size=(1280, 1200), k='-CANVAS-')],
            [sg.T('Time in seconds to display on screen')],
            [sg.Slider(range=(0.1, 60), default_value=self.init_dur, size=(40, 10), orientation='h',
                       k='-SLIDER-TIME-')],
            [sg.B('Choose', **gui_aux.t_kws(8))]]
        return [[sg.Col(l0, k='-COL1-'), sg.Col(l1, visible=False, k='-COL2-')]]

    def evaluate(self):
        w = self.window
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
        from matplotlib import ticker
        self.pars, syms, us, lims, pcs = reg.getPar(d=self.pars, to_return=['d', 's', 'l', 'lim', 'p'])
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


class GuiTreeData(sg.TreeData):
    def __init__(self, name='Model', root_key=None, build_tree=False, entries=None, headings=None,
                 col_widths=[20, 10, 80], **kwargs):
        super().__init__()
        if root_key is None:
            root_key = name
        self.name = name
        self.col_widths = col_widths
        self.w_width = np.sum(col_widths)
        self.root_key = root_key
        self.build_tree = build_tree
        self.headings = headings
        self.entries = self.get_entries() if entries is None else entries
        self.build()

    def get_value_arg(self, node, arg):
        if hasattr(node, arg):
            return getattr(node, arg)
        elif arg in self.headings:
            idx = self.headings.index(arg)
            try:
                return node.values[idx]
            except:
                return None

    def _NodeStr(self, node=None, level=1, k='name', v='description'):
        if node is None:
            node = self.root_node

        def str_pair(node, k, v, max_l=50):
            k0 = str(self.get_value_arg(node, k))
            v0 = str(self.get_value_arg(node, v))
            if v0 == ' ':
                return k0
            if max_l is not None:
                b = '{:<' + str(max_l - 4 * (level - 1)) + '}'
                k0 = b.format(k0)
            kv0 = k0 + ' : ' + v0
            return kv0

        """
        Does the magic of converting the TreeData into a nicely formatted string version

        :param node:  The node to begin printing the tree
        :type node: (TreeData.Node)
        :param level: The indentation level for string formatting
        :type level: (int)
        """
        return '\n'.join(
            [str_pair(node, k, v)] +
            [' ' * 4 * level + self._NodeStr(child, level + 1, k, v) for child in node.children])

    def get_df(self):
        if not self.build_tree and self.root_key in reg.storedConf('Tree'):
            df = pd.DataFrame.from_dict(reg.loadConf(id=self.root_key, conftype='Tree'))
        else:
            df = gui_aux.pars_to_tree(self.root_key)
            reg.saveConf(conf=df.to_dict(), conftype='Tree', id=self.root_key)
        return df

    def get_entries(self):
        self.df = self.get_df()
        if self.headings is None:
            self.headings = self.df.columns.values.tolist()[3:]
        entries = []
        for _, row in self.df.iterrows():
            d = row.to_dict()
            dd = {}
            dd['parent'] = d['parent'] if d['parent'] != 'root' else ''
            dd['key'] = d['key']
            dd['text'] = d['text']
            dd['values'] = [d[h] for h in self.headings]
            entries.append(dd)
        return entries

    def build(self):
        for entry in self.entries:
            self.insert(**entry)

    def save(self, **kwargs):
        with open(f'{reg.CONF_DIR}/glossary.txt', 'w') as f:
            f.write(self._NodeStr(**kwargs))

    def build_layout(self):
        return [
            [sg.Tree(self, headings=self.headings, auto_size_columns=False, show_expanded=True, justification='center',
                     max_col_width=1000, def_col_width=20, row_height=50, num_rows=30, col_widths=self.col_widths,
                     col0_width=20)]]

    def test(self):
        w = sg.Window('Parameter tree', self.build_layout(), size=(self.w_width * 20, 800))
        while True:
            e, v = w.read()
            if e == 'Ok':
                pass
            elif e in ['Cancel', None]:
                break
        w.close()


def detect_dataset(datagroup_id=None, path=None, raw=True, **kwargs):
    dic = {}
    if path in ['', None]:
        return dic
    if raw:
        conf = reg.loadConf(id=datagroup_id, conftype='Group').tracker.filesystem
        dF, df = conf.folder, conf.file
        dFp, dFs = dF.pref, dF.suf
        dfp, dfs, df_ = df.pref, df.suf, df.sep

        fn = path.split('/')[-1]
        if dFp is not None:
            if fn.startswith(dFp):
                dic[fn] = path
            else:
                ids, dirs = detect_dataset_in_subdirs(datagroup_id, path, fn, **kwargs)
                for id, dr in zip(ids, dirs):
                    dic[id] = dr
        elif dFs is not None:
            if fn.startswith(dFs):
                dic[fn] = path
            else:
                ids, dirs = detect_dataset_in_subdirs(datagroup_id, path, fn, **kwargs)
                for id, dr in zip(ids, dirs):
                    dic[id] = dr
        elif dfp is not None:
            ids, dirs = [f.split(df_)[1:][0] for f in os.listdir(path) if f.startswith(dfp)], [path]
            for id, dr in zip(ids, dirs):
                dic[id] = dr
        elif dfs is not None:
            ids = [f.split(df_)[:-1][0] for f in os.listdir(path) if f.endswith(dfs)]
            for id in ids:
                dic[id] = path
        elif df_ is not None:
            ids = aux.unique_list([f.split(df_)[0] for f in os.listdir(path) if df_ in f])
            for id in ids:
                dic[id] = path
        return dic
    else:
        from larvaworld.lib.process.dataset import LarvaDataset
        if os.path.exists(f'{path}/data'):
            dd = LarvaDataset(dir=path)
            dic[dd.id] = dd
        else:
            for ddr in [x[0] for x in os.walk(path)]:
                if os.path.exists(f'{ddr}/data'):
                    dd = LarvaDataset(dir=ddr)
                    dic[dd.id] = dd
        return dic


def detect_dataset_in_subdirs(datagroup_id, path, last_dir, full_ID=False):
    fn = last_dir
    ids, dirs = [], []
    if os.path.isdir(path):
        for f in os.listdir(path):
            dic = detect_dataset(datagroup_id, f'{path}/{f}', full_ID=full_ID, raw=True)
            for id, dr in dic.items():
                if full_ID:
                    ids += [f'{fn}/{id0}' for id0 in id]
                else:
                    ids.append(id)
                dirs.append(dr)
    return ids, dirs



