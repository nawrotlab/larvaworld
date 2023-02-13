import PySimpleGUI as sg
import copy
import os


from larvaworld.lib import reg
from larvaworld.gui import gui_aux
from larvaworld.gui.tabs import DrawBodyTab

class ModelTab(gui_aux.GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = ['physics', 'body']
        self.module_keys = list(reg.get_null('modules').keys())
        self.energetics_keys = list(reg.get_null('energetics').keys())
        self.canvas_size = (1200, 1000)


    def update(self, w, c, conf, id=None):
        for n in self.fields:
            c[n].update(w, conf[n])
        if conf['energetics'] is None :
            for k in self.energetics_keys:
                c[k].disable(w)
        else:
            for k, v in conf['energetics'].items():
                dic = conf['energetics'][k]
                c[k].update(w, dic)
                c[k].enable(w)


        module_dict = conf['brain']['modules']
        for k, v in module_dict.items():
            dic = conf['brain'][f'{k}_params']
            if k == 'olfactor':
                if dic is not None:
                    odor_gains = dic['odor_dict']
                    dic.pop('odor_dict')
                else:
                    odor_gains = {}
                c['odor_gains'].update(w, odor_gains)
            c[k].update(w, dic)
            if v :
                c[k].enable(w)
            else :
                c[k].disable(w)
        # c['Brain'].update(w, module_dict, use_prefix=False)
        for kk in ['nengo'] :
            w[f'TOGGLE_{kk}'].set_state(conf['brain'][kk])

    def get(self, w, v, c, as_entry=True):
        module_dict = dict(zip(self.module_keys, [w[f'TOGGLE_{k}'].get_state() for k in self.module_keys]))
        m = {}
        for n in self.fields:
            m[n] = None if c[n].disabled else c[n].get_dict(v, w)
        if all([c[k].disabled for k in self.energetics_keys]):
            m['energetics'] = None
        else :
            m['energetics'] = {k : c[k].get_dict(v, w) for k in self.energetics_keys}

        b = {}
        b['modules'] = module_dict
        for k in module_dict.keys():
            b[f'{k}_params'] = c[k].get_dict(v, w)
        if b['olfactor_params'] is not None:
            b['olfactor_params']['odor_dict'] = c['odor_gains'].dict
        for kk in ['nengo']:
            b[kk] = w[f'TOGGLE_{kk}'].get_state()
        m['brain'] = b
        return copy.deepcopy(m)

    def build_module_tab(self):
        Cmod, Cbody, Ceffector, Csens, Cmem, Cener = 'purple', 'orange', 'red', 'green', 'blue', 'yellow'
        col_dict = {
            'crawler': Ceffector,
            'turner': Ceffector,
            'interference': Cmod,
            'intermitter': Cmod,
            'olfactor': Csens,
            'toucher': Csens,
            'windsensor': Csens,
            'memory': Cmem,
            'body': Cbody,
            'physics': Cbody,
            'feeder': Ceffector,
            'DEB': Cener,
            'gut': Cener,
            # 'energetics': Cener,
        }
        dic = {

            'modulation/memory': ['intermitter', 'interference', 'memory'],
            'effectors': ['crawler', 'turner', 'feeder'],
            'sensors': ['olfactor', 'toucher', 'windsensor'],
            # 'modulation/memory': ['intermitter'],
            'body': ['body', 'physics'],
            'energetics': ['DEB', 'gut'],
            # 'energetics': ['energetics'],
        }

        ss = (500, 650)
        c = {}
        s1 = gui_aux.CollapsibleTable('odor_gains', index='ID', heading_dict={'mean': 'mean', 'std': 'std'}, state=True,
                                      col_widths=[13, 7, 6])
        c.update(s1.get_subdicts())
        tt = []
        pp = {}
        for k, vs in dic.items():
            ppp = []
            for v in vs:
                tS=18 if v in ['gut'] else 16
                cc = gui_aux.PadDict(v, toggle=True if v not in ['body', 'physics'] else None, background_color=col_dict[v], text_kws=gui_aux.t_kws(tS))
                c.update(cc.get_subdicts())
                if v=='olfactor' :
                    ll = cc.get_layout(size=(ss[0], int(ss[1]/2)))
                    ll.append(s1.get_layout())
                else :
                    ll = cc.get_layout(size=ss)
                ppp.append(sg.Col(ll, scrollable=True, vertical_scroll_only=True))
            pp[k] = [[sg.Pane(ppp, orientation='horizontal', show_handle=False)]]
            tt.append(sg.Tab(k, pp[k], key=f'{k} MODULES'))

        modules_tab = sg.TabGroup([tt], key='ACTIVE_MODULES', tab_location='topcenter', **gui_aux.tab_kws)
        l=sg.Col([[modules_tab]])
        return l, c

    def build_architecture_tab(self):
        fdir = f'{reg.ROOT_DIR}/gui/media/model_figures'
        # fdir = reg.Path["model"]
        fig_dict = {f: f'{fdir}/{f}' for f in sorted(os.listdir(fdir))}
        g2 = gui_aux.GraphList(self.name, tab=self, list_header='Model', fig_dict=fig_dict, subsample=3,
                               canvas_size=self.canvas_size)
        col1 = gui_aux.gui_col([g2], x_frac=0.2, y_frac=0.6, as_pane=True, pad=(20, 20))
        col2 = sg.Col([g2.canvas.get_layout(as_pane=True, pad=(0, 10))[0]], **gui_aux.col_kws)
        l2 = [[col1, col2]]
        return l2, {g2.name: g2}

    def build(self):
        sl0 = gui_aux.SelectionList(tab=self, buttons=['load', 'save', 'delete', 'tree', 'conf_tree'], root_key='Model')
        sl1 = gui_aux.SelectionList(tab=self, conftype='ModelGroup', disp='Model family :', buttons=[], single_line=True,
                                    width=15, text_kws=gui_aux.t_kws(10), sublists={'model families': sl0})
        b_nengo = gui_aux.named_bool_button('nengo', False)
        l00 = gui_aux.gui_col([sl1, sl0], x_frac=0.2, y_frac=0.6, as_pane=True, pad=(20, 20), add_to_bottom=[b_nengo])
        l01, c1 = self.build_module_tab()
        l1=[[l00, l01]]
        l2, g2 = self.build_architecture_tab()
        self.draw_tab = DrawBodyTab(name='body', gui=self.gui, conftype='Body')
        l3, c3, g3, d3 = self.draw_tab.build()
        self.selectionlists[self.draw_tab.conftype] = self.draw_tab.selectionlists[self.draw_tab.conftype]

        tabs={}
        tabs['modules'] = l1
        tabs['architecture'] = l2
        tabs['draw'] = l3
        l_tabs=[sg.Tab(k, v, key=f'{k}_TAB') for k,v in tabs.items()]
        l = [[sg.TabGroup([l_tabs], key='ACTIVE_MODEL_TAB', tab_location='topleft', **gui_aux.tab_kws)]]
        return l, {**c1, **c3}, {**g2, **g3}, {**{}, **d3}

    def eval(self, e, v, w, c, d, g):
        self.draw_tab.eval(e, v, w, c, d, g)
        return d,g


if __name__ == "__main__":
    from larvaworld.gui.tabs.larvaworld_gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['larva-model'])
    larvaworld_gui.run()
