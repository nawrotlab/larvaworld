import PySimpleGUI as sg
import copy
import os

from lib.conf.base.dtypes import null_dict
from lib.gui.aux.elements import CollapsibleDict, Collapsible, CollapsibleTable, GraphList, SelectionList, PadDict
from lib.gui.aux.functions import col_size, gui_cols, t_kws, gui_col
from lib.gui.tabs.tab import GuiTab
from lib.conf.base import paths


class ModelTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = ['physics', 'energetics', 'body']
        self.module_keys = list(null_dict('modules').keys())

    def update(self, w, c, conf, id=None):
        for n in self.fields:
            c[n].update(w, conf[n])
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

    def get(self, w, v, c, as_entry=True):
        module_dict = dict(zip(self.module_keys, [w[f'TOGGLE_{k}'].get_state() for k in self.module_keys]))
        m = {}
        for n in self.fields:
            m[n] = None if c[n].disabled else c[n].get_dict(v, w)

        b = {}
        b['modules'] = module_dict
        for k in module_dict.keys():
            b[f'{k}_params'] = c[k].get_dict(v, w)
        if b['olfactor_params'] is not None:
            b['olfactor_params']['odor_dict'] = c['odor_gains'].dict
        b['nengo'] = False
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
            'energetics': Cener,
        }
        dic = {

            'modulation/memory': ['intermitter', 'interference', 'memory'],
            'effectors': ['crawler', 'turner', 'feeder'],
            'sensors': ['olfactor', 'toucher', 'windsensor'],
            # 'modulation/memory': ['intermitter'],
            'body/energetics': ['body', 'physics', 'energetics'],
        }

        ss = (500, 650)
        c = {}
        s1 = CollapsibleTable('odor_gains', index='ID', heading_dict={'mean': 'mean', 'std': 'std'}, state=True,
                              col_widths=[13, 7, 6])
        c.update(s1.get_subdicts())
        tt = []
        pp = {}
        for k, vs in dic.items():
            ppp = []
            for v in vs:
                cc = PadDict(v, toggle=True if v not in ['body', 'physics'] else None, background_color=col_dict[v], text_kws=t_kws(16))
                c.update(cc.get_subdicts())
                if v=='olfactor' :
                    ll = cc.get_layout(size=(ss[0], int(ss[1]/2)))
                    ll.append(s1.get_layout())
                else :
                    ll = cc.get_layout(size=ss)
                ppp.append(sg.Col(ll, scrollable=True, vertical_scroll_only=True))
            pp[k] = [[sg.Pane(ppp, orientation='horizontal', show_handle=False)]]
            tt.append(sg.Tab(k, pp[k], key=f'{k} MODULES'))
        tab_kws = {'font': ("Helvetica", 14, "normal"), 'selected_title_color': 'darkblue', 'title_color': 'grey',
                   'tab_background_color': 'lightgrey'}
        l_tabs = sg.TabGroup([tt], key='ACTIVE_MODULES', tab_location='topcenter', **tab_kws)
        return l_tabs, c

    def build(self):
        l_tabs, c = self.build_module_tab()
        l0 = SelectionList(tab=self, buttons=['load', 'save', 'delete'])
        # c1 = [CollapsibleDict(n, **kws) for n, kws in zip(self.fields, [{}, {'toggle': True}, {}, {}])]
        # s1 = CollapsibleTable('odor_gains', index='ID', heading_dict={'mean': 'mean', 'std': 'std'}, state=True, col_widths = [8,6,6])
        # c2 = [CollapsibleDict(k, toggle=True) for k in self.module_keys]
        # l2 = [i.get_layout() for i in c2]
        # b = Collapsible('Brain', content=l2, state=True)

        fdir = paths.path('model')
        fig_dict = {f: f'{fdir}/{f}' for f in os.listdir(fdir)}
        g1 = GraphList(self.name, tab=self, list_header='Model', fig_dict=fig_dict, subsample=3,
                       canvas_size=col_size(x_frac= 0.9, y_frac=0.4))

        ll1 = gui_col([l0, g1], x_frac=0.2,y_frac=0.6, as_pane=True, pad=(20,20))
        # ll1 = gui_col([l0, s1, g1], x_frac=0.2,y_frac=0.6, as_pane=True, pad=(20,20))
        # ll3=gui_col([g1.canvas], x_frac=0.3, as_pane=True, pad=(20,20))
        ll2=sg.Col([[l_tabs]], size=col_size(0.8, 0.6))
        # ll = gui_cols(cols=[[l0, s1, g1], [g1.canvas]], x_fracs=[0.2,0.3], as_pane=True, pad=(20,20))
        # l.append([l_tabs])
        l1=[ll1,ll2]
        # l1=[ll1,ll2]
        l2=g1.canvas.get_layout(as_pane=True)

        l=[[sg.Col([l1,l2[0]])]]
        # l = [[sg.Pane([sg.vtop(l1), sg.vbottom(l2[0])], handle_size=30)]]

        return l, c, {g1.name: g1}, {}


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['larva-model', 'settings'])
    larvaworld_gui.run()
