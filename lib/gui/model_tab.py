import copy
import PySimpleGUI as sg

from lib.gui.gui_lib import CollapsibleDict, Collapsible, save_gui_conf, delete_gui_conf, b12_kws, \
    b6_kws, CollapsibleTable, graphic_button, t10_kws, t12_kws, t18_kws, w_kws, col_kws, col_size
import lib.conf.dtype_dicts as dtypes
from lib.gui.tab import GuiTab, SelectionList


class ModelTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, w, c, conf, id=None):
        for n in ['physics', 'energetics', 'body', 'odor']:
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
            c[k.upper()].update(w, dic)
        temp = copy.deepcopy(module_dict)
        for k in list(temp.keys()):
            temp[k.upper()] = temp.pop(k)
        c['Brain'].update(w, temp, use_prefix=False)

    def get(self, w, v, c, as_entry=True):
        module_dict = dict(zip(dtypes.module_keys, [w[f'TOGGLE_{k.upper()}'].get_state() for k in dtypes.module_keys]))
        m = {}

        for n in ['physics', 'energetics', 'body', 'odor']:
            m[n] = None if c[n].state is None else c[n].get_dict(v, w)

        b = {}
        b['modules'] = module_dict
        for k in module_dict.keys():
            b[f'{k}_params'] = c[k.upper()].get_dict(v, w)
        if b['olfactor_params'] is not None:
            b['olfactor_params']['odor_dict'] = c['odor_gains'].dict
        b['nengo'] = False
        m['brain'] = b

        return copy.deepcopy(m)

    def build(self):
        l0 = SelectionList(tab=self,conftype='Model',actions=['load', 'save', 'delete'])
        self.selectionlists = [l0]

        c1 = [CollapsibleDict(n, False, dict=dtypes.get_dict(n), type_dict=dtypes.get_dict_dtypes(n),
                              disp_name=n.capitalize(), **kwargs)
              for n, kwargs in zip(['physics', 'energetics', 'body', 'odor'], [{}, {'toggle': True}, {}, {}])]
        s1 = CollapsibleTable('odor_gains', False, headings=['id', 'mean', 'std'], dict={},
                              disp_name='Odor gains', type_dict=dtypes.get_dict_dtypes('odor_gain'))
        l1 = [i.get_section() for i in c1 + [s1]]
        c2 = [CollapsibleDict(k.upper(), False, dict=dtypes.get_dict(k), type_dict=dtypes.get_dict_dtypes(k),
                              toggle=True, disp_name=k.capitalize()) for k in dtypes.module_keys]
        l2 = [i.get_section() for i in c2]
        b = Collapsible('Brain', True, l2)

        l3=[sg.Col([b.get_section()], **col_kws, size=col_size(0.25)),
                                         sg.Col(l1, **col_kws, size=col_size(0.25))]
        c = {}
        for s in c2 + c1 + [s1, b]:
            c.update(s.get_subdicts())
        l = [[sg.Col([l0.l, l3], vertical_alignment='t')]]
        return l, c, {}, {}


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['larva-model'])
    larvaworld_gui.run()
