import copy
import threading
import PySimpleGUI as sg
import lib.conf.dtype_dicts as dtypes

from lib.gui.gui_lib import CollapsibleDict, Collapsible, \
    named_bool_button, save_gui_conf, delete_gui_conf, GraphList, CollapsibleTable, \
    graphic_button, t10_kws, t18_kws, w_kws, default_run_window, col_kws, col_size, t24_kws
from lib.gui.draw_env import draw_env
from lib.conf.conf import loadConfDict, loadConf
from lib.gui.tab import GuiTab, SelectionList


class EnvTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def update(self,w, c, conf, id=None):
        for n in ['border_list', 'larva_groups', 'arena', 'odorscape']:
            c[n].update(w, conf[n] if n in conf.keys() else {})
        for n in ['source_groups', 'source_units', 'food_grid']:
            c[n].update(w, conf['food_params'][n])


    def get(self,w, v, c, as_entry=False):
        env = {
            'food_params': {n: c[n].get_dict(v, w) for n in ['source_groups', 'source_units', 'food_grid']},
            **{n: c[n].get_dict(v, w) for n in ['larva_groups', 'border_list', 'arena', 'odorscape']}
        }

        env0 = copy.deepcopy(env)
        if not as_entry:
            for n, gr in env0['larva_groups'].items():
                if type(gr['model']) == str:
                    gr['model'] = loadConf(gr['model'], 'Model')
        return env0


    def build(self):
        s1 = CollapsibleTable('larva_groups', False, headings=['group', 'N', 'color', 'model'],
                              type_dict=dtypes.get_dict_dtypes('distro', class_name='Larva', basic=False))
        s2 = CollapsibleTable('source_groups', False, headings=['group', 'N', 'color', 'amount', 'odor_id'],
                              type_dict=dtypes.get_dict_dtypes('distro', class_name='Source', basic=False))
        s3 = CollapsibleTable('source_units', False, headings=['id', 'color', 'amount', 'odor_id'],
                              type_dict=dtypes.get_dict_dtypes('agent', class_name='Source'))
        s4 = CollapsibleTable('border_list', False, headings=['id', 'color', 'points'],
                              type_dict=dtypes.get_dict_dtypes('agent', class_name='Border'))
        c = {}
        for s in [s1, s2, s3, s4]:
            c.update(**s.get_subdicts())
        c1 = [CollapsibleDict(n, False, dict=dtypes.get_dict(n), type_dict=dtypes.get_dict_dtypes(n), **kw)
              for n, kw in zip(['arena', 'food_grid', 'odorscape'], [{}, {'toggle': True}, {}])]
        for s in c1:
            c.update(s.get_subdicts())
        l1 = [c[n].get_section() for n in ['source_groups', 'source_units', 'food_grid']]
        c2 = Collapsible('Sources', True, l1)
        c.update(c2.get_subdicts())
        l2 = [c[n].get_section() for n in ['arena', 'larva_groups', 'Sources', 'border_list', 'odorscape']]
        l1 = SelectionList(tab=self,conftype='Env',actions=['load', 'edit', 'save', 'delete'])
        self.selectionlists=[l1]
        l = [[sg.Col([l1.l, *l2], **col_kws, size=col_size(0.5))]]
        return l, c, {}, {}

    def edit(self,env=None):
        return draw_env(env)

    # def eval(self,e, v, w, c, d, g):
    #     if e == f'EDIT_{self.name}':
    #         env = self.get(w, v, c, extend=False)
    #         new_env = draw_env(env)
    #         self.update(new_env, w, c)
    #
    #     return d, g

if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['environment'])
    larvaworld_gui.run()

