import time
import PySimpleGUI as sg

from lib.aux.functions import get_pygame_key
from lib.conf.conf import saveConfDict, loadConfDict
from lib.conf.init_dtypes import store_controls
from lib.gui.aux.elements import CollapsibleDict, Collapsible
from lib.gui.aux.functions import t_kws, gui_col
from lib.gui.aux.buttons import GraphButton
import lib.conf.dtype_dicts as dtypes
from lib.gui.tabs.tab import GuiTab
from lib.aux import functions as fun

class SettingsTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = 'controls'
        self.k_reset = f'RESET_{self.k}'
        self.k_edit = f'EDIT_{self.k}'

    @ property
    def controls_dict(self):
        d=self.gui.dicts
        return d[self.k]

    @property
    def cur(self):
        return self.controls_dict['cur']

    # def set_cur(self, cur):
    #     self.controls_dict['cur'] = cur

    def control_k(self,k):
        return f'{self.k} {k}'

    @property
    def used_keys(self):
        d=self.controls_dict['keys']
        return fun.flatten_list([list(k.values()) for k in list(d.values())])

    def single_control_layout(self, k, v, prefix=None, editable=True):
        k0=f'{prefix} {k}' if prefix is not None else k
        l = [sg.T("", **t_kws(2)), sg.T(k, **t_kws(16)),
               sg.In(default_text=v, key=self.control_k(k0), disabled=True,
                     disabled_readonly_background_color='black', enable_events=True,
                     text_color='white', **t_kws(10), justification='center')]
        if editable :
            l+=[GraphButton('Document_2_Edit', f'{self.k_edit} {k0}', tooltip=f'Edit shortcut for {k}')]
        return l

    def single_control_collapsible(self, name, dic, editable=True) :
        l = [self.single_control_layout(k, v, prefix=name, editable=editable) for k, v in dic.items()]
        c = Collapsible(f'{self.k}_{name}', content=l, disp_name=name)
        return c

    def build_controls_collapsible(self, c):
        b_reset=GraphButton('Button_Burn', self.k_reset,tooltip='Reset all controls to the defaults. '
                                   'Restart Larvaworld after changing shortcuts.')
        conf = loadConfDict('Settings')
        l = []
        for title, dic in conf['keys'].items():
            cc = self.single_control_collapsible(title, dic)
            l += cc.get_layout(as_col=False)
            c.update(cc.get_subdicts())
        c_keyboard = Collapsible('Keyboard', content=l, next_to_header=[b_reset])
        c_mouse=self.single_control_collapsible('mouse', conf['mouse'], editable=False)

        c_controls = Collapsible('Controls', content=[[gui_col([c_keyboard, c_mouse], 0.33)]])
        for s in [c_keyboard, c_mouse]:
            c.update(s.get_subdicts())

        d=self.inti_control_dict(conf)

        return c_controls, d

    def inti_control_dict(self, conf):
        d = {self.k: conf}
        d[self.k]['cur'] = None
        return d

    def build(self):
        c = {}

        c1 = CollapsibleDict('Visualization', dict=dtypes.get_dict('visualization', mode='video', video_speed=60),
                             type_dict=dtypes.get_dict_dtypes('visualization'), toggled_subsections=None)
        c2 = CollapsibleDict('replay', default=True, toggled_subsections=False)

        c3, d = self.build_controls_collapsible(c)

        for s in [c1, c2, c3]:
            c.update(s.get_subdicts())
        l = [[
            gui_col([c1], 0.33),
            gui_col([c2], 0.33),
            gui_col([c3], 0.34),
        ]]
        return l, c, {}, d

    def update_controls(self, v, w):
        d0 = self.controls_dict
        cur=self.cur
        p1, p2 = cur.split(' ', 1)[0], cur.split(' ', 1)[-1]
        k_cur = self.control_k(cur)
        v_cur = v[k_cur]
        w[k_cur].update(disabled=True, value=v_cur)
        d0['keys'][p1][p2] = v_cur
        # d0['keys'][cur] = v_cur
        d0['pygame_keys'][cur] = get_pygame_key(v_cur)
        d0['cur'] = None
        saveConfDict(d0, 'Settings')

    def reset_controls(self, w):
        store_controls()
        conf = loadConfDict('Settings')
        d = self.inti_control_dict(conf)
        for title, dic in conf['keys'].items():
            for k0, v0 in dic.items():
                w[self.control_k(f'{title} {k0}')].update(disabled=True, value=v0)
        for k0, v0 in conf['mouse'].items():
            w[self.control_k(f'mouse {k0}')].update(disabled=True, value=v0)
        return d


    def eval(self, e, v, w, c, d, g):
        d0=self.controls_dict
        delay = 0.5
        cur = self.cur
        if e == self.k_reset:
            d=self.reset_controls(w)

        elif e.startswith(self.k_edit) and cur is None:
            cur = e.split(' ', 1)[-1]
            cur_key = self.control_k(cur)
            w[cur_key].update(disabled=False, value='', background_color='grey')
            w[cur_key].set_focus()
            d0['cur'] = cur

        elif cur is not None:
            cur_key = self.control_k(cur)
            v0 = v[cur_key]
            p1,p2=cur.split(' ', 1)[0], cur.split(' ', 1)[-1]
            if v0 == d0['keys'][p1][p2]:
                w[cur_key].update(disabled=True)
                d0['cur'] = None
            elif v0 in self.used_keys:
                w[cur_key].update(disabled=False, value='USED', background_color='red')
                w.refresh()
                time.sleep(delay)
                w[cur_key].update(disabled=False, value='', background_color='grey')
                w[cur_key].set_focus()
            else:
                self.update_controls(v, w)
        return d, g


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['settings'])
    larvaworld_gui.run()
