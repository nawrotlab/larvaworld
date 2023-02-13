import time
import PySimpleGUI as sg



from larvaworld.lib import reg
from larvaworld.gui import gui_aux

import larvaworld.lib.aux.dictsNlists as dNl

class SettingsTab(gui_aux.GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = 'controls'
        self.k_reset = f'RESET_{self.k}'
        self.k_edit = f'EDIT_{self.k}'
        self.Cvis, self.Ccon = 'green', 'red'

    @property
    def controls_dict(self):
        d = self.gui.dicts
        return d[self.k]

    @property
    def cur(self):
        return self.controls_dict['cur']

    def control_k(self, k):
        return f'{self.k} {k}'

    @property
    def used_keys(self):
        d = self.controls_dict['keys']
        return dNl.flatten_list([list(k.values()) for k in list(d.values())])

    def single_control_layout(self, k, v, prefix=None, editable=True):
        k0 = f'{prefix} {k}' if prefix is not None else k
        l = [sg.T(k, **gui_aux.t_kws(14)),
             sg.In(default_text=v, key=self.control_k(k0), disabled=True,
                   disabled_readonly_background_color='black', enable_events=True,
                   text_color='white', **gui_aux.t_kws(8), justification='center')]
        if editable:
            l += [gui_aux.GraphButton('Document_2_Edit', f'{self.k_edit} {k0}', tooltip=f'Edit shortcut for {k}')]
        return l

    def single_control_collapsible(self, name, dic, editable=True, **kwargs):
        l = [self.single_control_layout(k, v, prefix=name, editable=editable) for k, v in dic.items()]
        c = gui_aux.PadDict(f'{self.k}_{name}', content=l, disp_name=name, **kwargs)
        return c

    def build_controls_collapsible(self, c):
        kws = {'background_color': self.Ccon}
        b_reset = gui_aux.GraphButton('Button_Burn', self.k_reset, tooltip='Reset all controls to the defaults. '
                                                                           'Restart Larvaworld after changing shortcuts.')
        conf = reg.controls.load()
        cs = [self.single_control_collapsible(k, v, header_width=26, **kws) for k, v in conf['keys'].items()]
        cs.append(self.single_control_collapsible('mouse', conf['mouse'], editable=False, header_width=26, **kws))
        l = []
        for cc in cs:
            c.update(cc.get_subdicts())
            l += cc.get_layout(as_col=False)
        c_controls = gui_aux.PadDict('Controls', content=l, after_header=[b_reset], Ncols=3, header_width=90, **kws)
        d = self.inti_control_dict(conf)
        return c_controls, d

    def inti_control_dict(self, conf):
        d = {self.k: conf}
        d[self.k]['cur'] = None
        return d

    def build(self):
        kws = {'background_color': self.Cvis, 'header_width': 55, 'Ncols': 2, 'text_kws': gui_aux.t_kws(14),
               'value_kws': gui_aux.t_kws(12)}
        c = {}
        c1 = gui_aux.PadDict('visualization', **kws)
        c2 = gui_aux.PadDict('Replay', **kws)
        c3, d = self.build_controls_collapsible(c)
        for s in [c1, c2, c3]:
            c.update(s.get_subdicts())
        l = gui_aux.gui_cols(cols=[[c1, c2], [c3]], x_fracs=[0.4, 0.6])
        return l, c, {}, d

    def update_controls(self, v, w):
        d0 = self.controls_dict
        cur = self.cur
        p1, p2 = cur.split(' ', 1)[0], cur.split(' ', 1)[-1]
        k_cur = self.control_k(cur)
        v_cur = v[k_cur]
        w[k_cur].update(disabled=True, value=v_cur)
        d0['keys'][p1][p2] = v_cur
        # d0['keys'][cur] = v_cur
        d0['pygame_keys'][cur] = get_pygame_key(v_cur)
        d0['cur'] = None
        reg.controls.save(d0)

    def reset_controls(self, w):
        reg.controls.save()
        conf = reg.controls.load()
        d = self.inti_control_dict(conf)
        for title, dic in conf['keys'].items():
            for k0, v0 in dic.items():
                w[self.control_k(f'{title} {k0}')].update(disabled=True, value=v0)
        for k0, v0 in conf['mouse'].items():
            w[self.control_k(f'mouse {k0}')].update(disabled=True, value=v0)
        return d

    def eval(self, e, v, w, c, d, g):
        d0 = self.controls_dict
        delay = 0.5
        cur = self.cur
        if e == self.k_reset:
            d = self.reset_controls(w)

        elif e.startswith(self.k_edit) and cur is None:
            cur = e.split(' ', 1)[-1]
            cur_key = self.control_k(cur)
            w[cur_key].update(disabled=False, value='', background_color='grey')
            w[cur_key].set_focus()
            d0['cur'] = cur

        elif cur is not None:
            cur_key = self.control_k(cur)
            v0 = v[cur_key]
            p1, p2 = cur.split(' ', 1)[0], cur.split(' ', 1)[-1]
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

def get_pygame_key(key):
    pygame_keys = {
        'BackSpace': 'BACKSPACE',
        'tab': 'TAB',
        'del': 'DELETE',
        'clear': 'CLEAR',
        'Return': 'RETURN',
        'Escape': 'ESCAPE',
        'space': 'SPACE',
        'exclam': 'EXCLAIM',
        'quotedbl': 'QUOTEDBL',
        '+': 'PLUS',
        'comma': 'COMMA',
        '-': 'MINUS',
        'period': 'PERIOD',
        'slash': 'SLASH',
        'numbersign': 'HASH',
        'Down:': 'DOWN',
        'Up:': 'UP',
        'Right:': 'RIGHT',
        'Left:': 'LEFT',
        'dollar': 'DOLLAR',
        'ampersand': 'AMPERSAND',
        'parenleft': 'LEFTPAREN',
        'parenright': 'RIGHTPAREN',
        'asterisk': 'ASTERISK',
    }
    return f'K_{pygame_keys[key]}' if key in list(pygame_keys.keys()) else f'K_{key}'

if __name__ == "__main__":
    from larvaworld.gui.tabs.larvaworld_gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['set'])
    larvaworld_gui.run()
