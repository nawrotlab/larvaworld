import time

import PySimpleGUI as sg
import pygame

from lib.conf.conf import loadConfDict, saveConfDict
from lib.gui.gui_lib import graphic_button, default_run_window, CollapsibleDict, Collapsible, col_kws, col_size, \
    w_kws, t_kws, load_shortcuts
import lib.conf.dtype_dicts as dtypes
from lib.gui.tab import GuiTab


class SettingsTab(GuiTab):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)


    def build_shortcut_layout(self, collapsibles):
        conf=load_shortcuts()

        l = []
        for title, dic in dtypes.default_shortcuts.items():
            col_title=f'shortcuts_{title}'
            ll = [
                # [sg.T(title, **t16_kws)],
                *[[sg.T("", **t_kws(2)), sg.T(k, **t_kws(16)),
                   sg.In(default_text=conf['keys'][k], key=f'SHORT {k}', disabled=True,
                         disabled_readonly_background_color='black', enable_events=True,
                         text_color='white', **t_kws(10), justification='center'),
                   graphic_button('edit', f'EDIT_SHORTCUT  {k}', tooltip=f'Edit shortcut for {k}')] for k in list(dic.keys())],
                # [sg.T("", **t8_kws)]
            ]
            collapsibles[col_title]=Collapsible(col_title, state=False, disp_name=title, content=ll)
            l += collapsibles[col_title].get_layout(as_col=False)

        dicts = {'shortcuts': conf}
        dicts['shortcuts']['cur'] = None
        return l, dicts


    def build(self):
        collapsibles = {}
        l_short, dicts = self.build_shortcut_layout(collapsibles)
        l_mouse=[*[[sg.T("", **t_kws(2)), sg.T(k, **t_kws(16)),
                   sg.T(v, key=f'SHORT {k}', background_color='black',
                         text_color='white', **t_kws(10), justification='center')] for k,v in dtypes.mouse_controls.items()],
                # [sg.T("", **t8_kws)]
                 ]

        s1 = CollapsibleDict('Visualization', True, dict=dtypes.get_dict('visualization', mode='video', video_speed=60),
                             type_dict=dtypes.get_dict_dtypes('visualization'), toggled_subsections=None)
        s2 = CollapsibleDict('replay', True, default=True, toggled_subsections=False)
        s3 = Collapsible('Keyboard', True, content=l_short, next_to_header=[
                                                    graphic_button('burn', 'RESET_SHORTCUTS',
                                                                   tooltip='Reset all shortcuts to the defaults. '
                                                                           'Restart Larvaworld after changing shortcuts.')])
        s4 = Collapsible('Mouse', False, content=l_mouse)

        l_controls = [[sg.Col([s3.get_layout(), s4.get_layout()])]]

        s5 = Collapsible('Controls', True, content=l_controls)
        for s in [s1, s2, s3, s4, s5]:
            collapsibles.update(s.get_subdicts())
        l_set = [[sg.Col(s1.get_layout(as_col=False), **col_kws, size=col_size(1/3)),
                  sg.Col(s2.get_layout(as_col=False), **col_kws, size=col_size(1/3)),
                  sg.Col(s5.get_layout(as_col=False), **col_kws, size=col_size(1/3),
                         scrollable=False, vertical_scroll_only=True),
                  ]
                 ]
        return l_set, collapsibles, {}, dicts


    def eval(self,e, v, w, c, d, g):
        delay = 0.5
        cur = d['shortcuts']['cur']
        if e == 'RESET_SHORTCUTS':
            d['shortcuts']['cur'] = None
            for title, dic in dtypes.default_shortcuts.items():
                for k, v in dic.items():
                    w[f'SHORT {k}'].update(disabled=True, value=v)
                    d['shortcuts']['keys'][k] = v
                    d['shortcuts']['pygame_keys'][k] = dtypes.get_pygame_key(v)
                    saveConfDict(d['shortcuts'], 'Settings')

        elif 'EDIT_SHORTCUT' in e and cur is None:
                cur = e.split('  ')[-1]
                cur_key = f"SHORT {cur}"
                w[cur_key].update(disabled=False, value='', background_color='grey')
                w[cur_key].set_focus()
                d['shortcuts']['cur'] = cur

        elif cur is not None:
            cur_key = f"SHORT {cur}"
            v=v[cur_key]
            if v == d['shortcuts']['keys'][cur]:
                w[cur_key].update(disabled=True)
                d['shortcuts']['cur'] = None
            elif v in list(d['shortcuts']['keys'].values()):
                w[cur_key].update(disabled=False, value='USED', background_color='red')
                w.refresh()
                time.sleep(delay)
                w[cur_key].update(disabled=False, value='', background_color='grey')
                w[cur_key].set_focus()
            else:
                w[cur_key].update(disabled=True, value=v)
                d['shortcuts']['keys'][cur] = v
                d['shortcuts']['pygame_keys'][cur] = dtypes.get_pygame_key(v)
                d['shortcuts']['cur'] = None
                saveConfDict(d['shortcuts'], 'Settings')
        return d, g


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['settings'])
    larvaworld_gui.run()

