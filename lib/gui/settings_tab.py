import time

import PySimpleGUI as sg
import pygame

from lib.conf.conf import loadConfDict, saveConfDict
from lib.gui.gui_lib import t8_kws, graphic_button, default_run_window, CollapsibleDict, Collapsible, col_kws, col_size, \
    w_kws, t16_kws, t2_kws, t12_kws, load_shortcuts, t10_kws
import lib.conf.dtype_dicts as dtypes


def build_shortcut_layout(collapsibles):
    conf=load_shortcuts()

    l = []
    for title, dic in dtypes.default_shortcuts.items():
        col_title=f'shortcuts_{title}'
        ll = [
            # [sg.T(title, **t16_kws)],
            *[[sg.T("", **t2_kws), sg.T(k, **t16_kws),
               sg.In(default_text=conf['keys'][k], key=f'SHORT {k}', disabled=True,
                     disabled_readonly_background_color='black', enable_events=True,
                     text_color='white', **t10_kws, justification='center'),
               graphic_button('edit', f'EDIT_SHORTCUT  {k}', tooltip=f'Edit shortcut for {k}')] for k in list(dic.keys())],
            # [sg.T("", **t8_kws)]
        ]
        collapsibles[col_title]=Collapsible(col_title, state=False, disp_name=title, content=ll)
        l += collapsibles[col_title].get_section(as_col=False)

    dicts = {'shortcuts': conf}
    dicts['shortcuts']['cur'] = None
    return l, dicts


def build_settings_tab():
    collapsibles = {}
    l_short, dicts = build_shortcut_layout(collapsibles)
    l_mouse=[*[[sg.T("", **t2_kws), sg.T(k, **t16_kws),
               sg.T(v, key=f'SHORT {k}', background_color='black',
                     text_color='white', **t10_kws, justification='center')] for k,v in dtypes.mouse_controls.items()],
            # [sg.T("", **t8_kws)]
             ]

    s1 = CollapsibleDict('Visualization', True, dict=dtypes.get_dict('visualization', mode='video', video_speed=60),
                         type_dict=dtypes.get_dict_dtypes('visualization'), toggled_subsections=None)
    s2 = CollapsibleDict('Replay', False, dict=dtypes.get_dict('replay'), type_dict=dtypes.get_dict_dtypes('replay'),
                         toggled_subsections=False)
    s3 = Collapsible('Keyboard', True, content=l_short, next_to_header=[
                                                graphic_button('burn', 'RESET_SHORTCUTS',
                                                               tooltip='Reset all shortcuts to the defaults. '
                                                                       'Restart Larvaworld after changing shortcuts.')])
    s4 = Collapsible('Mouse', False, content=l_mouse)

    l_controls = [[sg.Col([s3.get_section(),s4.get_section()])]]

    s5 = Collapsible('Controls', True, content=l_controls)
    for s in [s1, s2, s3, s4, s5]:
        collapsibles.update(s.get_subdicts())
    l_set = [[sg.Col(s1.get_section(as_col=False), **col_kws, size=col_size(0.25)),
              sg.Col(s2.get_section(as_col=False), **col_kws, size=col_size(0.25)),
              sg.Col(s5.get_section(as_col=False), **col_kws, size=col_size(0.25),
                     scrollable=False, vertical_scroll_only=True),
              ]
             ]
    return l_set, collapsibles, {}, dicts


def eval_settings(event, values, window, collapsibles, dicts, graph_lists):
    delay = 0.5
    cur = dicts['shortcuts']['cur']
    if event == 'RESET_SHORTCUTS':
        dicts['shortcuts']['cur'] = None
        # window.ReturnKeyboardEvents = False
        for title, dic in dtypes.default_shortcuts.items():
            for k, v in dic.items():
                window[f'SHORT {k}'].update(disabled=True, value=v)
                dicts['shortcuts']['keys'][k] = v
                dicts['shortcuts']['pygame_keys'][k] = dtypes.get_pygame_key(v)
                saveConfDict(dicts['shortcuts'], 'Settings')

    elif 'EDIT_SHORTCUT' in event and cur is None:
            cur = event.split('  ')[-1]
            cur_key = f"SHORT {cur}"
            window[cur_key].update(disabled=False, value='', background_color='grey')
            window[cur_key].set_focus()
            dicts['shortcuts']['cur'] = cur

    elif cur is not None:
        cur_key = f"SHORT {cur}"
        v=values[cur_key]
        if v == dicts['shortcuts']['keys'][cur]:
            window[cur_key].update(disabled=True)
            dicts['shortcuts']['cur'] = None
        elif v in list(dicts['shortcuts']['keys'].values()):
            window[cur_key].update(disabled=False, value='USED', background_color='red')
            window.refresh()
            time.sleep(delay)
            window[cur_key].update(disabled=False, value='', background_color='grey')
            window[cur_key].set_focus()
        else:
            window[cur_key].update(disabled=True, value=v)
            dicts['shortcuts']['keys'][cur] = v
            dicts['shortcuts']['pygame_keys'][cur] = dtypes.get_pygame_key(v)
            dicts['shortcuts']['cur'] = None
            saveConfDict(dicts['shortcuts'], 'Settings')
    return dicts, graph_lists


if __name__ == "__main__":
    sg.theme('LightGreen')
    l, c, g, d = build_settings_tab()
    w = sg.Window('Settings', l, size=(1800, 1200), **w_kws, location=(300, 100))

    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        default_run_window(w, e, v, c, g)
        d, g = eval_settings(e, v, w, collapsibles=c, dicts=d, graph_lists=g)
    w.close()