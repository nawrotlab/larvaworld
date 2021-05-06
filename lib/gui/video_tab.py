import copy
import os

import PySimpleGUI as sg
import webbrowser


from lib.gui.gui_lib import CollapsibleDict, Collapsible, save_gui_conf, delete_gui_conf, b12_kws, \
    b6_kws, CollapsibleTable, graphic_button, t10_kws, t12_kws, t18_kws, w_kws, default_run_window, BtnLink
from lib.conf.conf import loadConfDict, loadConf
import lib.conf.dtype_dicts as dtypes

import lib.stor.paths as paths




def build_video_tab():
    dicts = {}
    graph_lists = {}
    collapsibles = {}
    link_pref = "http://computational-systems-neuroscience.de/wp-content/uploads/2021/04/"
    files = [f for f in os.listdir(paths.VideoSlideFolder) if f.endswith('png')]
    b_list = [sg.B(image_filename=os.path.join(paths.VideoSlideFolder, f), image_subsample=5, key=f'IMAGE {f.split(".")[0]}', enable_events=True, pad=(25,40),
                   metadata=BtnLink(link=f'{link_pref}{f.split(".")[0]}.mp4')) for f in files]
    n = 3
    b_list = [b_list[i * n:(i + 1) * n] for i in range((len(b_list) + n - 1) // n)]
    l_vid = [[sg.Col(b_list, vertical_scroll_only=True,scrollable=True, size=(1800, 1200))]]

    return l_vid, collapsibles, graph_lists, dicts


def eval_video_tab(event, values, window, collapsibles, dicts, graph_lists):
    if 'IMAGE' in event:
        link = window[event].metadata.link
        webbrowser.open(link)
    return dicts, graph_lists


if __name__ == "__main__":
    sg.theme('LightGreen')
    n = 'video'
    l, c, g, d = build_video_tab()
    w = sg.Window(f'{n} gui', l, size=(1800, 1200), **w_kws, location=(300, 100))

    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        default_run_window(w, e, v, c, g)
        d, g = eval_video_tab(e, v, w, collapsibles=c, dicts=d, graph_lists=g)
    w.close()
