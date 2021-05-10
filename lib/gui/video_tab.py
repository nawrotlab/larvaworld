import os
import PySimpleGUI as sg

from lib.gui.gui_lib import w_kws, default_run_window, window_size, ClickableImage
import lib.stor.paths as paths


def build_video_tab():
    link_pref = "http://computational-systems-neuroscience.de/wp-content/uploads/2021/04/"
    files = [f for f in os.listdir(paths.VideoSlideFolder) if f.endswith('png')]
    b_list = [ClickableImage(name=f.split(".")[0], link=f'{link_pref}{f.split(".")[0]}.mp4',
                             image_filename=os.path.join(paths.VideoSlideFolder, f),
                             image_subsample=5,pad=(25, 40)) for f in files]

    n = 3
    b_list = [b_list[i * n:(i + 1) * n] for i in range((len(b_list) + n - 1) // n)]
    l_vid = [[sg.Col(b_list, vertical_scroll_only=True,scrollable=True, size=window_size)]]

    return l_vid, {}, {}, {}


def eval_video_tab(event, values, window, collapsibles, dicts, graph_lists):
    if 'ClickableImage' in event:
        window[event].eval()

    return dicts, graph_lists


if __name__ == "__main__":
    sg.theme('LightGreen')
    n = 'video'
    l, c, g, d = build_video_tab()
    w = sg.Window(f'{n} gui', l, size=window_size, **w_kws, location=(300, 100))

    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        default_run_window(w, e, v, c, g)
        d, g = eval_video_tab(e, v, w, collapsibles=c, dicts=d, graph_lists=g)
    w.close()
