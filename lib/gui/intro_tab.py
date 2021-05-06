import copy
import os

import PySimpleGUI as sg
import webbrowser

from lib.gui.gui_lib import w_kws, default_run_window, BtnLink
import lib.stor.paths as paths


def build_intro_tab():
    dicts = {}
    graph_lists = {}
    collapsibles = {}

    c = {'size': (80, 1),
         'pad': (20, 5),
         'justification': 'center'
         }

    filenames=os.listdir(paths.IntroSlideFolder)

    filenames.sort()

    b_list = [
        sg.B(image_filename=os.path.join(paths.IntroSlideFolder, f), image_subsample=3,
             pad=(15, 70)) for f in filenames]
    l_title = sg.Col([[sg.T('', size=(5,5))],
        [sg.T('Larvaworld', font=("Cursive", 40, "italic"), **c)],
                     b_list,
                      [sg.T('Behavioral analysis and simulation platform', font=("Lobster", 15, "bold"), **c)],
                      [sg.T('for Drosophila larva', font=("Lobster", 15, "bold"), **c)]],
                     element_justification='center',
                     # justification='center',
                     # vertical_alignment='center'
                     )
    # files = [f for f in os.listdir(paths.IntroSlideFolder) if f.endswith('png')]


    l_intro = [[l_title]]

    return l_intro, collapsibles, graph_lists, dicts


def eval_intro_tab(event, values, window, collapsibles, dicts, graph_lists):
    pass
    # if 'IMAGE' in event:
    #     link = window[event].metadata.link
    #     webbrowser.open(link)
    # return dicts, graph_lists


if __name__ == "__main__":
    sg.theme('LightGreen')
    n = 'intro'
    l, c, g, d = build_intro_tab()
    w = sg.Window(f'{n} gui', l, size=(1800, 1200), **w_kws, location=(300, 100))

    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        default_run_window(w, e, v, c, g)
        d, g = eval_intro_tab(e, v, w, collapsibles=c, dicts=d, graph_lists=g)
    w.close()
