import os
import webbrowser

import PySimpleGUI as sg
import numpy as np
from lib.conf.conf import loadConfDict, saveConf, deleteConf, loadConf, expandConf
from lib.gui.gui_lib import ClickableImage, window_size, t10_kws, graphic_button, t24_kws, named_list_layout, t8_kws
import lib.stor.paths as paths

class ProgressBarLayout :
    def __init__(self, list):
        self.list=list
        n=self.list.disp
        self.k=f'{n}_PROGRESSBAR'
        # print(self.k)
        self.k_complete=f'{n}_COMPLETE'
        self.l = [sg.Text('Progress :', **t8_kws),
                  sg.ProgressBar(100, orientation='h', size=(8.8, 20), key=self.k,
                                 bar_color=('green', 'lightgrey'), border_width=3),
                  graphic_button('check', self.k_complete, visible=False,
                                 tooltip='Whether the current {n} was completed.')]

    def reset(self, w):
        w[self.k].update(0)
        w[self.k_complete].update(visible=False)

    def run(self, w, min=0, max=100):
        w[self.k_complete].update(visible=False)
        w[self.k].update(0, max=max)


class SelectionList:
    def __init__(self, tab, conftype, disp=None, actions=[], sublists={},idx=None, progress=False, **kwargs):
        self.tab = tab
        self.conftype = conftype
        self.actions = actions

        if disp is None:
            disps = [k for k, v in self.tab.gui.tab_dict.items() if v[1] == conftype]
            if len(disps) == 1:
                disp = disps[0]
            elif len(disps) > 1:
                raise ValueError('Each selectionList is associated with a single configuration type')
        self.disp = disp

        if not progress :
            self.progressbar=None
        else :
            self.progressbar = ProgressBarLayout(self)
        self.k0 = f'{self.conftype}_CONF'
        if idx is not None :
            self.k=f'{self.k0}{idx}'
        else :
            self.k = self.get_next(self.k0)

        self.l = self.build(**kwargs)
        self.sublists = sublists



    def w(self):
        if not hasattr(self.tab.gui, 'window'):
            return None
        else:
            return self.tab.gui.window

    def c(self):
        return self.tab.gui.collapsibles

    def d(self):
        return self.tab.gui.dicts

    def g(self):
        return self.tab.gui.graph_lists

    def set_g(self, g):
        self.tab.gui.graph_lists =g

    def set_d(self, d):
        self.tab.gui.dicts =d

    def build(self, append=[]):

        acts = self.actions
        n = self.disp
        bs = []
        if self.progressbar is not None :
            append+=self.progressbar.l
            # print('ssss')


        if 'load' in acts:
            bs.append(graphic_button('load', f'LOAD_{n}', tooltip=f'Load the configuration for a {n}.'))
        if 'edit' in acts:
            bs.append(graphic_button('edit', f'EDIT_{n}', tooltip=f'Configure an existing or create a new {n}.')),
        if 'save' in acts:
            bs.append(graphic_button('data_add', f'SAVE_{n}', tooltip=f'Save a new {n} configuration.'))
        if 'delete' in acts:
            bs.append(graphic_button('data_remove', f'DELETE_{n}',
                                     tooltip=f'Delete an existing {n} configuration.'))
        if 'run' in acts:
            bs.append(graphic_button('play', f'RUN_{n}', tooltip=f'Run the selected {n}.'))

        temp=[
            [sg.Text(n.capitalize(), **t10_kws), *bs],
            [sg.Combo(self.confs, key=self.k, enable_events=True,
                      tooltip=f'The currently loaded {n}.', readonly=True,
                      size=(24, self.Nconfs)
                      )]
        ]
        if self.progressbar is not None :
            temp.append(self.progressbar.l)
        # if len(append)>0 :
        #     temp.append(append)

        # tt=[sg.Text(n.capitalize(), **t10_kws), *bs]
        # print(self.k)
        l = [sg.Col(temp)]
        # print(self.k)
        return l

    def eval(self, e, v):
        w = self.w()
        c = self.c()
        n = self.disp
        id = v[self.k]
        k0 = self.conftype
        g=self.g()
        d=self.d()

        if e == f'LOAD_{n}' and id != '':
            conf = loadConf(id, k0)
            self.tab.update(w, c, conf, id)
            if self.progressbar is not None :
                self.progressbar.reset(w)
            for kk, vv in self.sublists.items():
                vv.update(w, conf[kk])

        elif e == f'SAVE_{n}':
            conf = self.tab.get(w, v, c, as_entry=True)
            for kk, vv in self.sublists.items():
                conf[kk] = v[vv.k]
            id = self.save(conf)
            self.update(w, id)
        elif e == f'DELETE_{n}' and id != '':
            deleteConf(id, k0)
            self.update(w)
        elif e == f'RUN_{n}' and id != '':
            # print([v[l.k] for l in self.tab.selectionlists])
            conf = self.tab.get(w, v, c, as_entry=False)
            # print(conf)
            for kk, vv in self.sublists.items():
                # print(self.k, kk, vv.k, vv.conftype)
                conf[kk] = expandConf(id=v[vv.k], conf_type=vv.conftype)
            # print(conf)
            d,g=self.tab.run(v,w,c,d,g, conf, id)
            self.set_d(d)
            self.set_g(g)
        elif e == f'EDIT_{n}':
            conf = self.tab.get(w, v, c, as_entry=False)
            new_conf = self.tab.edit(conf)
            self.tab.update(w, c, new_conf, id=None)
            # self.update(new_env, w, c)

    def update(self, w, id='', all=False):
        # w=self.w()
        w.Element(self.k).Update(values=self.confs, value=id)
        # w[self.k].update(values=list(loadConfDict(self.conftype).keys()), value=id)
        if all:
            for i in range(5):
                k = f'{self.k0}{i}'
                if k in w.AllKeysDict.keys():
                    w[k].update(values=self.confs, value=id)

    def save(self, conf):
        n = self.disp
        l = [
            named_list_layout(f'Store new {n}', f'{n}_ID', list(loadConfDict(self.conftype).keys()),
                              readonly=False, enable_events=False),
            [sg.Ok(), sg.Cancel()]]
        e, v = sg.Window(f'{n} configuration', l).read(close=True)
        if e == 'Ok':
            id = v[f'{n}_ID']
            saveConf(conf, self.conftype, id)
        return id

        # for i in range(3):
        #     k = f'{self.conf_k}{i}'
        #     w.Element(k, silent_on_error=True).Update(values=list(loadConfDict(k).keys()),value=id)

    def get_next(self, k0):
        w = self.w()
        if w is None:
            idx = 0
        else:
            idx = int(np.min([i for i in range(5) if f'{k0}{i}' not in w.AllKeysDict.keys()]))
        return f'{k0}{idx}'

    @property
    def confs(self):
        return list(loadConfDict(self.conftype).keys())

    @property
    def Nconfs(self):
        return len(self.confs)





class GuiTab:
    def __init__(self, name, gui):
        self.name = name
        self.gui = gui
        self.selectionlists = []

    def build(self):
        return None, {}, {}, {}

    def eval0(self, e, v):
        for i in self.selectionlists:
            i.eval(e, v)
        w = self.gui.window
        c = self.gui.collapsibles
        g = self.gui.graph_lists
        d = self.gui.dicts
        self.eval(e, v, w, c, d, g)

    def run(self, v, w,c, d, g, conf, id):
        pass
        # return d, g

    def eval(self, e, v, w, c, d, g):
        pass

    def get(self, w, v, c):
        return None

    def update(self, w, c, conf, id):
        pass


class IntroTab(GuiTab):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    def build(self):
        c = {'size': (80, 1),
             'pad': (20, 5),
             'justification': 'center'
             }

        filenames = os.listdir(paths.IntroSlideFolder)
        filenames.sort()

        b_list = [
            sg.B(image_filename=os.path.join(paths.IntroSlideFolder, f), image_subsample=3,
                 pad=(15, 70)) for f in filenames]
        l_title = sg.Col([[sg.T('', size=(5, 5))],
                          [sg.T('Larvaworld', font=("Cursive", 40, "italic"), **c)],
                          b_list,
                          [sg.T('Behavioral analysis and simulation platform', font=("Lobster", 15, "bold"), **c)],
                          [sg.T('for Drosophila larva', font=("Lobster", 15, "bold"), **c)]],
                         element_justification='center',
                         )

        l_intro = [[l_title]]

        return l_intro, {}, {}, {}


class VideoTab(GuiTab):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    def build(self):
        link_pref = "http://computational-systems-neuroscience.de/wp-content/uploads/2021/04/"
        files = [f for f in os.listdir(paths.VideoSlideFolder) if f.endswith('png')]
        b_list = [ClickableImage(name=f.split(".")[0], link=f'{link_pref}{f.split(".")[0]}.mp4',
                                 image_filename=os.path.join(paths.VideoSlideFolder, f),
                                 image_subsample=5, pad=(25, 40)) for f in files]

        n = 3
        b_list = [b_list[i * n:(i + 1) * n] for i in range((len(b_list) + n - 1) // n)]
        l = [[sg.Col(b_list, vertical_scroll_only=True, scrollable=True, size=window_size)]]

        return l, {}, {}, {}

    def eval(self, e, v, w, c, d, g):
        if 'ClickableImage' in e:
            w[e].eval()

        # return d, g

class TutorialTab(GuiTab):

    def build(self):
        c2 = {'size': (80, 1),
              'pad': (20, 5),
              'justification': 'left'
              }
        c1 = {'size': (70, 1),
              'pad': (10, 35)
              }

        col1 = [[sg.B(image_filename=os.path.join(paths.TutorialSlideFolder, '1.png'), key='BUTTON 1',
                      image_subsample=2, image_size=(70, 70), pad=(20, 10))],
                [sg.B(image_filename=os.path.join(paths.TutorialSlideFolder, '2.png'), key='BUTTON 2',
                      image_subsample=3, image_size=(70, 70), pad=(20, 10))],
                [sg.B(image_filename=os.path.join(paths.TutorialSlideFolder, '3.png'), key='BUTTON 3',
                      image_subsample=3, image_size=(70, 70), pad=(20, 10))],
                [sg.B(image_filename=os.path.join(paths.TutorialSlideFolder, '4.png'), key='BUTTON 4',
                      image_subsample=2, image_size=(70, 70), pad=(20, 10))]]
        col2 = [
            [sg.T('1. Run and analyze a single run experiment with selected larva model and environment',
                  font='Lobster 12', **c1)],
            [sg.T('2. Run and analyze a batch run experiment with selected larva model and environment',
                  font='Lobster 12', **c1)],
            [sg.T('3. Create your own experimental environment', font='Lobster 12', **c1)],
            [sg.T('4. Change settings and shortcuts', font='Lobster 12', **c1)],
        ]
        l_tut = [
            [sg.T('')],
            [sg.T('Tutorials', font=("Cursive", 40, "italic"), **c2)],
            [sg.T('Choose between following video tutorials:', font=("Lobster", 15, "bold"), **c2)],
            [sg.T('')],
            [sg.Column(col1), sg.Column(col2)],
            [sg.T('')],
            [sg.T('Further information:', font=("Lobster", 15, "bold"), **c2)],
            [sg.T('')],
            [sg.B(image_filename=os.path.join(paths.TutorialSlideFolder, 'Glossary.png'), key='GLOSSARY', image_subsample=3,
                  image_size=(70, 70), pad=(25, 10)),
             sg.T('Here you find a glossary explaining all variables in Larvaworld', font='Lobster 12', **c1)]

        ]

        return l_tut, {}, {}, {}

    def eval(self, event, values, window, collapsibles, dicts, graph_lists):
        if 'BUTTON 1' in event:
            webbrowser.open_new(paths.TutorialSlideFolder + "/1.mp4")
        if 'BUTTON 2' in event:
            webbrowser.open_new(paths.TutorialSlideFolder + "/2.mp4")
        if 'BUTTON 3' in event:
            webbrowser.open_new(paths.TutorialSlideFolder + "/3.mp4")
        if 'BUTTON 4' in event:
            webbrowser.open_new(paths.TutorialSlideFolder + "/4.mp4")
        if 'GLOSSARY' in event:
            webbrowser.open_new(paths.TutorialSlideFolder + "/Glossary.pdf")
        return dicts, graph_lists


if __name__ == "__main__":
    pass
    # sg.theme('LightGreen')
    # n = 'intro'
    # l, c, g, d = build_intro_tab()
    # w = sg.Window(f'{n} gui', l, size=(1800, 1200), **w_kws, location=(300, 100))
    #
    # while True:
    #     e, v = w.read()
    #     if e in (None, 'Exit'):
    #         break
    #     default_run_window(w, e, v, c, g)
    #     d, g = eval_intro_tab(e, v, w, collapsibles=c, dicts=d, graph_lists=g)
    # w.close()
