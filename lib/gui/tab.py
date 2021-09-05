import os
import webbrowser

import PySimpleGUI as sg
from lib.conf.conf import loadConf
from lib.gui.aux import window_size
from lib.gui.buttons import ClickableImage
import lib.stor.paths as paths


class GuiTab:
    def __init__(self, name, gui, conftype=None):
        self.name = name
        self.gui = gui
        self.conftype = conftype
        self.selectionlists = {}
        self.datalists = {}
        # self.graph_list=None

    @property
    def graph_list(self):
        gs = self.gui.graph_lists
        n = self.name
        if n in list(gs.keys()):
            return gs[n]
        else:
            return None

    @property
    def canvas_k(self):
        g=self.graph_list
        return g.canvas_key if g is not None else None

    @property
    def graphlist_k(self):
        g = self.graph_list
        return g.list_key if g is not None else None

    @property
    def base_list(self):
        return self.selectionlists[self.conftype] if self.conftype is not None else None

    @property
    def datalist(self):
        return self.datalists[self.name] if self.name in list(self.datalists.keys()) else None

    @property
    def base_dict(self):
        ds=self.gui.dicts
        n=self.name
        if n in list(ds.keys()) :
            return ds[n]
        else :
            return None

    def current_ID(self, v):
        l=self.base_list
        return v[l.k] if l is not None else None

    def current_conf(self, v):
        id=self.current_ID(v)
        return loadConf(id, self.conftype) if id is not None else None

    def build(self):
        return None, {}, {}, {}

    def eval0(self, e, v):

        w = self.gui.window
        c = self.gui.collapsibles
        g = self.gui.graph_lists
        d = self.gui.dicts
        for sl_name,sl in self.selectionlists.items():
            sl.eval(e, v, w, c, d, g)
        for dl_name,dl in self.datalists.items():
            dl.eval(e, v, w, c, d, g)
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

    def fork(self, func, kwargs):
        import os
        import signal
        import sys
        def handle_signal(signum, frame):
            print('Caught signal "%s"' % signum)
            if signum == signal.SIGTERM:
                print('SIGTERM. Exiting!')
                sys.exit(1)
            elif signum == signal.SIGHUP:
                print('SIGHUP')
            elif signum == signal.SIGUSR1:
                print('SIGUSR1 Calling wait()')
                pid, status = os.wait()
                print('PID was: %s.' % pid)

        print('Starting..')
        signal.signal(signal.SIGCHLD, handle_signal)
        signal.signal(signal.SIGHUP, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGUSR1, handle_signal)

        try:
            ff_pid = os.fork()
        except OSError as err:
            print('Unable to fork: %s' % err)
        if ff_pid > 0:
            # Parent.
            print('First fork.')
            print('Child PID: %d' % ff_pid)
        elif ff_pid == 0:
            res=func(**kwargs)
            # return res
            # sys.exit(0)




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

    def eval(self, e, v, w, c, d, g):
        if 'BUTTON 1' in e:
            webbrowser.open_new(paths.TutorialSlideFolder + "/1.mp4")
        if 'BUTTON 2' in e:
            webbrowser.open_new(paths.TutorialSlideFolder + "/2.mp4")
        if 'BUTTON 3' in e:
            webbrowser.open_new(paths.TutorialSlideFolder + "/3.mp4")
        if 'BUTTON 4' in e:
            webbrowser.open_new(paths.TutorialSlideFolder + "/4.mp4")
        if 'GLOSSARY' in e:
            webbrowser.open_new(paths.TutorialSlideFolder + "/Glossary.pdf")
        return d, g


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
