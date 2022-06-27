import os
import webbrowser

import PySimpleGUI as sg

from lib.registry.pars import preg
from lib.gui.tabs.tab import GuiTab
# from lib.conf.base.paths import path


class TutorialTab(GuiTab):

    def build(self):
        f0=preg.path_dict["tutorials"]
        # f0=path('tutorials')
        c2 = {'size': (80, 1),
              'pad': (20, 5),
              'justification': 'left'
              }
        c1 = {'size': (70, 1),
              'pad': (10, 35)
              }

        col1 = [[sg.B(image_filename=os.path.join(f0, '1.png'), key='BUTTON 1',
                      image_subsample=2, image_size=(70, 70), pad=(20, 10))],
                [sg.B(image_filename=os.path.join(f0, '2.png'), key='BUTTON 2',
                      image_subsample=3, image_size=(70, 70), pad=(20, 10))],
                [sg.B(image_filename=os.path.join(f0, '3.png'), key='BUTTON 3',
                      image_subsample=3, image_size=(70, 70), pad=(20, 10))],
                [sg.B(image_filename=os.path.join(f0, '4.png'), key='BUTTON 4',
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
            [sg.B(image_filename=os.path.join(f0, 'Glossary.png'), key='GLOSSARY', image_subsample=3,
                  image_size=(70, 70), pad=(25, 10)),
             sg.T('Here you find a glossary explaining all variables in Larvaworld', font='Lobster 12', **c1)]

        ]

        return l_tut, {}, {}, {}

    def eval(self, e, v, w, c, d, g):
        f0=preg.path_dict["tutorials"]
        if 'BUTTON 1' in e:
            webbrowser.open_new(f0 + "/1.mp4")
        if 'BUTTON 2' in e:
            webbrowser.open_new(f0 + "/2.mp4")
        if 'BUTTON 3' in e:
            webbrowser.open_new(f0 + "/3.mp4")
        if 'BUTTON 4' in e:
            webbrowser.open_new(f0 + "/4.mp4")
        if 'GLOSSARY' in e:
            webbrowser.open_new(f0 + "/Glossary.pdf")
        return d, g