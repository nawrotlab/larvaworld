import os

import PySimpleGUI as sg

from ...gui import gui_aux
from ... import ROOT_DIR

__all__ = [
    "IntroTab",
]


class IntroTab(gui_aux.GuiTab):
    def build(self):
        kws = {"size": (80, 1), "pad": (20, 5), "justification": "center"}
        f0 = f"{ROOT_DIR}/gui/media/intro"
        fs = sorted(os.listdir(f0))
        bl = [
            sg.B(image_filename=os.path.join(f0, f), image_subsample=3, pad=(15, 70))
            for f in fs
        ]
        l = [
            [
                sg.Col(
                    [
                        [sg.T("", size=(5, 5))],
                        [sg.T("Larvaworld", font=("Cursive", 40, "italic"), **kws)],
                        bl,
                        [
                            sg.T(
                                "Behavioral analysis and simulation platform",
                                font=("Lobster", 15, "bold"),
                                **kws,
                            )
                        ],
                        [
                            sg.T(
                                "for Drosophila larva",
                                font=("Lobster", 15, "bold"),
                                **kws,
                            )
                        ],
                    ],
                    element_justification="center",
                )
            ]
        ]
        return l, {}, {}, {}


if __name__ == "__main__":
    from .larvaworld_gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=["intro"])
    larvaworld_gui.run()
