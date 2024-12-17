import os

import PySimpleGUI as sg

from ...gui import gui_aux
from ... import ROOT_DIR

__all__ = [
    "VideoTab",
]


class VideoTab(gui_aux.GuiTab):
    def build(self):
        link0 = (
            "http://computational-systems-neuroscience.de/wp-content/uploads/2024/10/"
        )
        f0 = f"{ROOT_DIR}/gui/media/video"
        fs = [f for f in os.listdir(f0) if f.endswith("png")]
        ns = [f.split(".")[0] for f in fs]
        ffs = [os.path.join(f0, f) for f in fs]
        bl = [
            gui_aux.ClickableImage(
                name=n,
                link=f"{link0}{n}.mp4",
                image_filename=ff,
                image_subsample=5,
                pad=(25, 40),
            )
            for ff, n in zip(ffs, ns)
        ]
        n = 3
        bl = [bl[i * n : (i + 1) * n] for i in range((len(bl) + n - 1) // n)]
        l = [
            [
                sg.Col(
                    bl,
                    vertical_scroll_only=True,
                    scrollable=True,
                    size=gui_aux.window_size,
                )
            ]
        ]

        return l, {}, {}, {}

    def eval(self, e, v, w, c, d, g):
        if "ClickableImage" in e:
            w[e].eval()


if __name__ == "__main__":
    from .larvaworld_gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=["videos"])
    larvaworld_gui.run()
