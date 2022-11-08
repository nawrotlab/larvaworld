#!/usr/bin/env python
# !/usr/bin/env python
from lib.registry import reg
reg.init()

if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui
    my_gui = LarvaworldGui(tabs=None)
    my_gui.run()
