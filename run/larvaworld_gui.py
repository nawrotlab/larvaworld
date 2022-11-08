#!/usr/bin/env python
# !/usr/bin/env python
import sys

sys.path.insert(0, '..')
from lib.registry import reg

# reg.init0()
reg.init()

if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui
    my_gui = LarvaworldGui(tabs=None)
    my_gui.run()
