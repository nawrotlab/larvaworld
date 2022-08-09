#!/usr/bin/env python
# !/usr/bin/env python
import sys
sys.path.insert(0, '..')
from lib.registry import reg
reg.init()



if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=None)
    larvaworld_gui.run()
