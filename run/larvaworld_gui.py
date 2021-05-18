#!/usr/bin/env python
# !/usr/bin/env python
import sys

sys.path.insert(0, '..')
from lib.gui.gui import LarvaworldGui

if __name__ == "__main__":
    # larvaworld_gui = LarvaworldGui()
    larvaworld_gui = LarvaworldGui(tabs=['anal', 'settings'])
    # larvaworld_gui = LarvaworldGui(tabs=['intro','model', 'exp', 'batch', 'anal', 'video'])
    larvaworld_gui.run()
