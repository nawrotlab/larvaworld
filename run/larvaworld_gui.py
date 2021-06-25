#!/usr/bin/env python
# !/usr/bin/env python
import sys
import time

sys.path.insert(0, '..')
from lib.gui.gui import LarvaworldGui

if __name__ == "__main__":
    s0=time.time()
    larvaworld_gui = LarvaworldGui()
    s1 = time.time()
    print(s1-s0)
    # larvaworld_gui = LarvaworldGui(tabs=['exp'])
    # larvaworld_gui = LarvaworldGui(tabs=['model', 'exp', 'settings'])
    larvaworld_gui.run()
