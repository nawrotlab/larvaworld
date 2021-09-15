#!/usr/bin/env python
# !/usr/bin/env python
import sys
import time

from lib.gui.tabs.gui import LarvaworldGui

sys.path.insert(0, '..')

if __name__ == "__main__":
    s0=time.time()
    larvaworld_gui = LarvaworldGui()
    s1 = time.time()
    larvaworld_gui.run()
