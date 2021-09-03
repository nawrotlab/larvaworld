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
    larvaworld_gui.run()
