#!/usr/bin/env python
# !/usr/bin/env python
import sys
import time
sys.path.insert(0, '..')




if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui
    # ns0=['life-history', 'simulation']
    # ns0=['introduction', 'tutorials', 'larva-model', 'environment', 'life-history', 'simulation', 'essay', 'batch-run', 'analysis', 'import', 'videos', 'settings']

    larvaworld_gui = LarvaworldGui(tabs=None)
    larvaworld_gui.run()
