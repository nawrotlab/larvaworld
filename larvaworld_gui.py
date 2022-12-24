#!/usr/bin/env python
# !/usr/bin/env python

# from lib.registry import reg
# reg.init()
# from lib.gui.aux.tab_dict import tab_keys
from argparse import ArgumentParser
p=ArgumentParser()
p.add_argument('-t', '--tabs', type=str, nargs='+', help='The tabs to include in the GUI')

if __name__ == "__main__":
    args = p.parse_args()
    from gui.tabs.gui import LarvaworldGui
    my_gui = LarvaworldGui(tabs=args.tabs)
    my_gui.run()
