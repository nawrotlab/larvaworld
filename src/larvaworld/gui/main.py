#!/usr/bin/env python
# !/usr/bin/env python

from argparse import ArgumentParser

def main() :
    p=ArgumentParser()
    p.add_argument('-t', '--tabs', type=str, nargs='+', help='The tabs to include in the GUI')

    args = p.parse_args()
    from larvaworld.gui.tabs.larvaworld_gui import LarvaworldGui
    my_gui = LarvaworldGui(tabs=args.tabs)
    my_gui.run()

if __name__ == '__main__':
    main()