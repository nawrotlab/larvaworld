

from larvaworld.lib import aux
from larvaworld.cli.argparser import SimModeParser


def main():

    MP=SimModeParser()
    # raise
    # args = aux.AttrDict(vars(MP.p.parse_args()))
    MP.parse_args()
    MP.configure(show_args=True)
    # raise
    # MP.launch()


if __name__ == '__main__':
    main()