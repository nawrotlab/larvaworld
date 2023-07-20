

from larvaworld.lib import aux
from larvaworld.cli.argparser import SimModeParser


def main():

    MP=SimModeParser()
    # print(MP.cli_parser.parse_args())
    # raise
    # args = aux.AttrDict(vars(MP.p.parse_args()))
    MP.parse_args()
    MP.configure(show_args=False)
    # raise
    MP.launch()


if __name__ == '__main__':
    main()