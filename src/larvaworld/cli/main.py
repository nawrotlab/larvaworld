

from larvaworld.lib import reg, aux
from larvaworld.cli.argparser import SimModeParser, ParsDict


def main():

    # b=ParsDict.from_param(reg.gen.Replay())
    # MP=b.add()

    MP=SimModeParser()
    # print(MP.cli_parser.parse_args())
    # raise
    # args = aux.AttrDict(vars(MP.p.parse_args()))
    MP.parse_args()
    # args = aux.AttrDict(vars(MP.parse_args()))
    # print(args)
    # a=b.get(args)
    # print(a)
    # raise

    MP.configure(show_args=False)
    # raise
    MP.launch()


if __name__ == '__main__':
    main()