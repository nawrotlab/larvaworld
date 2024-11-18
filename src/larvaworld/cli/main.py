from .argparser import SimModeParser


def main():
    P = SimModeParser()
    P.parse_args()
    P.configure(show_args=P.args.show_parser_args)
    P.launch()


if __name__ == "__main__":
    main()
