from .argparser import SimModeParser


def launch(P, run, args):
    P.launch(run, args)


def main(cli_args=None, mainfun=launch):
    P = SimModeParser()
    args = P.parse_args(args=cli_args)
    run, run_kws = P.configure(args)  # type: ignore
    if args.show_parser_args:
        P.show_args(args=args, run_kws=run_kws)
    mainfun(P, run, args)


if __name__ == "__main__":
    main()
