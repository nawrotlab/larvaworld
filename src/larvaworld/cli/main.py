from __future__ import annotations
from typing import Any, Callable

from .argparser import SimModeParser


def launch(P: SimModeParser, run: Any, args: Any) -> None:
    P.launch(run, args)


def main(
    cli_args: list[str] | None = None,
    mainfun: Callable[[SimModeParser, Any, Any], None] = launch,
) -> None:
    P = SimModeParser()
    args = P.parse_args(args=cli_args)
    run, run_kws = P.configure(args)
    if args.show_parser_args:
        P.show_args(args=args, run_kws=run_kws)
    mainfun(P, run, args)


if __name__ == "__main__":
    main()
