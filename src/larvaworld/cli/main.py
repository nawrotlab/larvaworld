from __future__ import annotations
from argparse import ArgumentParser
import sys
from typing import Any, Callable

from .argparser import SimModeParser


def launch(P: SimModeParser, run: Any, args: Any) -> None:
    P.launch(run, args)


def _rerun_main(cli_args: list[str]) -> None:
    parser = ArgumentParser(
        prog="larvaworld rerun",
        description="Rerun a simulation from run_manifest.json",
    )
    parser.add_argument("manifest")
    parser.add_argument(
        "--reproducibility",
        choices=("strict", "parameters"),
        default="strict",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--with-media", action="store_true")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="Override an input by original path, dataset id, ref id, or role.",
    )
    args = parser.parse_args(cli_args)
    overrides: dict[str, str] = {}
    for item in args.input:
        key, separator, value = item.partition("=")
        if not separator or not key.strip() or not value.strip():
            parser.error("--input must use SOURCE=PATH syntax")
        overrides[key.strip()] = value.strip()
    from larvaworld.lib.sim.manifest import rerun_from_manifest

    rerun = rerun_from_manifest(
        args.manifest,
        reproducibility=args.reproducibility,
        output_dir=args.output_dir,
        allow_version_mismatch=args.allow_version_mismatch,
        input_overrides=overrides or None,
        with_media=args.with_media,
    )
    print(f"Rerun manifest: {rerun.manifest_path}")


def main(
    cli_args: list[str] | None = None,
    mainfun: Callable[[SimModeParser, Any, Any], None] = launch,
) -> None:
    effective_args = list(sys.argv[1:] if cli_args is None else cli_args)
    if effective_args and effective_args[0] == "rerun":
        _rerun_main(effective_args[1:])
        return
    P = SimModeParser()
    args = P.parse_args(args=effective_args)
    run, run_kws = P.configure(args)
    if args.show_parser_args:
        P.show_args(args=args, run_kws=run_kws)
    mainfun(P, run, args)


if __name__ == "__main__":
    main()
