"""
Entry point for the larvaworld command-line interface.

Dispatches to either the ``rerun`` subcommand, which replays a simulation from a
stored ``run_manifest.json``, or to the standard :class:`SimModeParser` flow that
configures and launches one of the supported simulation modes.
"""

from __future__ import annotations
from argparse import ArgumentParser
import sys
from typing import Any, Callable

from .argparser import SimModeParser


def launch(P: SimModeParser, run: Any, args: Any) -> None:
    """Launch a configured simulation run.

    This is the default ``mainfun`` used by :func:`main`; it simply delegates to
    :meth:`SimModeParser.launch`.

    Args:
        P: The parser that configured the run.
        run: The configured simulation run object.
        args: The parsed command-line arguments.
    """
    P.launch(run, args)


def _rerun_main(cli_args: list[str]) -> None:
    """Handle the ``larvaworld rerun`` subcommand.

    Parses the rerun-specific options and replays the simulation described by a
    ``run_manifest.json`` file, printing the path of the resulting manifest.

    Args:
        cli_args: Command-line arguments following the ``rerun`` keyword.

    Raises:
        SystemExit: If an ``--input`` override is not in ``SOURCE=PATH`` form.
    """
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
    """Run the larvaworld CLI.

    Args:
        cli_args: Arguments to parse. Defaults to ``sys.argv[1:]``.
        mainfun: Callable invoked with the parser, the configured run, and the
            parsed arguments. Overridable to intercept the run instead of
            launching it (used by the test-suite).
    """
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
