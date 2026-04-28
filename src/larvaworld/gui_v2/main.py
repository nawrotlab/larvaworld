from __future__ import annotations

from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser(description="Launch the Larvaworld gui_v2 desktop shell.")
    parser.add_argument(
        "--geometry",
        default="1360x860",
        help="Window geometry in WIDTHxHEIGHT form.",
    )
    args = parser.parse_args()

    try:
        from larvaworld.gui_v2.shell import LarvaworldGuiV2Shell
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            raise RuntimeError(
                "gui_v2 requires PySide6 in the active Python environment. "
                "Install the project dependencies again after adding PySide6."
            ) from exc
        raise

    shell = LarvaworldGuiV2Shell(geometry=args.geometry)
    shell.run()


if __name__ == "__main__":
    main()
