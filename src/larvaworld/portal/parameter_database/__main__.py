"""Standalone launcher: `python -m larvaworld.portal.parameter_database`.

Serves just the Parameter Database view on its own, without the rest of the
portal (landing page, workspace, other apps). Configure via env vars, same
as the main portal:
- LARVAWORLD_PORTAL_PORT (default: 5006)
- LARVAWORLD_PORTAL_OPEN_BROWSER (default: true on Windows/macOS)
"""

from __future__ import annotations

import os
import sys


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_open_browser() -> bool:
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


def main() -> None:
    import panel as pn

    from larvaworld.portal.parameter_database.parameter_db_app import (
        build_standalone_page,
    )

    port = int(os.getenv("LARVAWORLD_PORTAL_PORT", "5006"))
    open_browser = _env_flag("LARVAWORLD_PORTAL_OPEN_BROWSER", _default_open_browser())

    pn.extension("tabulator")
    pn.serve(
        build_standalone_page,
        port=port,
        show=open_browser,
        threaded=False,
        num_procs=1,
        use_xheaders=False,
    )


if __name__ == "__main__":
    main()
