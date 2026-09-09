from __future__ import annotations

import os
import subprocess
import sys


def test_canvas_widgets_root_import_is_lazy() -> None:
    code = r"""
import sys
from larvaworld.portal.canvas_widgets import (
    CanvasArena,
    CanvasObject,
    EnvironmentCanvasState,
    env_params_to_canvas_state,
)

assert CanvasArena is not None
assert CanvasObject is not None
assert EnvironmentCanvasState is not None
assert env_params_to_canvas_state is not None

for name in (
    "panel",
    "bokeh",
    "larvaworld.portal.canvas_widgets.environment_canvas",
):
    assert name not in sys.modules, name

from larvaworld.portal.canvas_widgets import EnvironmentCanvas

assert EnvironmentCanvas is not None
assert "larvaworld.portal.canvas_widgets.environment_canvas" in sys.modules
    """
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
