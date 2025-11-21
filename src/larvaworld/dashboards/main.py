from __future__ import annotations
from typing import Dict, Callable

import panel as pn


def main() -> None:
    from .model_inspector import model_inspector_app
    from .module_inspector import module_inspector_app
    from .track_viewer import track_viewer_app
    from .experiment_viewer import experiment_viewer_app
    from .lateral_oscillator_inspector import lateral_oscillator_app

    apps: Dict[str, Callable[..., object]] = {
        "larva_models": model_inspector_app,
        "locomotory_modules": module_inspector_app,
        "lateral_oscillator": lateral_oscillator_app,
        "track_viewer": track_viewer_app,
        "experiment_viewer": experiment_viewer_app,
    }
    pn.serve(apps, port=5006)


if __name__ == "__main__":
    main()
