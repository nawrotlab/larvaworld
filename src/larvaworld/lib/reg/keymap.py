"""
Keymap/shortcuts for interactive pygame visualization of simulations.
"""

from __future__ import annotations
from typing import Any, Dict

import json

from ... import CONF_DIR
from ...lib import util

__all__: list[str] = [
    "ControlRegistry",
]


def get_pygame_key(key: str) -> str:
    pygame_keys = util.AttrDict(
        {
            "BackSpace": "BACKSPACE",
            "tab": "TAB",
            "del": "DELETE",
            "clear": "CLEAR",
            "Return": "RETURN",
            "Escape": "ESCAPE",
            "space": "SPACE",
            "exclam": "EXCLAIM",
            "quotedbl": "QUOTEDBL",
            "+": "PLUS",
            "comma": "COMMA",
            "-": "MINUS",
            "period": "PERIOD",
            "slash": "SLASH",
            "numbersign": "HASH",
            "Down:": "DOWN",
            "Up:": "UP",
            "Right:": "RIGHT",
            "Left:": "LEFT",
            "dollar": "DOLLAR",
            "ampersand": "AMPERSAND",
            "parenleft": "LEFTPAREN",
            "parenright": "RIGHTPAREN",
            "asterisk": "ASTERISK",
        }
    )
    return f"K_{pygame_keys[key]}" if key in pygame_keys else f"K_{key}"


def init_shortcuts() -> util.AttrDict:
    d = util.AttrDict(
        {
            "draw": {
                "visible_trails": "p",
                "▲ trail duration": "+",
                "▼ trail duration": "-",
                "trail_color": "x",
                "draw_head": "h",
                "draw_centroid": "e",
                "draw_midline": "m",
                "draw_contour": "c",
                "draw_sensors": "j",
                "draw_orientations": "k",
                "draw_segs": "l",
            },
            "color": {
                "black_background": "g",
                "random_colors": "r",
                "color_behavior": "b",
            },
            "aux": {
                "visible_clock": "t",
                "visible_scale": "n",
                "visible_state": "s",
                "visible_ids": "tab",
            },
            "screen": {
                "move up": "UP",
                "move down": "DOWN",
                "move left": "LEFT",
                "move right": "RIGHT",
            },
            "simulation": {
                "larva_collisions": "y",
                "pause": "space",
                "snapshot": "i",
                "delete item": "del",
            },
            "inspect": {
                "focus_mode": "f",
                "odor gains": "z",
                "dynamic graph": "q",
            },
            "landscape": {
                "odor_aura": "u",
                "windscape": "w",
                "plot odorscapes": "o",
                **{f"odorscape {i}": i for i in range(10)},
                # 'move_right': 'RIGHT',
            },
        }
    )

    return d


def init_controls() -> util.AttrDict:
    k = init_shortcuts()
    d = util.AttrDict(
        {
            "keys": {},
            "mouse": {
                "select item": "left click",
                "add item": "left click",
                "select item mode": "right click",
                "inspect item": "right click",
                "screen zoom in": "scroll up",
                "screen zoom out": "scroll down",
            },
        }
    )
    ds = {}
    for title, dic in k.items():
        ds.update(dic)
        d.keys[title] = dic
    d.pygame_keys = {k: get_pygame_key(v) for k, v in ds.items()}
    return d


class ControlRegistry:
    """
    Registry for keyboard and mouse controls in pygame visualizations.

    Manages keyboard shortcuts and mouse controls for interactive simulation
    visualization. Controls are saved to and loaded from a configuration file.

    Attributes:
        path: Path to the controls configuration file
        conf: AttrDict containing control mappings with sections:
            - keys: Keyboard shortcuts organized by category
            - mouse: Mouse control mappings
            - pygame_keys: Pygame key constant mappings

    Example:
        >>> controls = ControlRegistry()
        >>> controls.conf.keys['draw']['visible_trails']  # 'p'
        >>> controls.conf.mouse['select item']  # 'left click'
        >>> controls.save()  # Save current configuration
        >>> loaded = controls.load()  # Load from file
    """

    def __init__(self) -> None:
        self.path = f"{CONF_DIR}/controls.txt"
        self.conf = init_controls()
        self.save(self.conf)

    def save(self, conf: util.AttrDict | None = None) -> None:
        if conf is None:
            conf = self.conf
        with open(self.path, "w") as fp:
            json.dump(conf, fp)

    def load(self) -> util.AttrDict:
        with open(self.path) as tfp:
            c = json.load(tfp)
        return util.AttrDict(c)
