"""
Keymap/shortcuts for interactive pygame visualization of simulations.
"""

from __future__ import annotations
from typing import Any, Dict

import json
import string

from ... import CONF_DIR
from ...lib import util

__all__: list[str] = [
    "ControlRegistry",
    "build_pygame_keys",
    "default_controls",
    "merge_controls",
    "validate_key_name",
    "validate_shortcut_conf",
]


_SUPPORTED_SPECIAL_KEYS = {
    "space",
    "tab",
    "del",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "+",
    "-",
}


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


def default_controls() -> util.AttrDict:
    """Return a fresh controls configuration without filesystem writes."""
    return util.AttrDict(init_controls()).get_copy()


def build_pygame_keys(keys: dict[str, dict[str, str]]) -> dict[str, str]:
    """Build flattened action -> pygame key mapping from nested shortcut keys."""
    result: dict[str, str] = {}
    for section in keys.values():
        if not isinstance(section, dict):
            continue
        for action, key in section.items():
            if not isinstance(action, str) or not isinstance(key, str):
                continue
            result[action] = get_pygame_key(key)
    return result


def merge_controls(
    defaults: dict[str, Any], overrides: dict[str, Any]
) -> util.AttrDict:
    """Merge workspace overrides over defaults and rebuild derived pygame keys."""
    base = util.AttrDict(defaults).get_copy()
    keys = util.AttrDict(base.get("keys", {})).get_copy()
    override_keys = overrides.get("keys", {}) if isinstance(overrides, dict) else {}
    if isinstance(override_keys, dict):
        for section_name, section_values in override_keys.items():
            if not isinstance(section_name, str) or not isinstance(
                section_values, dict
            ):
                continue
            current = util.AttrDict(keys.get(section_name, {})).get_copy()
            for action, value in section_values.items():
                if isinstance(action, str) and isinstance(value, str):
                    current[action] = value
            keys[section_name] = current
    base["keys"] = keys
    base["pygame_keys"] = build_pygame_keys(keys)
    return util.AttrDict(base)


def validate_key_name(key: str) -> bool:
    """Return True when key name is supported by the V1a shortcut editor."""
    if not isinstance(key, str):
        return False
    normalized = key.strip()
    if not normalized:
        return False
    if normalized in _SUPPORTED_SPECIAL_KEYS:
        return True
    if len(normalized) == 1 and (
        normalized in string.ascii_letters or normalized in string.digits
    ):
        return True
    return False


def validate_shortcut_conf(keys: dict[str, dict[str, str]]) -> list[str]:
    """Validate shortcut keys and return a list of human-readable errors."""
    errors: list[str] = []
    if not isinstance(keys, dict):
        return ["Shortcut keys configuration must be a mapping."]

    assigned: dict[str, str] = {}
    for section_name, section_values in keys.items():
        if not isinstance(section_name, str) or not isinstance(section_values, dict):
            errors.append(f'Invalid shortcut section "{section_name}".')
            continue
        for action_name, value in section_values.items():
            path = f"{section_name}.{action_name}"
            if not isinstance(action_name, str):
                errors.append(f"Invalid action name at {path}.")
                continue
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{path}: key must be a non-empty string.")
                continue
            key = value.strip()
            if not validate_key_name(key):
                errors.append(f'{path}: unsupported key "{key}".')
                continue
            previous = assigned.get(key)
            if previous is not None and previous != path:
                errors.append(
                    f'{path}: key "{key}" is already assigned to "{previous}".'
                )
            else:
                assigned[key] = path
    return errors


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
