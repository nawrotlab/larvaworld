from __future__ import annotations

from larvaworld.lib.reg import keymap


def test_default_controls_contains_expected_sections() -> None:
    controls = keymap.default_controls()
    assert "keys" in controls
    assert "mouse" in controls
    assert "pygame_keys" in controls
    assert controls.keys["simulation"]["pause"] == "space"


def test_build_pygame_keys_maps_supported_keys() -> None:
    result = keymap.build_pygame_keys(
        {
            "simulation": {"pause": "space"},
            "aux": {"visible_ids": "tab"},
            "screen": {"move up": "UP"},
            "draw": {"▲ trail duration": "+"},
        }
    )
    assert result["pause"] == "K_SPACE"
    assert result["visible_ids"] == "K_TAB"
    assert result["move up"] == "K_UP"
    assert result["▲ trail duration"] == "K_PLUS"


def test_merge_controls_applies_overrides_and_rebuilds_pygame_keys() -> None:
    defaults = keymap.default_controls()
    merged = keymap.merge_controls(
        defaults,
        {"keys": {"simulation": {"pause": "tab"}}},
    )
    assert merged.keys["simulation"]["pause"] == "tab"
    assert merged.pygame_keys["pause"] == "K_TAB"


def test_validate_shortcut_conf_rejects_empty_invalid_and_duplicate_keys() -> None:
    errors = keymap.validate_shortcut_conf(
        {
            "simulation": {"pause": "", "snapshot": "nope"},
            "draw": {"trail_color": "p"},
            "aux": {"visible_clock": "p"},
        }
    )
    assert any("pause" in error and "non-empty" in error for error in errors)
    assert any('unsupported key "nope"' in error for error in errors)
    assert any('key "p" is already assigned' in error for error in errors)
