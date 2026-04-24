from __future__ import annotations

import panel as pn
import pytest

from larvaworld.lib.param.spatial import Area
from larvaworld.lib.reg.generators import EnvConf, FoodConf
from larvaworld.portal.config_widgets import (
    build_area_widget,
    build_border_widget,
    build_env_params_widget,
    build_food_conf_widget,
)


def _find_widget(
    viewable: pn.viewable.Viewable,
    name: str,
    widget_type: type[pn.widgets.Widget] | tuple[type[pn.widgets.Widget], ...] = pn.widgets.Widget,
):
    for widget in viewable.select(widget_type):
        if getattr(widget, "name", None) == name:
            return widget
    raise AssertionError(f"Could not find widget {name!r}.")


def _scape_switches(viewable: pn.viewable.Viewable) -> list[pn.widgets.Switch]:
    return [widget for widget in viewable.select(pn.widgets.Switch)]


def test_area_widget_syncs_bidirectionally() -> None:
    area = Area()
    widget = build_area_widget(area)

    assert pn.Column(widget).get_root() is not None

    width = _find_widget(widget, "Arena width", pn.widgets.FloatInput)
    height = _find_widget(widget, "Arena height", pn.widgets.FloatInput)
    geometry = _find_widget(widget, "Geometry", pn.widgets.Select)
    torus = _find_widget(widget, "Torus", pn.widgets.Checkbox)

    width.value = 0.24
    height.value = 0.18
    geometry.value = "rectangular"
    torus.value = True

    assert area.dims == pytest.approx((0.24, 0.18))
    assert area.geometry == "rectangular"
    assert area.torus is True

    area.param.update(dims=(0.31, 0.27), geometry="circular", torus=False)

    assert width.value == pytest.approx(0.31)
    assert height.value == pytest.approx(0.27)
    assert geometry.value == "circular"
    assert torus.value is False


def test_border_widget_adds_edits_and_deletes_items() -> None:
    env_conf = EnvConf()
    widget = build_border_widget(env_conf)

    assert pn.Column(widget).get_root() is not None

    select = _find_widget(widget, "Selected border", pn.widgets.Select)
    assert select.disabled is True
    assert select.options == {"No borders yet": ""}

    new_id = _find_widget(widget, "New border ID", pn.widgets.TextInput)
    add_button = _find_widget(widget, "Add border", pn.widgets.Button)
    delete_button = _find_widget(widget, "Delete border", pn.widgets.Button)

    new_id.value = "border_a"
    add_button.clicks += 1

    assert "border_a" in env_conf.border_list
    assert select.disabled is False
    assert select.value == "border_a"

    color = _find_widget(widget, "Color", pn.widgets.ColorPicker)
    vertices = _find_widget(widget, "Vertices", pn.widgets.LiteralInput)
    color.value = "#112233"
    vertices.value = [(0.0, 0.0), (0.05, 0.0), (0.05, 0.04)]

    assert env_conf.border_list["border_a"].color == "#112233"
    assert env_conf.border_list["border_a"].vertices == [
        (0.0, 0.0),
        (0.05, 0.0),
        (0.05, 0.04),
    ]

    delete_button.clicks += 1

    assert "border_a" not in env_conf.border_list
    assert select.disabled is True
    assert select.options == {"No borders yet": ""}


def test_food_widget_supports_grid_units_and_group_distribution() -> None:
    food_conf = FoodConf()
    widget = build_food_conf_widget(food_conf)

    assert pn.Column(widget).get_root() is not None

    unit_id = _find_widget(widget, "New source unit ID", pn.widgets.TextInput)
    add_unit = _find_widget(widget, "Add source unit", pn.widgets.Button)
    delete_unit = _find_widget(widget, "Delete source unit", pn.widgets.Button)

    unit_id.value = "unit_a"
    add_unit.clicks += 1
    assert "unit_a" in food_conf.source_units

    unit_color = _find_widget(widget, "Color", pn.widgets.ColorPicker)
    unit_color.value = "#225577"
    assert food_conf.source_units["unit_a"].color == "#225577"

    delete_unit.clicks += 1
    assert "unit_a" not in food_conf.source_units

    enable_grid = _find_widget(widget, "Enable Food grid", pn.widgets.Checkbox)
    enable_grid.value = True
    assert food_conf.food_grid is not None

    group_id = _find_widget(widget, "New source group ID", pn.widgets.TextInput)
    add_group = _find_widget(widget, "Add source group", pn.widgets.Button)
    delete_group = _find_widget(widget, "Delete source group", pn.widgets.Button)

    group_id.value = "group_a"
    add_group.clicks += 1
    assert "group_a" in food_conf.source_groups

    center_x = _find_widget(widget, "Distribution center X", pn.widgets.FloatInput)
    center_y = _find_widget(widget, "Distribution center Y", pn.widgets.FloatInput)
    center_x.value = 0.08
    center_y.value = -0.02

    assert food_conf.source_groups["group_a"].distribution.loc == pytest.approx(
        (0.08, -0.02)
    )

    delete_group.clicks += 1
    assert "group_a" not in food_conf.source_groups


def test_env_widget_builds_optional_scapes_and_updates_env_config() -> None:
    env_conf = EnvConf()
    widget = build_env_params_widget(env_conf)

    assert pn.Column(widget).get_root() is not None

    with pytest.raises(AssertionError):
        _find_widget(widget, "Odorscape type", pn.widgets.Select)

    enable_odorscape, enable_windscape, enable_thermoscape = _scape_switches(widget)

    enable_odorscape.value = True
    odorscape_type = _find_widget(widget, "Odorscape type", pn.widgets.Select)
    odorscape_type.value = "DiffusionValueLayer"
    sigma_x = _find_widget(widget, "Diffusion sigma X", pn.widgets.FloatInput)
    sigma_y = _find_widget(widget, "Diffusion sigma Y", pn.widgets.FloatInput)
    sigma_x.value = 1.2
    sigma_y.value = 0.8

    assert env_conf.odorscape is not None
    assert "DiffusionValueLayer" in type(env_conf.odorscape).__name__
    assert env_conf.odorscape.gaussian_sigma == pytest.approx((1.2, 0.8))

    enable_windscape.value = True
    wind_speed = _find_widget(widget, "Wind speed", pn.widgets.FloatInput)
    wind_speed.value = 12.5
    assert env_conf.windscape is not None
    assert env_conf.windscape.wind_speed == pytest.approx(12.5)

    enable_thermoscape.value = True
    plate_temp = _find_widget(widget, "Plate temperature", pn.widgets.FloatInput)
    thermo_positions = _find_widget(
        widget, "Thermal source positions", pn.widgets.LiteralInput
    )
    thermo_dtemps = _find_widget(
        widget, "Thermal source dTemps", pn.widgets.LiteralInput
    )
    plate_temp.value = 24.5
    thermo_positions.value = {"hotspot": [0.08, -0.04]}
    thermo_dtemps.value = {"hotspot": 6.5}

    assert env_conf.thermoscape is not None
    assert env_conf.thermoscape.plate_temp == pytest.approx(24.5)
    assert env_conf.thermoscape.thermo_sources == {"hotspot": [0.08, -0.04]}
    assert env_conf.thermoscape.thermo_source_dTemps == {"hotspot": 6.5}
