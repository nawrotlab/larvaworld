from __future__ import annotations

import panel as pn

from larvaworld.lib import reg
from larvaworld.lib.reg.generators import ExpConf
from larvaworld.portal.config_widgets import (
    build_larva_group_widget,
    build_larva_groups_widget,
)


def _find_widget(
    viewable: pn.viewable.Viewable,
    name: str,
    widget_type: type[pn.widgets.Widget]
    | tuple[type[pn.widgets.Widget], ...] = pn.widgets.Widget,
):
    for widget in viewable.select(widget_type):
        if getattr(widget, "name", None) == name:
            return widget
    raise AssertionError(f"Could not find widget {name!r}.")


def test_larva_group_widget_builds_for_generated_group() -> None:
    group = reg.gen.LarvaGroup(model="explorer", group_id="group_a")
    widget = build_larva_group_widget(group)

    assert pn.Column(widget).get_root() is not None


def test_larva_group_widget_updates_core_fields_and_nested_classes() -> None:
    group = reg.gen.LarvaGroup(model="explorer", group_id="group_a")
    widget = build_larva_group_widget(group)

    color = _find_widget(widget, "Color", pn.widgets.ColorPicker)
    population = _find_widget(widget, "N", pn.widgets.LiteralInput)
    odor_id = _find_widget(widget, "Id", pn.widgets.TextInput)
    age = _find_widget(widget, "Age", pn.widgets.LiteralInput)

    color.value = "#112233"
    population.value = 7
    odor_id.value = "apple"
    age.value = 12.0

    assert group.color == "#112233"
    assert group.distribution.N == 7
    assert group.odor.id == "apple"
    assert group.life_history.age == 12.0

    nested = group.nestedConf
    assert nested["color"] == "#112233"
    assert nested["distribution"]["N"] == 7
    assert nested["odor"]["id"] == "apple"
    assert nested["life_history"]["age"] == 12.0


def test_larva_groups_widget_coerces_dict_payloads_and_roundtrips_nested_conf() -> None:
    exp = ExpConf()
    exp.larva_groups["group_a"] = {
        "group_id": "group_a",
        "model": "explorer",
        "sample": None,
        "color": "#334455",
        "distribution": {"N": 3, "loc": (0.0, 0.0), "scale": (0.0, 0.0)},
        "odor": {"id": "apple"},
        "life_history": {"age": 8.0},
    }

    widget = build_larva_groups_widget(exp)
    assert pn.Column(widget).get_root() is not None
    assert isinstance(exp.larva_groups["group_a"], type(reg.gen.LarvaGroup()))

    selected = _find_widget(widget, "Selected larva group", pn.widgets.Select)
    assert selected.value == "group_a"
    population = _find_widget(widget, "N", pn.widgets.LiteralInput)
    sample = _find_widget(widget, "Ref configuration ID", pn.widgets.Select)

    population.value = 5
    sample.value = None

    group = exp.larva_groups["group_a"]
    nested = group.nestedConf
    assert group.distribution.N == 5
    assert nested["distribution"]["N"] == 5
    assert nested["sample"] is None
