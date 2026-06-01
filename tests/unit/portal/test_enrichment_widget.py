from __future__ import annotations

import panel as pn
import pytest

from larvaworld.lib.param.enrichment import EnrichConf
from larvaworld.lib.reg.generators import ExpConf
from larvaworld.portal.config_widgets import build_enrichment_widget
from larvaworld.portal.config_widgets.conftype_widget import ConftypeWidgetController


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


def test_enrichment_widget_uses_core_param_fields() -> None:
    enrichment = EnrichConf()
    widget = build_enrichment_widget(enrichment)

    assert pn.Column(widget).get_root() is not None

    proc_keys = _find_widget(widget, "Proc keys", pn.widgets.MultiChoice)
    anot_keys = _find_widget(widget, "Anot keys", pn.widgets.MultiChoice)
    tor_durs = _find_widget(widget, "Tor durs", pn.widgets.LiteralInput)
    recompute = _find_widget(widget, "Recompute", pn.widgets.Checkbox)
    mode = _find_widget(widget, "Mode", pn.widgets.Select)

    proc_keys.value = ["spatial", "source"]
    anot_keys.value = ["patch_residency"]
    tor_durs.value = [8, 16]
    recompute.value = True
    mode.value = "full"

    assert enrichment.proc_keys == ["spatial", "source"]
    assert enrichment.anot_keys == ["patch_residency"]
    assert enrichment.tor_durs == [8, 16]
    assert enrichment.recompute is True
    assert enrichment.mode == "full"


def test_enrichment_widget_edits_preprocessing_config() -> None:
    enrichment = EnrichConf()
    widget = build_enrichment_widget(enrichment)

    rescale_by = _find_widget(widget, "Rescale by", pn.widgets.LiteralInput)
    filter_f = _find_widget(widget, "Filter f", pn.widgets.LiteralInput)
    transposition = _find_widget(widget, "Transposition", pn.widgets.Select)
    interpolate_nans = _find_widget(widget, "Interpolate nans", pn.widgets.Checkbox)

    rescale_by.value = 0.002
    filter_f.value = 1.5
    transposition.value = "arena"
    interpolate_nans.value = True

    assert enrichment.pre_kws.rescale_by == pytest.approx(0.002)
    assert enrichment.pre_kws.filter_f == pytest.approx(1.5)
    assert enrichment.pre_kws.transposition == "arena"
    assert enrichment.pre_kws.interpolate_nans is True


def test_exp_conftype_widget_uses_typed_enrichment_helper() -> None:
    controller = ConftypeWidgetController(ExpConf, conftype="Exp", allow_reset=False)

    widgets = {
        getattr(widget, "name", None)
        for widget in controller.view.select(pn.widgets.Widget)
        if getattr(widget, "name", None)
    }

    assert "Proc keys" in widgets
    assert "Anot keys" in widgets
    assert "Rescale by" in widgets
