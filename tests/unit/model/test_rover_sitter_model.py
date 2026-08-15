"""
Tests for the rover/sitter behavioural phenotypes derived from the generic
(species-average) Drosophila DEB model.
"""

from __future__ import annotations

import json

import pytest

from larvaworld.lib.model.deb import deb_equations as de
from larvaworld.lib.model.deb import rover_sitter_model as rs

#: Yield coefficients carried by the pre-existing larvaworld species files
#: (``models/deb_{default,rover,sitter}.csv``). The phenotypes must reproduce them.
LEGACY_Y_E_X = {"drosophila": 0.763636, "rover": 0.85, "sitter": 0.5}

T_25C = 298.15


@pytest.fixture(scope="module")
def generic() -> de.DEBPars:
    return rs.load_drosophila()


@pytest.fixture(scope="module")
def models(generic: de.DEBPars) -> dict:
    return rs.phenotypes(base=generic)


# ---------------------------------------------------------------------------
# Loading the generic species model
# ---------------------------------------------------------------------------


def test_generic_model_loads_from_amp_export(generic: de.DEBPars) -> None:
    assert generic.metadata["species"] == "Drosophila_melanogaster"
    assert generic.metadata["typified_model"] == "abp"
    assert generic.lb_info == 1
    assert 0.0 < generic.l_b < 1.0
    assert generic.E_0 > 0.0


def test_generic_model_carries_the_amp_primary_parameters(generic: de.DEBPars) -> None:
    """Primary values must come through untouched -- the export is the reference."""
    with open(rs.DROSOPHILA_AMP_JSON, encoding="utf-8") as fh:
        raw = {e["symbol"]: e["value"] for e in json.load(fh)["parameters"]}
    for symbol in (
        "z",
        "v",
        "kap",
        "p_M",
        "k_J",
        "E_G",
        "E_Hb",
        "E_Hp",
        "E_He",
        "kap_X",
        "kap_P",
        "kap_R",
        "T_A",
        "T_ref",
        "del_M",
        "F_m",
    ):
        assert getattr(generic, symbol) == raw[symbol], symbol


def test_degenerate_kap_V_is_overridden_not_clamped(generic: de.DEBPars) -> None:
    """
    The AmP export reports kap_V = -1.526e-54 (the fit collapsed it). Loading must
    substitute the documented physical value, and loading *without* an override
    must fail loudly rather than silently accept a degenerate parameter.
    """
    with open(rs.DROSOPHILA_AMP_JSON, encoding="utf-8") as fh:
        raw = {e["symbol"]: e["value"] for e in json.load(fh)["parameters"]}
    assert raw["kap_V"] < 0.0

    assert generic.kap_V == rs.KAP_V_OVERRIDE

    with pytest.raises(ValueError, match="kap_V"):
        de.DEBPars.from_amp_json(rs.DROSOPHILA_AMP_JSON)


def test_caller_can_override_kap_V() -> None:
    p = rs.load_drosophila(kap_V=0.5)
    assert p.kap_V == 0.5


# ---------------------------------------------------------------------------
# Phenotype derivation
# ---------------------------------------------------------------------------


def test_phenotypes_returns_generic_and_both_phenotypes(models: dict) -> None:
    assert set(models) == {"drosophila", "rover", "sitter"}


def test_generic_kap_X_is_the_amp_value(generic: de.DEBPars) -> None:
    assert generic.kap_X == pytest.approx(0.8)


@pytest.mark.parametrize("name", ["drosophila", "rover", "sitter"])
def test_phenotypes_reproduce_legacy_yield_coefficients(
    name: str, models: dict
) -> None:
    """
    y_E_X = kap_X mu_X / mu_E, so expressing the phenotype contrast through the
    primary symbol kap_X must land exactly on the legacy y_E_X values.
    """
    assert models[name].y_E_X == pytest.approx(LEGACY_Y_E_X[name], abs=1e-6)


def test_only_the_differentiating_parameter_differs(models: dict) -> None:
    rover, sitter = models["rover"], models["sitter"]
    differing = {
        f
        for f in de.DEBPars.__dataclass_fields__
        if f != "metadata" and getattr(rover, f) != getattr(sitter, f)
    }
    assert differing == {rs.DEFAULT_PHENOTYPE_PARAM}


def test_rover_extracts_more_from_the_same_food(models: dict) -> None:
    """
    kap_X raises the yield of reserve on food, which lowers the half-saturation
    coefficient K. At a given food density X the scaled functional response
    f = X/(X+K) is therefore higher for the rover.
    """
    rover, sitter = models["rover"], models["sitter"]
    assert rover.y_E_X > sitter.y_E_X
    assert rover.K < sitter.K

    X = 0.5 * (rover.K + sitter.K)
    f_rover = X / (X + rover.K)
    f_sitter = X / (X + sitter.K)
    assert f_rover > f_sitter


def test_assimilation_at_equal_f_is_phenotype_independent(models: dict) -> None:
    """
    Table S1 assimilation p_A = {p_Am} s_M f V^(2/3) does not involve kap_X: the
    phenotypes differ in how much food a given f costs, not in what f delivers.
    """
    kw = dict(E=5.0, V=1e-3, E_H=10.0, f=0.8, s_M=3.0)
    p_rover = de.powers(de.Stage.LARVA, pars=models["rover"], **kw)
    p_sitter = de.powers(de.Stage.LARVA, pars=models["sitter"], **kw)
    assert p_rover.p_A == pytest.approx(p_sitter.p_A)


def test_convenience_constructors_match_make_phenotype(generic: de.DEBPars) -> None:
    assert (
        rs.rover(base=generic).kap_X == rs.make_phenotype("rover", base=generic).kap_X
    )
    assert (
        rs.sitter(base=generic).kap_X == rs.make_phenotype("sitter", base=generic).kap_X
    )


# ---------------------------------------------------------------------------
# Choosing a different differentiating parameter
# ---------------------------------------------------------------------------


def test_differentiating_parameter_is_selectable(generic: de.DEBPars) -> None:
    values = {"rover": 0.85, "sitter": 0.75}
    r = rs.make_phenotype("rover", base=generic, param="kap", values=values)
    s = rs.make_phenotype("sitter", base=generic, param="kap", values=values)
    assert (r.kap, s.kap) == (0.85, 0.75)
    assert r.kap_X == s.kap_X == generic.kap_X  # untouched
    # compound parameters must be rederived from the new primary value
    assert r.p_Am == pytest.approx(r.z * r.p_M / r.kap)


def test_extra_overrides_are_applied(generic: de.DEBPars) -> None:
    r = rs.make_phenotype("rover", base=generic, p_M=1000.0)
    assert r.p_M == 1000.0
    assert r.k_M == pytest.approx(1000.0 / r.E_G)


def test_unknown_phenotype_raises(generic: de.DEBPars) -> None:
    with pytest.raises(ValueError, match="unknown phenotype"):
        rs.make_phenotype("wanderer", base=generic)


def test_unknown_parameter_raises(generic: de.DEBPars) -> None:
    with pytest.raises(ValueError, match="no default phenotype values"):
        rs.make_phenotype("rover", base=generic, param="v")

    with pytest.raises(ValueError, match="not a DEBPars parameter"):
        rs.make_phenotype(
            "rover", base=generic, param="nonsense", values={"rover": 1.0}
        )


def test_incomplete_values_raise(generic: de.DEBPars) -> None:
    with pytest.raises(ValueError, match="missing an entry"):
        rs.make_phenotype("sitter", base=generic, param="kap", values={"rover": 0.85})


# ---------------------------------------------------------------------------
# Both phenotypes remain viable models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["drosophila", "rover", "sitter"])
def test_phenotype_completes_a_life_cycle(name: str, models: dict) -> None:
    pars = models[name]
    st, tr = de.run(
        pars,
        engine="closed",
        dt=1.0 / 24.0,
        f=1.0,
        T=T_25C,
        until_stage=de.Stage.IMAGO,
    )
    assert st.stage == de.Stage.IMAGO
    assert list(tr.events) == list(de.STAGES)
    assert st.alive
