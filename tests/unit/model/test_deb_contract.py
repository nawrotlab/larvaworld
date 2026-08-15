"""
Contract tests for the DEB model's integration surface.

These pin the boundary between ``lib/model/deb`` and the rest of larvaworld -- the
stored-config keys, the recorded dict, the reporter codenames, the attributes the
gut reads back, and the sim-loop stepping protocol. They are deliberately about the
*interface*, not the physics: the equations are covered by
``test_deb_equations.py``.

They exist so that the DEB core can be replaced without silently breaking the
registry, the portal model inspector, the data collectors or the plots.
"""

from __future__ import annotations

import numpy as np
import pytest

from larvaworld.lib import util
from larvaworld.lib.model.deb.deb import DEB, DEB_basic, DEB_model
from larvaworld.lib.model.deb.gut import Gut
from larvaworld.lib.param import NestedConf, Substrate, class_defaults, class_objs

#: The complete set of DEB parameters that reach a stored larva-model config.
#: `module_modes.energetics_kws` builds it with exactly this exclusion, so every
#: parameter declared on `DEB_model` is excluded by construction and no
#: physiological parameter is persisted. Changing this set changes every stored
#: model config and the portal model inspector.
CONFIG_KEYS = {
    "aging",
    "assimilation_mode",
    "dt",
    "hunger_as_EEB",
    "hunger_gain",
    "species",
    "starvation_strategy",
}

EXCLUDED = [DEB_model, "substrate", "id"]

#: Per-step series recorded by `DEB.update_dict`.
DEB_SERIES = {
    "age",
    "mass",
    "length",
    "reserve",
    "reserve_density",
    "hunger",
    "pupation_buffer",
    "f",
    "deb_p_A",
    "sim_p_A",
    "EEB",
}

#: Per-step series recorded by `Gut.update_dict`.
GUT_SERIES = {
    "R_absorbed",
    "mol_ingested",
    "gut_p_A",
    "M_X",
    "M_P",
    "M_Pu",
    "M_g",
    "M_c",
    "R_M_c",
    "R_M_g",
    "R_M_X",
    "R_M_P",
    "R_M_X_M_P",
}

#: Scalars added by `DEB.finalize_dict`.
FINALIZE_SCALARS = {
    "species",
    "birth",
    "pupation",
    "death",
    "id",
    "epochs",
    "epoch_qs",
    "fr",
    "feed_freq_estimate",
    "f_mean",
    "f_deviation_mean",
    "Nfeeds",
    "mean_feed_freq",
    "gut_residence_time",
}

#: Additional scalars present only when an intermitter is attached.
INTERMITTER_SCALARS = {
    "feed_freq_simulated",
    "crawl ratio",
    "pause ratio",
    "feed ratio",
}

#: Codenames `reg.parDB.build_deb_pars` resolves as `agent.deb.<name>` every
#: recorded step, via `util.rgetattr`.
REPORTER_ATTRS = [
    "ingested_gut_volume_ratio",
    "volume_ingested",
    "ingested_body_volume_ratio",
    "ingested_body_area_ratio",
    "ingested_body_mass_ratio",
]

#: Attributes `gut.py` reads back off its owning DEB instance.
GUT_READBACK = [
    "L",
    "V",
    "dt",
    "substrate",
    "mu_E",
    "w_P",
    "w_E",
    "d_V",
    "d_X",
    "base_f",
    "J_X_Am",
    "Lb",
]


@pytest.fixture(scope="module")
def deb() -> DEB:
    return DEB(id="contract", species="rover", substrate=Substrate(type="standard"))


@pytest.fixture(scope="module")
def grown() -> dict:
    return DEB.default_growth(id="contract", species="rover")


# ---------------------------------------------------------------------------
# Class identity
# ---------------------------------------------------------------------------


def test_class_hierarchy_is_preserved() -> None:
    """`excluded=[DEB_model, ...]` only works while DEB_model is a base of DEB."""
    assert issubclass(DEB, DEB_basic)
    assert issubclass(DEB_basic, DEB_model)
    assert issubclass(DEB_model, NestedConf)


def test_DEB_model_is_constructible_with_no_arguments() -> None:
    """`class_defaults` instantiates a bare DEB_model to compute the exclusion."""
    assert DEB_model() is not None


def test_deb_owns_a_gut(deb: DEB) -> None:
    assert isinstance(deb.gut, Gut)
    assert deb.gut.deb is deb


# ---------------------------------------------------------------------------
# Stored-config surface
# ---------------------------------------------------------------------------


def test_config_default_keys_are_exactly_the_contract() -> None:
    assert set(class_defaults(DEB, excluded=EXCLUDED)) == CONFIG_KEYS


def test_config_param_objects_match_the_contract() -> None:
    """Protects the portal model inspector, which introspects the same set."""
    assert set(dict(class_objs(DEB, excluded=EXCLUDED))) == CONFIG_KEYS


def test_no_physiological_parameter_reaches_a_stored_config() -> None:
    keys = set(class_defaults(DEB, excluded=EXCLUDED))
    for name in ("p_Am", "E_G", "v", "kap", "E_M", "Lm", "k_J", "E_Hb", "T", "T_A"):
        assert name not in keys, f"{name} must not be persisted in model configs"


@pytest.mark.parametrize("species", ["default", "rover", "sitter"])
def test_species_values_used_by_stored_configs_still_load(species: str) -> None:
    """The 10 stored rover*/sitter* model configs persist these strings."""
    d = DEB(id="t", species=species)
    assert d.species == species


def test_assimilation_mode_gut_is_selectable(deb: DEB) -> None:
    """Pinned by tests/unit/portal/test_model_inspector_*.py."""
    assert "gut" in DEB.param.assimilation_mode.objects


# ---------------------------------------------------------------------------
# Recorded dict
# ---------------------------------------------------------------------------


def test_finalize_dict_key_set(grown: dict) -> None:
    assert set(grown) == DEB_SERIES | GUT_SERIES | FINALIZE_SCALARS


def test_finalize_dict_series_are_aligned(grown: dict) -> None:
    n = len(grown["age"])
    assert n > 0
    for key in DEB_SERIES | GUT_SERIES:
        assert (
            len(grown[key]) == n
        ), f"{key} has {len(grown[key])} entries, expected {n}"


def test_finalize_dict_scalars_are_usable_by_plot_debs(grown: dict) -> None:
    """`plot/deb.py:plot_debs` reads exactly these."""
    assert isinstance(grown["id"], str)
    assert isinstance(grown["species"], str)
    assert np.isfinite(grown["birth"])
    assert np.isfinite(grown["pupation"])
    assert grown["pupation"] > grown["birth"]  # the GUI uses this as a slider range
    assert grown["fr"] > 0
    assert isinstance(grown["epochs"], list)
    assert len(grown["epochs"]) == len(grown["epoch_qs"])


def test_finalize_dict_gains_ratio_keys_with_an_intermitter() -> None:
    from larvaworld.lib.model.modules.intermitter import OfflineIntermitter

    d = DEB(id="t", species="rover", intermitter=OfflineIntermitter())
    d.grow_larva(epochs=[])
    out = d.finalize_dict()
    assert INTERMITTER_SCALARS <= set(out)


# ---------------------------------------------------------------------------
# Data collection and gut coupling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attr", REPORTER_ATTRS)
def test_reporter_codenames_resolve(attr: str, deb: DEB) -> None:
    """`reg.parDB` collects these as `deb.<attr>` through `util.rgetattr`."""
    value = util.rgetattr(deb, attr)
    assert isinstance(value, (int, float, np.floating))
    assert np.isfinite(value)


@pytest.mark.parametrize("attr", GUT_READBACK)
def test_gut_readback_attributes_exist(attr: str, deb: DEB) -> None:
    value = getattr(deb, attr)
    assert value is not None
    if isinstance(value, (int, float, np.floating)):
        assert np.isfinite(value)


def test_agent_facing_state_is_finite(deb: DEB) -> None:
    """`LarvaMotile` reads exactly these three off the DEB every step."""
    for attr in ("Lw", "Ww", "V"):
        value = getattr(deb, attr)
        assert np.isfinite(value) and value > 0, attr


# ---------------------------------------------------------------------------
# Sim-loop stepping protocol
# ---------------------------------------------------------------------------


def test_run_check_buffers_until_the_deb_timestep_elapses() -> None:
    """
    The simulation ticks at 0.1 s while the DEB steps at 60 s, so `run_check`
    accumulates ingested volume and elapsed time and steps once per DEB timestep.
    """
    d = DEB(id="t", species="rover", substrate=Substrate(type="standard"))
    d.grow_larva(epochs=[])
    dt_sim = 0.1
    n = int(d.dt_in_sec / dt_sim)

    age0 = d.age
    for _ in range(n - 1):
        d.run_check(dt=dt_sim, X_V=1e-9)
    assert d.age == age0, "the DEB must not advance before its timestep elapses"
    assert d.X_V_buffer > 0

    d.run_check(dt=dt_sim, X_V=1e-9)
    assert d.age == pytest.approx(age0 + d.dt)
    assert d.X_V_buffer == 0
    assert d.time_buffer == 0


def test_grow_larva_advances_age_and_reaches_the_larval_stage() -> None:
    d = DEB(id="t", species="rover")
    d.grow_larva(epochs=[])
    assert d.age > 0
    assert d.stage in ("larva", "pupa", "imago", "dead")


def test_hunger_drives_the_intermitter() -> None:
    """DEB writes its hunger drive into the intermitter's EEB every update."""
    from larvaworld.lib.model.modules.intermitter import OfflineIntermitter

    im = OfflineIntermitter()
    d = DEB(id="t", species="rover", intermitter=im, hunger_as_EEB=True)
    d.update_hunger()
    assert im.EEB == pytest.approx(d.hunger)
    assert 0.0 <= d.hunger <= 1.0
