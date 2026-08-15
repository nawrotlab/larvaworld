"""
Tests for the ground-truth DEB equations module.

The module under test is a transcription of Tables S1 (fluxes) and S2 (state
variable dynamics) of the Drosophila DEB model specification. These tests check
the transcription itself, the DEB invariants that follow from it, the embryo
solver, and agreement with the published AmP predictions for the species.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from larvaworld.lib.model.deb import deb_equations as de

#: The AmP predictions are reported at 25 degC.
T_25C = 298.15


@pytest.fixture(scope="module")
def pars() -> de.DEBPars:
    return de.DEBPars()


@pytest.fixture(scope="module")
def amp() -> dict:
    return de.amp_predictions(de_json())


def de_json() -> str:
    from larvaworld.lib.model.deb.rover_sitter_model import DROSOPHILA_AMP_JSON

    return DROSOPHILA_AMP_JSON


# ---------------------------------------------------------------------------
# Compound parameters (parscomp_st.m / addchem.m)
# ---------------------------------------------------------------------------


def test_compound_parameter_identities(pars: de.DEBPars) -> None:
    assert pars.p_Am == pytest.approx(pars.z * pars.p_M / pars.kap)
    assert pars.E_m == pytest.approx(pars.p_Am / pars.v)
    assert pars.g == pytest.approx(pars.E_G / (pars.kap * pars.E_m))
    assert pars.k_M == pytest.approx(pars.p_M / pars.E_G)
    assert pars.k == pytest.approx(pars.k_J / pars.k_M)
    assert pars.L_m == pytest.approx(pars.v / (pars.k_M * pars.g))
    # L_m == z when L_m_ref == 1 cm
    assert pars.L_m == pytest.approx(pars.z)
    assert pars.M_V == pytest.approx(pars.d_V / pars.w_V)
    assert pars.y_V_E * pars.y_E_V == pytest.approx(1.0)
    assert pars.kap_G == pytest.approx(pars.mu_V * pars.M_V / pars.E_G)
    assert pars.E_V == pytest.approx(pars.d_V * pars.mu_V / pars.w_V)
    assert pars.y_E_X == pytest.approx(pars.kap_X * pars.mu_X / pars.mu_E)
    assert pars.p_Xm == pytest.approx(pars.p_Am / pars.kap_X)
    assert pars.J_X_Am == pytest.approx(pars.y_X_E * pars.J_E_Am)
    assert pars.K == pytest.approx(pars.J_X_Am / pars.F_m)


def test_molecular_weights_from_chemical_indices(pars: de.DEBPars) -> None:
    """w_O = n_O^T . [12, 1, 16, 14]; the defaults give 23.9 g/mol throughout."""
    expected = np.asarray([12.0, 1.0, 16.0, 14.0]) @ np.asarray(pars.n_O, dtype=float)
    assert [pars.w_X, pars.w_V, pars.w_E, pars.w_P] == pytest.approx(expected.tolist())
    assert pars.w_V == pytest.approx(23.9)


def test_compound_pars_match_amp_pseudodata_predictions(pars: de.DEBPars) -> None:
    """AmP reports kap_G and k as predicted pseudo-data; ours must reproduce them."""
    assert pars.kap_G == pytest.approx(0.8021, rel=1e-3)
    assert pars.k == pytest.approx(0.001022, rel=1e-3)


def test_with_rederives_compound_parameters(pars: de.DEBPars) -> None:
    doubled = pars.with_(z=2.0 * pars.z)
    assert doubled.p_Am == pytest.approx(2.0 * pars.p_Am)
    assert doubled.E_m == pytest.approx(2.0 * pars.E_m)
    assert pars.p_Am == pytest.approx(pars.z * pars.p_M / pars.kap)  # original intact


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_degenerate_kap_V_is_rejected() -> None:
    """The raw AmP abp export has kap_V = -1.526e-54; it must not be accepted."""
    with pytest.raises(ValueError, match="kap_V"):
        de.DEBPars(kap_V=-1.526e-54)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"kap": 1.0}, "kap"),
        ({"kap": 0.0}, "kap"),
        ({"kap_X": 1.5}, "kap_X"),
        ({"p_M": -1.0}, "p_M"),
        ({"E_Hb": 1e6}, "E_Hb"),  # would exceed E_Hp
    ],
)
def test_invalid_parameters_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        de.DEBPars(**kwargs)


def test_E_He_below_E_Hp_is_allowed(pars: de.DEBPars) -> None:
    """Maturity resets at pupation, so E_He < E_Hp is legitimate (and is the AmP case)."""
    assert pars.E_He < pars.E_Hp
    assert pars.E_Hb < pars.E_Hp


# ---------------------------------------------------------------------------
# Embryo solver
# ---------------------------------------------------------------------------


def test_get_lb_exact_for_k_equals_one() -> None:
    lb, info = de.get_lb(g=0.5, k=1.0, v_Hb=1e-3)
    assert info == 1
    assert lb == pytest.approx(1e-3 ** (1.0 / 3.0))


def test_get_lb_in_physical_domain(pars: de.DEBPars) -> None:
    """v_Hb > 1 for this species, which defeats the naive Newton start."""
    assert pars.v_Hb > 1.0
    assert pars.lb_info == 1
    assert 0.0 < pars.l_b < 1.0


@pytest.mark.parametrize("v_Hb", [1e-6, 1e-4, 1e-2, 0.5, 1.31, 2.0])
def test_get_lb_domain_across_maturity_levels(v_Hb: float) -> None:
    lb, info = de.get_lb(g=0.3609, k=0.001022, v_Hb=v_Hb)
    assert info == 1, f"no convergence for v_Hb={v_Hb}"
    assert 0.0 < lb < 1.0


def test_beta0_is_antisymmetric_and_zero_on_diagonal() -> None:
    assert de.beta0(0.3, 0.3) == pytest.approx(0.0, abs=1e-12)
    assert de.beta0(0.1, 0.4) == pytest.approx(-de.beta0(0.4, 0.1))


def test_initial_reserve_is_positive_and_scales(pars: de.DEBPars) -> None:
    assert pars.E_0 > 0.0
    assert pars.u_E0 > 0.0
    assert pars.E_0 == pytest.approx(pars.p_Am * pars.u_E0 / pars.U_coeff)


# ---------------------------------------------------------------------------
# Table S1 -- fluxes
# ---------------------------------------------------------------------------


def test_acceleration_clamping() -> None:
    # before birth both lengths are inf -> s_M == 1
    assert de.acceleration(0.01, math.inf, math.inf) == 1.0
    # between birth and pupation -> L / L_b
    assert de.acceleration(0.2, 0.05, math.inf) == pytest.approx(4.0)
    # never below 1
    assert de.acceleration(0.01, 0.05, math.inf) == 1.0
    # frozen at L_p / L_b once past pupation length
    assert de.acceleration(0.4, 0.05, 0.3) == pytest.approx(6.0)


def test_non_feeding_stages_have_zero_assimilation(pars: de.DEBPars) -> None:
    for stage in (de.Stage.EGG, de.Stage.PUPA):
        p = de.powers(stage, E=1.0, V=1e-4, E_H=0.05, pars=pars, f=1.0, s_M=2.0)
        assert p.p_A == 0.0


def test_feeding_stages_assimilation_matches_table_S1(pars: de.DEBPars) -> None:
    V, f, s_M, TC = 1e-3, 0.7, 3.0, 1.4
    for stage in (de.Stage.LARVA, de.Stage.IMAGO):
        p = de.powers(stage, E=5.0, V=V, E_H=1.0, pars=pars, f=f, s_M=s_M, TC=TC)
        assert p.p_A == pytest.approx(TC * pars.p_Am * s_M * f * V ** (2.0 / 3.0))


def test_mobilization_matches_table_S1(pars: de.DEBPars) -> None:
    E, V, s_M, TC = 5.0, 1e-3, 3.0, 1.0
    V23 = V ** (2.0 / 3.0)

    # egg: no acceleration in the Table S1 expression
    p = de.powers(de.Stage.EGG, E=E, V=V, E_H=0.05, pars=pars, s_M=s_M, TC=TC)
    assert p.p_C == pytest.approx(
        E * (pars.E_G * pars.v * V23 + pars.p_M * V) / (pars.kap * E + pars.E_G * V)
    )

    # larva and pupa: identical mobilization, with acceleration
    expected = (
        E
        * (pars.E_G * pars.v * s_M * V23 + pars.p_M * V)
        / (pars.kap * E + pars.E_G * V)
    )
    for stage in (de.Stage.LARVA, de.Stage.PUPA):
        p = de.powers(stage, E=E, V=V, E_H=0.05, pars=pars, s_M=s_M, TC=TC)
        assert p.p_C == pytest.approx(expected)

    # imago: pure reserve turnover
    p = de.powers(de.Stage.IMAGO, E=E, V=V, E_H=0.05, pars=pars, s_M=s_M, TC=TC)
    assert p.p_C == pytest.approx(E * pars.v * s_M / V ** (1.0 / 3.0))


def test_maintenance_growth_and_maturation_match_table_S1(pars: de.DEBPars) -> None:
    E, V, E_H, s_M = 5.0, 1e-3, 0.4, 3.0
    for stage in de.STAGES:
        p = de.powers(stage, E=E, V=V, E_H=E_H, pars=pars, f=1.0, s_M=s_M)
        assert p.p_S == pytest.approx(pars.p_M * V)
        assert p.p_J == pytest.approx(pars.k_J * E_H)
        if stage == de.Stage.IMAGO:
            assert p.p_G == 0.0
            assert p.p_R == pytest.approx(p.p_C - p.p_S - p.p_J)
        else:
            assert p.p_G == pytest.approx(pars.kap * p.p_C - p.p_S)
            assert p.p_R == pytest.approx((1.0 - pars.kap) * p.p_C - p.p_J)


def test_surface_specific_maintenance_is_not_applied(pars: de.DEBPars) -> None:
    """{p_T} appears in no row of Table S1, so varying it must change nothing."""
    kw = dict(E=5.0, V=1e-3, E_H=0.4, f=1.0, s_M=3.0)
    a = de.powers(de.Stage.LARVA, pars=pars, **kw)
    b = de.powers(de.Stage.LARVA, pars=pars.with_(p_T=100.0), **kw)
    assert a == b


def test_external_p_A_overrides_table_S1(pars: de.DEBPars) -> None:
    """The hook a gut model / behavioural simulation uses to drive assimilation."""
    p = de.powers(
        de.Stage.LARVA, E=5.0, V=1e-3, E_H=0.4, pars=pars, f=1.0, s_M=3.0, p_A=42.0
    )
    assert p.p_A == 42.0


def test_unknown_stage_raises(pars: de.DEBPars) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        de.powers("larvae", E=1.0, V=1e-3, E_H=0.1, pars=pars)


@pytest.mark.parametrize("V", [1e-6, 1e-4, 1e-2, 1.0])
@pytest.mark.parametrize("stage", [de.Stage.EGG, de.Stage.LARVA, de.Stage.PUPA])
def test_p_C_density_form_agrees(stage: str, V: float, pars: de.DEBPars) -> None:
    """
    Table S1 writes mobilization in the flux form

        p_C = E ([E_G] v s_M V^(2/3) + [p_M] V) / (kap E + [E_G] V)

    which is algebraically identical to the density form

        p_C = E ([E_G] v s_M / L + [p_M]) / (kap E / V + [E_G])

    obtained by dividing numerator and denominator by V. The two agree only if the
    numerator's maintenance term is the volume-specific [p_M] and not the flux
    [p_M] V -- substituting one for the other is a dimensional error that this test
    detects, and it is exactly the discrepancy the density form invites.
    """
    E, s_M = 5.0, 3.0
    s = 1.0 if stage == de.Stage.EGG else s_M  # Table S1 omits s_M in the egg row
    L = V ** (1.0 / 3.0)

    got = de.powers(stage, E=E, V=V, E_H=0.4, pars=pars, s_M=s_M).p_C
    density_form = (
        E * (pars.E_G * pars.v * s / L + pars.p_M) / (pars.kap * E / V + pars.E_G)
    )
    assert got == pytest.approx(density_form, rel=1e-12)


# ---------------------------------------------------------------------------
# Table S2 -- dynamics
# ---------------------------------------------------------------------------


def test_maturation_routing_matches_table_S2(pars: de.DEBPars) -> None:
    """dE_H/dt = p_R except in the imago; dE_R/dt = p_R only in the imago."""
    for stage in de.STAGES:
        p = de.powers(stage, E=5.0, V=1e-3, E_H=0.4, pars=pars, f=1.0, s_M=3.0)
        dE, dV, dE_H, dE_R = de.derivatives(stage, p, pars)
        assert dE == pytest.approx(p.p_A - p.p_C)
        assert dV == pytest.approx(p.p_G / pars.E_G)
        if stage == de.Stage.IMAGO:
            assert dE_H == 0.0 and dE_R == pytest.approx(p.p_R)
        else:
            assert dE_R == 0.0 and dE_H == pytest.approx(p.p_R)


@pytest.mark.parametrize("stage", de.STAGES)
def test_energy_balance_closes(stage: str, pars: de.DEBPars) -> None:
    """p_A == dE + p_G + p_S + p_J + p_R: no energy is created or lost."""
    p = de.powers(stage, E=5.0, V=1e-3, E_H=0.4, pars=pars, f=1.0, s_M=3.0)
    dE, _dV, _dE_H, _dE_R = de.derivatives(stage, p, pars)
    residual = p.p_A - (dE + p.p_G + p.p_S + p.p_J + p.p_R)
    assert residual == pytest.approx(0.0, abs=1e-9 * max(abs(p.p_C), 1.0))


# ---------------------------------------------------------------------------
# Stage machine
# ---------------------------------------------------------------------------


def test_pupation_resets_maturity_and_resorbs_structure(pars: de.DEBPars) -> None:
    st = de.DEBState(E=100.0, V=2e-2, E_H=pars.E_Hp, stage=de.Stage.LARVA, L_b=0.03)
    E_before, V_before = st.E, st.V

    assert de.transition(st, pars) is True

    assert st.stage == de.Stage.PUPA
    assert st.E_H == 0.0
    assert st.L_p == pytest.approx(V_before ** (1.0 / 3.0))
    assert st.V == pars.V_seed
    # larval structure returns to reserve with efficiency kap_V
    assert st.E == pytest.approx(E_before + pars.kap_V * pars.E_G * V_before)


def test_birth_records_structural_length(pars: de.DEBPars) -> None:
    st = de.DEBState(E=1.0, V=1e-4, E_H=pars.E_Hb, stage=de.Stage.EGG)
    assert de.transition(st, pars) is True
    assert st.stage == de.Stage.LARVA
    assert st.L_b == pytest.approx((1e-4) ** (1.0 / 3.0))


def test_no_transition_below_threshold(pars: de.DEBPars) -> None:
    st = de.DEBState(E=1.0, V=1e-4, E_H=0.5 * pars.E_Hb, stage=de.Stage.EGG)
    assert de.transition(st, pars) is False
    assert st.stage == de.Stage.EGG


def test_initial_state_is_an_egg(pars: de.DEBPars) -> None:
    st = de.initial_state(pars)
    assert st.stage == de.Stage.EGG
    assert st.E == pytest.approx(pars.E_0)
    assert st.E_H == 0.0 and st.E_R == 0.0 and st.age == 0.0
    assert st.alive
    assert st.s_M() == 1.0


# ---------------------------------------------------------------------------
# Integration engines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["stepped", "closed"])
def test_full_life_cycle_visits_all_stages_in_order(
    engine: str, pars: de.DEBPars
) -> None:
    dt = 1.0 / (24.0 * 60.0) if engine == "stepped" else 1.0 / 24.0
    st, tr = de.run(
        pars,
        engine=engine,
        dt=dt,
        f=1.0,
        T=T_25C,
        until_stage=de.Stage.IMAGO,
        max_steps=2_000_000,
        record_every=60,
    )
    assert st.stage == de.Stage.IMAGO
    assert list(tr.events) == list(de.STAGES)
    ages = list(tr.events.values())
    assert ages == sorted(ages), "stage entry ages must increase"
    assert np.all(np.diff(tr.t) >= 0.0), "age must be non-decreasing"
    assert np.isfinite(tr.E).all() and (tr.V > 0).all()


@pytest.mark.parametrize("f", [0.5, 0.8, 1.0])
def test_reserve_density_converges_to_f(f: float, pars: de.DEBPars) -> None:
    """
    The fundamental DEB invariant: at constant food, scaled reserve density e
    converges to f. This holds only if the mobilization flux is dimensionally
    consistent, so it is the acceptance test for the Table S1 p_C expression.
    """
    st, _ = de.run(
        pars,
        engine="closed",
        dt=1.0 / 24.0,
        f=f,
        T=T_25C,
        until_maturity=0.95 * pars.E_Hp,
    )
    assert st.stage == de.Stage.LARVA
    e = st.E / (st.V * pars.E_m)
    assert e == pytest.approx(f, abs=1e-6)


def test_stepped_engine_converges_to_closed(pars: de.DEBPars) -> None:
    """
    Both engines integrate the same right-hand side, so explicit Euler must
    approach the adaptive solution as dt shrinks.

    The comparison stays inside the larval stage: across a stage transition the
    two engines locate the event at slightly different times, which is a
    discretisation artefact of the event itself rather than of the integration.
    """

    def seed() -> de.DEBState:
        return de.DEBState(
            E=0.9 * pars.E_m * 1e-3,
            V=1e-3,
            E_H=10.0,
            stage=de.Stage.LARVA,
            L_b=pars.L_b_pred,
        )

    t_end = 0.25
    ref, _ = de.run(
        pars,
        state=seed(),
        engine="closed",
        dt=1.0 / 240.0,
        f=1.0,
        T=T_25C,
        until_age=t_end,
    )
    assert ref.stage == de.Stage.LARVA, "the comparison must stay within one stage"

    errors = []
    for dt in (1.0 / (24 * 60), 1.0 / (24 * 600), 1.0 / (24 * 6000)):
        st, _ = de.run(
            pars,
            state=seed(),
            engine="stepped",
            dt=dt,
            f=1.0,
            T=T_25C,
            until_age=t_end,
            max_steps=10_000_000,
        )
        errors.append(abs(st.V - ref.V) / ref.V)

    assert errors == sorted(errors, reverse=True), f"not converging: {errors}"
    assert errors[-1] < 1e-4, f"insufficient accuracy at the finest dt: {errors}"


def test_stop_criteria(pars: de.DEBPars) -> None:
    st, _ = de.run(pars, engine="closed", dt=1.0 / 24.0, f=1.0, until_age=1.5)
    assert st.age == pytest.approx(1.5, abs=1e-6)

    st, _ = de.run(
        pars, engine="closed", dt=1.0 / 24.0, f=1.0, until_stage=de.Stage.LARVA
    )
    assert st.stage == de.Stage.LARVA

    target = 0.5 * pars.E_Hp
    st, _ = de.run(pars, engine="closed", dt=1.0 / 24.0, f=1.0, until_maturity=target)
    assert st.E_H >= target


def test_callable_functional_response(pars: de.DEBPars) -> None:
    st, _ = de.run(
        pars,
        engine="stepped",
        dt=1.0 / (24 * 60),
        f=lambda t: 1.0,
        T=T_25C,
        until_age=1.0,
    )
    ref, _ = de.run(
        pars, engine="stepped", dt=1.0 / (24 * 60), f=1.0, T=T_25C, until_age=1.0
    )
    assert st.E == pytest.approx(ref.E)


def test_step_applies_external_assimilation(pars: de.DEBPars) -> None:
    """A single step with an injected p_A -- the sim-loop entry point."""
    st = de.DEBState(E=1.0, V=1e-4, E_H=1.0, stage=de.Stage.LARVA, L_b=0.03)
    fed = st.copy()
    de.step(st, pars, dt=1e-4, p_A=0.0)
    de.step(fed, pars, dt=1e-4, p_A=10.0)
    assert fed.E > st.E


def test_unknown_engine_raises(pars: de.DEBPars) -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        de.run(pars, engine="rk4")


def test_temperature_correction(pars: de.DEBPars) -> None:
    assert de.temperature_correction(pars, None) == 1.0
    assert de.temperature_correction(pars, pars.T_ref) == pytest.approx(1.0)
    expected = math.exp(pars.T_A / pars.T_ref - pars.T_A / T_25C)
    assert de.temperature_correction(pars, T_25C) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# tempcorr -- 1/3/5-parameter Arrhenius (DEBtool_M/lib/misc/tempcorr.m)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pars_T",
    [
        [12000.0],
        [12000.0, 277.0, 20000.0],
        [12000.0, 350.0, 190000.0],
        [12000.0, 277.0, 328.0, 20000.0, 190000.0],
    ],
)
def test_tempcorr_is_unity_at_the_reference(pars_T: list) -> None:
    """tempcorr(T_ref, T_ref, .) == 1 whatever the parameters."""
    assert float(de.tempcorr(320.0, 320.0, pars_T)) == pytest.approx(1.0)


def test_tempcorr_one_parameter_is_plain_arrhenius() -> None:
    T, T_ref, T_A = 300.0, 293.15, 8000.0
    expected = math.exp(T_A / T_ref - T_A / T)
    assert float(de.tempcorr(T, T_ref, [T_A])) == pytest.approx(expected)
    assert float(de.tempcorr(T, T_ref, T_A)) == pytest.approx(expected)


def test_tempcorr_torpor_never_exceeds_plain_arrhenius() -> None:
    """Torpor only ever slows rates down, so 1 parameter is always the largest."""
    T_ref, temps = 320.0, np.array([300.0, 310.0, 320.0, 330.0, 340.0])
    one = de.tempcorr(temps, T_ref, [12000.0])
    five = de.tempcorr(temps, T_ref, [12000.0, 277.0, 328.0, 20000.0, 190000.0])
    assert np.all(five <= one + 1e-12)


def test_tempcorr_low_temperature_torpor_only_acts_below_reference() -> None:
    T_ref, temps = 320.0, np.array([300.0, 310.0, 330.0, 340.0])
    one = de.tempcorr(temps, T_ref, [12000.0])
    low = de.tempcorr(temps, T_ref, [12000.0, 277.0, 20000.0])  # T_L < T_ref
    assert np.all(low[:2] < one[:2])  # below T_ref: suppressed
    assert low[2:] == pytest.approx(one[2:])  # above T_ref: unaffected


def test_tempcorr_high_temperature_torpor_only_acts_above_reference() -> None:
    T_ref, temps = 320.0, np.array([300.0, 310.0, 330.0, 340.0])
    one = de.tempcorr(temps, T_ref, [12000.0])
    high = de.tempcorr(temps, T_ref, [12000.0, 350.0, 190000.0])  # T_H > T_ref
    assert high[:2] == pytest.approx(one[:2])
    assert np.all(high[2:] < one[2:])


def test_tempcorr_rejects_bad_parameter_vectors() -> None:
    with pytest.raises(ValueError, match="1, 3 or 5"):
        de.tempcorr(300.0, 293.15, [1.0, 2.0])
    with pytest.raises(ValueError, match="must lie between"):
        de.tempcorr(300.0, 320.0, [12000.0, 330.0, 340.0, 20000.0, 190000.0])


def test_temperature_correction_selects_the_right_branch(pars: de.DEBPars) -> None:
    plain = de.temperature_correction(pars, 280.0)
    chilled = de.temperature_correction(pars.with_(T_L=277.0, T_AL=20000.0), 280.0)
    assert chilled < plain

    both = pars.with_(T_L=277.0, T_AL=20000.0, T_H=320.0, T_AH=190000.0)
    assert de.temperature_correction(both, both.T_ref) == pytest.approx(1.0)
    assert de.temperature_correction(both, 330.0) < de.temperature_correction(
        pars, 330.0
    )


def test_torpor_parameters_must_be_supplied_in_pairs(pars: de.DEBPars) -> None:
    with pytest.raises(ValueError, match="T_L and T_AL"):
        pars.with_(T_L=277.0)
    with pytest.raises(ValueError, match="T_H and T_AH"):
        pars.with_(T_H=320.0)
    with pytest.raises(ValueError, match="T_L .* must not exceed T_ref"):
        pars.with_(T_L=400.0, T_AL=20000.0)
    with pytest.raises(ValueError, match="T_H .* must not be below T_ref"):
        pars.with_(T_H=100.0, T_AH=20000.0)


# ---------------------------------------------------------------------------
# Per-stage simulation and life-cycle chaining
# ---------------------------------------------------------------------------


def test_resolve_stage_accepts_the_embryo_alias() -> None:
    assert de.resolve_stage("embryo") == de.Stage.EGG
    assert de.resolve_stage("larva") == de.Stage.LARVA
    with pytest.raises(ValueError, match="unknown stage"):
        de.resolve_stage("nymph")


def test_run_stage_stops_at_the_next_stage(pars: de.DEBPars) -> None:
    st, tr = de.run_stage(pars, stage="egg", engine="closed", dt=1 / 24, f=1.0, T=T_25C)
    assert st.stage == de.Stage.LARVA
    assert set(tr.stage) == {de.Stage.EGG, de.Stage.LARVA}


def test_run_stage_accepts_the_embryo_alias(pars: de.DEBPars) -> None:
    a, _ = de.run_stage(
        pars, stage="embryo", engine="closed", dt=1 / 24, f=1.0, T=T_25C
    )
    b, _ = de.run_stage(pars, stage="egg", engine="closed", dt=1 / 24, f=1.0, T=T_25C)
    assert a.age == pytest.approx(b.age)


def test_run_stage_runs_a_stage_in_isolation(pars: de.DEBPars) -> None:
    """A pupa started from a synthetic state must use the pupal fluxes only."""
    seed = de.DEBState(
        E=50.0, V=1e-6, E_H=0.0, stage=de.Stage.PUPA, L_b=0.036, L_p=0.29
    )
    st, tr = de.run_stage(
        pars, state=seed, stage="pupa", engine="closed", dt=1 / 240, f=1.0, T=T_25C
    )
    assert de.Stage.LARVA not in tr.stage
    assert de.Stage.EGG not in tr.stage
    assert st.stage == de.Stage.IMAGO


def test_run_life_cycle_matches_run_until_imago(pars: de.DEBPars) -> None:
    """
    Chaining run_stage must reproduce a single run(until_stage="imago").

    Not bit-identical: each run_stage call sizes its own solve_ivp output grid from
    the age it starts at, so the adaptive solver places steps slightly differently
    and locates each event a few 1e-8 apart. The tolerance is well inside that and
    far below any real divergence.
    """
    lh = de.run_life_cycle(pars, engine="closed", dt=1 / 24, f=1.0, T=T_25C)
    st, tr = de.run(
        pars, engine="closed", dt=1 / 24, f=1.0, T=T_25C, until_stage=de.Stage.IMAGO
    )
    assert lh.reached == de.STAGES
    for stage in de.STAGES:
        assert lh.events[stage] == pytest.approx(tr.events[stage], rel=1e-6)
    assert lh.final.age == pytest.approx(st.age, rel=1e-6)


def test_run_life_cycle_durations_are_consistent(pars: de.DEBPars) -> None:
    lh = de.run_life_cycle(pars, engine="closed", dt=1 / 24, f=1.0, T=T_25C)
    for stage in (de.Stage.EGG, de.Stage.LARVA, de.Stage.PUPA):
        entered = lh.events[stage]
        nxt = de.STAGES[de.STAGES.index(stage) + 1]
        assert lh.durations[stage] == pytest.approx(lh.events[nxt] - entered, rel=1e-9)
    assert lh.time_to_pupation == pytest.approx(
        lh.age_at_pupation - lh.age_at_birth, rel=1e-12
    )


def test_run_life_cycle_reference_lengths_match_amp(
    pars: de.DEBPars, amp: dict
) -> None:
    lh = de.run_life_cycle(pars, engine="closed", dt=1 / 24, f=1.0, T=T_25C)
    assert lh.Lw_b == pytest.approx(amp["Lb"], rel=5e-3)
    assert lh.Lw_p == pytest.approx(amp["Lj"], rel=1e-2)


def test_L_p_is_the_larval_size_not_the_post_reset_seed(pars: de.DEBPars) -> None:
    """
    Larval structure is resorbed at pupation, so the state entering the pupal stage
    carries only the seed volume. L_p must still report the size reached.
    """
    lh = de.run_life_cycle(pars, engine="closed", dt=1 / 24, f=1.0, T=T_25C)
    assert lh.L_at("pupa") == pytest.approx(pars.V_seed ** (1 / 3))
    assert lh.L_p > 100 * lh.L_at("pupa")
    assert lh.L_p > lh.L_b


def test_run_life_cycle_rejects_until_stage(pars: de.DEBPars) -> None:
    with pytest.raises(ValueError, match="drives the stage sequence itself"):
        de.run_life_cycle(pars, until_stage="pupa")


def test_life_cycle_trajectory_is_continuous(pars: de.DEBPars) -> None:
    lh = de.run_life_cycle(pars, engine="closed", dt=1 / 24, f=1.0, T=T_25C)
    tr = lh.trajectory
    assert np.all(np.diff(tr.t) >= 0.0)
    assert tr.t.size == len(tr.stage) == tr.E.size
    assert set(tr.stage) == set(de.STAGES)


@pytest.mark.parametrize("engine", ["stepped", "closed"])
def test_run_life_cycle_works_with_both_engines(engine: str, pars: de.DEBPars) -> None:
    dt = 1 / (24 * 60) if engine == "stepped" else 1 / 24
    lh = de.run_life_cycle(
        pars, engine=engine, dt=dt, f=1.0, T=T_25C, max_steps=2_000_000
    )
    assert lh.reached == de.STAGES


def test_format_life_history_lists_every_stage(pars: de.DEBPars) -> None:
    lh = de.run_life_cycle(pars, engine="closed", dt=1 / 24, f=1.0, T=T_25C)
    text = de.format_life_history(lh)
    for stage in de.STAGES:
        assert stage in text
    for label in ("oviposition", "hatching", "pupation", "emergence"):
        assert label in text
    # the pupal column reports the size reached at pupation, not the reset seed
    assert f"{10 * lh.Lw_p:.4g}" in text


def test_weights(pars: de.DEBPars) -> None:
    V, E = 1e-3, 5.0
    assert de.dry_weight(pars, V, E) == pytest.approx(
        V * pars.d_V + E * pars.w_E / pars.mu_E
    )
    assert de.wet_weight(pars, V, E) == pytest.approx(
        V + E * pars.w_E / (pars.mu_E * pars.d_V)
    )
    # wet weight exceeds dry weight for any positive state
    assert de.wet_weight(pars, V, E) > de.dry_weight(pars, V, E)


# ---------------------------------------------------------------------------
# Validation against the published AmP predictions
# ---------------------------------------------------------------------------


def test_reproduces_amp_predictions(pars: de.DEBPars, amp: dict) -> None:
    """
    Independent validation: the AmP results page reports predicted age at birth,
    time to pupation and the physical lengths at those events. Our integration of
    Tables S1/S2 from the same primary parameters must land on them.

    Tolerances are loose on the times because AmP obtains them from its own
    closed-form ``get_tj_habp`` scheme rather than by integration.
    """
    st, tr = de.run(
        pars,
        engine="closed",
        dt=1.0 / 24.0,
        f=1.0,
        T=T_25C,
        until_stage=de.Stage.IMAGO,
    )
    ab = tr.events["larva"]
    tj = tr.events["pupa"] - ab

    assert ab == pytest.approx(amp["ab"], rel=0.05)
    assert tj == pytest.approx(amp["tj"], rel=0.05)
    # physical lengths, Lw = L / del_M, as reached by integration
    assert st.L_b / pars.del_M == pytest.approx(amp["Lb"], rel=5e-3)
    assert st.L_p / pars.del_M == pytest.approx(amp["Lj"], rel=1e-2)


def test_analytic_length_at_birth_matches_amp(pars: de.DEBPars, amp: dict) -> None:
    """
    The analytic route -- get_lb on the compound parameters -- is independent of the
    integrator and reproduces the AmP prediction to five significant figures.
    """
    assert pars.L_b_pred / pars.del_M == pytest.approx(amp["Lb"], rel=1e-4)


def test_amp_predictions_block_is_not_used_for_parameterisation() -> None:
    """
    data_predictions holds *physical* lengths. Loading them into structural-length
    slots would introduce a systematic 1/del_M error, so from_amp_json must ignore
    that block entirely.
    """
    from larvaworld.lib.model.deb.rover_sitter_model import load_drosophila

    p = load_drosophila()
    preds = de.amp_predictions(de_json())
    # Lb is a prediction (0.05788 cm, physical); it must not have become a parameter.
    assert not hasattr(p, "Lb")
    assert p.L_b_pred != pytest.approx(preds["Lb"])
    assert p.L_b_pred / p.del_M == pytest.approx(preds["Lb"], rel=1e-3)
