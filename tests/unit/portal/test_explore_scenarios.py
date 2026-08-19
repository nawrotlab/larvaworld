"""Tests for the curated Explore scenario catalog."""

from __future__ import annotations

import pytest

from larvaworld.portal.explore.scenarios import (
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    SCENARIOS,
    scenario_by_id,
    scenarios_by_category,
    validate_scenarios,
)


def test_catalog_is_valid_standalone() -> None:
    validate_scenarios()


def test_catalog_is_not_empty_and_stays_small() -> None:
    # The whole point is curation: a beginner must not face a wall of options.
    assert 1 <= len(SCENARIOS) <= 20


def test_every_scenario_id_is_unique() -> None:
    ids = [s.id for s in SCENARIOS]
    assert len(ids) == len(set(ids))


def test_every_scenario_references_a_real_experiment() -> None:
    reg = pytest.importorskip("larvaworld.lib.reg")
    known = set(reg.conf.Exp.confIDs)
    validate_scenarios(known_exp_ids=known)


def test_step_caps_are_bounded() -> None:
    # Bounded runtime is what makes "press once and watch" safe on a laptop.
    for scenario in SCENARIOS:
        assert 0 < scenario.step_cap <= 2000
        assert 0 < scenario.n_agents <= 50


def test_every_category_used_is_declared_and_ordered() -> None:
    used = {s.category for s in SCENARIOS}
    assert used <= set(CATEGORY_TITLES)
    grouped = scenarios_by_category()
    assert list(grouped) == [c for c in CATEGORY_ORDER if c in used]


def test_grouping_covers_every_scenario() -> None:
    grouped = scenarios_by_category()
    assert sum(len(v) for v in grouped.values()) == len(SCENARIOS)


def test_scenario_lookup() -> None:
    first = SCENARIOS[0]
    assert scenario_by_id(first.id) is first
    assert scenario_by_id("no-such-scenario") is None


def test_scenarios_carry_beginner_facing_copy() -> None:
    for scenario in SCENARIOS:
        # Titles must be prose, not raw registry IDs.
        assert scenario.title != scenario.exp_id
        assert " " in scenario.title
        assert scenario.teaser.strip()
        assert len(scenario.explanation.split()) >= 15


def test_validation_rejects_unknown_experiment() -> None:
    with pytest.raises(ValueError, match="unknown experiment"):
        validate_scenarios(known_exp_ids=set())
