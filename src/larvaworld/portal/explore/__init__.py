"""Explore: the zero-configuration entry point to Larvaworld.

Kept import-light on purpose - importing this package must not pull in Panel or
the simulation stack, so the portal keeps a cheap startup.
"""

from larvaworld.portal.explore.scenarios import (
    CATEGORY_ORDER,
    CATEGORY_TITLES,
    SCENARIOS,
    Scenario,
    scenario_by_id,
    scenarios_by_category,
    validate_scenarios,
)

__all__ = [
    "CATEGORY_ORDER",
    "CATEGORY_TITLES",
    "SCENARIOS",
    "Scenario",
    "scenario_by_id",
    "scenarios_by_category",
    "validate_scenarios",
]
