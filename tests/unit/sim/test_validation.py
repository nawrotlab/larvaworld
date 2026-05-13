from __future__ import annotations

from copy import deepcopy

from larvaworld.lib import reg, util
from larvaworld.lib.sim.validation import (
    validate_experiment_environment_compatibility,
)


def _base_parameters() -> dict:
    return {
        "env_params": {
            "arena": {"geometry": "rectangular", "dims": (0.2, 0.2)},
            "food_params": {
                "food_grid": None,
                "source_groups": {},
                "source_units": {},
            },
            "border_list": {},
            "odorscape": None,
            "windscape": None,
            "thermoscape": None,
        },
        "larva_groups": {
            "explorer": {
                "distribution": {
                    "loc": (0.0, 0.0),
                    "scale": (0.01, 0.01),
                    "mode": "uniform",
                    "shape": "rect",
                }
            }
        },
    }


def test_validation_accepts_valid_rectangular_configuration() -> None:
    report = validate_experiment_environment_compatibility(_base_parameters())
    assert not report.has_errors
    assert report.warnings == ()


def test_validation_treats_boundary_point_as_valid() -> None:
    parameters = _base_parameters()
    parameters["larva_groups"]["explorer"]["distribution"]["loc"] = (0.1, 0.0)
    parameters["larva_groups"]["explorer"]["distribution"]["scale"] = (0.0, 0.0)
    report = validate_experiment_environment_compatibility(parameters)
    assert not report.has_errors


def test_validation_rejects_invalid_arena_dims() -> None:
    parameters = _base_parameters()
    parameters["env_params"]["arena"]["dims"] = (0.2,)
    report = validate_experiment_environment_compatibility(parameters)
    assert report.has_errors
    assert any(issue.path == "env_params.arena.dims" for issue in report.errors)


def test_validation_rejects_arena_dims_with_more_than_two_values() -> None:
    parameters = _base_parameters()
    parameters["env_params"]["arena"]["dims"] = (0.2, 0.2, 0.1)
    report = validate_experiment_environment_compatibility(parameters)
    assert report.has_errors
    assert any(issue.path == "env_params.arena.dims" for issue in report.errors)


def test_validation_warns_on_unequal_circular_dims() -> None:
    parameters = _base_parameters()
    parameters["env_params"]["arena"]["geometry"] = "circular"
    parameters["env_params"]["arena"]["dims"] = (0.2, 0.16)
    report = validate_experiment_environment_compatibility(parameters)
    assert not report.has_errors
    assert any(issue.path == "env_params.arena.dims" for issue in report.warnings)


def test_validation_rejects_larva_group_center_outside_arena() -> None:
    parameters = _base_parameters()
    parameters["larva_groups"]["explorer"]["distribution"]["loc"] = (0.12, 0.0)
    report = validate_experiment_environment_compatibility(parameters)
    assert report.has_errors
    assert any(
        issue.path == "larva_groups.explorer.distribution.loc"
        for issue in report.errors
    )


def test_validation_warns_on_distribution_loc_with_more_than_two_values() -> None:
    parameters = _base_parameters()
    parameters["larva_groups"]["explorer"]["distribution"]["loc"] = (0.0, 0.0, 0.1)
    report = validate_experiment_environment_compatibility(parameters)
    assert not report.has_errors
    assert any(
        issue.path == "larva_groups.explorer.distribution.loc"
        for issue in report.warnings
    )


def test_validation_rejects_deterministic_envelope_outside_arena() -> None:
    parameters = _base_parameters()
    parameters["larva_groups"]["explorer"]["distribution"]["loc"] = (0.095, 0.0)
    parameters["larva_groups"]["explorer"]["distribution"]["scale"] = (0.02, 0.02)
    report = validate_experiment_environment_compatibility(parameters)
    assert report.has_errors
    assert any(
        issue.path == "larva_groups.explorer.distribution" for issue in report.errors
    )


def test_validation_warns_on_distribution_scale_with_more_than_two_values() -> None:
    parameters = _base_parameters()
    parameters["larva_groups"]["explorer"]["distribution"]["scale"] = (
        0.01,
        0.01,
        0.1,
    )
    report = validate_experiment_environment_compatibility(parameters)
    assert not report.has_errors
    assert any(
        issue.path == "larva_groups.explorer.distribution.scale"
        for issue in report.warnings
    )


def test_validation_warns_when_normal_3sigma_envelope_leaves_arena() -> None:
    parameters = _base_parameters()
    parameters["larva_groups"]["explorer"]["distribution"]["mode"] = "normal"
    parameters["larva_groups"]["explorer"]["distribution"]["shape"] = "rect"
    parameters["larva_groups"]["explorer"]["distribution"]["scale"] = (0.2, 0.2)
    report = validate_experiment_environment_compatibility(parameters)
    assert not report.has_errors
    assert any(
        issue.path == "larva_groups.explorer.distribution" for issue in report.warnings
    )


def test_validation_warns_for_ambiguous_shape_mode_combo() -> None:
    parameters = deepcopy(_base_parameters())
    parameters["larva_groups"]["explorer"]["distribution"]["shape"] = "triangle"
    report = validate_experiment_environment_compatibility(parameters)
    assert not report.has_errors
    assert any(
        "Ambiguous distribution shape/mode" in issue.message
        for issue in report.warnings
    )


def test_validation_registry_chemotaxis_no_longer_false_positive() -> None:
    parameters = util.AttrDict(reg.conf.Exp.getID("chemotaxis").get_copy())
    report = validate_experiment_environment_compatibility(parameters)
    assert not report.has_errors


def test_validation_registry_tactile_detection_strict_mode_errors() -> None:
    parameters = util.AttrDict(reg.conf.Exp.getID("tactile_detection").get_copy())
    report = validate_experiment_environment_compatibility(
        parameters,
        experiment_id="tactile_detection",
    )
    assert report.has_errors
    assert any(
        issue.path == "larva_groups.toucher.distribution" for issue in report.errors
    )


def test_validation_registry_tactile_detection_legacy_mode_warns() -> None:
    parameters = util.AttrDict(reg.conf.Exp.getID("tactile_detection").get_copy())
    report = validate_experiment_environment_compatibility(
        parameters,
        allow_registry_legacy=True,
        experiment_id="tactile_detection",
    )
    assert not report.has_errors
    assert any(
        issue.path == "larva_groups.toucher.distribution" for issue in report.warnings
    )
