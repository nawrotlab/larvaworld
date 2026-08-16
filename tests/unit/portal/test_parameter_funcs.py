"""Tests for the reusable parameter lookup/load/save functions."""

from __future__ import annotations

import pytest

from larvaworld.lib import reg
from larvaworld.portal.parameter_database.parameter_funcs import (
    get_param_instance,
    register_new_param,
    remove_param,
)


def _cloned_config(source_k: str, *, k: str, p: str, d: str) -> dict:
    config = dict(get_param_instance(source_k).to_config())
    config["k"] = k
    config["p"] = p
    config["d"] = d
    return config


def test_get_param_instance_known_key() -> None:
    lp = get_param_instance("t")
    assert lp.disp == reg.par.kdict["t"].disp
    assert lp.sym == reg.par.kdict["t"].sym


def test_get_param_instance_unknown_key_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_param_instance("__not_a_real_key__")


def test_clone_via_to_config_and_register_round_trip() -> None:
    config = _cloned_config(
        "t", k="x_clone_test_k", p="x_clone_test_p", d="x_clone_test_d"
    )

    new_k = register_new_param(config)

    assert new_k == "x_clone_test_k"
    assert reg.par.kdict[new_k].p == "x_clone_test_p"
    assert reg.getPar(k=new_k, to_return="p") == "x_clone_test_p"


def test_register_new_param_duplicate_key_raises_value_error() -> None:
    config = _cloned_config(
        "t", k="x_dup_data_test_k", p="x_dup_data_test_p", d="x_dup_data_test_d"
    )
    register_new_param(config)

    config2 = dict(config)
    config2["p"] = "x_dup_data_test_p_2"
    with pytest.raises(ValueError):
        register_new_param(config2)


def test_remove_param_deletes_registered_param() -> None:
    config = _cloned_config(
        "t",
        k="x_remove_data_test_k",
        p="x_remove_data_test_p",
        d="x_remove_data_test_d",
    )
    new_k = register_new_param(config)
    assert new_k in reg.par.dict

    remove_param(new_k)

    assert new_k not in reg.par.dict
    assert new_k not in reg.par.kdict


def test_remove_param_unknown_key_raises_key_error() -> None:
    with pytest.raises(KeyError):
        remove_param("__not_a_real_key__")


def test_register_new_param_requires_name() -> None:
    with pytest.raises(ValueError):
        register_new_param({"p": ""})
