"""Tests for LarvaworldParam.to_config / save_config / from_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from larvaworld.lib import reg
from larvaworld.lib.reg.data_aux import LarvaworldParam
from larvaworld.lib.util import AttrDict

#: (key, expected `v` param class name) covering every value-type actually
#: present in the registry. "t"/"N_ts" resolve to PositiveNumber/
#: PositiveInteger (not plain Number/Integer) since resolve_param_class
#: prefers the more specific custom class for a lower bound of exactly 0.
_SAMPLE_KEYS = [
    ("t", "PositiveNumber"),
    ("N_ts", "PositiveInteger"),
    ("tor", "Magnitude"),
    ("on_food", "Boolean"),
]


@pytest.fixture(params=_SAMPLE_KEYS, ids=[k for k, _ in _SAMPLE_KEYS])
def sample_param(request) -> LarvaworldParam:
    k, expected_type = request.param
    reg.par.update_kdict([k])
    lp = reg.par.kdict[k]
    assert type(lp.param["v"]).__name__ == expected_type
    return lp


def test_to_config_returns_attrdict_with_reconstruction_metadata(
    sample_param: LarvaworldParam,
) -> None:
    config = sample_param.to_config()

    assert isinstance(config, AttrDict)
    assert config.k == sample_param.k
    assert config.p == sample_param.p
    assert config.v == sample_param.v
    assert config.doc == sample_param.description
    assert "name" not in config


def test_from_config_reconstructs_equivalent_instance(
    sample_param: LarvaworldParam,
) -> None:
    config = sample_param.to_config()

    new_lp = LarvaworldParam.from_config(config)

    assert new_lp.k == sample_param.k
    assert new_lp.p == sample_param.p
    assert new_lp.v == sample_param.v
    assert new_lp.dtype == sample_param.dtype
    assert new_lp.u == sample_param.u
    assert new_lp.description == sample_param.description


def test_save_config_and_from_config_round_trip_via_file(
    sample_param: LarvaworldParam, tmp_path: Path
) -> None:
    file_path = str(tmp_path / f"{sample_param.k}_config.pkl")

    sample_param.save_config(file_path)
    loaded = AttrDict.load(file_path)
    new_lp = LarvaworldParam.from_config(loaded)

    assert new_lp.v == sample_param.v
    assert new_lp.dtype == sample_param.dtype
    assert new_lp.u == sample_param.u
    assert new_lp.description == sample_param.description
    try:
        assert new_lp.param["v"].bounds == sample_param.param["v"].bounds
    except AttributeError:
        pass  # Boolean has no bounds; nothing to compare.
