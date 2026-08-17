import pytest
import random
from larvaworld.lib.model import SpaceDict
from larvaworld.lib.util import AttrDict


@pytest.fixture
def space_dict():
    return SpaceDict(
        base_model="explorer",
        space_mkeys=["interference", "crawler"],
        Pmutation=0.1,
        Cmutation=0.1,
        init_mode="random",
    )


def test_space_dict_initialization(space_dict):
    assert isinstance(space_dict, SpaceDict)
    assert space_dict.base_model is not None
    assert isinstance(space_dict.space_mkeys, list)
    assert isinstance(space_dict.Pmutation, float)
    assert isinstance(space_dict.Cmutation, float)
    assert isinstance(space_dict.init_mode, str)


def test_space_dict_build(space_dict):
    space_objs = space_dict.build()
    assert isinstance(space_objs, AttrDict)
    assert len(space_objs) > 0


def test_space_dict_defaults(space_dict):
    defaults = space_dict.defaults
    assert isinstance(defaults, AttrDict)
    assert len(defaults) > 0


def test_space_dict_randomize(space_dict):
    random.seed(0)
    g = space_dict.defaults
    randomized_g = space_dict.randomize()
    assert isinstance(g, AttrDict)
    assert len(g) > 0
    assert isinstance(randomized_g, AttrDict)
    assert len(randomized_g) > 0
    assert g != randomized_g


def test_space_dict_mutate(space_dict):
    random.seed(0)
    g = space_dict.defaults.get_copy()
    mutated_g = space_dict.mutate(g)
    assert isinstance(g, AttrDict)
    assert len(g) > 0
    assert isinstance(mutated_g, AttrDict)
    assert len(mutated_g) > 0


def test_space_dict_create_first_generation(space_dict):
    random.seed(0)
    N = 5
    first_gen = space_dict.create_first_generation(N)
    assert isinstance(first_gen, list)
    assert len(first_gen) == N
    for g in first_gen:
        assert isinstance(g, AttrDict)
        assert len(g) > 0


def test_space_dict_excludes_effector_params_by_default():
    sd = SpaceDict(base_model="explorer", space_mkeys=["turner"])
    assert "brain.turner.input_noise" not in sd.space_ks
    assert "brain.turner.output_noise" not in sd.space_ks
    assert len(sd.space_ks) > 0


def test_space_dict_include_effector_params_adds_noise_keys():
    sd = SpaceDict(
        base_model="explorer", space_mkeys=["turner"], include_effector_params=True
    )
    assert "brain.turner.input_noise" in sd.space_ks
    assert "brain.turner.output_noise" in sd.space_ks
    # non-Effector turner params are still present too
    assert "brain.turner.base_activation" in sd.space_ks


def test_space_dict_space_pkeys_narrows_to_exact_allowlist():
    sd = SpaceDict(
        base_model="explorer",
        space_mkeys=["turner"],
        include_effector_params=True,
        space_pkeys=["input_noise", "output_noise"],
    )
    assert sd.space_ks == ["brain.turner.input_noise", "brain.turner.output_noise"]
    assert sd.defaults == {
        "brain.turner.input_noise": 0.0,
        "brain.turner.output_noise": 0.0,
    }


def test_space_dict_defaults_unaffected_when_new_params_unset(space_dict):
    """Regression guard: today's exact behavior is unchanged when the new
    params are left at their defaults (include_effector_params=False,
    space_pkeys=None)."""
    baseline = SpaceDict(
        base_model="explorer",
        space_mkeys=["interference", "crawler"],
        Pmutation=0.1,
        Cmutation=0.1,
        init_mode="random",
    )
    assert baseline.space_ks == space_dict.space_ks
