import pytest

from larvaworld.lib.reg.parDB import ParamRegistry

_VALID_CATEGORIES = {
    "initial",
    "angular",
    "spatial",
    "chunks",
    "sim_pars",
    "deb_pars",
}


@pytest.fixture(scope="module")
def pclass() -> ParamRegistry:
    return ParamRegistry()


def test_build_populates_category_dict_for_all_keys(pclass: ParamRegistry) -> None:
    assert set(pclass.category_dict) == set(pclass.dict)
    assert set(pclass.category_dict.values()) <= _VALID_CATEGORIES


def test_category_of_returns_custom_for_unknown_key(pclass: ParamRegistry) -> None:
    assert pclass.category_of("__not_a_real_key__") == "custom"


def test_add_returns_key(pclass: ParamRegistry) -> None:
    k = pclass.add(p="x_test_param_add", dtype=float)
    assert k in pclass.dict
    assert pclass.dict[k].p == "x_test_param_add"


def test_add_and_instantiate_registers_into_kdict(pclass: ParamRegistry) -> None:
    k = pclass.add_and_instantiate(
        p="x_new_test_param", k="x_new_k", dtype=float, v0=1.0
    )

    assert k in pclass.dict
    assert k in pclass.kdict
    assert pclass.category_dict[k] == "custom"
    assert pclass.kdict[k].v == 1.0


def test_add_and_instantiate_rejects_duplicate_key_without_overwrite(
    pclass: ParamRegistry,
) -> None:
    k = pclass.add_and_instantiate(
        p="x_dup_test_param", k="x_dup_k", dtype=float, v0=2.0
    )

    with pytest.raises(ValueError):
        pclass.add_and_instantiate(
            p="x_dup_test_param_2", k="x_dup_k", dtype=float, v0=3.0
        )

    # Left untouched by the failed attempt.
    assert pclass.kdict[k].v == 2.0

    k2 = pclass.add_and_instantiate(
        p="x_dup_test_param_3", k="x_dup_k", dtype=float, v0=4.0, overwrite=True
    )
    assert k2 == k
    assert pclass.kdict[k].v == 4.0


def test_getpar_still_works_after_registering_new_param(pclass: ParamRegistry) -> None:
    k = pclass.add_and_instantiate(
        p="x_getpar_test_param", k="x_getpar_k", dtype=float, v0=5.0
    )

    assert pclass.getPar(k=k, to_return="p") == "x_getpar_test_param"


def test_remove_deletes_from_dict_kdict_and_category(pclass: ParamRegistry) -> None:
    k = pclass.add_and_instantiate(
        p="x_remove_test_param", k="x_remove_k", dtype=float, v0=6.0
    )
    assert k in pclass.dict and k in pclass.kdict and k in pclass.category_dict

    pclass.remove(k)

    assert k not in pclass.dict
    assert k not in pclass.kdict
    assert k not in pclass.category_dict


def test_remove_unknown_key_raises_key_error(pclass: ParamRegistry) -> None:
    with pytest.raises(KeyError):
        pclass.remove("__not_a_real_key__")
