"""
Behavioural phenotypes (rover / sitter) derived from the generic Drosophila DEB model.

The generic model is the species-specific AmP parameter set for
*Drosophila melanogaster* -- ``models/amp/Drosophila_melanogaster.json``, taken
verbatim from the Add-my-Pet database. It stands for the average representative
individual of the species and is the reference point: replacing that file with a
newer AmP export is the supported way to update the model.

The two behavioural phenotypes are *not* independent parameter sets. They are the
generic model with a single differentiating parameter overridden -- by default
``kap_X``, the digestion efficiency of food to reserve. Which parameter carries
the phenotype difference is a modelling choice, so it is a parameter of
:func:`make_phenotype` rather than a hard-coded assumption.

Why ``kap_X``
-------------
``kap_X`` is the primary symbol behind the yield coefficient the legacy larvaworld
species files differentiate on::

    y_E_X = kap_X * mu_X / mu_E = kap_X * 525000 / 550000

The legacy ``models/deb_*.csv`` files carry ``y_E_X`` directly, and they map back
onto ``kap_X`` exactly:

===========  =========  ==========
file         ``y_E_X``  ``kap_X``
===========  =========  ==========
default      0.763636   0.800000
rover        0.85       0.890476
sitter       0.50       0.523810
===========  =========  ==========

The generic value 0.8 is precisely the ``kap_X`` the AmP export carries, so the
legacy "default" species and the AmP species model already agree on this
parameter. Expressing the phenotypes through ``kap_X`` therefore preserves the
established rover/sitter contrast while moving the parameterisation onto the
primary DEB symbol.

Biologically: rovers digest ingested food into reserve more efficiently than
sitters, so at equal functional response a rover assimilates more per unit of food.

See :mod:`larvaworld.lib.model.deb.deb_equations` for the equations themselves.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

import os

from .deb_equations import DEBPars

__all__: list[str] = [
    "DROSOPHILA_AMP_JSON",
    "PHENOTYPES",
    "SPECIES",
    "DEFAULT_SPECIES",
    "DEFAULT_PHENOTYPE_PARAM",
    "PHENOTYPE_VALUES",
    "KAP_V_OVERRIDE",
    "load_drosophila",
    "make_phenotype",
    "rover",
    "sitter",
    "phenotypes",
    "get_species_pars",
]


#: In-repo copy of the AmP parameter export. Replace this file to update the model.
DROSOPHILA_AMP_JSON: str = os.path.join(
    os.path.dirname(__file__), "models", "amp", "Drosophila_melanogaster.json"
)

#: The behavioural phenotypes this module produces.
PHENOTYPES: tuple[str, ...] = ("rover", "sitter")

#: Parameter that carries the phenotype difference, unless overridden.
DEFAULT_PHENOTYPE_PARAM: str = "kap_X"

#: Phenotype values per differentiating parameter. Values for ``kap_X`` reproduce
#: the legacy ``y_E_X`` of ``models/deb_rover.csv`` / ``deb_sitter.csv`` exactly.
PHENOTYPE_VALUES: dict[str, dict[str, float]] = {
    "kap_X": {"rover": 0.890476, "sitter": 0.523810},
}

#: The AmP export reports ``kap_V = -1.526e-54``: the abp fit collapsed this
#: parameter onto its lower bound, and MATLAB ``predict_Drosophila_melanogaster.m``
#: rejects the parameter set outright when ``kap_V < 0``. ``DEBPars`` refuses such a
#: value rather than silently clamping it, so a physical value must be supplied
#: explicitly. This one is inherited from the pre-existing larvaworld species files
#: (``models/deb_*.csv``), where it has always been 0.99148.
KAP_V_OVERRIDE: float = 0.99148


def load_drosophila(json_path: Optional[str] = None, **overrides: Any) -> DEBPars:
    """
    Load the generic (species-average) Drosophila DEB model from an AmP export.

    Parameters
    ----------
    json_path : AmP parameter export; defaults to the in-repo copy
    **overrides : primary parameters to override after loading

    Notes
    -----
    ``kap_V`` is overridden with :data:`KAP_V_OVERRIDE` unless the caller supplies
    its own value -- see that constant for why. Only the ``parameters`` block of the
    export is read; ``data_predictions`` holds *physical* lengths and is available
    through :func:`~larvaworld.lib.model.deb.deb_equations.amp_predictions` for
    validation only.
    """
    overrides.setdefault("kap_V", KAP_V_OVERRIDE)
    return DEBPars.from_amp_json(json_path or DROSOPHILA_AMP_JSON, overrides=overrides)


def make_phenotype(
    phenotype: str,
    base: Optional[DEBPars] = None,
    param: str = DEFAULT_PHENOTYPE_PARAM,
    values: Optional[dict[str, float]] = None,
    **overrides: Any,
) -> DEBPars:
    """
    Derive a behavioural phenotype from the generic species model.

    Parameters
    ----------
    phenotype : one of :data:`PHENOTYPES`
    base : generic model; :func:`load_drosophila` is used when omitted
    param : the differentiating parameter (default :data:`DEFAULT_PHENOTYPE_PARAM`)
    values : ``{phenotype: value}`` for ``param``; defaults to
        ``PHENOTYPE_VALUES[param]``
    **overrides : further primary-parameter overrides, applied last

    Returns
    -------
    DEBPars : a new parameter set with all compound parameters rederived.
    """
    if phenotype not in PHENOTYPES:
        raise ValueError(
            f"unknown phenotype {phenotype!r}; expected one of {PHENOTYPES}"
        )
    if values is None:
        if param not in PHENOTYPE_VALUES:
            raise ValueError(
                f"no default phenotype values for {param!r}; pass values= explicitly. "
                f"Known: {sorted(PHENOTYPE_VALUES)}"
            )
        values = PHENOTYPE_VALUES[param]
    if phenotype not in values:
        raise ValueError(f"values= is missing an entry for phenotype {phenotype!r}")

    base = load_drosophila() if base is None else base
    if param not in base.__dataclass_fields__:
        raise ValueError(f"{param!r} is not a DEBPars parameter")

    return base.with_(**{param: values[phenotype]}, **overrides)


def rover(base: Optional[DEBPars] = None, **kwargs: Any) -> DEBPars:
    """The rover phenotype. See :func:`make_phenotype`."""
    return make_phenotype("rover", base=base, **kwargs)


def sitter(base: Optional[DEBPars] = None, **kwargs: Any) -> DEBPars:
    """The sitter phenotype. See :func:`make_phenotype`."""
    return make_phenotype("sitter", base=base, **kwargs)


def phenotypes(
    base: Optional[DEBPars] = None,
    param: str = DEFAULT_PHENOTYPE_PARAM,
    values: Optional[dict[str, float]] = None,
    **overrides: Any,
) -> dict[str, DEBPars]:
    """
    Build the generic model plus both phenotypes in one call.

    Returns ``{"default": generic, "rover": ..., "sitter": ...}``, keyed by the
    same names :data:`SPECIES` uses.
    """
    base = load_drosophila() if base is None else base
    out = {DEFAULT_SPECIES: base}
    for name in PHENOTYPES:
        out[name] = make_phenotype(
            name, base=base, param=param, values=values, **overrides
        )
    return out


#: Name of the generic, unmodified species model.
DEFAULT_SPECIES: str = "default"

#: Selectable model names. The ten stored rover*/sitter* larva-model configs
#: persist ``"rover"`` and ``"sitter"``.
SPECIES: tuple[str, ...] = (DEFAULT_SPECIES, *PHENOTYPES)


@lru_cache(maxsize=None)
def get_species_pars(species: str = DEFAULT_SPECIES) -> DEBPars:
    """
    Resolve a species name to its parameter set.

    The AmP export is the source in every case. ``"default"`` uses it unaltered --
    the average representative individual of the species -- while ``"rover"`` and
    ``"sitter"`` additionally override the differentiating parameter.

    Cached, because constructing a :class:`DEBPars` runs the embryo solver and a
    ``DEB`` instance is built every time the model registry resolves defaults.
    The result is shared, so treat it as immutable -- use
    :meth:`DEBPars.with_` to vary a parameter.
    """
    if species not in SPECIES:
        raise ValueError(f"unknown species {species!r}; expected one of {SPECIES}")
    if species in PHENOTYPES:
        return make_phenotype(species)
    return load_drosophila()
