"""
Dynamic Energy Budget (DEB) Theory - Standard Parameters for the Generalized Animal

This file provides the baseline parameter values for the DEB 'generalized animal'
at a reference temperature of 293.15 K (20 °C). These standardized parameters serve
as pseudo-data anchors within the Add-my-Pet database framework to calibrate and
estimate missing parameters for specific real-world animal species.

Units and Definitions:
- p_Am: Max surface-area-specific assimilation rate [J * cm^-2 * d^-1]
- v: Energy conductance [cm * d^-1]
- kappa: Allocation fraction of mobilized reserve to soma [dimensionless]
- kappa_R: Reproduction efficiency [dimensionless]
- p_M: Volume-specific somatic maintenance costs [J * cm^-3 * d^-1]
- p_T: Surface-area-specific somatic maintenance costs [J * cm^-2 * d^-1]
- k_J: Maturity maintenance rate coefficient [d^-1]
- E_G: Specific costs for structure [J * cm^-3]
- E_Hb: Maturity level at birth [J]
- E_Hp: Maturity level at puberty [J]
- h_a: Aging acceleration [d^-2]
- s_G: Gompertz stress coefficient [dimensionless]
- T_A: Arrhenius temperature [K]
- del_M: Shape coefficient [dimensionless]
- mu_E: Chemical potential of reserves [J * mol^-1]
- d_V: Dry/wet weight ratio for structure [dimensionless]
- d_E: Dry/wet weight ratio for reserves [dimensionless]
"""

from ...util import AttrDict

__all__: list[str] = ["deb_generalized_animal", "compare_model_to_generalized_animal"]

deb_generalized_animal = AttrDict(
    {
        "p_Am": 110.0,
        "v": 0.02,
        "kap": 0.8,
        "p_M": 18.0,
        "E_G": 2800.0,
        "E_Hb": 2750.0,
        "E_Hp": 166000.0,
        "h_a": 1e-07,
        "s_G": 0.0,
        "T_A": 8000.0,
        "del_M": 0.5,
        "E_He": 0.7665,
    }
)


def compare_model_to_generalized_animal(species: str = "Drosophila_melanogaster"):
    """
    Compare a given model's parameters to the generalized animal parameters.

    Args:
        model_params (dict): A dictionary of model parameters to compare.

    Returns:
        dict: A dictionary containing the differences between the model parameters
              and the generalized animal parameters.
    """
    from .amp_import import from_amp_json

    metadata, values, results, nonfree_pars, free_pars = from_amp_json(species)

    gen_ks = deb_generalized_animal.keylist
    nonfree = [k for k in nonfree_pars if k in gen_ks]
    free = [k for k in free_pars if k in gen_ks]
    free_equal = []
    nonfree_unequal = []
    unused = [k for k in gen_ks if k not in nonfree and k not in free]
    additional = [k for k in free_pars + nonfree_pars if k not in gen_ks]
    for k in nonfree:
        if not bool(values[k] == deb_generalized_animal[k]):
            nonfree_unequal.append(k)
    for k in free:
        if not bool(values[k] != deb_generalized_animal[k]):
            free_equal.append(k)
    comparison = AttrDict(
        {
            "free": free_pars,
            "nonfree": nonfree_pars,
            "free_equal": free_equal,
            "nonfree_unequal": nonfree_unequal,
            "unused": unused,
            "additional": additional,
        }
    )
    return metadata, values, results, comparison
