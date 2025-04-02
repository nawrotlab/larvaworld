from brian2 import (
    NeuronGroup,
    SpikeMonitor,
    StateMonitor,
    PopulationRateMonitor,
    TimedArray,
    run,
)
from brian2.units import *


def osn_params(custom_params: dict = {}):
    """
    factory method for model parameters
    :param custom_params:
    :return:
    """
    params = {
        "C_m": 150 * pF,  # Membrane capacitance ORN
        "g_l": 10 * nS,  # Leak conductance ORN
        "E_l": -70 * mV,  # Leak potential ORN
        "E_Ia": -90 * mV,  # Adaptation reversal potential
        "baseline_input_current": 300 * pA,  # baseline stimulus for neuronal activity
        "threshold": -43 * mV,
        "reset_term": (
            """
                 v = -67 * mV      # after each spike the membrane is resetted to -63mV
                 g_Ia += 0.085 * nS  # after each spike the adaptation variable rises by 0.1nS
                 """
        ),
        "tau_Ia": 5000 * ms,  # Adaptation variable time constant
        "g_Ia": 2.7 * nS,
        "v_0": -67 * mV,  # initial membrane potential
    }
    params.update(custom_params)
    return params


def get_osn_model(
    stimulus: TimedArray, params: dict = osn_params(), **kwargs
) -> (NeuronGroup, dict):
    """
    builder function for brian2 model
    :param stimulus:
    :param params:
    :param kwargs:
    :return:
    """
    merged_params = params
    merged_params.update(kwargs)
    merged_params.update({"S": stimulus})
    print(
        f"OSN params: {merged_params} | kwargs: {kwargs} | stimulus: {stimulus.values[:10]} ..."
    )

    ORN_eqs = """
          dv/dt = (g_l*(E_l-v)+g_Ia*(E_Ia-v)+I)/C_m : volt (unless refractory)
          dg_Ia/dt = -g_Ia/tau_Ia : siemens # conductance adaptation
          I = (S(t)+baseline_input_current) : ampere
          """

    OSN = NeuronGroup(
        1,
        ORN_eqs,
        threshold="v > threshold",
        reset=merged_params["reset_term"],
        refractory=2 * ms,
        method="euler",
        namespace=merged_params,
    )

    OSN.v = merged_params["v_0"]
    OSN.g_Ia = merged_params["g_Ia"]

    monitors = {
        "state": StateMonitor(OSN, ["I", "v", "g_Ia"], record=True),
        "rate": PopulationRateMonitor(OSN),
        "spikes": SpikeMonitor(OSN),
    }
    return OSN, monitors, merged_params
