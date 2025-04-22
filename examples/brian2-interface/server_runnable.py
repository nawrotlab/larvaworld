from brian2 import *
from model import get_osn_model, osn_params
from utils import timeit
import numpy as np
import os


def concentration_to_current(max_signal, Amax, hill_coeff=5.5):
    """
    convert a concentration value into a current through a hill function
    :param max_signal:
    :param Amax:
    :param hill_coeff:
    :return:
    """
    EC_50 = ((max_signal + 50) / 2) * uM
    hill_fn = lambda x: (
        Amax
        * ((x * uM) ** hill_coeff)
        / ((EC_50) ** hill_coeff + (x * uM) ** hill_coeff)
    )
    return hill_fn


def get_model_instance(input: TimedArray, sim_dt=0.1 * ms, **kwargs):
    """
    factory method to obtain an instance of the brian2 model
    :param input:
    :param sim_dt:
    :param kwargs:
    :return:
    """
    # prepare model
    defaultclock.dt = sim_dt
    NG, monitors, params = get_osn_model(input, params=osn_params(), **kwargs)
    net = Network(NG)
    net.add(monitors.values())
    return net, monitors


@timeit
def execute_model(
    id,
    sim_id,
    step_id,
    odor_id,
    odor_concentration,
    concentration_max_mmol,
    baseline_input_current,
    A_max,
    hill_coeff,
    warmup_time,
    T_sim,
    rng_seed,
    tau_Ia=5000 * ms,
    g_Ia=2.7 * nS,
    sim_dt=0.1 * ms,
):
    """
    execute a model instance for a given simulation step and run it for some time
    this function takes care of restoring brian2 model state from previous runs and to snapshot
    the model state after the current run
    :param id:
    :param sim_id: simulation identifier
    :param step_id: simulation step
    :param odor_id:
    :param odor_concentration:
    :param concentration_max_mmol:
    :param baseline_input_current:
    :param A_max:
    :param hill_coeff:
    :param warmup_time: warmup time used to calibrate the brian2 model before running for the first time
    :param T_sim: duration of the simulation
    :param rng_seed:
    :param tau_Ia:
    :param g_Ia:
    :param sim_dt:
    :return:
    """
    seed(rng_seed)
    net, mons = None, None
    cache_dir = "cache/sim-" + str(sim_id) + "/"
    model_id = str(id)
    model_cache_file = cache_dir + "brian_model_" + model_id
    os.makedirs(cache_dir, exist_ok=True)
    sigmoid_fn = concentration_to_current(
        concentration_max_mmol, A_max, hill_coeff=hill_coeff
    )
    concentration_current = sigmoid_fn(odor_concentration) * pA
    concentration_max_current = sigmoid_fn(concentration_max_mmol)

    print(
        "*** Running OSN model odor_id={} concentration={} (input_current={} max_current={}) step_id={} for {} with dt={}".format(
            odor_id,
            odor_concentration,
            concentration_current,
            concentration_max_current,
            step_id,
            T_sim,
            defaultclock.dt,
        )
    )

    if not os.path.exists(model_cache_file):
        print("assembling new model instance ....")
        warmup_input = TimedArray(
            ([concentration_max_current] * warmup_time) * pA, dt=1 * ms
        )
        net, mons = get_model_instance(
            warmup_input,
            tau_Ia=tau_Ia,
            g_Ia=g_Ia,
            baseline_input_current=baseline_input_current,
        )
        net.store("init" + model_id, filename=model_cache_file)
        net.run(warmup_time * ms)
        net.store("post_warmup" + model_id, filename=model_cache_file)
        print("storing: step{}_{}".format(model_id, 0))
        net.store("state_{}".format(model_id), filename=model_cache_file)

    # restore state from previous step & run for one iteration
    if step_id is not None and step_id > 0:
        current_input = TimedArray(([concentration_current] * T_sim), dt=1 * ms)
        net, mons = get_model_instance(
            current_input,
            tau_Ia=tau_Ia,
            g_Ia=g_Ia,
            baseline_input_current=baseline_input_current,
        )
        print(
            "sim_id='{}' id='{}' restoring state with previous step_id={}".format(
                sim_id, model_id, step_id - 1
            )
        )
        net.restore("state_{}".format(model_id), filename=model_cache_file)
        print(
            "loaded: step{}_{} network clock: {}".format(model_id, step_id - 1, net.t)
        )
        net.run(T_sim * ms)

    # snapshot and save the current simulation state
    if step_id is not None:
        print(
            "sim_id='{}' id='{}' storing state with step_id={}".format(
                sim_id, model_id, step_id
            )
        )
        print("storing: step{}_{} network clock: {}".format(model_id, step_id, net.t))
        net.store("state_{}".format(model_id), filename=model_cache_file)

    if net is not None:
        print("network clock: {}".format(net.t))

    if mons is not None:
        mean_rate = np.mean(mons["rate"].rate / Hz)
        mean_voltage = np.mean(mons["state"].v[0] / mV)
        mean_current = np.mean(mons["state"].I[0] / pA)
        mean_adaptation = np.mean(mons["state"].g_Ia[0] / nS)
        # compute simulation results which we plan to pass back to larvaworld
        print(
            "*** Result mean_rate={} mean_v={} mean_I={} mean_gIa={}".format(
                mean_rate, mean_voltage, mean_current, mean_adaptation
            )
        )
        return (mean_rate, mean_voltage, mean_current, mean_adaptation)

    return 0, 0, 0, 0
