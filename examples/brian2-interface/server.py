"""server.py
model server example
Usage:
    server.py test [options]
    server.py [options]

Options:
    -p --port=<port>      The port number to communicate over [default: 5795]
    -h --host=<host>      The host name to communicate over [default: localhost]
    -s --socket=<socket>  A socket file to use, instead of a host:port
    -N --threads=<int>    Number of parallel workers to execute models
"""

from concurrent.futures.thread import ThreadPoolExecutor
import docopt
from larvaworld.lib.ipc import Server, Client, BrianInterfaceMessage
from utils import print_time_summary
from server_runnable import execute_model
import traceback
from brian2 import *

N_threads = 1


def process_model(msg: BrianInterfaceMessage):
    """
    process a single BrianInterfaceMessage message
    :param msg:
    :return:
    """
    print(
        "*** process_model simId={} | modelId={} | step={} | params={}".format(
            msg.sim_id, msg.model_id, msg.step, msg.params
        )
    )

    # default simulation values
    odor_concentration = (
        msg.param("concentration_mmol")
        if msg.param("concentration_mmol") is not None
        else 450
    )
    T_sim = msg.param("T") if msg.param("T") else 100
    rng_seed = int(msg.param("seed")) if msg.param("seed") else 4242
    warmup_duration = msg.param("warmup") if msg.param("warmup") else 200
    odor_id = msg.param("odor_id") if msg.param("odor_id") is not None else 0
    A_max = msg.param("A_max") if msg.param("A_max") is not None else 450
    hill_coeff = msg.param("hill_coeff") if msg.param("hill_coeff") is not None else 1.5

    tau_Ia = (
        msg.param("tau_Ia") * ms if msg.param("hill_coeff") is not None else 5000 * ms
    )
    g_Ia = msg.param("g_Ia") * nS if msg.param("hill_coeff") is not None else 2.7 * nS
    concentration_max_mmol = (
        msg.param("concentration_max_mmol")
        if msg.param("concentration_max_mmol") is not None
        else 450
    )

    baseline_input_current = (
        msg.param("baseline_input_current") * pA
        if msg.param("baseline_input_current") is not None
        else 300 * pA
    )

    OSN_rate, OSN_voltage, OSN_current, OSN_adaptation = execute_model(
        id=msg.model_id,
        sim_id=msg.sim_id,
        step_id=msg.step,
        odor_id=odor_id,
        odor_concentration=odor_concentration,
        concentration_max_mmol=concentration_max_mmol,
        baseline_input_current=baseline_input_current,
        A_max=A_max,
        hill_coeff=hill_coeff,
        warmup_time=warmup_duration,
        T_sim=T_sim,
        rng_seed=rng_seed,
        tau_Ia=tau_Ia,
        g_Ia=g_Ia,
    )

    print_time_summary()
    return msg.with_params(
        OSN_rate=OSN_rate,
        OSN_voltage=OSN_voltage,
        OSN_current=OSN_current,
        OSN_adaptation=OSN_adaptation,
    )


def server_process_request(objects):
    """
    handler function to be executed for received messages
    :param objects:
    :return:
    """
    print(
        "*** processing {} LarvaMessage's N_workers={}".format(len(objects), N_threads)
    )
    with ThreadPoolExecutor(max_workers=N_threads) as executor:
        try:
            future = executor.map(process_model, objects)
            return future
        except Exception as e:
            print(f"ERROR in process_model:")
            print(e)
            traceback.print_exc()
            return []


# main entrypoint of the model server
# To run a self-test execute:
# python3 server.py test
if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    server_address = args["--socket"] or (args["--host"], int(args["--port"]))
    N_threads = int(args["--threads"]) if args["--threads"] is not None else 1

    if args["test"]:
        print(
            "** connecting to LarvaOSNServer at: {}:{}".format(
                server_address[0], server_address[1]
            )
        )
        testMessages = [
            BrianInterfaceMessage(
                1, 1, step_id=i, odor_id=i % 3, odor_concentration=1, T=100, warmup=300
            )
            for i in range(5)
        ]
        with Client(server_address) as client:
            print("*** sending TEST LavaMessages: {}".format(testMessages))
            response = client.send(testMessages)
            print("Response: {}".format(response))
    else:
        print(
            "** started LarvaOSNServer at: {}:{} N_workers={} - hit CTRL+C to stop".format(
                server_address[0], server_address[1], N_threads
            )
        )
        s = Server(server_address, server_process_request)
        s.allow_reuse_address = True
        s.serve_forever()
