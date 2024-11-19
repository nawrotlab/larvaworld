import datetime
from uuid import uuid4

from .... import vprint
from ...ipc import BrianInterfaceMessage, Client


class RemoteBrianModelInterface:
    # generates random agent id prefixed with current date + time
    # format: 2023-11-06_16-30_<randomId>
    @staticmethod
    def getRandomModelId():
        def numberToBase(n, b):
            if n == 0:
                return [0]
            digits = []
            while n:
                digits.append(int(n % b))
                n //= b
            return digits[::-1]

        rand_id = uuid4()
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")
        urlsafe_66_alphabet = (
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        short_id = "".join(
            urlsafe_66_alphabet[x % len(urlsafe_66_alphabet)]
            for x in numberToBase(rand_id.int, 66)
        )
        return "_".join([date_str, short_id])

    # remote_dt: duration of remote simulation per step in ms
    def __init__(self, server_host="localhost", server_port=5795, remote_dt=100):
        self.server_host = server_host
        self.server_port = server_port
        self.t_sim = int(remote_dt)
        self.step_cache = {}

    def executeRemoteModelStep(
        self, sim_id, model_instance_id, t_sim, t_warmup=0, **kwargs
    ):
        # t_sim: duration of remote model simulation in ms
        # warmup: duration of remote model warmup in ms
        if model_instance_id not in self.step_cache:
            self.step_cache[model_instance_id] = 0

        msg = BrianInterfaceMessage(
            sim_id,
            model_instance_id,
            self.step_cache[model_instance_id],
            T=t_sim,
            warmup=t_warmup,
            **kwargs,
        )
        # send model parameters to remote model server & wait for result response
        with Client((self.server_host, self.server_port)) as client:
            vprint(f"RemoteBrianModelInterface: BrianInterfaceMessage sent: {msg}")
            [response] = client.send([msg])  # this is a LarvaMessage object again
            self.step_cache[response.model_id] += 1
            vprint(
                f"RemoteBrianModelInterface: BrianInterfaceMessage received: {response}"
            )
            return response
