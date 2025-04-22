from larvaworld.lib.model.modules.sensor import Olfactor, OSNOlfactor, Thermosensor
import numpy as np


class LocalOlfactor(OSNOlfactor):
    def __init__(self, server_port=5795, remote_dt=100, remote_warmup=500, **kwargs):
        self.last_osn_activity = None
        print("**** LocalOlfactor ****")
        print(kwargs)
        super().__init__(
            response_key="OSN_rate",
            server_host="localhost",
            server_port=server_port,
            remote_dt=remote_dt,
            remote_warmup=remote_warmup,
            **kwargs,
        )

    def update(self):
        agent_id = (
            self.brain.agent.unique_id if self.brain is not None else self.agent_id
        )
        sim_id = self.brain.agent.model.id if self.brain is not None else self.sim_id
        # construct the payload with the data that gets sent to the model server
        # this includes the currently sensed concentration values from the environment
        msg_kws = {
            "odor_id": 0,
            # The concentration change :
            "concentration_mmol": list(self.input.values())[0]
            * 220,  # 1st ODOR concentration
            "concentration_change_mmol": self.first_odor_concentration_change,  # 1st ODOR concentration change
            "concentration_max_mmol": 440,  # in mMol
            "baseline_input_current": 100,  # in pA
            #'tau_Ia': 300,
            "g_Ia": 0,
            #'hill_coeff': 5.5,
            "max_rate": 100,
            "min_rate": 40,
        }

        response = self.brianInterface.executeRemoteModelStep(
            sim_id, agent_id, self.remote_dt, t_warmup=15000, **msg_kws
        )
        current_osn_activity = response.param(self.response_key)

        if self.last_osn_activity is None:
            self.last_osn_activity = current_osn_activity

        delta_response = (
            (current_osn_activity - self.last_osn_activity) / self.last_osn_activity
        ) * 100
        self.output = delta_response
        self.last_osn_activity = current_osn_activity
        # self.output = -scaled_response #response.param(self.response_key) #self.dt * np.sum([10 10 * * response.param(self.response_key) for id in self.gain_ids])
        print(
            f"LocalOlfactor output: rate: {response.param(self.response_key)} delta_response: {delta_response}"
        )
        # super().update()


class CustomBehaviorModule(Thermosensor):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    # use levy walker model conf as base model
