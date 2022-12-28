
from .ipc import Server, Client

class LarvaMessage(ipc.Message):
    def __init__(self, sim_id, model_id, **params):
        self.sim_id = sim_id
        self.model_id = model_id
        self.params = params

    def _get_args(self):
        return [self.sim_id, self.model_id], self.params

    def with_params(self, **params):
        return LarvaMessage(self.sim_id, self.model_id, **params)

    def param(self, key):
        try:
            return self.params[key]
        except Exception as e:
            return None