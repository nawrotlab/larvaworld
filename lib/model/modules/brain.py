import numpy as np

from lib.model.modules.basic import Oscillator_coupling
from lib.model.modules.crawler import Crawler
from lib.model.modules.feeder import Feeder
from lib.model.modules.intermitter import Intermitter
from lib.model.modules.memory import RLmemory, SimpleMemory
from lib.model.modules.olfactor import Olfactor
from lib.model.modules.turner import Turner

class Brain():
    def __init__(self, agent, modules, conf):
        self.agent = agent
        self.modules = modules
        self.conf = conf
        self.olfactory_activation = 0
        # self.crawler, self.turner, self.feeder, self.olfactor, self.intermitter = None, None, None, None, None

    def sense_odors(self, pos):
        cons = {}
        for id, layer in self.agent.model.odor_layers.items():
            v = layer.get_value(pos)
            cons[id] = v + np.random.normal(scale=v * self.olfactor.noise)
        return cons


class DefaultBrain(Brain):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dt = self.agent.model.dt
        m=self.modules
        c=self.conf

        self.coupling = Oscillator_coupling(**c['interference_params']) if m['interference'] else Oscillator_coupling()
        self.crawler = Crawler(dt=dt, **c['crawler_params']) if m['crawler'] else None
        self.turner = Turner(dt=dt, **c['turner_params']) if m['turner'] else None
        self.feeder = Feeder(dt=dt, model=self.agent.model, **c['feeder_params']) if m['feeder'] else None
        self.intermitter = Intermitter(dt=dt, crawler=self.crawler, feeder=self.feeder,
                                           **c['intermitter_params']) if m['intermitter'] else None
        o=self.olfactor = Olfactor(dt=dt, **c['olfactor_params']) if m['olfactor'] else None
        # self.memory = SimpleMemory(brain=self, dt=dt, decay_coef=o.decay_coef, gain=o.gain, **c['memory_params']) if m['memory'] else None
        self.memory = RLmemory(brain=self, dt=dt, decay_coef=o.decay_coef, gain=o.gain, **c['memory_params']) if m['memory'] else None

    def run(self, pos):
        if self.intermitter:
            self.intermitter.step()

        # Step the feeder
        feed_motion = self.feeder.step() if self.feeder else False

        if self.memory:
            reward=self.agent.food_detected is not None
            self.olfactor.gain, self.olfactor.decay_coef = self.memory.step(self.olfactor.get_gain(),
                                                                            self.olfactor.get_dCon(),
                                                                            reward,
                                                                            self.olfactor.decay_coef)
        lin = self.crawler.step(self.agent.sim_length) if self.crawler else 0
        self.olfactory_activation = self.olfactor.step(self.sense_odors(pos)) if self.olfactor else 0
        # print(np.round(list(self.sense_odors(pos).values()),4))
        # ... and finally step the turner...
        ang = self.turner.step(inhibited=self.coupling.step(crawler=self.crawler, feeder=self.feeder),
                               attenuation=self.coupling.attenuation,
                               A_olf=self.olfactory_activation) if self.turner else 0

        return lin, ang, feed_motion
