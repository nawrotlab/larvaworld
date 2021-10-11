import numpy as np

from lib.model.modules.basic import Oscillator_coupling
from lib.model.modules.crawler import Crawler
from lib.model.modules.feeder import Feeder
from lib.model.modules.intermitter import Intermitter
from lib.model.modules.memory import RLOlfMemory, RLTouchMemory
from lib.model.modules.olfactor import Olfactor, Toucher
from lib.model.modules.turner import Turner


class Brain():
    def __init__(self, agent, modules, conf):
        self.agent = agent
        self.modules = modules
        self.conf = conf
        self.olfactory_activation = 0
        self.touch_activation = 0

        # self.crawler, self.turner, self.feeder, self.olfactor, self.intermitter = None, None, None, None, None

    def sense_odors(self, pos):
        cons = {}
        for id, layer in self.agent.model.odor_layers.items():
            v = layer.get_value(pos)
            cons[id] = v + np.random.normal(scale=v * self.olfactor.noise)
        return cons

    def sense_food(self):
        a=self.agent
        sensors = a.get_sensors()
        return {s: int(a.detect_food(a.get_sensor_position(s)) is not None) for s in sensors}
        # reward = np.mean(list(dic.values()))
        # if self.touch_dict is not None:
        #     dx = {s: dic[s] - self.touch_dict[s] for s in ss}
        # else:
        #     dx = {s: 0 for s in ss}
        # self.touch_dict = dic
        #
        # return dx, reward




class DefaultBrain(Brain):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dt = self.agent.model.dt
        m = self.modules
        c = self.conf

        self.coupling = Oscillator_coupling(**c['interference_params']) if m['interference'] else Oscillator_coupling()
        self.crawler = Crawler(dt=dt, **c['crawler_params']) if m['crawler'] else None
        self.turner = Turner(dt=dt, **c['turner_params']) if m['turner'] else None
        self.feeder = Feeder(dt=dt, model=self.agent.model, **c['feeder_params']) if m['feeder'] else None
        self.intermitter = Intermitter(brain=self, dt=dt, crawler=self.crawler, feeder=self.feeder,
                                       **c['intermitter_params']) if m['intermitter'] else None
        o = self.olfactor = Olfactor(brain=self, dt=dt, **c['olfactor_params']) if m['olfactor'] else None
        self.memory = RLOlfMemory(brain=self, dt=dt, gain=o.gain,
                                  **c['memory_params']) if (m['memory'] and c['memory_params']['mode'] =='olf') else None
        t=self.toucher = Toucher(brain=self, dt=dt, gain_dict={s:0.0 for s in self.agent.get_sensors()})
        self.touch_memory = RLTouchMemory(brain=self, dt=dt, gain=t.gain,
                                          **c['memory_params']) if (m['memory'] and c['memory_params']['mode'] =='touch') else None


    def run(self, pos):
        if self.intermitter:
            self.intermitter.step()

        # Step the feeder
        feed_motion = self.feeder.step() if self.feeder else False
        reward = self.agent.food_detected is not None
        if self.memory:
            self.olfactor.gain = self.memory.step(self.olfactor.get_dCon(),reward)
        lin = self.crawler.step(self.agent.sim_length) if self.crawler else 0
        self.olfactory_activation = self.olfactor.step(self.sense_odors(pos)) if self.olfactor else 0

        # print(np.round(list(self.sense_odors(pos).values()),4))
        # ... and finally step the turner...
        if self.touch_memory:
            self.toucher.gain=self.touch_memory.step(self.toucher.get_dCon(),reward)
        self.touch_activation = self.toucher.step(self.sense_food()) if self.toucher else 0
        ang = self.turner.step(inhibited=self.coupling.step(crawler=self.crawler, feeder=self.feeder),
                               attenuation=self.coupling.attenuation,
                               A_in=self.olfactory_activation + self.touch_activation) if self.turner else 0

        return lin, ang, feed_motion
