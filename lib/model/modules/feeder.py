from lib.model.modules.basic import Oscillator


class Feeder(Oscillator):
    def __init__(self, model, feed_radius, V_bite,
                 feeder_initial_freq=2, feeder_freq_range=[1, 3], **kwargs):
        super().__init__(initial_freq=feeder_initial_freq, freq_range=feeder_freq_range, **kwargs)
        self.model = model
        self.feed_radius = feed_radius
        self.V_bite = V_bite
        # self.feed_success = None
        self.stop_effector()

    def step(self):
        self.complete_iteration = False
        if self.effector:
            super().oscillate()
        return self.complete_iteration