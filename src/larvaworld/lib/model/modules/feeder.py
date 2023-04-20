import param

from larvaworld.lib.model.modules.basic import Oscillator


class Feeder(Oscillator):
    feed_radius = param.Magnitude(default=0.05,label='feeding radius', doc='The accessible radius for a feeding motion as fraction of body length.')
    V_bite = param.Magnitude(default=0.001,label='mouthook capacity', doc='The volume of a feeding motion as fraction of body volume.')

    def __init__(self, initial_freq=2, freq_range=(1.0, 3.0),**kwargs):
        super().__init__(initial_freq=initial_freq, freq_range=freq_range,**kwargs)

    def step(self):
        self.complete_iteration=False
        if self.active :
            self.oscillate()
        return self.complete_iteration