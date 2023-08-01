import param

from larvaworld.lib.model.modules.oscillator import Oscillator
from larvaworld.lib.param import PositiveNumber


class Feeder(Oscillator):
    freq = PositiveNumber(2.0,bounds =(1.0, 3.0))
    feed_radius = param.Magnitude(0.05,label='feeding radius', doc='The accessible radius for a feeding motion as fraction of body length.')
    V_bite = param.Magnitude(0.001,label='mouthook capacity', doc='The volume of a feeding motion as fraction of body volume.')

    def __init__(self, freq_range=(1.0, 3.0),**kwargs):
        super().__init__(freq_range=freq_range,**kwargs)
        self.stop_effector()

    def step(self):
        self.complete_iteration=False
        if self.active :
            self.oscillate()
        # return self.complete_iteration

    def suppresion_relief(self, phi_range):
        return self.phi_in_range(phi_range)