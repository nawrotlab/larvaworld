import param

from ...param import PositiveNumber
from .oscillator import Oscillator

__all__ = [
    "Feeder",
]


class Feeder(Oscillator):
    freq = PositiveNumber(2.0, bounds=(1.0, 3.0))
    feed_radius = param.Magnitude(
        0.05,
        step=0.001,
        label="feeding radius",
        doc="The accessible radius for a feeding motion as fraction of body length.",
    )
    V_bite = param.Magnitude(
        0.001,
        step=0.0001,
        label="mouthook capacity",
        doc="The volume of a feeding motion as fraction of body volume.",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stop_effector()

    def step(self):
        self.complete_iteration = False
        if self.active:
            self.oscillate()
        # return self.complete_iteration

    def suppresion_relief(self, phi_range):
        return self.phi_in_range(phi_range)
