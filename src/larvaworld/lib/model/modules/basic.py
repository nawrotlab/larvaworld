import numpy as np
import param

from larvaworld.lib import aux
from larvaworld.lib.model.modules.oscillator import Timer, Oscillator
from larvaworld.lib.param import PositiveNumber


class Effector(Timer):
    input_noise = param.Magnitude(0.0, precedence=-3,label='input noise', doc='The noise applied at the input of the module.')
    output_noise = param.Magnitude(0.0, precedence=-3,label='output noise', doc='The noise applied at the output of the module.')
    input_range = param.Range(precedence=-3,label='input range',doc='The input range of the module.')
    output_range = param.Range(precedence=-3,label='output range',doc='The output range of the module.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = 0
        self.output = 0



    def update_output(self, output):
        return self.apply_noise(output, self.output_noise, self.output_range)


    def update_input(self,input):
        return self.apply_noise(input, self.input_noise, self.input_range)



    def apply_noise(self, value,noise=0, range=None):
        if type(value) in [int,float]:
            value *= (1 + np.random.normal(scale=noise))
            if range is not None and len(range)==2:
                A0, A1 = range
                if value > A1:
                    value = A1
                elif value < A0:
                    value = A0
        elif isinstance(value, dict) :
            for k,v in value.items() :
                value[k]=self.apply_noise(v, noise)
        else :
            pass
        return value

    def get_output(self, t):
        return self.output

    def update(self):
        pass

    def act(self,**kwargs):
        pass

    def inact(self,**kwargs):
        pass

    def step(self,A_in=0, **kwargs):
        self.input = self.update_input(A_in)
        self.update()
        if self.active :
            self.act(**kwargs)
        else :
            self.inact(**kwargs)
        self.output = self.update_output(self.output)
        return self.output



class StepEffector(Effector):
    initial_amp = PositiveNumber(1.0, allow_None=True, label='oscillation amplitude', doc='The initial amplitude of the oscillation.')
    amp_range = param.Range(label='oscillation amplitude range', doc='The amplitude range of the oscillator.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.amp = self.initial_amp


    @property
    def Act_coef(self):
        return self.amp

    @property
    def Act_Phi(self):
        return 1

    @property
    def Act(self):
        c = self.Act_coef
        Aphi = self.Act_Phi
        return c*Aphi


    def set_amp(self, v):
        self.amp = v

    def get_amp(self, t):
        return self.amp



    def act(self):
        self.output =self.Act

    def inact(self):
        self.output =0





class StepOscillator(Oscillator, StepEffector):


    def act(self):
        self.oscillate()
        self.output =self.Act



class SinOscillator(StepOscillator):


    @property
    def Act_Phi(self):
        return np.sin(self.phi)