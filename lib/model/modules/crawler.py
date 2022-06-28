# import numpy as np
# from scipy import signal

# from lib.model.modules.basic import Oscillator, Effector
# from lib.registry.pars import preg


class Crawler:
    def __init__(self, dt,waveform, **kwargs):
        from lib.registry.pars import preg
        D = preg.larva_conf_dict
        # self.activity = 0
        # self.output = 0
        # self.noise = crawler_noise
        self.ef0 = D.mdicts2['crawler'].mode[waveform].class_func(**kwargs, dt=dt)
        # self.ef0.start_effector()


    def step(self, A_in=0):
        self.activity = self.ef0.step(A_in)
        self.activation = self.ef0.input
        return self.activity
