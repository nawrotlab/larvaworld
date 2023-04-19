import numpy as np
import param


class Effector(param.Parameterized):
    dt = param.Number(default=0.1, label='timestep', doc='The timestep of the simulation in seconds.')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active = False
        self.ticks = 0
        self.total_ticks = 0

    def count_time(self):
        self.ticks += 1
        self.total_ticks += 1

    @ property
    def t(self):
        return self.ticks * self.dt

    @property
    def total_t(self):
        return self.total_ticks * self.dt


    def start_effector(self):
        self.active = True

    def stop_effector(self):
        self.active = False
        self.ticks = 0



    def reset(self):
        self.ticks = 0
        self.total_ticks = 0

    # def update(self):
    #     if self.active :
    #         self.count_time()




class Oscillator(Effector):
    initial_freq = param.Number(label='oscillation frequency', doc='The initial frequency of the oscillator.')
    freq_range = param.NumericTuple(label='oscillation frequency range', doc='The frequency range of the oscillator.')
    random_phi = param.Boolean(default=True, label='random oscillation phase', doc='Whether to randomize the initial phase of the oscillator.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.initial_freq = float(np.random.normal(loc=initial_freq, scale=initial_freq_std, size=1))
        self.freq = self.initial_freq
        self.complete_iteration = False
        self.iteration_counter = 0
        # self.d_phi = 2 * np.pi * self.dt * self.freq
        # self.timesteps_per_iteration = int(round((1 / self.freq) / self.dt))
        self.phi = np.random.rand() * 2 * np.pi if self.random_phi else 0

    def set_freq(self, v):
        self.freq = v

    def get_freq(self, t):
        return self.freq

    @property
    def timesteps_per_iteration(self):
        if self.freq != 0:
            return int(round((1 / self.freq) / self.dt))
        else :
            return None

    def set_initial_freq(self, value):
        if self.freq_range:
            value = np.clip(value, self.freq_range[0], self.freq_range[1])
        self.initial_freq = value

    def oscillate(self):
        self.count_time()
        self.phi += 2 * np.pi * self.dt * self.freq
        if self.phi >= 2 * np.pi:
            self.phi %= 2 * np.pi
            self.ticks = 0
            self.complete_iteration = True
            self.iteration_counter += 1
        else:
            self.complete_iteration = False

    def reset(self):
        self.ticks = 0
        self.total_ticks = 0
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0

    def phi_in_range(self, phi_range):
        return phi_range[0] < self.phi < phi_range[1]

class StepEffector(Effector):
    initial_amp = param.Number(label='oscillation amplitude', doc='The initial amplitude of the oscillation.')
    amp_range = param.NumericTuple(label='oscillation amplitude range', doc='The amplitude range of the oscillator.')
    input_noise = param.Number(default=0.0, label='input noise', doc='The noise applied at the input of the module.')
    output_noise = param.Number(default=0.0, label='output noise', doc='The noise applied at the output of the module.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input = 0
        self.output = 0

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

    def update_input(self,A_in=0):
        return A_in

    def update_output(self):
        if self.active :
            a= self.Act
        else :
            a=0
        return a*(1 + np.random.normal(scale=self.output_noise))

    def step(self,A_in=0, **kwargs):
        self.input = self.update_input(A_in)* (1 + np.random.normal(scale=self.input_noise))
        self.update()
        self.output = self.update_output()
        return self.output


    def set_amp(self, v):
        self.amp = v

    def get_amp(self, t):
        return self.amp

    def update(self):
        if self.active:
            self.count_time()
# class StepEffector(Effector, StepModule):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)




class StepOscillator(Oscillator, StepEffector):

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     self.start_effector()

    @property
    def Act_Phi(self):
        return np.sin(self.phi)

    def update(self):
        self.complete_iteration = False
        if self.active:
            self.oscillate()

