import numpy as np
import param


class Timer(param.Parameterized) :
    dt = param.Number(default=0.1, label='timestep', doc='The timestep of the simulation in seconds.')
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ticks = 0
        self.total_ticks = 0
        # self.mode = mode

        self.active = True
        # self.ticks = 0
        # self.total_ticks = 0
        self.complete_iteration = False

    def count_time(self):
        self.ticks += 1
        self.total_ticks += 1

    @ property
    def t(self):
        return self.ticks * self.dt

    @property
    def total_t(self):
        return self.total_ticks * self.dt

    def reset(self):
        self.ticks = 0
        self.total_ticks = 0

    def start_effector(self):
        self.active = True

    def stop_effector(self):
        self.active = False
        self.ticks = 0




class Effector(Timer):
    input_noise = param.Number(default=0.0, label='input noise', doc='The noise applied at the input of the module.')
    output_noise = param.Number(default=0.0, label='output noise', doc='The noise applied at the output of the module.')
    input_range = param.List(label='input range',doc='The input range of the module.')
    output_range = param.List(label='output range',doc='The output range of the module.')

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


class Oscillator(Timer):
    initial_freq = param.Number(label='oscillation frequency', doc='The initial frequency of the oscillator.')
    freq_range = param.List(label='oscillation frequency range', doc='The frequency range of the oscillator.')
    random_phi = param.Boolean(default=True, label='random oscillation phase', doc='Whether to randomize the initial phase of the oscillator.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.initial_freq = float(np.random.normal(loc=initial_freq, scale=initial_freq_std, size=1))
        self.freq = self.initial_freq

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
        self.complete_iteration = False
        self.phi += 2 * np.pi * self.dt * self.freq
        if self.phi >= 2 * np.pi:
            self.phi %= 2 * np.pi
            # self.ticks = 0
            self.complete_iteration = True
            self.iteration_counter += 1


    def reset(self):
        # self.ticks = 0
        # self.total_ticks = 0
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0

    def phi_in_range(self, phi_range):
        return phi_range[0] < self.phi < phi_range[1]

    def update(self):
        self.complete_iteration = False

    # def act(self):
    #     self.oscillate()



class StepEffector(Effector):
    initial_amp = param.Number(default=1.0, allow_None=True, label='oscillation amplitude', doc='The initial amplitude of the oscillation.')
    amp_range = param.List(label='oscillation amplitude range', doc='The amplitude range of the oscillator.')

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

    # def update(self):
    #     self.complete_iteration = False


    def act(self):
        self.oscillate()
        self.output =self.Act



class SinOscillator(StepOscillator):

    @property
    def Act_Phi(self):
        return np.sin(self.phi)