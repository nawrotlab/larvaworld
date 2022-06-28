import numpy as np
from scipy import signal


class Effector:
    def __init__(self, dt, **kwargs):
        self.dt = dt
        self.t = 0
        self.total_t = 0
        # self.noise = noise
        self.effector = False
        self.ticks = 0
        self.total_ticks = 0

    def count_time(self):
        self.t += self.dt
        self.total_t += self.dt

    def count_ticks(self):
        self.ticks += 1
        self.total_ticks += 1

    def reset_ticks(self):
        self.ticks = 0


    def start_effector(self):
        self.effector = True

    def stop_effector(self):
        self.effector = False
        self.t = 0

    def active(self):
        return self.effector

    def reset(self):
        self.t = 0
        self.total_t = 0




class Oscillator(Effector):
    def __init__(self, initial_freq=None, initial_freq_std=0, random_phi=True, **kwargs):
        super().__init__(**kwargs)
        self.initial_freq = float(np.random.normal(loc=initial_freq, scale=initial_freq_std, size=1))
        self.freq = self.initial_freq
        self.complete_iteration = False
        self.iteration_counter = 0
        self.d_phi = 2 * np.pi * self.dt * self.freq
        self.timesteps_per_iteration = int(round((1 / self.freq) / self.dt))
        self.phi = np.random.rand() * 2 * np.pi if random_phi else 0

    def set_freq(self, v):
        self.freq = v
        if self.freq!=0 :
            self.timesteps_per_iteration = int(round((1 / self.freq) / self.dt))
        else :
            self.timesteps_per_iteration = None

    def get_freq(self, t):
        return self.freq

    def oscillate(self):
        super().count_time()
        self.phi += self.d_phi
        if self.phi >= 2 * np.pi:
            self.phi %= 2 * np.pi
            self.t = 0
            self.complete_iteration = True
            self.iteration_counter += 1
        else:
            self.complete_iteration = False

    def reset(self):
        self.t = 0
        self.total_t = 0
        self.phi = 0
        self.complete_iteration = False
        self.iteration_counter = 0

class StepModule:
    def __init__(self, initial_amp, amp_range=None,input_noise=0,output_noise=0, **kwargs):
        self.active = True
        self.initial_amp = initial_amp
        self.amp = initial_amp
        self.amp_range = amp_range
        self.input_noise = input_noise
        self.output_noise = output_noise
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

    def step(self,A_in=0, **kwargs):
        self.update()
        A=self.update_input(A_in)
        self.input = A* (1 + np.random.normal(scale=self.input_noise))
        if self.active :

            a= self.Act
        else :
            a=0
        self.output = a*(1 + np.random.normal(scale=self.output_noise))
        return self.output

    def update(self):
        pass

    def set_amp(self, v):
        self.amp = v

    def get_amp(self, t):
        return self.amp


class StepEffector(Effector, StepModule):
    def __init__(self, initial_amp=1, amp_range=None,input_noise=0,output_noise=0, **kwargs):
        Effector.__init__(self,**kwargs)
        # super(Effector, self).__init__(**kwargs)
        StepModule.__init__(self,initial_amp=initial_amp, amp_range=amp_range,input_noise=input_noise,output_noise=output_noise)
        # super(StepModule, self).__init__(initial_amp=initial_amp, amp_range=amp_range,input_noise=input_noise,output_noise=output_noise)
        self.start_effector()

    def update(self):
        if self.effector :
            self.active=True
            super().count_time()
        else :
            self.active = False




class StepOscillator(Oscillator, StepModule) :
    def __init__(self, initial_amp, amp_range=None,input_noise=0,output_noise=0, **kwargs):
        Oscillator.__init__(self,**kwargs)
        # super(Oscillator, self).__init__(**kwargs)
        StepModule.__init__(self,initial_amp=initial_amp, amp_range=amp_range,input_noise=input_noise,output_noise=output_noise)
        # super(StepModule, self).__init__(initial_amp=initial_amp, amp_range=amp_range,input_noise=input_noise,output_noise=output_noise)
        self.start_effector()

    def update(self):
        if self.effector :
            self.active=True
            self.complete_iteration = False
            super().oscillate()
        else :
            self.active = False


    @property
    def Act_Phi(self):
        return np.sin(self.phi)


class StrideOscillator(StepOscillator) :
    def __init__(self, stride_dst_mean=None, stride_dst_std=0.0, **kwargs):
        super().__init__(**kwargs)
        self.stride_dst_mean, self.stride_dst_std = [np.max([0.0, ii]) for ii in [stride_dst_mean, stride_dst_std]]
        self.step_to_length = self.new_stride

    @property
    def new_stride(self):
        return np.random.normal(loc=self.stride_dst_mean, scale=self.stride_dst_std)

    @property
    def Act(self):
        c = self.Act_coef
        Aphi = self.Act_Phi
        if self.complete_iteration:
            self.step_to_length = self.new_stride
        return self.freq * self.step_to_length * (1 + c*Aphi)



class GaussOscillator(StrideOscillator):
    def __init__(self, gaussian_window_std,**kwargs):
        super().__init__(**kwargs)

        self.gauss_w=signal.gaussian(360, std=gaussian_window_std * 360, sym=False)

    @property
    def Act_Phi(self):
        idx = [int(np.rad2deg(self.phi))]
        return self.gauss_w[idx]


class SquareOscillator(StrideOscillator):
    def __init__(self, square_signal_duty, **kwargs):
        super().__init__(**kwargs)
        self.square_signal_duty = square_signal_duty

    @ property
    def Act_Phi(self):
        return signal.square(self.phi, duty=self.square_signal_duty)

class PhaseOscillator(StrideOscillator):
        def __init__(self, max_vel_phase,max_scaled_vel, **kwargs):
            super().__init__(**kwargs)
            self.max_vel_phase = max_vel_phase
            self.max_scaled_vel = max_scaled_vel

        @property
        def Act_Phi(self):
            return np.cos(self.phi - self.max_vel_phase)

        @property
        def Act_coef(self):
            return self.max_scaled_vel



class ConEffector(StepEffector):

    def step00(self,**kwargs):
        super().count_time()
        return None



