from nengo import *
import numpy as np
from nengo.networks import EnsembleArray

from lib.model.modules.brain import Brain
from lib.model.modules.basic import Oscillator_coupling
from lib.model.modules.intermitter import NengoIntermitter


class NengoBrain(Network, Brain):

    def __init__(self, agent, modules, conf, **kwargs):
        super().__init__(**kwargs)
        Brain.__init__(self, agent, modules, conf)
        dt = self.agent.model.dt
        m = self.modules
        c = self.conf
        self.food_feedback = False
        self.osc_coupling = Oscillator_coupling(**c['interference_params'])
        self.feeder = NengoEffector(**c['feeder_params'])
        self.turner = NengoEffector(**c['turner_params'])
        self.crawler = NengoEffector(**c['crawler_params'])
        if m['olfactor']:
            self.olfactor = NengoEffector(**c['olfactor_params'])
        else:
            self.olfactor = None
        nm = self.nengo_manager = NengoManager(crawler=self.crawler,
                                               turner=self.turner,
                                               feeder=self.feeder,
                                               osc_coupling=self.osc_coupling,
                                               Nodors=self.agent.model.Nodors)
        if m['intermitter']:
            self.intermitter = NengoIntermitter(dt=dt, brain=self, nengo_manager=nm, **c['intermitter_params'])
            self.intermitter.start_effector()
        else:
            self.intermitter = None
        self.build()
        self.sim = Simulator(self, dt=0.01)
        self.Nsteps = int(dt / self.sim.dt)

    def build(self):
        m = self.nengo_manager
        with self:
            if self.olfactor is not None:
                N = m.Nodors
                odors = Node(m.get_odor_concentrations, size_in=N)

                odor_memory = EnsembleArray(n_neurons=100, n_ensembles=N, ens_dimensions=2)
                odor_change = EnsembleArray(n_neurons=200, n_ensembles=N, ens_dimensions=1, radius=0.01,
                                            intercepts=dists.Uniform(0, 0.1))
                for i in range(N):
                    Connection(odors[i], odor_memory.ensembles[i][0], transform=[[1]], synapse=0.01)
                    Connection(odor_memory.ensembles[i][0], odor_memory.ensembles[i][1], transform=1, synapse=1.0)
                    Connection(odor_memory.ensembles[i][0], odor_change.input[i], transform=1, synapse=0.1)
                    Connection(odor_memory.ensembles[i][1], odor_change.input[i], transform=-1, synapse=0.1)

                # Collect data for plotting
                self.p_odor = Probe(odors)
                # self.ens_probe = nengo.Probe(ens.output, synapse=0.01)
                self.p_change = Probe(odor_change.output)
                # self.p_change = odor_change.probes

            x = Ensemble(n_neurons=200, dimensions=3, neuron_type=Direct())
            y = Ensemble(n_neurons=200, dimensions=3, neuron_type=Direct())

            synapse = 1.0

            # synapse=0.1

            def linear_oscillator(x):
                dr = 1 - x[0] ** 2 - x[1] ** 2
                s = 2 * np.pi * x[2] / 2
                if s > 0.1:
                    v = synapse * -x[1] * s + x[0] * dr + x[0]
                    n = synapse * x[0] * s + x[1] * dr + x[1]
                    return [v, n]
                else:
                    return [0, 1]

            def angular_oscillator(x):
                dr = 1 - x[0] ** 2 - x[1] ** 2
                s = 2 * np.pi * x[2]
                if s > 0.1:
                    return [synapse * -x[1] * s + x[0] * dr + x[0],
                            synapse * x[0] * s + x[1] * dr + x[1]]
                else:
                    return [0, 1]

            def feeding_oscillator(x):
                dr = 1 - x[0] ** 2 - x[1] ** 2
                s = 2 * np.pi * x[2]
                if s > 0.1:
                    return [synapse * -x[1] * s + x[0] * dr + x[0],
                            synapse * x[0] * s + x[1] * dr + x[1]]
                else:
                    return [0, 1]

            def oscillator_interference(x):
                coup = m.osc_coupling
                c0, c1 = coup.crawler_phi_range
                # cr = 1 - coup.feeder_interference_free_window / np.pi
                f0, f1 = coup.feeder_phi_range
                # fr = 1 - coup.feeder_interference_free_window / np.pi
                r = coup.attenuation
                if x[0] > 0 or x[2] > 0:
                    v = [x[0], 0, x[2]]
                else:
                    v = x
                return v

            def intermittency(x):
                s, f, p = self.intermitter.active_bouts
                if s is None:
                    x[0] = 0
                elif s > 0:
                    x[1] *= 0.1
                if f is None:
                    x[2] = 0
                elif f > 0:
                    x[1] *= 0.1
                return x

            def crawler(x):
                return np.abs(x) * 2 * m.scaled_stride_step

            def turner(x):
                return x * m.turner.get_amp(0)

            def feeder(x):
                if x > 0.99:
                    return 1
                else:
                    return 0

            Connection(x, x[:2], synapse=synapse, function=linear_oscillator)
            Connection(y, y[:2], synapse=synapse, function=angular_oscillator)

            linear_freq_node = Node(m.crawler.get_freq, size_out=1)
            angular_freq_node = Node(m.turner.get_freq, size_out=1)

            linear_freq = Ensemble(n_neurons=50, dimensions=1, neuron_type=Direct())
            angular_freq = Ensemble(n_neurons=50, dimensions=1, neuron_type=Direct())

            Connection(linear_freq_node, linear_freq)
            Connection(angular_freq_node, angular_freq)

            Connection(linear_freq, x[2])
            Connection(angular_freq, y[2])

            interference = Ensemble(n_neurons=200, dimensions=3, neuron_type=Direct())
            Connection(x[0], interference[0], synapse=0)
            Connection(y[0], interference[1], synapse=0)

            speeds = Ensemble(n_neurons=200, dimensions=3, neuron_type=Direct())
            Connection(interference, speeds, synapse=0.01, function=intermittency)

            linear_s = Node(size_in=1)
            angular_s = Node(size_in=1)

            Connection(speeds[0], linear_s, synapse=0, function=crawler)
            Connection(speeds[1], angular_s, synapse=0, function=turner)

            # Collect data for plotting
            self.p_speeds = Probe(speeds)
            self.p_linear_s = Probe(linear_s)
            self.p_angular_s = Probe(angular_s)

            if self.feeder.initial_freq == 0:
                z = Ensemble(n_neurons=200, dimensions=3, neuron_type=Direct())
                Connection(z, z[:2], synapse=synapse, function=feeding_oscillator)
                feeding_freq_node = Node(m.feeder.get_freq, size_out=1)
                feeding_freq = Ensemble(n_neurons=50, dimensions=1, neuron_type=Direct())
                Connection(feeding_freq_node, feeding_freq)
                Connection(feeding_freq, z[2])
                feeding_s = Node(size_in=1)
                Connection(z[0], interference[2], synapse=0)
                Connection(speeds[2], feeding_s, synapse=0, function=feeder)
                self.p_feeding_s = Probe(feeding_s)

                if self.food_feedback:
                    f_cur = Node(m.get_food_detected, size_out=1)
                    f_suc = Node(m.get_feed_success, size_out=1)
                    Connection(f_cur, linear_freq)
                    Connection(f_suc, feeding_freq)
                    Connection(f_cur, linear_freq, synapse=1, transform=-1)
                    Connection(f_cur, feeding_freq, synapse=1, transform=1)
                    Connection(f_suc, feeding_freq, synapse=0.01, transform=1)
                    Connection(f_suc, linear_freq, synapse=0.01, transform=-1)

    def mean_odor_change(self, data, Nticks):
        c = data[self.p_change]
        mean_c = np.mean(c[-Nticks:], axis=0)[0]
        return mean_c

    def mean_lin_s(self, data, Nticks):
        s = data[self.p_linear_s]
        mean_s = np.mean(s[-Nticks:], axis=0)[0]
        return mean_s

    def mean_ang_s(self, data, Nticks):
        s = data[self.p_angular_s]
        mean_s = np.mean(s[-Nticks:], axis=0)[0]
        return mean_s

    def feed_event(self, data, Nticks):
        s = data[self.p_feeding_s]
        event = np.any(s[-Nticks:] >= 1)
        return event

    def run(self, pos):
        l = self.agent.sim_length
        N = self.Nsteps
        m = self.nengo_manager
        m.set_odor_concentrations(self.sense_odors(pos) if self.olfactor else {})
        m.set_food_detected(self.agent.food_detected)
        m.set_feed_success(self.agent.feed_success)
        self.intermitter.step()
        self.sim.run_steps(N, progress_bar=False)
        d = self.sim.data

        ang = self.mean_ang_s(d, N) + np.random.normal(scale=self.turner.noise)
        lin = self.mean_lin_s(d, N) * l + np.random.normal(scale=self.crawler.noise * l)
        feed = self.feed_event(d, N)
        self.olfactory_activation = 100 * self.mean_odor_change(d, N) if self.olfactor else 0
        return lin, ang, feed


class NengoManager:

    def __init__(self, crawler, turner, feeder, osc_coupling, Nodors=1):
        self.crawler = crawler
        self.turner = turner
        self.feeder = feeder
        self.osc_coupling = osc_coupling

        self.state = 'wait'
        self.activation = 0
        self.Nodors = Nodors
        self.odor_concentrations = np.zeros(Nodors)
        self.food_detected, self.feed_success = False, None

        self.scaled_stride_step = self.crawler.step_to_length_mu

    def add_value(self, value):
        self.activation += value

    def return_state(self, t):
        return self.state

    def return_activation(self, t):
        return self.activation

    def set_odor_concentrations(self, value):
        self.odor_concentrations = value

    def set_food_detected(self, value):
        prev = self.food_detected
        cur = 1 if value is not None else 0
        return cur - prev

    def get_food_detected(self, t):
        return self.food_detected

    def set_feed_success(self, value):
        if value == True:
            self.feed_success = 1
        elif value == False:
            self.feed_success = -1
        elif value is None:
            self.feed_success = 0

    def get_feed_success(self, t):
        return self.feed_success

    def get_odor_concentrations(self, t, N):
        return self.odor_concentrations


class NengoEffector:
    def __init__(self, initial_freq=None, default_freq=None, freq_range=None, initial_amp=None, amp_range=None,
                 noise=0.0, **kwargs):
        self.initial_freq = initial_freq
        self.freq = initial_freq
        if default_freq is None:
            default_freq = initial_freq
        self.default_freq = default_freq
        self.freq_range = freq_range
        self.initial_amp = initial_amp
        self.amp = initial_amp
        self.amp_range = amp_range
        self.noise = noise

        #     Todo get rid of this
        self.complete_iteration = False
        self.__dict__.update(kwargs)

    def get_freq(self, t):
        return self.freq

    def set_freq(self, v):
        self.freq = v

    def get_amp(self, t):
        return self.amp

    def set_amp(self, v):
        self.amp = v

    def set_default_freq(self, value):
        value = np.clip(value, self.freq_range[0], self.freq_range[1])
        self.default_freq = value

    def active(self):
        if self.freq != 0:
            return True
        else:
            return False
