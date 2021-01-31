from nengo import *
import numpy as np
from nengo.networks import EnsembleArray
from numpy import pi


class FlyBrain(Network):

    def build(self, input_manager, olfactor=False, num_odor_layers=1, three_oscillators=False,
              intermitter=False, mousebrain_weights=None,
              turner_noise=None, turner_amp_range=None,
              turner_initial_amp=None,
              crawler_initial_amp=None, crawler_noise=None,
              scaled_stride_step=None, crawler_interference_free_window=None,
              feed_radius=None, **kwargs):
        Nodors = num_odor_layers
        self.scaled_stride_step = scaled_stride_step
        self.crawler_interference_free_window = crawler_interference_free_window

        with self:
            if olfactor:
                odors = Node(input_manager.get_odor_concentrations, size_in=Nodors)
                odor_memory = EnsembleArray(n_neurons=100, n_ensembles=Nodors, ens_dimensions=2)
                odor_change = EnsembleArray(n_neurons=200, n_ensembles=Nodors, ens_dimensions=1, radius=0.01,
                                            intercepts=dists.Uniform(0, 0.1))
                for i in range(num_odor_layers):
                    Connection(odors[i], odor_memory.ensembles[i][0], transform=[[1]], synapse=0.01)
                    Connection(odor_memory.ensembles[i][0], odor_memory.input[i][1], transform=1, synapse=1)
                    Connection(odor_memory.ensembles[i][0], odor_change.input[i], transform=1, synapse=0.01)
                    Connection(odor_memory.ensembles[i][1], odor_change.input[i], transform=-1, synapse=0.01)

                # Collect data for plotting
                self.p_odor = Probe(odors)
                # self.ens_probe = nengo.Probe(ens.output, synapse=0.01)
                self.p_change = Probe(odor_change)

            if three_oscillators:
                x = Ensemble(n_neurons=200, dimensions=3, neuron_type=Direct())
                y = Ensemble(n_neurons=200, dimensions=3, neuron_type=Direct())
                z = Ensemble(n_neurons=200, dimensions=3, neuron_type=Direct())
                synapse = 1.0
                # synapse=0.1

                # e = 10 ** (-6)

                def linear_oscillator(x):
                    dr = 1 - x[0] ** 2 - x[1] ** 2
                    s = 2 * pi * x[2]/2
                    if s > 0.1:
                        v = synapse * -x[1] * s + x[0] * dr + x[0]
                        n = synapse * x[0] * s + x[1] * dr + x[1]
                        return [v, n]
                    else:
                        return [0, 1]

                def angular_oscillator(x):
                    dr = 1 - x[0] ** 2 - x[1] ** 2
                    s = 2 * pi * x[2]
                    # How to make the oscillator return to baseline state(x[0]=0) when s= 0
                    if s > 0.1:
                        # return [synapse * -x[1] * s + x[0] * (r - x[0]**2 - x[1]**2) + x[0] - 0.5*x[0],
                        return [synapse * -x[1] * s + x[0] * dr + x[0],
                                synapse * x[0] * s + x[1] * dr + x[1]]
                    else:
                        return [0, 1]

                def feeding_oscillator(x):
                    dr = 1 - x[0] ** 2 - x[1] ** 2
                    s = 2 * pi * x[2]
                    if s > 0.1:
                        # return [synapse * -x[1] * s + x[0] * (r - x[0]**2 - x[1]**2) + x[0] - 0.5*x[0],
                        return [synapse * -x[1] * s + x[0] * dr + x[0],
                                synapse * x[0] * s + x[1] * dr + x[1]]
                    else:
                        return [0, 1]

                def oscillator_interference(x):
                    # To get 0.25 of the whole cycle we need to set the threshold at cos(45)=0.5253
                    # thr = input_manager.get_interference_threshold()
                    pars = input_manager.interference_pars
                    c0 = pars['crawler_interference_start']
                    cr = 1-pars['feeder_interference_free_window']/np.pi
                    f0 = pars['crawler_interference_start']
                    fr = 1-pars['feeder_interference_free_window']/np.pi
                    r = pars['interference_ratio']
                    if x[0] > cr or x[2] > fr:
                    # if c0 <= np.pi*(x[0]+1) <= c0 + cr or f0 <= np.pi*(x[2]+1)<=f0 + fr:
                        v = [x[0], 0, x[2]]
                        # v = [x[0], x[1] * r, x[2]]
                    else:
                        v = x
                    return v

                def crawler(x):
                    # print(x)
                    # s=input_manager.get_linear_freq(0)
                    # if s>0.1 :
                    #     # This fits the empirical curve AND correctly computes scaled_stride_step during analysis
                    #     v = (x + 1 + 1.0)/1.6 *input_manager.scaled_stride_step
                    #     return v
                    # else :
                    #     return 0
                    return np.abs(x)*input_manager.scaled_stride_step

                def turner(x):
                    amp = turner_initial_amp
                    return x * amp

                def feeder(x):
                    if x > 0.999:
                        return 1
                    else:
                        return 0

                Connection(x, x[:2], synapse=synapse, function=linear_oscillator)
                Connection(y, y[:2], synapse=synapse, function=angular_oscillator)
                Connection(z, z[:2], synapse=synapse, function=feeding_oscillator)
                linear_freq_node = Node(input_manager.get_linear_freq, size_out=1)
                angular_freq_node = Node(input_manager.get_angular_freq, size_out=1)
                feeding_freq_node = Node(input_manager.get_feeding_freq, size_out=1)

                linear_freq = Ensemble(n_neurons=50, dimensions=1, neuron_type=Direct())
                angular_freq = Ensemble(n_neurons=50, dimensions=1, neuron_type=Direct())
                feeding_freq = Ensemble(n_neurons=50, dimensions=1, neuron_type=Direct())

                Connection(linear_freq_node, linear_freq)
                Connection(angular_freq_node, angular_freq)
                Connection(feeding_freq_node, feeding_freq)

                Connection(linear_freq, x[2])
                Connection(angular_freq, y[2])
                Connection(feeding_freq, z[2])

                interference = Ensemble(n_neurons=200, dimensions=3, neuron_type=Direct())
                Connection(x[0], interference[0], synapse=0)
                Connection(y[0], interference[1], synapse=0)
                Connection(z[0], interference[2], synapse=0)

                speeds = Ensemble(n_neurons=200, dimensions=3, neuron_type=Direct())
                Connection(interference, speeds, synapse=0.01, function=oscillator_interference)

                linear_s = Node(size_in=1)
                angular_s = Node(size_in=1)
                feeding_s = Node(size_in=1)

                Connection(speeds[0], linear_s, synapse=0, function=crawler)
                Connection(speeds[1], angular_s, synapse=0, function=turner)
                Connection(speeds[2], feeding_s, synapse=0, function=feeder)

                # Collect data for plotting
                self.p_speeds = Probe(speeds)
                self.p_linear_s = Probe(linear_s)
                self.p_angular_s = Probe(angular_s)
                self.p_feeding_s = Probe(feeding_s)

                # self.p_lin_f_Node = Probe(linear_freq_node)
                # print(self.p_linear_s)

    def mean_odor_change(self, data, Nticks):
        c = data[self.p_change]
        # mean_c = c[-1]
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
        event = np.any(s[-Nticks:] == 1)
        return event


class NengoManager:

    def __init__(self, num_odor_layers=1, turner_initial_freq=None, turner_noise=0.0, turner_amp_range=None,
                 turner_initial_amp=None, turner_freq_range=None,
                 crawler_freq_range=None, crawler_initial_freq=None, crawler_initial_amp=None, crawler_noise=0.0,
                 scaled_stride_step=None,
                 # interference_free_window=None,
                 interference_pars=None,
                 feeder_freq_range=None, feeder_initial_freq=None, feeder_noise=0.0, **kwargs):
        self.state = 'wait'
        self.activation = 0
        self.odor_concentrations = np.zeros(num_odor_layers)

        self.interference_pars = interference_pars
        self.interference_threshold = 0
        self.linear_osc_amp = 1

        self.scaled_stride_step = scaled_stride_step

        self.active_crawler_freq = crawler_initial_freq
        self.active_turner_freq = turner_initial_freq
        self.active_feeder_freq = feeder_initial_freq

        self.crawler_freq = crawler_initial_freq
        self.turner_freq = turner_initial_freq
        self.feeder_freq = feeder_initial_freq

        self.crawler_freq_range = crawler_freq_range
        self.turner_freq_range = turner_freq_range
        self.feeder_freq_range = feeder_freq_range

        self.crawler_noise = crawler_noise
        self.turner_noise = turner_noise
        self.feeder_noise = feeder_noise

    def add_value(self, value):
        self.activation += value

    def return_state(self, t):
        return self.state

    def return_activation(self, t):
        return self.activation

    def set_odor_concentrations(self, value):
        self.odor_concentrations = value

    def get_odor_concentrations(self, t):
        return self.odor_concentrations

    def get_frequencies(self, t):
        return [self.crawler_freq, self.turner_freq, self.feeder_freq]

    def get_linear_freq(self, t):
        return self.crawler_freq

    def get_angular_freq(self, t):
        return self.turner_freq

    def get_feeding_freq(self, t):
        return self.feeder_freq

    def set_active_crawler_freq(self, value):
        value = np.clip(value, self.crawler_freq_range[0], self.crawler_freq_range[1])
        self.active_crawler_freq = value

    def set_active_turner_freq(self, value):
        value = np.clip(value, self.turner_freq_range[0], self.turner_freq_range[1])
        self.active_turner_freq = value

    def set_active_feeder_freq(self, value):
        value = np.clip(value, self.feeder_freq_range[0], self.feeder_freq_range[1])
        self.active_feeder_freq = value

    def set_crawler_freq(self, value):
        self.crawler_freq = value

    def set_turner_freq(self, value):
        self.turner_freq = value

    def set_feeder_freq(self, value):
        self.feeder_freq = value

    def set_frequencies(self, values):
        self.set_crawler_freq(values[0])
        self.set_turner_freq(values[1])
        self.set_feeder_freq(values[2])

    def retrieve_active_frequencies(self, booleans):
        if booleans[0]:
            self.set_crawler_freq(self.active_crawler_freq)
        if booleans[1]:
            self.set_turner_freq(self.active_turner_freq)
        if booleans[2]:
            self.set_feeder_freq(self.active_feeder_freq)

    # Get the threshold of the (-)cosinusoidal oscillation over which the turner will be free
    # def get_interference_threshold(self):
    #     return self.interference_threshold

    # def set_interference_threshold(self, value):
    #     self.interference_threshold = value

    # Get the amplitude that will amplify the (-)cosinusoidal oscillation to get the linear speed
    def get_linear_osc_amp(self):
        return self.linear_osc_amp

    def set_linear_osc_amp(self, value):
        self.linear_osc_amp = value
