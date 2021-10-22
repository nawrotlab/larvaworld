from nengo import *
import numpy as np
from nengo.networks import EnsembleArray

from lib.aux.dictsNlists import save_dict
from lib.model.modules.brain import Brain
from lib.model.modules.basic import Oscillator_coupling
from lib.model.modules.intermitter import NengoIntermitter
from lib.model.modules.sensor import WindSensor


class NengoBrain(Network, Brain):

    def __init__(self, agent, modules, conf, **kwargs):
        super().__init__(**kwargs)
        Brain.__init__(self, agent, modules, conf)
        dt = self.agent.model.dt
        m = self.modules
        c = self.conf
        self.food_feedback = False
        if m['feeder']:
            self.feeder = NengoEffector(**c['feeder_params'])
        self.turner = NengoEffector(**c['turner_params'])
        self.crawler = NengoEffector(**c['crawler_params'])
        self.osc_coupling = Oscillator_coupling(brain=self, **c['interference_params'])
        if m['intermitter']:
            self.intermitter = NengoIntermitter(dt=dt, brain=self, **c['intermitter_params'])
            self.intermitter.start_effector()
        else:
            self.intermitter = None
        self.build()
        self.sim = Simulator(self, dt=0.01, progress_bar=False)
        self.Nsteps = int(dt / self.sim.dt)

    def build(self):
        o = self.olfactor
        ws = self.windsensor
        a = self.agent
        N1, N2=50,10
        with self:
            if o is not None:
                N = o.Ngains
                odors = Node(o.get_X_values, size_in=N)

                olfMem = EnsembleArray(100, N, 2)
                dCon = EnsembleArray(200, N, 1, radius=0.01, intercepts=dists.Uniform(0, 0.1))
                dConOut = Ensemble(200, N, neuron_type=Direct())
                for i in range(N):
                    Connection(odors[i], olfMem.ensembles[i][0], transform=[[1]], synapse=0.01)
                    Connection(olfMem.ensembles[i][0], olfMem.ensembles[i][1], transform=1, synapse=1.0)
                    Connection(olfMem.ensembles[i][0], dCon.input[i], transform=1, synapse=0.1)
                    Connection(olfMem.ensembles[i][1], dCon.input[i], transform=-1, synapse=0.1)
                    Connection(dCon.ensembles[i], dConOut[i], transform=1, synapse=0.0)

                # Collect data for plotting
                self.p_odor = Probe(odors)
                self.p_change = Probe(dConOut)

            x = Ensemble(N1, 3, neuron_type=Direct())
            y = Ensemble(N1, 3, neuron_type=Direct())

            s1 = 1.0

            def linOsc(x):
                dr = 1 - x[0] ** 2 - x[1] ** 2
                s = 2 * np.pi * x[2] / 2
                if s > 0.1:
                    v = s1 * -x[1] * s + x[0] * dr + x[0]
                    n = s1 * x[0] * s + x[1] * dr + x[1]
                    return [v, n]
                else:
                    return [0, 1]

            def angOsc(x):
                dr = 1 - x[0] ** 2 - x[1] ** 2
                s = 2 * np.pi * x[2]
                if s > 0.1:
                    return [s1 * -x[1] * s + x[0] * dr + x[0],
                            s1 * x[0] * s + x[1] * dr + x[1]]
                else:
                    return [0, 1]

            def feeOsc(x):
                dr = 1 - x[0] ** 2 - x[1] ** 2
                s = 2 * np.pi * x[2]
                if s > 0.1:
                    return [s1 * -x[1] * s + x[0] * dr + x[0],
                            s1 * x[0] * s + x[1] * dr + x[1]]
                else:
                    return [0, 1]

            def oscillator_interference(x):
                coup = self.osc_coupling
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
                return np.abs(x) * 2 * self.crawler.step_to_length_mu

            def turner(x):
                return x * self.turner.get_amp(0)

            def feeder(x):
                if x > 0.99:
                    return 1
                else:
                    return 0

            Connection(x, x[:2], synapse=s1, function=linOsc)
            Connection(y, y[:2], synapse=s1, function=angOsc)

            linFrIn = Node(self.crawler.get_freq, size_out=1)
            angFrIn = Node(self.turner.get_freq, size_out=1)

            linFr = Ensemble(N2, 1, neuron_type=Direct())
            angFr = Ensemble(N2, 1, neuron_type=Direct())

            Connection(linFrIn, linFr)
            Connection(angFrIn, angFr)

            Connection(linFr, x[2])
            Connection(angFr, y[2])

            interference = Ensemble(N1, 3, neuron_type=Direct())
            Connection(x[0], interference[0], synapse=0)
            Connection(y[0], interference[1], synapse=0)

            Vs = Ensemble(N1, 3, neuron_type=Direct())
            Connection(interference, Vs, synapse=0.01, function=intermittency)

            linV = Node(size_in=1)
            angV = Node(size_in=1)

            Connection(Vs[0], linV, synapse=0, function=crawler)
            Connection(Vs[1], angV, synapse=0, function=turner)

            # Collect data for plotting
            self.p_speeds = Probe(Vs)
            self.p_linV = Probe(linV)
            self.p_angV = Probe(angV)

            if self.feeder is not None:
                z = Ensemble(N1, 3, neuron_type=Direct())
                Connection(z, z[:2], synapse=s1, function=feeOsc)
                feeFrIn = Node(self.feeder.get_freq, size_out=1)
                feeFr = Ensemble(N2, 1, neuron_type=Direct())
                Connection(feeFrIn, feeFr)
                Connection(feeFr, z[2])
                feeV = Node(size_in=1)
                Connection(z[0], interference[2], synapse=0)
                Connection(Vs[2], feeV, synapse=0, function=feeder)
                self.p_feeV = Probe(feeV)


                if self.food_feedback:
                    f_cur = Node(a.get_on_food, size_out=1)
                    f_suc = Node(a.get_feed_success, size_out=1)
                    Connection(f_cur, linFr)
                    Connection(f_suc, feeFr)
                    Connection(f_cur, linFr, synapse=s1, transform=-1)
                    Connection(f_cur, feeFr, synapse=s1, transform=1)
                    Connection(f_suc, feeFr, synapse=0.01, transform=1)
                    Connection(f_suc, linFr, synapse=0.01, transform=-1)


            if ws is not None:
                Ch = Node(ws.get_activation, size_out=1)

                LNa = Ensemble(N2, 1, neuron_type=Direct())
                LNb = Ensemble(N2, 1, neuron_type=Direct())
                Ha = Ensemble(N2, 1, neuron_type=Direct())
                Hb = Ensemble(N2, 1, neuron_type=Direct())
                B1 = Ensemble(N2, 1, neuron_type=Direct())
                B2 = Ensemble(N2, 1, neuron_type=Direct())
                Hunch = Ensemble(N2, 1, neuron_type=Direct())
                Bend = Ensemble(N2, 1, neuron_type=Direct())
                Connection(Ch, LNa, synapse=0.01, transform=1)
                Connection(Ch, LNb, synapse=0.01, transform=1)
                Connection(Ch, Hb, synapse=0.01, transform=0.3)
                Connection(Ch, B1, synapse=0.01, transform=1)
                Connection(Ch, B2, synapse=0.01, transform=0.3)
                Connection(LNa, LNb, synapse=0.01, transform=-1)
                Connection(LNb, LNa, synapse=0.01, transform=-1)
                Connection(LNa, B2, synapse=0.01, transform=-1)
                Connection(LNb, B1, synapse=0.01, transform=-1)
                Connection(LNa, B1, synapse=0.01, transform=-0.1)
                Connection(LNb, B2, synapse=0.01, transform=-0.1)
                Connection(LNa, Ha, synapse=0.01, transform=-0.2)
                Connection(LNb, Ha, synapse=0.01, transform=-0.2)
                Connection(LNa, Hb, synapse=0.01, transform=-0.2)
                Connection(LNb, Hb, synapse=0.01, transform=-0.2)
                Connection(Ha, LNa, synapse=0.01, transform=-0.2)
                Connection(Ha, LNb, synapse=0.01, transform=-0.2)
                Connection(Hb, LNa, synapse=0.01, transform=-0.6)
                Connection(Hb, LNb, synapse=0.01, transform=-0.6)
                Connection(B1, Ha, synapse=0.01, transform=0.1)
                Connection(B2, Ha, synapse=0.01, transform=0.1)
                Connection(B2, Hb, synapse=0.01, transform=0.1)
                Connection(B2, Hunch, synapse=0.01, transform=-0.3)
                Connection(B2, Bend, synapse=0.01, transform=0.3)
                Connection(B1, Hunch, synapse=0.01, transform=0.3)
                Connection(B1, Bend, synapse=0.01, transform=0.3)
                Connection(Hunch, linFr, synapse=0.0, transform=ws.weights['hunch_lin'])
                Connection(Hunch, angFr, synapse=0.0, transform=ws.weights['hunch_ang'])
                Connection(Bend, linFr, synapse=0.0, transform=ws.weights['bend_lin'])
                Connection(Bend, angFr, synapse=0.0, transform=ws.weights['bend_ang'])

            if True :
                if self.feeder is not None :
                    self.feed_probes = {k: Probe(v) for k, v in zip(['feeFrIn', 'feeFr', 'feeV'], [feeFrIn, feeFr, feeV])}
                    if self.food_feedback :
                        self.feed_probes.update({k: Probe(v) for k, v in zip(['f_cur', 'f_suc'], [f_cur, f_suc])})
                else:
                    self.feed_probes = {}
                if ws is not None :
                    self.anemo_probes = {k: Probe(v) for k, v in zip(['Ch', 'LNa', 'LNb', 'Ha', 'Hb', 'B1', 'B2', 'Bend', 'Hunch'],
                                                                     [Ch, LNa, LNb, Ha, Hb, B1, B2, Bend, Hunch])}
                else :
                    self.anemo_probes={}
                self.loco_probes = {k: Probe(v) for k, v in
                                    zip(['Vs', 'linV', 'angV', 'interference', 'angFr', 'linFr', 'linFrIn', 'angFrIn'],
                                        [Vs, linV, angV, interference, angFr, linFr, linFrIn, angFrIn])}
                self.dict = {
                    'anemotaxis':{k: [] for k in self.anemo_probes.keys()},
                    'locomotion':{k: [] for k in self.loco_probes.keys()},
                    'feeding':{k: [] for k in self.feed_probes.keys()},
                             }
            else :
                self.dict=None

    def update_dict(self, data):
        for n,m in zip([self.anemo_probes, self.loco_probes, self.feed_probes], ['anemotaxis', 'locomotion', 'feeding']) :
            for k, v in n.items() :
                self.dict[m][k].append(np.mean(data[v][-self.Nsteps:]))


    def mean_odor_change(self, data):
        if self.olfactor is not None :
            return np.mean(data[self.p_change][-self.Nsteps:], axis=0)[0]
        else :
            return 0

    def mean_lin_s(self, data):
        return np.mean(data[self.p_linV][-self.Nsteps:], axis=0)[0]

    def mean_ang_s(self, data):
        return np.mean(data[self.p_angV][-self.Nsteps:], axis=0)[0]

    def feed_event(self, data):
        if self.feeder is not None:
            return np.any(data[self.p_feeV][-self.Nsteps:] >= 1)
        else :
            return False

    def run(self, pos):
        l = self.agent.sim_length
        if self.olfactor:
            self.olfactor.X = self.sense_odors(pos)
        if self.windsensor:
            self.wind_activation = self.windsensor.step(self.sense_wind())
        self.intermitter.step()
        self.sim.run_steps(self.Nsteps, progress_bar=False)
        d = self.sim.data

        ang = self.mean_ang_s(d) + np.random.normal(scale=self.turner.noise)
        lin = self.mean_lin_s(d) * l + np.random.normal(scale=self.crawler.noise * l)
        feed = self.feed_event(d)
        self.olfactory_activation = 100 * self.mean_odor_change(d)
        if self.dict is not None :
            self.update_dict(d)
        self.sim.clear_probes()
        return lin, ang, feed

    def save_dicts(self, path):
        if self.dict is not None:
            save_dict(self.dict, f'{path}/{self.agent.unique_id}.txt', use_pickle=False)

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
