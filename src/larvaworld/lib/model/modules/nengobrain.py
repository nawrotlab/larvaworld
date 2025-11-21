from __future__ import annotations
from typing import Any
import numpy as np
from nengo import Connection, Direct, Ensemble, Network, Node, Probe, Simulator, dists
from nengo.networks import EnsembleArray

from ... import util
from . import Brain

__all__: list[str] = [
    "NengoBrain",
]


class NengoBrain(Network, Brain):
    def __init__(
        self,
        conf: Any,
        agent: Any | None = None,
        dt: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        Brain.__init__(self, conf=conf, agent=agent, dt=dt)
        self.food_feedback = False
        # self.locomotor = Locomotor(conf=conf,dt=self.dt)
        self.build()
        self.sim = Simulator(self, dt=0.01, progress_bar=False)
        self.Nsteps = int(self.dt / self.sim.dt)

    def build(self) -> None:
        o = self.olfactor
        ws = self.windsensor
        cra = self.locomotor.crawler
        tur = self.locomotor.turner
        fee = self.locomotor.feeder
        a = self.agent
        N1, N2 = 50, 10

        with self:
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
                    return [
                        s1 * -x[1] * s + x[0] * dr + x[0],
                        s1 * x[0] * s + x[1] * dr + x[1],
                    ]
                else:
                    return [0, 1]

            def feeOsc(x):
                dr = 1 - x[0] ** 2 - x[1] ** 2
                s = 2 * np.pi * x[2]
                if s > 0.1:
                    return [
                        s1 * -x[1] * s + x[0] * dr + x[0],
                        s1 * x[0] * s + x[1] * dr + x[1],
                    ]
                else:
                    return [0, 1]

            def oscillator_interference(x):
                coup = self.osc_coupling
                c0, c1 = coup.crawler_phi_range
                f0, f1 = coup.feeder_phi_range
                r = coup.attenuation
                if x[0] > 0 or x[2] > 0:
                    v = [x[0], 0, x[2]]
                else:
                    v = x
                return v

            def intermittency(x):
                s, f, p, r = self.locomotor.intermitter.active_bouts
                if s is None:
                    x[0] = 0
                elif s > 0:
                    x[1] *= 1 - x[0]
                if f is None:
                    x[2] = 0
                elif f > 0:
                    x[1] *= 1 - x[2]
                return x

            def crawler(x):
                if x <= 0:
                    return 0
                else:
                    return x * 2 * self.locomotor.crawler.get_amp(0)
                    # return x * 2 * self.locomotor.crawler.step_to_length_mu

            def turner(x):
                return x * self.locomotor.turner.get_amp(0)

            def feeder(x):
                if x > 0.99:
                    return 1
                else:
                    return 0

            if cra and tur:
                linFrIn = Node(cra.get_freq, size_out=1, label="Crawl Freq Stim")
                angFrIn = Node(tur.get_freq, size_out=1, label="Bend Freq Stim")

                linFr = Ensemble(N2, 1, neuron_type=Direct(), label="Crawl Freq")
                angFr = Ensemble(N2, 1, neuron_type=Direct(), label="Bend Freq")

                Connection(linFrIn, linFr)
                Connection(angFrIn, angFr)

                # raise
                x = Ensemble(N1, 3, neuron_type=Direct(), label="Crawl Osc")
                y = Ensemble(N1, 3, neuron_type=Direct(), label="Bend Osc")
                Connection(x, x[:2], synapse=s1, function=linOsc)
                Connection(y, y[:2], synapse=s1, function=angOsc)
                Connection(linFr, x[2])
                Connection(angFr, y[2])

                interference = Ensemble(
                    N1, 3, neuron_type=Direct(), label="Interference"
                )
                Connection(x[0], interference[0], synapse=0)
                Connection(y[0], interference[1], synapse=0)

                Vs = Ensemble(N1, 3, neuron_type=Direct(), label="Osc Vels")
                Connection(interference, Vs, synapse=0.01, function=intermittency)
                linV = Node(size_in=1, label="Crawl Vel")
                angV = Node(size_in=1, label="Bend Vel")

                Connection(Vs[0], linV, synapse=0, function=crawler)
                Connection(Vs[1], angV, synapse=0, function=turner)

                # Collect data for plotting
                self.p_speeds = Probe(Vs)
                self.p_linV = Probe(linV)
                self.p_angV = Probe(angV)

            if fee:
                feeFrIn = Node(fee.get_freq, size_out=1, label="Feed Freq Stim")
                feeFr = Ensemble(N2, 1, neuron_type=Direct(), label="Feed Freq")
                Connection(feeFrIn, feeFr)
                z = Ensemble(N1, 3, neuron_type=Direct(), label="Feed Osc")
                Connection(z, z[:2], synapse=s1, function=feeOsc)
                Connection(feeFr, z[2])
                feeV = Node(size_in=1, label="Feed Vel")
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

            if o is not None:
                N = len(o.gain)
                odors = Node(o.get_X_values, size_in=N, label="Olf Stim")

                olfMem = EnsembleArray(100, N, 2, label="Olf Mem")
                dCon = EnsembleArray(
                    200,
                    N,
                    1,
                    radius=0.01,
                    intercepts=dists.Uniform(0, 0.1),
                    label="Olf Perception",
                )
                dConOut = Ensemble(200, N, neuron_type=Direct(), label="Olf Modulation")
                for i in range(N):
                    Connection(
                        odors[i], olfMem.ensembles[i][0], transform=[[1]], synapse=0.01
                    )
                    Connection(
                        olfMem.ensembles[i][0],
                        olfMem.ensembles[i][1],
                        transform=1,
                        synapse=1.0,
                    )
                    Connection(
                        olfMem.ensembles[i][0], dCon.input[i], transform=1, synapse=0.1
                    )
                    Connection(
                        olfMem.ensembles[i][1], dCon.input[i], transform=-1, synapse=0.1
                    )
                    Connection(dCon.ensembles[i], dConOut[i], transform=10, synapse=0.0)
                    Connection(dConOut[i], angFr, transform=1, synapse=0.0)

                # Collect data for plotting
                self.p_odor = Probe(odors)
                self.p_change = Probe(dConOut)

            if ws is not None:
                Ch = Node(ws.get_output, size_out=1, label="Ch")
                LNa = Ensemble(N2, 1, neuron_type=Direct(), label="LNa")
                LNb = Ensemble(N2, 1, neuron_type=Direct(), label="LNb")
                Ha = Ensemble(N2, 1, neuron_type=Direct(), label="Ha")
                Hb = Ensemble(N2, 1, neuron_type=Direct(), label="Hb")
                B1 = Ensemble(N2, 1, neuron_type=Direct(), label="B1")
                B2 = Ensemble(N2, 1, neuron_type=Direct(), label="B2")
                Hunch = Ensemble(N2, 1, neuron_type=Direct(), label="Hunch Output")
                Bend = Ensemble(N2, 1, neuron_type=Direct(), label="Bend Output")

                ws_list = [
                    [Ch, LNa, 0.01, 1],
                    [Ch, LNb, 0.01, 1],
                    [Ch, Hb, 0.01, 0.3],
                    [Ch, B1, 0.01, 1],
                    [Ch, B2, 0.01, 0.3],
                    [LNa, LNb, 0.01, -1],
                    [LNb, LNa, 0.01, -1],
                    [LNa, B1, 0.01, -0.1],
                    [LNb, B1, 0.01, -1],
                    [LNa, B2, 0.01, -1],
                    [LNb, B2, 0.01, -0.1],
                    [LNa, Ha, 0.01, -0.2],
                    [LNb, Ha, 0.01, -0.2],
                    [LNa, Hb, 0.01, -0.2],
                    [LNb, Hb, 0.01, -0.2],
                    [Ha, LNa, 0.01, -0.2],
                    [Hb, LNa, 0.01, -0.6],
                    [Ha, LNb, 0.01, -0.2],
                    [Hb, LNb, 0.01, -0.6],
                    [B1, Ha, 0.01, 0.1],
                    [B1, Hb, 0.01, 0.1],
                    [B2, Hb, 0.01, 0.1],
                    [B1, Hunch, 0.01, 0.3],
                    [B1, Bend, 0.01, 0.3],
                    [B2, Hunch, 0.01, -0.3],
                    [B2, Bend, 0.01, 0.3],
                ]
                for i0, i1, syn, tr in ws_list:
                    Connection(i0, i1, synapse=syn, transform=tr)
                mode = "freq"
                if mode == "freq":
                    lin_target, ang_target = linFr, angFr
                elif mode == "vel":
                    lin_target, ang_target = linV, angV

                Connection(
                    Hunch, lin_target, synapse=0.0, transform=ws.weights["hunch_lin"]
                )
                Connection(
                    Hunch, ang_target, synapse=0.0, transform=ws.weights["hunch_ang"]
                )
                Connection(
                    Bend, lin_target, synapse=0.0, transform=ws.weights["bend_lin"]
                )
                Connection(
                    Bend, ang_target, synapse=0.0, transform=ws.weights["bend_ang"]
                )

            if True:
                D = util.AttrDict(
                    **{
                        k: Probe(v)
                        for k, v in zip(
                            ["Vs", "linV", "angV", "interference"],
                            [Vs, linV, angV, interference],
                        )
                    },
                    **{
                        k: Probe(v)
                        for k, v in zip(
                            ["angFr", "linFr", "linFrIn", "angFrIn"],
                            [angFr, linFr, linFrIn, angFrIn],
                        )
                    },
                )
                if fee is not None:
                    D.update(
                        {
                            k: Probe(v)
                            for k, v in zip(
                                ["feeFrIn", "feeFr", "feeV"], [feeFrIn, feeFr, feeV]
                            )
                        }
                    )
                    if self.food_feedback:
                        D.update(
                            {
                                k: Probe(v)
                                for k, v in zip(["f_cur", "f_suc"], [f_cur, f_suc])
                            }
                        )
                if ws is not None:
                    D.update(
                        {
                            k: Probe(v)
                            for k, v in zip(
                                [
                                    "Ch",
                                    "LNa",
                                    "LNb",
                                    "Ha",
                                    "Hb",
                                    "B1",
                                    "B2",
                                    "Bend",
                                    "Hunch",
                                ],
                                [Ch, LNa, LNb, Ha, Hb, B1, B2, Bend, Hunch],
                            )
                        }
                    )
                self.probes = D
                self.dict = {k: [] for k in D}
            else:
                self.dict = None

    def update_dict(self, data: Any) -> None:
        for k, p in self.probes.items():
            self.dict[k].append(np.mean(data[p][-self.Nsteps :], axis=0))

    def step(
        self, pos: Any, length: float, on_food: bool = False
    ) -> tuple[float, float, bool]:
        L = self.locomotor
        N = self.Nsteps
        MS = self.modalities
        kws = {"pos": pos}

        O = MS["olfaction"]
        if O.sensor:
            O.sensor.X = O.func(**kws)
        W = MS["windsensation"]
        if W.sensor:
            W.A = W.sensor.step(W.func(**kws))

        self.sim.run_steps(N, progress_bar=False)
        d = self.sim.data
        MS["olfaction"].A = (
            100 * np.mean(d[self.p_change][-N:], axis=0)[0] if O.sensor else 0
        )

        ang = np.mean(d[self.p_angV][-N:], axis=0)[0] * (
            1 + np.random.normal(scale=L.turner.output_noise)
        )
        lin = (
            np.mean(d[self.p_linV][-N:], axis=0)[0]
            * (1 + np.random.normal(scale=L.crawler.output_noise))
            * length
        )
        feed_motion = np.any(d[self.p_feeV][-N:] >= 1) if L.feeder else False
        L.step_intermitter(
            stride_completed=False, feed_motion=feed_motion, on_food=on_food
        )

        if self.dict is not None:
            self.update_dict(d)
        self.sim.clear_probes()
        return lin, ang, feed_motion

    def save_dicts(self, path: str) -> None:
        if self.dict is not None:
            util.save_dict(self.dict, f"{path}/{self.agent.unique_id}.txt")
