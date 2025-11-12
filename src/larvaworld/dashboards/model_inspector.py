"""
Interactive dashboard to inspect any model configuration available
"""

from __future__ import annotations
from typing import Any

import holoviews as hv
import pandas as pd
import panel as pn
from holoviews.streams import Buffer
from panel.template import DarkTheme


from larvaworld.lib import reg
from larvaworld.lib import util
from larvaworld.lib.model import DefaultBrain, Effector
from larvaworld.lib.model import moduleDB as MD
from larvaworld.lib.param import class_objs

__all__: list[str] = [
    "model_inspector_app",
]

w, h = 800, 500
MODULES = MD.LocoModsBasic


class ModelInspector:
    def __init__(self) -> None:
        self.running = False
        self.t = 0
        self.mID = None

    def build(self, mID: str):
        if mID != self.mID:
            self.reset()
            if hasattr(self, "brain"):
                del (
                    self.brain,
                    self.inspect,
                    self.streams,
                    self.dmaps,
                    self.collector,
                    self.plot,
                    self.attrs,
                )
            self.mID = mID
            m = CT.getID(self.mID)

            self.brain = DefaultBrain(conf=m.brain, dt=0.1)
            self.inspect = self.switch(self.brain)
            self.prepare_streams()
        return self.inspect

    def reset(self) -> None:
        self.running = False
        self.t = 0

    def initialize(self) -> None:
        self.running = True
        # return self.runner()

    def switch(self, brain: DefaultBrain):
        l = []
        L = brain.locomotor
        for k in MODULES:
            if k in MD.LocoMods:
                obj = getattr(brain.locomotor, k)
            elif k in MD.ids:
                obj = getattr(brain, k)
            if obj:
                A = obj.__class__
                try:
                    c = pn.Card(
                        pn.Param(
                            obj,
                            expand_button=True,
                            default_precedence=3,
                            show_name=False,
                            parameters=class_objs(
                                A, excluded=[Effector, "phi", "name"]
                            ).keylist,
                        ),
                        max_width=280,
                        margin=20,
                        header=pn.pane.Markdown(f"### {k} : {A.name}", align="center"),
                        header_background=MD.ModuleColorDict[k],
                    )
                    l.append(c)
                except:
                    pass
        return pn.GridBox(*l, ncols=2, nrows=2)

    def prepare_streams(self) -> None:
        self.collector = reg.par.output_reporters(ks=["A_T", "A_C"], agents=[self])
        self.attrs = ["x"] + self.collector.keylist
        df = pd.DataFrame({a: [] for a in self.attrs}, columns=self.attrs)

        self.streams = {}
        self.dmaps = []
        for a in self.collector.keylist:
            self.streams[a] = Buffer(df[["x", a]], length=100, index=False)
            self.dmaps.append(
                hv.DynamicMap(hv.Curve, streams=[self.streams[a]]).opts(
                    xlabel="time (sec)", ylabel=a, width=800, height=300
                )
            )

        self.plot = hv.Layout(self.dmaps).cols(1)

    def run(self, v: bool):
        if v:
            self.initialize()
        dt = self.brain.dt
        while self.running:
            lin, ang, feed_motion = self.brain.locomotor.step(A_in=0)
            df = pd.DataFrame(
                [
                    (
                        util.rgetattr(self, self.collector[a])
                        if a != "x"
                        else self.t * dt
                        for a in self.attrs
                    )
                ],
                columns=self.attrs,
            )
            for a in self.collector.keylist:
                self.streams[a].send(df[["x", a]])
            self.t += 1
            if self.t > 500:
                self.reset()

        return self.plot


inspector = ModelInspector()
CT = reg.conf.Model
Msel = pn.widgets.Select(value="explorer", name="larva-model", options=CT.confIDs)
Mrun = pn.widgets.Button(name="Run")

model_inspector_app: "pn.template.MaterialTemplate" = pn.template.MaterialTemplate(
    title="larvaworld : Larva Model inspector", theme=DarkTheme, sidebar_width=w
)

model_inspector_app.sidebar.append(
    pn.Column(pn.Row(Msel, Mrun, width=300, height=80), pn.bind(inspector.build, Msel))
)
model_inspector_app.main.append(pn.bind(inspector.run, Mrun))
model_inspector_app.servable()
