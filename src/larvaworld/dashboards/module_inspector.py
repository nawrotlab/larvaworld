from __future__ import annotations
from typing import Any

import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
from panel.template import DarkTheme

import larvaworld.lib.model as model
import larvaworld.lib.param


__all__: list[str] = [
    "module_inspector_app",
]


def module_tester(M, temp):
    defaults = larvaworld.lib.param.class_defaults(M)
    args = defaults.keylist
    m0 = M()
    attrs = ["input", "activation", "phi", "output"]
    attrs = [a for a in attrs if hasattr(m0, a)]
    try:
        A_in_min, A_in_max = M.param["input_range"].default
    except:
        A_in_min, A_in_max = -1, 1
    A_in = pn.widgets.FloatSlider(name="input", start=A_in_min, end=A_in_max, value=0)

    c = pn.Param(
        M.param,
        expand_button=True,
        default_precedence=3,
        show_name=False,
        parameters=args,
        # widgets=widgets
    )

    N = 100
    trange = np.arange(N)

    # Interactive data pipeline
    def module_sampler(A_in):
        m = M()
        df = pd.DataFrame(columns=attrs, index=trange)
        for i in range(N):
            m.step(A_in=A_in)
            df.loc[i] = {k: getattr(m, k) for k in attrs}
        df.index *= m.dt
        return df

    title = M.__name__
    p2 = pn.GridBox(*[c.widget(arg) for arg in args], ncols=1)

    p1 = pn.Column(
        pn.pane.Markdown(f"### {title}", align="center"),
        A_in,
        p2,
        max_width=w,
        max_height=h,
    )

    plot = hvplot.bind(module_sampler, A_in).interactive()

    card = pn.Card(
        plot.hvplot(min_height=h).output().options(xlabel="time (sec)", ylabel="Units"),
        title=title,
    )

    temp.main.append(card)

    return p1


def bind_to_value(widget, temp):
    return pn.bind(module_tester, widget, temp)


w, h = 400, 500
w2 = int(w / 2) - 20
module_inspector_app: "pn.template.MaterialTemplate" = pn.template.MaterialTemplate(
    title="larvaworld : Behavioral Module inspector", theme=DarkTheme, sidebar_width=w
)

Ms = [
    model.Crawler,
    model.PhaseOscillator,
    model.NeuralOscillator,
    model.ConstantTurner,
    model.SinTurner,
]
Msel = pn.widgets.Select(name="module", options={MM.__name__: MM for MM in Ms})
module_inspector_app.sidebar.append(
    pn.Column(Msel, bind_to_value(Msel, temp=module_inspector_app))
)

module_inspector_app.servable()
