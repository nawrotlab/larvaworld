from __future__ import annotations
from typing import Any

import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
from panel.template import DarkTheme

import larvaworld.lib.model as model

__all__: list[str] = [
    "lateral_oscillator_app",
]

M = model.NeuralOscillator
module_attrs = ["input", "activation", "output"]
title = "Neural lateral oscillator"
sidebar_width, sidebar_height = 400, 500
widget_kws = {"type": pn.widgets.NumberInput, "width": int(sidebar_width / 2) - 20}
args1 = [
    "dt",
    "tau",
    "n",
    "m",
    "w_ee",
    "w_ce",
    "w_ec",
    "w_cc",
    "input_noise",
    "output_noise",
]

c = pn.Param(
    M.param,
    expand_button=True,
    default_precedence=3,
    show_name=False,
    widgets={
        "base_activation": {"type": pn.widgets.FloatSlider},
        "activation_range": {"type": pn.widgets.RangeSlider},
        **{arg: widget_kws for arg in args1},
    },
)

# Data and Widgets

A_in_min, A_in_max = M.param["input_range"].default
A_in = pn.widgets.FloatSlider(name="input", start=A_in_min, end=A_in_max, value=0)

p2 = pn.GridBox(*[c.widget(arg) for arg in args1], ncols=2)

p1 = pn.Column(
    pn.pane.Markdown(f"### {title}", align="center"),
    A_in,
    c.widget("base_activation"),
    c.widget("activation_range"),
    p2,
    max_width=sidebar_width,
    max_height=sidebar_height,
)

N = 100
trange = np.arange(N)


# Interactive data pipeline
def module_tester(A_in):
    m = M()
    df = pd.DataFrame(columns=module_attrs, index=trange)
    for i in range(N):
        m.step(A_in=A_in)
        df.loc[i] = {k: getattr(m, k) for k in module_attrs}
    df.index *= m.dt
    return df


plot = hvplot.bind(module_tester, A_in).interactive()
lateral_oscillator_app: "pn.template.MaterialTemplate" = pn.template.MaterialTemplate(
    title="larvaworld : Lateral Oscillator module",
    theme=DarkTheme,
    sidebar_width=sidebar_width,
)
lateral_oscillator_app.sidebar.append(p1)
lateral_oscillator_app.main.append(
    pn.Card(
        plot.hvplot(min_height=sidebar_height)
        .output()
        .options(xlabel="time (sec)", ylabel="Neural units"),
        title=title,
    )
)
lateral_oscillator_app.servable()
