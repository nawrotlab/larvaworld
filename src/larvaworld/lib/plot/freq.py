"""
Frequency-related plotting
"""

from __future__ import annotations
from typing import Any, Optional, Sequence

import numpy as np
from scipy.fft import fftfreq

from .. import plot, util, funcs

__all__: list[str] = [
    "plot_fft_multi",
]


@funcs.graph("freq powerspectrum", required={"ks": ["v", "fov"]})
def plot_fft_multi(
    ks: Sequence[str] = ("v", "fov"),
    name: str = "frequency_powerspectrum",
    axx: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot FFT power spectra for multiple parameters.

    Creates Fourier analysis plots showing frequency power spectra and
    dominant frequency distributions for velocity and angular velocity,
    with inset probability histogram.

    Args:
        ks: Parameter keys to analyze. Defaults to ('v', 'fov') for forward
            and angular velocity
        name: Plot name for saving. Defaults to 'frequency_powerspectrum'
        axx: Inset axes for probability histogram. Auto-created if None
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_fft_multi(ks=['v', 'fov'], datasets=[d1, d2])
    """
    P = plot.AutoPlot(ks=ks, name=name, build_kws={"w": 15, "h": 12}, **kwargs)
    if axx is None:
        axx = P.axs[0].inset_axes([0.64, 0.65, 0.3, 0.25])
    palette = util.AttrDict(
        {
            "v": {
                "color": "green",
                "fr_range": (1.0, 2.5),
                "label": "forward speed",
            },
            "fov": {
                "color": "purple",
                "fr_range": (0.1, 0.8),
                "label": "angular speed",
            },
        }
    )
    plist = util.ItemList(palette.values())
    col0ks, lks, fr_rs = plist.color, plist.label, plist.fr_range

    prob_kws = {
        "bins": np.linspace(0, 2, 40),
        "alpha": 0.5,
        "ax": axx,
        "labels": P.labels,
    }

    xfs = [fftfreq(c.Nticks, c.dt)[: c.Nticks // 2] for c in P.datasets.config]
    dts = P.datasets.config.dt

    for i, k in enumerate(palette):
        colsk = [util.mix2colors(col, col0ks[i]) for col in P.colors]

        fsk = []
        for j, l in enumerate(P.labels):
            res = (
                P.dkdict[l][k]
                .groupby("AgentID")
                .apply(util.fft_max, dt=dts[j], fr_range=fr_rs[i], return_amps=True)
                .values
            )
            fsk.append([r[0] for r in res])
            yk = np.array([r[1] for r in res])

            plot.plot_quantiles(
                yk, x=xfs[j], axis=P.axs[0], label=lks[i], color=colsk[j]
            )
        plot.prob_hist(vs=fsk, colors=colsk, **prob_kws)
    P.conf_ax(
        0,
        ylim=(0, 8),
        xlim=(0, 3.5),
        ylab="Amplitude (a.u.)",
        xlab="Frequency (Hz)",
        title="Fourier analysis",
        titlefontsize=25,
        yMaxN=5,
    )
    P.data_leg(0, loc="upper left", labels=lks, colors=col0ks, fontsize=15)
    P.conf_ax(
        ax=axx,
        ylab="Probability",
        xlab="Dominant frequency (Hz)",
        yMaxN=2,
        major_ticklabelsize=10,
        minor_ticklabelsize=10,
    )
    if P.Ndatasets > 1:
        P.data_leg(0, loc="upper center", fontsize=15)
    return P.get()
