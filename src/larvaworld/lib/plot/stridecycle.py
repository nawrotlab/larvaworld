"""
Stide-cycle-related plotting
"""

from __future__ import annotations
from typing import Any, Optional, Sequence

import os

import numpy as np

from .. import plot, reg, util, funcs
from ..util import nam

__all__: list[str] = [
    "plot_vel_during_strides",
    "stride_cycle",
    "stride_cycle_all_points",
    "plot_stride_Dbend",
    "plot_stride_Dorient",
    "plot_interference",
]


def plot_vel_during_strides(
    dataset: Any,
    use_component: bool = False,
    save_to: Optional[str] = None,
    return_fig: bool = False,
    show: bool = False,
) -> Any:
    """
    Plot linear and angular velocities during stride epochs.

    Creates two-panel figure showing how velocities vary across stride
    duration, with optional component velocity analysis.

    Args:
        dataset: Dataset containing stride epoch data
        use_component: Use component velocities instead of scalar. Defaults to False
        save_to: Directory to save plots. Uses dataset plot_dir if None
        return_fig: Whether to return figure object. Defaults to False
        show: Whether to display plot. Defaults to False

    Returns:
        Figure object if return_fig is True, else None

    Example:
        >>> fig = plot_vel_during_strides(dataset, use_component=True, return_fig=True)
    """
    chunk = "stride"
    D = dataset.epoch_dicts[chunk]

    Npoints = 64
    d = dataset

    if save_to is None:
        save_to = os.path.join(d.plot_dir, "plot_vel_during_strides")
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    save_as_lin = "linear_velocities_during_strides.pdf"
    save_as_ang = "angular_velocity_during_strides.pdf"
    filepath_lin = os.path.join(save_to, save_as_lin)
    filepath_ang = os.path.join(save_to, save_as_ang)
    filepaths = [filepath_lin, filepath_ang]

    svels = nam.scal(nam.vel(d.points))
    lvels = nam.scal(nam.lin(nam.vel(d.points[1:])))
    hov = nam.vel(nam.orient("front"))

    if use_component:
        lin_vels = lvels
    else:
        lin_vels = svels
    lin_vels = [lin_vels[0], lin_vels[int(len(lin_vels) / 2)], lin_vels[-1]]
    ang_vels = [hov]
    vels = [lin_vels, ang_vels]
    vels_list = lin_vels + ang_vels

    all_vel_timeseries = [[] for i in range(len(vels_list))]

    for id in d.config.agent_ids:
        for start, stop in D[id]:
            for i, vel in enumerate(vels_list):
                vel_timeserie = d.s.loc[(slice(start, stop), id), vel].values
                all_vel_timeseries[i].append(vel_timeserie)

    durations = [len(i) for i in all_vel_timeseries[0]]
    lin_vel_timeseries = all_vel_timeseries[:-1]
    ang_vel_timeseries = [[np.abs(a) for a in all_vel_timeseries[-1]]]
    vel_timeseries = [lin_vel_timeseries, ang_vel_timeseries]

    lin_cs = ["black", "seagreen", "mediumturquoise"]
    ang_cs = ["black"]
    cs = [lin_cs, ang_cs]
    lin_labels = [r"$\bf{head}$", r"$\bf{mid}$", r"$\bf{tail}$"]
    ang_labels = [r"$\dot{\theta}_{or}$"]
    labels = [lin_labels, ang_labels]
    ylabels = [r"scaled velocity $(sec^{-1})$", "angular velocity $(deg/sec)$"]

    from matplotlib import pyplot as plt

    for i in [0, 1]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for serie, vel, col, c, l in zip(
            vel_timeseries[i], vels[i], cs[i], cs[i], labels[i]
        ):
            array = [
                np.interp(
                    x=np.linspace(0, 2 * np.pi, Npoints),
                    xp=np.linspace(0, 2 * np.pi, dur),
                    fp=ts,
                    left=0,
                    right=0,
                )
                for dur, ts in zip(durations, serie)
            ]
            plot.plot_quantiles(df=array, axis=ax, color_mean=c, color=col, label=l)

        Nticks = 5
        ticks = np.linspace(0, Npoints - 1, Nticks)
        ax.set_xticks(ticks=ticks)
        ax.set_xticklabels(
            labels=[
                r"$0$",
                r"$\frac{\pi}{2}$",
                r"$\pi$",
                r"$\frac{3\pi}{2}$",
                r"$2\pi$",
            ]
        )
        ax.set_xlim([0, Npoints - 1])
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel(r"$\phi_{stride}$")
        l = ax.legend(loc="upper right")
        for j, text in enumerate(l.get_texts()):
            text.color = cs[i][j]
        plt.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95, wspace=0.01)
        fig.savefig(filepaths[i], dpi=300)
        print(f"Plot saved as {filepaths[i]}")


@funcs.graph("stride cycle", required={"ks": ["sv", "fov", "rov", "foa", "b"]})
def stride_cycle(
    name: Optional[str] = None,
    shorts: Sequence[str] = ("sv", "fov", "rov", "foa", "b"),
    modes: Optional[Sequence[str]] = None,
    subfolder: str = "stride",
    Nbins: int = 64,
    individuals: bool = False,
    pooled: bool = True,
    title: str = "Stride cycle analysis",
    **kwargs: Any,
) -> Any:
    """
    Create stride cycle curves for kinematic parameters.

    Generates multi-panel plot showing how velocity, angular velocity,
    and bend vary throughout the stride cycle phase (0 to 2π).

    Args:
        name: Plot name for saving. Auto-generated if None
        shorts: Parameter keys to plot. Defaults to velocity and angular parameters
        modes: Processing modes for each parameter. Auto-determined if None
        subfolder: Subfolder for saving. Defaults to 'stride'
        Nbins: Number of phase bins. Defaults to 64
        individuals: Plot individual trajectories. Defaults to False
        pooled: Show pooled quantiles. Defaults to True
        title: Figure title. Defaults to 'Stride cycle analysis'
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = stride_cycle(datasets=[d1, d2], individuals=True, Nbins=128)
    """
    if name is None:
        name = (
            "stride_cycle_curves_all_larvae" if individuals else "stride_cycle_curves"
        )

    Nsh = len(shorts)
    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        build_kws={"N": Nsh, "w": 8, "h": 5, "sharex": True},
        **kwargs,
    )

    x = np.linspace(0, 2 * np.pi, Nbins)
    for ii, sh in enumerate(shorts):
        if modes is None:
            mode = "abs" if sh == "sv" else "norm"
        else:
            mode = modes[ii]

        for l, d, c in P.data_palette:
            kws = {"label": l, "color": c}
            if individuals:
                try:
                    df = d.cycle_curves[sh][mode]
                    if pooled:
                        plot.plot_quantiles(df=df, axis=P.axs[ii], x=x, **kws)
                    else:
                        for j in range(df.shape[0]):
                            P.axs[ii].plot(x, df[j, :], color=c)
                        P.axs[ii].plot(x, np.nanquantile(df, q=0.5, axis=0), **kws)
                except:
                    pass
            else:
                try:
                    P.axs[ii].plot(x, np.array(d.pooled_cycle_curves[sh][mode]), **kws)
                except:
                    pass
        P.conf_ax(
            ii,
            xticks=np.linspace(0, 2 * np.pi, 5),
            xlim=[0, 2 * np.pi],
            xticklabels=[
                r"$0$",
                r"$\frac{\pi}{2}$",
                r"$\pi$",
                r"$\frac{3\pi}{2}$",
                r"$2\pi$",
            ],
            xlab=r"$\phi_{stride}$",
            ylab=reg.getPar(sh, to_return="symunit"),
            xvis=True if ii == Nsh - 1 else False,
        )
    P.axs[0].legend(loc="upper left", fontsize=15)
    P.conf_fig(
        title=title,
        title_kws={"w": "bold", "s": 20},
        align=True,
        adjust_kws={"BT": (0.1, 0.9), "LR": (0.2, 0.9), "H": 0.01},
    )
    return P.get()


@funcs.graph("stride cycle multi", required={"ks": ["sv", "fov", "rov", "foa", "b"]})
def stride_cycle_all_points(
    name: str = "stride cycle multi",
    idx: int = 0,
    Nbins: int = 64,
    short: Optional[str] = "fov",
    subfolder: str = "stride",
    maxNpoints: int = 5,
    axx: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Create stride cycle plot with all body points.

    Generates detailed stride cycle visualization showing angular velocities
    at multiple body points and orientation changes during strides.

    Args:
        name: Plot name for saving. Defaults to 'stride cycle multi'
        idx: Agent index to plot. Defaults to 0
        Nbins: Number of phase bins. Defaults to 64
        short: Parameter key to analyze. Defaults to 'fov'
        subfolder: Subfolder for saving. Defaults to 'stride'
        maxNpoints: Maximum body points to show. Defaults to 5
        axx: Inset axes for additional plot. Creates new if None
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = stride_cycle_all_points(datasets=[d1], idx=0, maxNpoints=7)
    """
    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        build_kws={"Nrows": 2, "w": 15, "h": 6, "sharex": True},
        **kwargs,
    )
    pi2 = 2 * np.pi
    x = np.linspace(0, pi2, Nbins)
    l, sv, fv, fov = reg.getPar(["l", "sv", "fv", "fov"])
    for d in P.datasets:
        s, e, c = d.data
        id = d.ids[idx]
        ss = s.xs(id, level="AgentID")
        D = d.chunk_dicts[id]
        strides = D.stride
        if short is not None:
            par, ylab1 = reg.getPar(short, to_return=["d", "l"])
            da = D.stride_Dor
            aa = util.stride_interp(ss[par].values, strides, Nbins)

            plot.plot_quantiles(
                df=np.vstack([aa[da > 0], -aa[da < 0]]),
                axis=P.axs[1],
                color="blue",
                x=x,
                label="experiment",
            )
        else:
            ylab1 = None

        points0 = nam.midline(c.Npoints, type="point")
        if c.Npoints > maxNpoints:
            points = (
                [points0[0]]
                + [
                    points0[2 + int(ii * (c.Npoints - 2) / (maxNpoints - 2))]
                    for ii in range(maxNpoints - 2)
                ]
                + [points0[-1]]
            )
        else:
            points = points0
        if len(points) == 5:
            pointcols = [
                "black",
                "darkblue",
                "darkgreen",
                "seagreen",
                "mediumturquoise",
            ]
        else:
            from matplotlib import cm

            pointcols = cm.rainbow(np.linspace(0, 1, len(points)))
        y0max = 0.7
        for p, col in zip(points, pointcols):
            v_p = nam.vel(p)
            a = (
                ss[v_p]
                if v_p in ss.columns
                else util.eudist(ss[nam.xy(p)].values) / c.dt
            )
            a = a / e[l].loc[id]
            aa = np.zeros([len(strides), Nbins])
            for ii, (s0, s1) in enumerate(strides):
                aa[ii, :] = np.interp(x, np.linspace(0, pi2, s1 - s0), a[s0:s1])
            aa_mu = np.nanquantile(aa, q=0.5, axis=0)
            aa_max = np.max(aa_mu)
            phi_max = x[np.argmax(aa_mu)]
            plot.plot_quantiles(df=aa, axis=P.axs[0], color=col, x=x, label=p)
            P.axs[0].axvline(
                phi_max,
                ymax=aa_max / y0max,
                color=col,
                alpha=1,
                linestyle="dashed",
                linewidth=2,
                zorder=20,
            )
            P.axs[0].scatter(
                phi_max,
                aa_max + 0.02 * y0max,
                color=col,
                marker="v",
                linewidth=2,
                zorder=20,
            )
        for i, ymax, ylab in zip(
            [0, 1], [y0max, None], [r"scaled velocity $(s^{-1})$", ylab1]
        ):
            P.conf_ax(
                i,
                ylim=(0, ymax),
                xlim=(0, pi2),
                ylab=ylab,
                xlab=r"$\phi_{stride}$",
                legfontsize=15,
                leg_loc="upper left",
                yMaxN=5,
                xticks=np.linspace(0, pi2, 5),
                xticklabels=[
                    r"$0$",
                    r"$\frac{\pi}{2}$",
                    r"$\pi$",
                    r"$\frac{3\pi}{2}$",
                    r"$2\pi$",
                ],
            )

        try:
            att = "attenuation"
            ps = [nam.max(f"phi_{nam.vel(p)}") for i, p in enumerate(points0)]
            aa = np.zeros([c.Npoints, c.N]) * np.nan
            for i, p in enumerate(ps):
                aa[i, :] = e[p].values - e[nam.max(f"phi_{att}")].values
            if axx is None:
                axx = P.axs[1].inset_axes([0.7, 0.7, 0.3, 0.25])
            axx.violinplot(aa.T, widths=0.9)
            axx.axhline(0, color="green", alpha=0.5, linestyle="dashed", linewidth=1)
            P.conf_ax(
                ax=axx,
                ylab=r"$\Delta\phi$",
                xlab="# point",
                xticks=np.arange(c.Npoints + 1),
                yticks=[-np.pi / 2, 0, np.pi / 2, np.pi],
                xticklabels=[None] + np.arange(1, c.Npoints + 1, 1).tolist(),
                yticklabels=[r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"],
                minor_ticklabelsize=12,
                major_ticklabelsize=12,
            )

        except:
            pass
    P.adjust((0.15, 0.9), (0.2, 0.9), 0.1, 0.15)
    return P.get()


@funcs.graph(
    "stride Dbend",
    required={
        "pars": [
            nam.at("bend", nam.start("stride")),
            nam.at("bend", nam.stop("stride")),
        ]
    },
)
def plot_stride_Dbend(
    name: str = "stride_bend_change",
    show_text: bool = False,
    subfolder: str = "stride",
    **kwargs: Any,
) -> Any:
    """
    Plot bend angle changes during strides.

    Creates scatter plot showing relationship between initial bend angle
    and bend change over stride, with linear regression fits.

    Args:
        name: Plot name for saving. Defaults to 'stride_bend_change'
        show_text: Show regression equation text. Defaults to False
        subfolder: Subfolder for saving. Defaults to 'stride'
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_stride_Dbend(datasets=[d1, d2], show_text=True)
    """
    P = plot.AutoPlot(name=name, subfolder=subfolder, **kwargs)
    ax = P.axs[0]
    b0p, b1p, bdp = nam.atStartStopChunk("bend", "stride")
    fits = {}
    for i, (d, l, c) in enumerate(zip(P.datasets, P.labels, P.colors)):
        if not util.SuperList([b0p, b1p]).exist_in(d.step_ps):
            d.track_par_in_chunk(chunk="stride", par="bend")
        b0 = d.get_par(b0p).dropna().values.flatten()[:500]
        b1 = d.get_par(b1p).dropna().values.flatten()[:500]
        sign_b = np.sign(b0)
        b0 *= sign_b
        b1 *= sign_b
        db = b1 - b0
        ax.scatter(x=b0, y=db, marker="o", s=2.0, alpha=0.6, color=c, label=l)
        m, k = np.polyfit(b0, db, 1)
        m = np.round(m, 2)
        k = np.round(k, 2)
        fits[l] = [m, k]
        ax.plot(b0, m * b0 + k, linewidth=4, color=c)
        if show_text:
            ax.text(
                0.3,
                0.9 - i * 0.1,
                rf"${l} : \Delta\theta_{{b}}={m} \cdot \theta_{{b}}$",
                fontsize=12,
                transform=ax.transAxes,
            )
            print(f"Bend correction during strides for {l} fitted as : db={m}*b + {k}")
    P.conf_ax(
        xlab=r"$\theta_{bend}$ at stride start $(deg)$",
        ylab=r"$\Delta\theta_{bend}$ over stride $(deg)$",
        yMaxN=5,
    )
    P.adjust((0.25, 0.95), (0.2, 0.95), 0.01)
    return P.get()


@funcs.graph("stride Dor", required={"ks": ["str_fo", "str_ro"]})
def plot_stride_Dorient(
    name: str = "stride_orient_change",
    absolute: bool = True,
    subfolder: str = "stride",
    Nbins: int = 200,
    **kwargs: Any,
) -> Any:
    """
    Plot orientation changes during strides.

    Creates histograms showing distribution of front and rear orientation
    changes over stride cycles.

    Args:
        name: Plot name for saving. Defaults to 'stride_orient_change'
        absolute: Use absolute values. Defaults to True
        subfolder: Subfolder for saving. Defaults to 'stride'
        Nbins: Number of histogram bins. Defaults to 200
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_stride_Dorient(datasets=[d1, d2], absolute=False, Nbins=150)
    """
    P = plot.AutoPlot(
        ks=["str_fo", "str_ro"],
        ranges=[80, 80],
        absolute=absolute,
        name=name,
        subfolder=subfolder,
        build_kws={"Ncols": "Nks"},
        **kwargs,
    )
    P.plot_hist(alpha=0.5, nbins=Nbins)
    P.adjust((0.12, 0.99), (0.2, 0.95), 0.01)
    return P.get()


@funcs.graph("interference", required={"ks": ["sv", "fov", "rov", "bv", "l"]})
def plot_interference(
    mode: str = "orientation",
    agent_idx: Optional[int] = None,
    subfolder: str = "interference",
    **kwargs: Any,
) -> Any:
    """
    Plot interference patterns between body segments.

    Creates scatter plots showing relationships between front/rear angular
    velocities, body length, and orientations to detect inter-segment interference.

    Args:
        mode: Analysis mode ('orientation' or other). Defaults to 'orientation'
        agent_idx: Specific agent index to analyze. Analyzes all if None
        subfolder: Subfolder for saving. Defaults to 'interference'
        **kwargs: Additional arguments passed to AutoPlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_interference(datasets=[d1, d2], mode='orientation', agent_idx=0)
    """
    name = (
        f"interference_{mode}"
        if agent_idx is None
        else f"interference_{mode}_agent_idx_{agent_idx}"
    )

    ks = ["sv"]
    if mode == "orientation":
        ks.append("fov")
    elif mode == "orientation_x2":
        ks.append("fov")
        ks.append("rov")
    elif mode == "bend":
        ks.append("bv")
    elif mode == "spinelength":
        ks.append("l")

    Npars = len(ks)

    pars, ylabs = reg.getPar(ks, to_return=["d", "l"])
    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        Nrows=Npars,
        figsize=(10, Npars * 5),
        sharex=True,
        **kwargs,
    )

    ylim = [0, 60] if mode in ["bend", "orientation", "orientation_x2"] else None

    for i, (p, ylab) in enumerate(zip(pars, ylabs)):
        for l, d, c in P.data_palette:
            df = d.read(f"stride.{p}")
            if agent_idx is not None:
                df = df.loc[c.agent_ids[agent_idx]].values
            else:
                df = df.values
            if mode in ["bend", "orientation"]:
                df = np.abs(df)
            Npoints = df.shape[1] - 1
            plot.plot_quantiles(df=df, axis=P.axs[i], color=c, label=l)
        P.conf_ax(
            i,
            ylab=ylab,
            ylim=ylim if i != 0 else [0.0, 0.6],
            yMaxN=4,
            leg_loc="upper right",
            xlab=r"$\phi_{stride}$",
            xlim=[0, Npoints],
            xticks=np.linspace(0, Npoints, 5),
            xticklabels=[
                r"$0$",
                r"$\frac{\pi}{2}$",
                r"$\pi$",
                r"$\frac{3\pi}{2}$",
                r"$2\pi$",
            ],
            xvis=True if i == Npars - 1 else False,
        )
    P.adjust((0.12, 0.95), (0.2 / Npars, 0.97), 0.05, 0.1)
    return P.get()
