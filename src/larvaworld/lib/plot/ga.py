"""
Genetic-algorithm optimization-progress plotting.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence

import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from .. import plot, funcs, reg

__all__: list[str] = [
    "ga_progress_plot",
]


def _ga_param_ylabel(k: str, module: Optional[str]) -> str:
    """
    Resolve a GA-optimized parameter's y-axis label via the parameter
    database, following the project's symbol/unit convention (e.g.
    turner input noise renders as the LaTeX-subscripted "In_T"), instead
    of the raw DataFrame column name.

    Tries, in order: `k` itself as a registry key; `brain.<module>.<k>`
    as a registered natural-language path (the convention every brain-
    module parameter in parDB.py uses, e.g. "brain.turner.input_noise");
    falling back to the bare column name if neither resolves (e.g. for a
    GA space over a parameter that has no dedicated database entry).
    """
    try:
        return reg.getPar(k=k, to_return="symunit")
    except Exception:
        pass
    if module is not None:
        try:
            return reg.getPar(p=f"brain.{module}.{k}", to_return="symunit")
        except Exception:
            pass
    return k


@funcs.graph("ga progress", required={"args": ["df"]})
def ga_progress_plot(
    df: pd.DataFrame,
    ks: Optional[Sequence[str]] = None,
    module: Optional[str] = None,
    name: str = "ga_progress",
    **kwargs: Any,
) -> Any:
    """
    Visualize genetic-algorithm optimization progress across generations.

    One panel per optimized parameter, plus a separate final "Fitness"
    panel -- kept apart from the optimized parameters, since fitness is
    the GA's own selection score, not a variable being tuned -- each
    showing the per-generation mean (with a std band across that
    generation's genomes), as produced by `GAlauncher.store_genomes()`.
    The x-axis uses integer-only generation ticks (a generation index has
    no meaningful fractional value), and the optimized-parameter y-axis
    labels are pulled from the parameter database (symbol + unit) instead
    of raw column names, so they read consistently with every other plot
    in the package -- see `_ga_param_ylabel`.

    Args:
        df: Per-generation genome DataFrame, as built by
            `GAlauncher.store_genomes()` -- one row per genome, with a
            "generation" column, one column per optimized parameter, and
            a "fitness" column.
        ks: Optimized-parameter column names to plot. Defaults to every
            df column other than "generation"/"fitness".
        module: Brain module the optimized parameters in `ks` belong to
            (e.g. "turner"), used to resolve y-axis labels via the
            parameter database's `brain.<module>.<k>` path when `k`
            itself isn't a registered key. Optional; labels fall back to
            the raw column name when unresolved.
        name: Plot name for saving. Defaults to "ga_progress".
        **kwargs: Additional arguments passed to AutoBasePlot.

    Returns:
        Plot output (figure object or None based on return_fig setting).

    Example:
        >>> fig = ga_progress_plot(
        ...     df, ks=["input_noise", "output_noise"], module="turner"
        ... )
    """
    if ks is None:
        ks = [c for c in df.columns if c not in ("generation", "fitness")]

    generation_counts = df.groupby("generation").size()
    Nagents = int(generation_counts.max())
    Ngenerations = int(df["generation"].nunique())

    panels = list(ks) + ["fitness"]
    P = plot.AutoBasePlot(
        name=name,
        build_kws={"N": len(panels), "Ncols": 1, "sharex": True, "w": 8, "h": 3},
        **kwargs,
    )

    for ax, k in zip(P.axs[:-1], ks):
        sns.lineplot(data=df, x="generation", y=k, marker="o", errorbar="sd", ax=ax)
        ax.set_ylabel(_ga_param_ylabel(k, module))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fitness_ax = P.axs[-1]
    sns.lineplot(
        data=df,
        x="generation",
        y="fitness",
        marker="o",
        errorbar="sd",
        ax=fitness_ax,
        color="firebrick",
    )
    fitness_ax.set_ylabel("Fitness")
    fitness_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fitness_ax.set_xlabel(f"Generation ({Nagents} larvae/generation)")

    P.fig.suptitle(
        "Genetic algorithm optimization progress\n"
        f"({Ngenerations} generations, {Nagents} larvae/generation)"
    )
    P.fig.tight_layout(rect=(0, 0, 1, 0.94))
    return P.get()
