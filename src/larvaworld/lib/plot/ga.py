"""
Genetic-algorithm optimization-progress plotting.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence

import pandas as pd
import seaborn as sns

from .. import plot, funcs

__all__: list[str] = [
    "ga_progress_plot",
]


@funcs.graph("ga progress", required={"args": ["df"]})
def ga_progress_plot(
    df: pd.DataFrame,
    ks: Optional[Sequence[str]] = None,
    name: str = "ga_progress",
    **kwargs: Any,
) -> Any:
    """
    Visualize GA optimization progress across generations.

    One panel per optimized parameter plus a final fitness panel, each
    showing the per-generation mean (with a std band across the
    generation's genomes) as produced by GAlauncher.store_genomes().

    Args:
        df: Per-generation genome DataFrame, as built by
            GAlauncher.store_genomes() -- one row per genome, with a
            "generation" column, one column per optimized parameter, and
            a "fitness" column.
        ks: Optimized-parameter column names to plot. Defaults to every
            df column other than "generation"/"fitness".
        name: Plot name for saving. Defaults to 'ga_progress'
        **kwargs: Additional arguments passed to AutoBasePlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = ga_progress_plot(df, ks=["input_noise", "output_noise"])
    """
    if ks is None:
        ks = [c for c in df.columns if c not in ("generation", "fitness")]
    panels = list(ks) + ["fitness"]

    P = plot.AutoBasePlot(
        name=name,
        build_kws={"N": len(panels), "Ncols": 1, "sharex": True, "w": 8, "h": 3},
        **kwargs,
    )
    for ax, k in zip(P.axs, panels):
        sns.lineplot(data=df, x="generation", y=k, marker="o", errorbar="sd", ax=ax)
        ax.set_ylabel(k)
    P.axs[-1].set_xlabel("Generation")
    P.fig.tight_layout()
    return P.get()
