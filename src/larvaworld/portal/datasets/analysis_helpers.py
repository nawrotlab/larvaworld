from __future__ import annotations

import os
from functools import lru_cache
from hashlib import sha256
from typing import Any, Callable

from larvaworld.lib.reg.graph import GraphRegistry


def hash_dataset_ids(dataset_ids: tuple[str, ...]) -> str:
    """Create a cache key from a tuple of dataset IDs."""
    data = "|".join(sorted(dataset_ids))
    return sha256(data.encode()).hexdigest()[:16]


def get_valid_plots_for_datasets(
    graph_registry: GraphRegistry,
    dataset_ids: list[str],
    datasets: list[Any],
    default_kwargs: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """
    For each graphgroup, determine which registered plot functions work for the given datasets.
    Uses try/except invocation (same as legacy ButtonGraphList.refresh_figs).
    Returns a dict mapping graphgroup_id -> list of valid (available) function IDs.

    `GraphRegistry.build_graphgroups()` returns `{group_id: [entry, ...]}`,
    each entry a `{"key": ..., "plotID": ..., "args": {...}}` dict -- the
    same `plotID` can appear multiple times per group under different
    "key"s/args (e.g. several "timeplot" entries for different `ks`), so
    availability is checked and reported per distinct `plotID` (deduplicated).
    """
    if default_kwargs is None:
        default_kwargs = {}

    valid_by_group: dict[str, list[str]] = {}
    graphgroups = graph_registry.build_graphgroups()

    for group_id, entries in graphgroups.items():
        seen_fids: set[str] = set()
        valid_fids: list[str] = []
        for entry in entries:
            fid = entry.get("plotID")
            if fid is None or fid in seen_fids or fid not in graph_registry.dict:
                continue
            seen_fids.add(fid)
            try:
                graph_registry.run(
                    fid,
                    datasets=datasets,
                    labels=dataset_ids,
                    return_fig=True,
                    **default_kwargs,
                )
                valid_fids.append(fid)
            except Exception:
                pass
        valid_by_group[group_id] = valid_fids

    return valid_by_group


def run_plot_for_datasets(
    graph_registry: GraphRegistry,
    plot_id: str,
    datasets: list[Any],
    dataset_ids: list[str],
    default_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Run a registered plot function and return the figure.

    Always passes `return_fig=True`. A plot function built on the common
    `plot.util.process_plot` finalization returns `(fig, save_to,
    filename)` in that case rather than saving the figure itself --
    `process_plot`'s own `return_fig` branch deliberately skips
    `save_plot()`, leaving persistence to the caller. Unpack that tuple
    here, save when `save_to` is given, and return a plain `fig` either way.
    """
    if default_kwargs is None:
        default_kwargs = {}
    result = graph_registry.run(
        plot_id,
        datasets=datasets,
        labels=dataset_ids,
        return_fig=True,
        **default_kwargs,
    )
    if isinstance(result, tuple) and len(result) == 3 and hasattr(result[0], "savefig"):
        fig, save_to, filename = result
        if save_to:
            from larvaworld.lib.plot.util import save_plot

            os.makedirs(save_to, exist_ok=True)
            save_plot(fig, os.path.join(save_to, filename), filename)
        return fig
    return result
