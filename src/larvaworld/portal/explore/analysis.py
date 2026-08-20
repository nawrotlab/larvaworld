"""Non-persistent analysis recipes for short Explore simulation previews."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "PreviewAnalysisResult",
    "PreviewFigure",
    "PreviewMetric",
    "SCENARIO_ANALYSIS_RECIPES",
    "build_preview_analysis",
    "close_preview_figures",
    "preview_enrichment",
    "render_preview_figure_images",
]


@dataclass(frozen=True)
class PreviewMetric:
    """One population-level metric shown next to an Explore playback."""

    label: str
    value: float
    unit: str = ""


@dataclass(frozen=True)
class PreviewFigure:
    """A preview plot before or after worker-side PNG rendering."""

    title: str
    figure: Any | None = None
    png: bytes | None = None


@dataclass
class PreviewAnalysisResult:
    """Transient plots, metrics, and non-fatal warnings for one preview."""

    metrics: list[PreviewMetric] = field(default_factory=list)
    figures: list[PreviewFigure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _PlotSpec:
    graph_id: str
    title: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _AnalysisRecipe:
    proc_keys: tuple[str, ...]
    plots: tuple[_PlotSpec, _PlotSpec]
    metric_kind: str
    dispersal: bool = False


def _plots(*plots: _PlotSpec) -> tuple[_PlotSpec, _PlotSpec]:
    if len(plots) != 2:
        raise ValueError("Every Explore analysis recipe must define exactly two plots.")
    return plots[0], plots[1]


SCENARIO_ANALYSIS_RECIPES: dict[str, _AnalysisRecipe] = {
    "dish": _AnalysisRecipe(
        proc_keys=("spatial", "angular"),
        plots=_plots(
            _PlotSpec("pathlength", "Path length"),
            _PlotSpec("timeplots", "Speed and turning", {"ks": ["sv", "fov"]}),
        ),
        metric_kind="locomotion",
    ),
    "dispersion": _AnalysisRecipe(
        proc_keys=("spatial",),
        plots=_plots(
            _PlotSpec("trajectories", "Trajectories"),
            _PlotSpec("dispersal", "Dispersal over time", {"range": (0, 40)}),
        ),
        metric_kind="dispersion",
        dispersal=True,
    ),
    "focus": _AnalysisRecipe(
        proc_keys=("spatial", "angular"),
        plots=_plots(
            _PlotSpec("pathlength", "Path length"),
            _PlotSpec("timeplots", "Body bend and velocity", {"ks": ["b", "sv"]}),
        ),
        metric_kind="locomotion",
    ),
    "chemotaxis": _AnalysisRecipe(
        proc_keys=("spatial", "angular", "source"),
        plots=_plots(
            _PlotSpec("trajectories", "Trajectories"),
            _PlotSpec(
                "timeplots",
                "Odor sensing and turning",
                {"ks": ["c_odor1", "dc_odor1", "A_olf"]},
            ),
        ),
        metric_kind="source",
    ),
    "chemorbit": _AnalysisRecipe(
        proc_keys=("spatial", "angular", "source"),
        plots=_plots(
            _PlotSpec("trajectories", "Trajectories"),
            _PlotSpec(
                "timeplots",
                "Odor sensing and turning",
                {"ks": ["c_odor1", "dc_odor1", "A_olf"]},
            ),
        ),
        metric_kind="source",
    ),
    "CS_UCS_off": _AnalysisRecipe(
        proc_keys=("PI",),
        plots=_plots(
            _PlotSpec("trajectories", "Trajectories"),
            _PlotSpec("preference_index", "Odor preference"),
        ),
        metric_kind="preference",
    ),
    "patchy_food": _AnalysisRecipe(
        proc_keys=("spatial", "angular", "source"),
        plots=_plots(
            _PlotSpec("trajectories", "Trajectories"),
            _PlotSpec("food intake (timeplot)", "Food intake"),
        ),
        metric_kind="foraging",
    ),
    "uniform_food": _AnalysisRecipe(
        proc_keys=("spatial", "angular", "source"),
        plots=_plots(
            _PlotSpec("trajectories", "Trajectories"),
            _PlotSpec("food intake (timeplot)", "Food intake"),
        ),
        metric_kind="foraging",
    ),
    "anemotaxis": _AnalysisRecipe(
        proc_keys=("spatial", "angular", "wind"),
        plots=_plots(
            _PlotSpec("trajectories", "Trajectories"),
            _PlotSpec("timeplots", "Wind response", {"ks": ["A_wind", "anemotaxis"]}),
        ),
        metric_kind="wind",
    ),
    "RvsS": _AnalysisRecipe(
        proc_keys=("spatial",),
        plots=_plots(
            _PlotSpec("pathlength", "Path length by phenotype"),
            _PlotSpec("food intake (timeplot)", "Food intake by phenotype"),
        ),
        metric_kind="phenotype",
    ),
    "maze": _AnalysisRecipe(
        proc_keys=("spatial", "angular", "source"),
        plots=_plots(
            _PlotSpec("trajectories", "Trajectories"),
            _PlotSpec("pathlength", "Path length"),
        ),
        metric_kind="maze",
    ),
}


def preview_enrichment(scenario_id: str) -> dict[str, Any]:
    """Return the smallest safe enrichment configuration for a scenario."""
    recipe = SCENARIO_ANALYSIS_RECIPES[scenario_id]
    config: dict[str, Any] = {
        "pre_kws": {},
        "proc_keys": list(recipe.proc_keys),
        "anot_keys": [],
        "mode": "minimal",
        "recompute": False,
        "tor_durs": [],
        "dsp_starts": [],
        "dsp_stops": [],
    }
    if recipe.dispersal:
        config.update({"dsp_starts": [0.0], "dsp_stops": [40.0]})
    return config


def _numeric_values(frame: Any, column: str) -> np.ndarray:
    try:
        values = np.asarray(frame[column], dtype=float)
    except (KeyError, TypeError, ValueError):
        return np.array([], dtype=float)
    return values[np.isfinite(values)]


def _last_per_agent(frame: Any, column: str) -> np.ndarray:
    try:
        values = frame[column].groupby(level="AgentID").last()
    except (KeyError, TypeError, ValueError, AttributeError):
        return np.array([], dtype=float)
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _endpoint_values(datasets: Sequence[Any], *columns: str) -> np.ndarray:
    for column in columns:
        values = [
            _numeric_values(dataset.e, column)
            for dataset in datasets
            if column in dataset.e
        ]
        values = [value for value in values if value.size]
        if values:
            return np.concatenate(values)
    return np.array([], dtype=float)


def _step_last_values(datasets: Sequence[Any], *columns: str) -> np.ndarray:
    for column in columns:
        values = [
            _last_per_agent(dataset.s, column)
            for dataset in datasets
            if column in dataset.s
        ]
        values = [value for value in values if value.size]
        if values:
            return np.concatenate(values)
    return np.array([], dtype=float)


def _xy_start_end(dataset: Any) -> tuple[np.ndarray, np.ndarray] | None:
    for x_col, y_col in (("x", "y"), ("centroid_x", "centroid_y")):
        if x_col not in dataset.s or y_col not in dataset.s:
            continue
        try:
            xy = dataset.s[[x_col, y_col]].dropna()
            first = xy.groupby(level="AgentID").first().values.astype(float)
            last = xy.groupby(level="AgentID").last().values.astype(float)
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if first.size and last.size:
            return first, last
    return None


def _displacements(datasets: Sequence[Any]) -> np.ndarray:
    values = []
    for dataset in datasets:
        xy = _xy_start_end(dataset)
        if xy is not None:
            values.extend(np.linalg.norm(xy[1] - xy[0], axis=1))
    return np.asarray(values, dtype=float)


def _source_distances(datasets: Sequence[Any]) -> np.ndarray:
    values = []
    for dataset in datasets:
        source_xy = getattr(dataset.config, "source_xy", {}) or {}
        if not source_xy:
            continue
        xy = _xy_start_end(dataset)
        if xy is None:
            continue
        source = np.asarray(next(iter(source_xy.values())), dtype=float)
        values.extend(np.linalg.norm(xy[1] - source, axis=1))
    return np.asarray(values, dtype=float)


def _add_metric(
    result: PreviewAnalysisResult,
    label: str,
    values: np.ndarray,
    *,
    unit: str = "",
    reducer: Any = np.mean,
) -> None:
    if values.size:
        result.metrics.append(PreviewMetric(label, float(reducer(values)), unit))
    else:
        result.warnings.append(f"{label} metric was unavailable for this preview.")


def _locomotion_metrics(datasets: Sequence[Any], result: PreviewAnalysisResult) -> None:
    _add_metric(
        result, "Mean path length", _endpoint_values(datasets, "cum_d"), unit="m"
    )
    _add_metric(
        result, "Mean speed", _numeric_values_all(datasets, "sv", "v"), unit="m/s"
    )


def _numeric_values_all(datasets: Sequence[Any], *columns: str) -> np.ndarray:
    for column in columns:
        values = [
            _numeric_values(dataset.s, column)
            for dataset in datasets
            if column in dataset.s
        ]
        values = [value for value in values if value.size]
        if values:
            return np.concatenate(values)
    return np.array([], dtype=float)


def _foraging_metrics(datasets: Sequence[Any], result: PreviewAnalysisResult) -> None:
    _add_metric(
        result,
        "Mean food intake",
        _step_last_values(datasets, "ingested_food_volume", "f_am"),
        unit="mg",
    )
    on_food = _endpoint_values(datasets, "on_food_tr")
    if not on_food.size:
        on_food = _numeric_values_all(datasets, "on_food")
    _add_metric(
        result, "Time on food", on_food, unit="%", reducer=lambda x: 100 * np.mean(x)
    )


def _preference_metrics(datasets: Sequence[Any], result: PreviewAnalysisResult) -> None:
    pis = []
    left, right = [], []
    for dataset in datasets:
        pi = getattr(dataset.config, "PI", {}) or {}
        if "PI" in pi:
            pis.append(float(pi["PI"]))
        xy = _xy_start_end(dataset)
        if xy is None:
            continue
        arena_dims = dataset.config.env_params.arena.dims
        threshold = 0.1 * float(arena_dims[0])
        xs = xy[1][:, 0]
        left.append(np.mean(xs <= -threshold))
        right.append(np.mean(xs >= threshold))
    _add_metric(result, "Preference index", np.asarray(pis), unit="")
    _add_metric(
        result,
        "Left occupancy",
        np.asarray(left),
        unit="%",
        reducer=lambda x: 100 * np.mean(x),
    )
    _add_metric(
        result,
        "Right occupancy",
        np.asarray(right),
        unit="%",
        reducer=lambda x: 100 * np.mean(x),
    )


def _wind_metrics(datasets: Sequence[Any], result: PreviewAnalysisResult) -> None:
    _add_metric(
        result, "Final anemotaxis", _endpoint_values(datasets, "anemotaxis"), unit="m"
    )
    bearing = _numeric_values_all(datasets, "bearing_to_wind")
    _add_metric(result, "Mean bearing to wind", np.abs(bearing), unit="deg")


def _phenotype_metrics(datasets: Sequence[Any], result: PreviewAnalysisResult) -> None:
    for dataset in datasets:
        label = str(getattr(dataset.config, "id", "Group"))
        _add_metric(
            result,
            f"{label} path length",
            _numeric_values(dataset.e, "cum_d"),
            unit="m",
        )
        _add_metric(
            result,
            f"{label} food intake",
            _last_per_agent(dataset.s, "ingested_food_volume"),
            unit="mg",
        )


def _build_metrics(
    scenario_id: str,
    datasets: Sequence[Any],
    result: PreviewAnalysisResult,
) -> None:
    kind = SCENARIO_ANALYSIS_RECIPES[scenario_id].metric_kind
    if kind == "locomotion":
        _locomotion_metrics(datasets, result)
    elif kind == "dispersion":
        _add_metric(
            result,
            "Median final displacement",
            _displacements(datasets),
            unit="m",
            reducer=np.median,
        )
        _add_metric(
            result, "Mean path length", _endpoint_values(datasets, "cum_d"), unit="m"
        )
    elif kind in {"source", "maze"}:
        label = (
            "Final distance to target" if kind == "maze" else "Final distance to source"
        )
        _add_metric(result, label, _source_distances(datasets), unit="m")
        _add_metric(
            result, "Mean path length", _endpoint_values(datasets, "cum_d"), unit="m"
        )
    elif kind == "preference":
        _preference_metrics(datasets, result)
    elif kind == "foraging":
        _foraging_metrics(datasets, result)
    elif kind == "wind":
        _wind_metrics(datasets, result)
    elif kind == "phenotype":
        _phenotype_metrics(datasets, result)


def _preference_index_figure(datasets: Sequence[Any]) -> Any:
    """Create the single-run PI chart that the multi-condition registry lacks."""
    from matplotlib.figure import Figure

    labels, values = [], []
    for dataset in datasets:
        pi = getattr(dataset.config, "PI", {}) or {}
        if "PI" not in pi:
            continue
        labels.append(str(getattr(dataset.config, "id", "group")))
        values.append(float(pi["PI"]))
    if not values:
        raise ValueError("Preference Index was not available for this preview.")
    figure = Figure(figsize=(5.5, 3.2), tight_layout=True)
    axis = figure.subplots()
    axis.bar(labels, values, color="#5c7f5a")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylim(-1, 1)
    axis.set_ylabel("Preference index")
    return figure


def _build_figure(spec: _PlotSpec, datasets: Sequence[Any]) -> Any:
    if spec.graph_id == "preference_index":
        figure = _preference_index_figure(datasets)
    else:
        from larvaworld.lib import reg

        figure = reg.graphs.run(
            spec.graph_id,
            datasets=datasets,
            save_to=None,
            show=False,
            **spec.kwargs,
        )
    return figure


def build_preview_analysis(
    scenario_id: str,
    datasets: Sequence[Any],
) -> PreviewAnalysisResult:
    """Build the two curated plots and population metrics for one preview."""
    recipe = SCENARIO_ANALYSIS_RECIPES[scenario_id]
    result = PreviewAnalysisResult()
    _build_metrics(scenario_id, datasets, result)
    for spec in recipe.plots:
        try:
            figure = _build_figure(spec, datasets)
            if figure is None:
                raise ValueError("The graph registry returned no figure.")
            result.figures.append(PreviewFigure(spec.title, figure))
        except Exception as exc:
            result.warnings.append(f"{spec.title} was unavailable: {exc}")
    return result


def render_preview_figure_images(
    result: PreviewAnalysisResult,
    *,
    dpi: int = 144,
) -> None:
    """Render and close preview figures in their creating worker thread.

    Matplotlib figures are not thread-safe. Explore creates them in its
    background worker, so passing the live figures to Panel would make the
    Bokeh document render objects owned by another thread. Converting them to
    immutable PNG bytes here keeps all Matplotlib work on one thread and makes
    result rendering a lightweight image update.
    """
    rendered: list[PreviewFigure] = []
    for item in result.figures:
        if item.png is not None:
            rendered.append(item)
            continue
        if item.figure is None:
            result.warnings.append(f"{item.title} had no renderable figure.")
            continue

        buffer = BytesIO()
        try:
            figure = item.figure
            figure.canvas.print_figure(
                buffer,
                format="png",
                facecolor=figure.get_facecolor(),
                edgecolor=figure.get_edgecolor(),
                dpi=dpi,
                bbox_inches="tight",
            )
            rendered.append(PreviewFigure(title=item.title, png=buffer.getvalue()))
        except Exception as exc:
            result.warnings.append(
                f"{item.title} could not be rendered: {type(exc).__name__}: {exc}"
            )
        finally:
            close_preview_figures([item])
    result.figures = rendered


def close_preview_figures(figures: Sequence[PreviewFigure]) -> None:
    """Release Matplotlib resources when leaving an Explore result page."""
    from matplotlib import pyplot as plt

    for item in figures:
        if item.figure is not None:
            plt.close(item.figure)
