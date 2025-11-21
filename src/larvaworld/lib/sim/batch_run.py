from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .single_run import ExpRun

import itertools

import agentpy as ap
import numpy as np
import pandas as pd
import param

from ... import vprint
from .. import reg, util
from ..param import ClassAttr, NestedConf, PositiveInteger, PositiveNumber
from ..plot import plot_2d, plot_3pars, plot_heatmap_PI

__all__: list[str] = [
    "OptimizationOps",
    "BatchRun",
    "space_search_sample",
]


class OptimizationOps(NestedConf):
    """
    Configuration for optimization-guided parameter space search.

    Defines stopping criteria and optimization goals for batch runs
    that use optimization instead of grid search to explore parameter space.

    Attributes:
        fit_par: Fitness parameter name to optimize.
        minimize: If True, minimize fitness; if False, maximize.
        absolute: If True, use absolute values of fitness parameter.
        max_Nsims: Maximum number of simulations before halting.
        threshold: Target fitness value to reach.
        operator: Aggregation operator ('mean' or 'std') across agents.
    """

    fit_par = param.String(None, doc="The fitness parameter to be optimized")
    minimize = param.Boolean(True, doc="Whether to minimize the fitness parameter.")
    absolute = param.Boolean(
        False, doc="Whether to evaluate the absolute values of the parameter."
    )
    max_Nsims = PositiveInteger(
        5, label="max # simulations", doc="The maximum number of simulations to run."
    )
    threshold = PositiveNumber(0.001, doc="The threshold to reach during optimization.")
    operator = param.Selector(
        objects=["mean", "std"], doc="The operator to apply across agents"
    )

    def check(self, fits: np.ndarray) -> None:
        """
        Check optimization stopping criteria and log status.

        Evaluates whether to halt optimization based on maximum
        simulations reached or threshold achieved.

        Args:
            fits: Array of fitness values from completed simulations.

        Side Effects:
            Prints status message via vprint.
        """
        if fits.shape[0] >= self.max_Nsims:
            vprint("Maximum number of simulations reached. Halting search", 2)
        elif self.threshold_reached(fits):
            vprint("Best result reached threshold. Halting search", 2)
        else:
            vprint("Not reached threshold. Expanding space search", 2)
            pass

    def threshold_reached(self, fits: np.ndarray) -> bool:
        """
        Check if fitness threshold has been reached.

        Compares best fitness value (min or max based on optimization
        direction) against the target threshold.

        Args:
            fits: Array of fitness values from completed simulations.

        Returns:
            True if threshold reached, False otherwise.
        """
        if self.minimize:
            return np.nanmin(fits) <= self.threshold
        else:
            return np.nanmax(fits) >= self.threshold


class BatchRun(reg.generators.SimConfiguration, ap.Experiment):
    """
    Batch execution of parameter space search experiments.

    Runs multiple simulations with varying parameters to explore
    parameter space either via grid search or optimization-guided search.
    Extends agentpy.Experiment for parallel execution and result aggregation.

    Attributes:
        optimization: OptimizationOps configuration for guided search.

    Example:
        >>> batch_conf = reg.conf.Batch.getID('chemorbit')
        >>> batch = BatchRun(experiment='chemorbit', **batch_conf)
        >>> batch.simulate(n_jobs=4)
    """

    optimization = ClassAttr(OptimizationOps, doc="The optimization configuration")

    def __init__(
        self,
        experiment: str,
        space_search: dict[str, Any],
        id: str | None = None,
        space_kws: dict[str, Any] = {},
        exp: Any = None,
        exp_kws: dict[str, Any] = {},
        store_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Simulation mode 'Batch' launches a batch-run of a specified experiment type that performs a parameter-space search.

        Extends the agentpy.Experiment class.
        Controls the execution of multiple single simulations ('Exp' mode, see ExpRun class) with slightly different
        parameter-sets to cover a predefined parameter-space.

        Args:
            experiment: The preconfigured type of batch-run to launch.
            space_search: Dictionary that configures the parameter-space to be covered. Each entry is a parameter name
                         and the respective arguments:
                         - Direct value: param_name: value
                         - Discrete values: param_name: {'values': [v1, v2, ...]}
                         - Continuous range: param_name: {'range': (min, max)}
                         - Grid range: param_name: {'range': (min, max), 'Ngrid': N}
            id: Unique ID of the batch-run simulation. If not specified it is automatically set according to the
                batch-run type.
            space_kws: Additional arguments for the parameter-space construction.
            exp: The type of experiment for single runs launched by the batch-run.
            exp_kws: Additional arguments for the single runs.
            store_data: Whether to store batch-run results.
            **kwargs: Arguments passed to parent class (ap.Experiment).

        Example:
            >>> batch_conf = reg.conf.Batch.getID('chemorbit')
            >>> batch = BatchRun(experiment='chemorbit', **batch_conf)
            >>> batch.simulate(n_jobs=4)
        """
        # FIXME This causes error when the ap.Experiment.name is set to the model_class.name ('ExpRun')
        # See line 56 in site-packages/agentpy/experiment.py
        # Local import to avoid sim/__init__ indirection and potential cycles
        from .single_run import ExpRun  # type: ignore

        ap.Experiment.__init__(
            self,
            model_class=ExpRun,
            sample=space_search_sample(space_search, **space_kws),
            store_data=False,
            **kwargs,
        )

        reg.generators.SimConfiguration.__init__(
            self, runtype="Batch", experiment=experiment, id=id, store_data=store_data
        )
        self.df_path = f"{self.dir}/results.h5"
        self.exp_conf = reg.conf.Exp.expand(exp) if isinstance(exp, str) else exp
        self.exp_conf.update(**exp_kws)

        self.datasets = {}
        self.results = None
        self.figs = {}

    def _single_sim(self, run_id: Any):
        """Perform a single simulation."""
        i = 0 if run_id[0] is None else run_id[0]
        m = self.model(
            parameters=self.exp_conf.update_existingnestdict_by_suffix(self.sample[i]),
            _run_id=run_id,
            **self._model_kwargs,
        )
        self.datasets[i] = m.simulate(
            display=False, seed=self._random[run_id] if self._random else None
        )
        if "variables" in m.output and self.record is False:
            del m.output["variables"]  # Remove dynamic variables from record
        return m.output

    def default_processing(self, d: Any | None = None):
        P = self.optimization
        p = P.fit_par
        if p in d.end_ps:
            vals = d.e[p].values
        elif p in d.step_ps:
            vals = d.s[p].groupby("AgentID").mean()
        else:
            raise ValueError("Could not retrieve fit parameter from dataset")
        if P.absolute:
            vals = np.abs(vals)
        if P.operator == "mean":
            fit = np.mean(vals)
        elif P.operator == "std":
            fit = np.std(vals)
        else:
            raise ValueError("An operator must be set to True")
        return fit

    def end(self) -> None:
        self.par_df = self.output._combine_pars()
        self.par_names = self.par_df.columns.values.tolist()

        try:
            self.par_df["fit"] = [
                self.default_processing(self.datasets[i][0]) for i in self.par_df.index
            ]
            self.optimization.check(self.par_df["fit"].values)
        except:
            pass

    def simulate(self, **kwargs: Any):
        self.run(**kwargs)
        if "PI" in self.experiment:
            self.PI_heatmap()
        self.plot_results()
        util.storeH5(self.par_df, key="results", path=self.df_path, mode="w")
        return self.par_df, self.figs

    def plot_results(self) -> None:
        p_ns = self.par_names
        target_ns = [p for p in self.par_df.columns if p not in p_ns]
        kws = {"df": self.par_df, "save_to": self.plot_dir, "show": True}
        for t in target_ns:
            if len(p_ns) == 1:
                self.figs[f"{p_ns[0]}VS{t}"] = plot_2d(labels=p_ns + [t], pref=t, **kws)
            elif len(p_ns) == 2:
                self.figs.update(plot_3pars(vars=p_ns, target=t, pref=t, **kws))
            elif len(p_ns) > 2:
                for i, pair in enumerate(itertools.combinations(p_ns, 2)):
                    self.figs.update(
                        plot_3pars(vars=list(pair), target=t, pref=f"{i}_{t}", **kws)
                    )

    def PI_heatmap(self, **kwargs: Any) -> None:
        PIs = [self.datasets[i][0].config.PI["PI"] for i in self.par_df.index]
        Lgains = self.par_df.values[:, 0].astype(int)
        Rgains = self.par_df.values[:, 1].astype(int)
        df = pd.DataFrame(
            index=pd.Series(np.unique(Lgains), name="left_gain"),
            columns=pd.Series(np.unique(Rgains), name="right_gain"),
            dtype=float,
        )
        for Lgain, Rgain, PI in zip(Lgains, Rgains, PIs):
            df[Rgain].loc[Lgain] = PI
        df.to_csv(f"{self.plot_dir}/PIs.csv", index=True, header=True)
        self.figs["PI_heatmap"] = plot_heatmap_PI(save_to=self.plot_dir, z=df, **kwargs)


def space_search_sample(space_dict: dict[str, Any], n: int = 1, **kwargs: Any):
    """
    Convert parameter space dictionary to agentpy Sample object.

    Transforms user-friendly parameter space specification into
    agentpy sampling objects (Values, Range, IntRange) for batch execution.

    Args:
        space_dict: Parameter space specification with structure:
                   - Direct value: param_name: value
                   - Discrete values: param_name: {'values': [v1, v2, ...]}
                   - Continuous range: param_name: {'range': (min, max)}
                   - Grid range: param_name: {'range': (min, max), 'Ngrid': N}
        n: Number of samples to generate (default: 1 for grid search).
        **kwargs: Additional args passed to ap.Sample constructor.

    Returns:
        agentpy.Sample object for batch parameter iteration.

    Example:
        >>> space = {
        >>>     'N': 50,
        >>>     'food_grid': {'values': [5, 10, 15]},
        >>>     'duration': {'range': (1.0, 10.0), 'Ngrid': 10}
        >>> }
        >>> sample = space_search_sample(space)
    """
    D = {}
    for p, args in space_dict.items():
        if not isinstance(args, dict) or ("values" not in args and "range" not in args):
            D[p] = args
        elif "values" in args and args["values"] is not None:
            D[p] = ap.Values(*args["values"])
        else:
            r0, r1 = args["range"]
            if "Ngrid" in args:
                vs = np.linspace(r0, r1, args["Ngrid"])
                if type(r0) == int and type(r1) == int:
                    vs = vs.astype(int)
                D[p] = ap.Values(*vs.tolist())
            else:
                if type(r0) == int and type(r1) == int:
                    D[p] = ap.IntRange(r0, r1)
                elif type(r0) == float and type(r1) == float:
                    D[p] = ap.Range(r0, r1)
    return ap.Sample(D, n=n, **kwargs)


if __name__ == "__main__":
    e = "chemorbit"
    batch_conf = reg.conf.Batch.getID(e)

    m = BatchRun(batch_type=e, **batch_conf)
    m.simulate(n_jobs=1)
    # m.PI_heatmap(show=True)
