from __future__ import annotations
from typing import Any
import argparse
import subprocess

import pandas as pd

from .. import util
from ... import ROOT_DIR, SIM_DIR
from ..process import LarvaDataset

__all__: list[str] = [
    "Exec",
]


class Exec:
    """
    Subprocess execution wrapper for simulations.

    Manages simulation execution either synchronously (in-process) or
    asynchronously (as external subprocess) for non-blocking operation.
    """

    def __init__(
        self,
        mode: str,
        conf: dict,
        experiment: str,
        run_externally: bool = True,
        progressbar: Any | None = None,
        w_progressbar: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize subprocess execution wrapper for simulations.

        Creates an execution wrapper that can run simulations either
        synchronously or as external subprocess for non-blocking operation.

        Args:
            mode: Execution mode ('sim' for single run or 'batch' for batch execution).
            conf: Configuration dict (structure depends on mode).
                  For 'sim': Must contain 'id', 'sim_params', 'larva_groups'.
                  For 'batch': Must contain 'id' and batch parameters.
            experiment: Experiment type string (e.g., 'chemorbit', 'dispersion').
            run_externally: If True, launches as subprocess (non-blocking).
                           If False, runs synchronously in current process.
            progressbar: Optional progress bar widget for GUI integration.
            w_progressbar: Optional secondary progress bar widget.
            **kwargs: Additional args passed to subprocess.Popen.

        Example:
            >>> exec = Exec('sim', sim_conf, 'chemorbit', run_externally=True)
            >>> exec.run()
            >>> # Later check completion:
            >>> if exec.check():
            >>>     results = exec.results
        """
        self.run_externally = run_externally
        self.mode = mode
        self.conf = conf
        self.progressbar = progressbar
        self.w_progressbar = w_progressbar
        self.type = experiment
        self.done = False

    def terminate(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.kill()

    def run(self, **kwargs: Any) -> None:
        f0, f1 = (
            f"{ROOT_DIR}/lib/sim/exec_conf.txt",
            f"{ROOT_DIR}/lib/sim/exec_run.py",
        )
        if self.run_externally:
            util.save_dict(self.conf, f0)
            self.process = subprocess.Popen(["python", f1, self.mode, f0], **kwargs)
        else:
            res = self.exec_run()
            self.results = self.retrieve(res)
            self.done = True

    def check(self) -> bool:
        if not self.done:
            if self.run_externally:
                if self.process.poll() is not None:
                    self.results = self.retrieve()
                    self.done = True
                    return True
            return False
        else:
            return True

    def retrieve(
        self, res: pd.DataFrame | list[LarvaDataset] | None = None
    ) -> pd.DataFrame | tuple[dict[str, dict[str, Any]], dict[str, Any]] | None:
        """
        Retrieve and process results from subprocess execution.

        Collects simulation results from subprocess execution and
        processes them according to experiment type and mode.

        Args:
            res: Results from subprocess - can be:
                 - pd.DataFrame: Single batch endpoint data (batch mode).
                 - list[LarvaDataset]: Multiple dataset results (sim mode).
                 - None: Load from disk based on self.conf (default for external runs).

        Returns:
            Processed results, type depends on self.mode:
            - For 'batch' mode: DataFrame with batch results or None if load fails.
            - For 'sim' mode: tuple of (entry_dict, fig_dict) where:
              * entry_dict: {id: {'dataset': list[LarvaDataset], 'figs': dict}}
              * fig_dict: Figure dictionary (currently None - TODO)
            - None: If results cannot be retrieved.

        Example:
            >>> # After external batch run completes:
            >>> results = exec.retrieve()  # Loads from disk
            >>> if isinstance(results, pd.DataFrame):
            >>>     print(f"Batch results: {len(results)} rows")
            >>>
            >>> # For sim mode with manual results:
            >>> datasets = [LarvaDataset(dir=path) for path in paths]
            >>> entry, figs = exec.retrieve(res=datasets)
        """
        if self.mode == "batch":
            if res is None and self.run_externally:
                f = f'{SIM_DIR}/batch_runs/{self.type}/{self.conf["id"]}/results.h5'
                try:
                    res = pd.read_hdf(f, key="results")
                except:
                    res = None
            return res
        elif self.mode == "sim":
            id = self.conf["id"]
            if res is None and self.run_externally:
                dir0 = f"{SIM_DIR}/single_runs/{self.conf['sim_params']['path']}/{id}"
                res = [
                    LarvaDataset(dir=f"{dir0}/{id}.{gID}")
                    for gID in self.conf["larva_groups"]
                ]

            if res is not None:
                # TODO sim analysis independent from SingleRun class. Currently exec does not run analysis for "sim" mode
                # fig_dict, results = sim_analysis(res, self.type)
                fig_dict, results = None, None
                entry = {id: {"dataset": res, "figs": fig_dict}}
            else:
                entry, fig_dict = None, None
            return entry, fig_dict

    def exec_run(self):
        if self.mode == "sim":
            # Local import to avoid importing the sim package and potential cycles
            from .single_run import ExpRun  # type: ignore

            self.process = ExpRun(parameters=self.conf)
            res = self.process.simulate()
        elif self.mode == "batch":
            # Local import to avoid importing the sim package and potential cycles
            from .batch_run import BatchRun  # type: ignore

            self.process = None
            k = BatchRun(**self.conf)
            res = k.simulate()

        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run given batch-exec/simulation")
    parser.add_argument(
        "mode",
        choices=["sim", "batch"],
        help="Whether we are running a single simulation or a batch-exec",
    )
    parser.add_argument(
        "conf_file",
        type=str,
        help="The configuration file of the batch-exec/simulation",
    )
    args = parser.parse_args()
    conf = util.load_dict(args.conf_file)
    k = Exec(args.mode, conf)
    k.exec_run()
