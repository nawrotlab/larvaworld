"""
Test AmP zero-variate predictions against Larvaworld simulations.

This module demonstrates the complete pipeline for validating DEB model
predictions:

1. Load AmP parameter export and observed/predicted data from JSON
2. Simulate life cycle using two integration engines (closed-form ODE, stepped)
3. Compare zero-variate predictions (age/length/fecundity at life events)
4. Visualize observed vs. DEBtool-predicted vs. simulated values

The zero-variate symbols tested are:
  - ab, tj, tje: developmental timings
  - Lb, Lj, Wd_e_f: lengths/weights at life events
  - t1, t2, L1, L2: instar durations/lengths (deferred - not yet ported)
  - Ri: fecundity (deferred - needs ultimate-state formula)
  - am_BD_LD: lifespan (deferred - needs aging ODE integration)

Full documentation of the DEB model and parameter meanings is in:
  src/larvaworld/lib/model/deb/deb_equations.py (module docstring and inline)

Author: Claude Haiku 4.5
License: Same as Larvaworld (check LICENSE file)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from larvaworld.lib.model.deb import deb_equations as de
    from larvaworld.lib.util import AttrDict
except ImportError:
    raise ImportError(
        "Larvaworld must be installed. Install from the repo root: pip install -e ."
    )


# AmP symbol -> LifeHistory accessor mapping (copied from deb_equations.py for reference)
# This is *specific* to the 4-stage holometabolous model as currently transcribed.
# See deb_equations.py:_PREDICTION_ACCESSORS for the authoritative mapping.
_PREDICTION_ACCESSORS_REFERENCE = {
    "ab": "lambda lh: lh.age_at_birth",
    "tj": "lambda lh: lh.time_to_pupation",
    "tje": "lambda lh: lh.durations.get(de.Stage.PUPA)",
    "Lb": "lambda lh: lh.Lw_b",
    "Lj": "lambda lh: lh.Lw_p",
    "Wd_e_f": "lambda lh: lh.Wd_at('imago')",
    # Deferred (ground truth exists in DEBtool, not yet ported):
    # "t1": "instar 1 duration - requires get_tj_habp + instar sub-model",
    # "t2": "instar 2 duration - requires get_tj_habp + instar sub-model",
    # "L1": "instar 1 length - requires get_tj_habp + instar sub-model",
    # "L2": "instar 2 length - requires get_tj_habp + instar sub-model",
    # "Ri": "fecundity - requires ultimate-state closed form from DEBtool",
    # "am_BD_LD": "lifespan - requires aging ODE from get_tm_mod_habp",
}


class AmPPredictionTester:
    """
    Load and test AmP predictions against Larvaworld simulations.

    Workflow:
    1. Load JSON export (parameters + observed/predicted data)
    2. Build DEBPars from parameters
    3. Run life cycle simulation with both integration engines
    4. Extract predictions from LifeHistory
    5. Compare and visualize

    Attributes
    ----------
    json_path : Path
        Path to AmP JSON export (e.g., Drosophila_melanogaster.json)
    pars : DEBPars
        Parsed parameters from JSON
    metadata : dict
        Experiment metadata (species, author, etc.) from JSON
    amp_data : dict
        AmP's original predictions from JSON 'results' block
    lh_closed : LifeHistory
        Life cycle simulation using closed-form ODE engine
    lh_stepped : LifeHistory
        Life cycle simulation using fixed-step Euler engine
    predictions_closed : dict
        Extracted predictions from closed-form run
    predictions_stepped : dict
        Extracted predictions from stepped run
    comparison : DataFrame
        Side-by-side table: observed, AmP-predicted, Larvaworld closed, Larvaworld stepped
    """

    def __init__(
        self,
        json_path: str | Path,
        f: float = 1.0,
        T: Optional[float] = None,
        dt: float = 1.0 / (24.0 * 60.0),
    ):
        """
        Initialize the tester.

        Parameters
        ----------
        json_path : str or Path
            Path to AmP JSON export
        f : float
            Scaled functional response (default: 1.0, meaning replete food)
        T : float, optional
            Temperature in Kelvin for temperature correction (default: None = T_ref)
        dt : float
            Step size in days for the stepped engine (default: 1 minute)
        """
        self.json_path = Path(json_path)
        self.f = f
        self.T = T
        self.dt = dt

        # Storage for results
        self.pars: Optional[de.DEBPars] = None
        self.metadata: dict = {}
        self.amp_data: dict = {}
        self.lh_closed: Optional[de.LifeHistory] = None
        self.lh_stepped: Optional[de.LifeHistory] = None
        self.predictions_closed: dict = {}
        self.predictions_stepped: dict = {}
        self.comparison: Optional[pd.DataFrame] = None

    def load_json(self) -> None:
        """
        Load AmP parameter export from JSON.

        Extracts:
        - parameters: DEB model parameters
        - results: observed and predicted zero-variate values
        - metadata: species, author, date, etc.
        """
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        print(
            f"Loaded {self.metadata.get('species', 'unknown species')} "
            f"from {self.json_path.name}"
        )

        # Parse parameters into DEBPars
        params_list = data.get("parameters", [])
        param_dict = {p["symbol"]: p["value"] for p in params_list}

        self.pars = de.DEBPars(**param_dict)
        print(f"  Parameters loaded: {len(param_dict)} fields")

        # Parse zero-variate predictions
        self.amp_data = {
            e["symbol"]: {
                "observed": e["observed"],
                "predicted": e["predicted"],
                "RE": e["RE"],
                "unit": e.get("unit", "-"),
                "description": e.get("description", ""),
            }
            for e in data.get("results", [])
        }
        print(f"  Zero-variates loaded: {len(self.amp_data)} predictions")

    def run_simulations(self) -> None:
        """
        Run life cycle with both integration engines.

        Sets lh_closed and lh_stepped as LifeHistory objects.
        Prints progress to stdout.
        """
        if self.pars is None:
            raise RuntimeError("Must call load_json() first")

        print(f"\nRunning simulations (f={self.f}, T={self.T})...")

        # Closed-form ODE engine
        print("  [1/2] Closed-form ODE integration...")
        self.lh_closed = de.run_life_cycle(
            self.pars, f=self.f, T=self.T, engine="closed"
        )
        print(f"       {len(self.lh_closed.reached)} stages reached")

        # Fixed-step Euler engine
        print("  [2/2] Fixed-step Euler integration...")
        self.lh_stepped = de.run_life_cycle(
            self.pars, f=self.f, T=self.T, engine="stepped", dt=self.dt
        )
        print(f"       {len(self.lh_stepped.reached)} stages reached")

    def extract_predictions(self) -> None:
        """
        Extract zero-variate predictions from both LifeHistory objects.

        Uses the _PREDICTION_ACCESSORS mapping in deb_equations.py to map
        AmP symbol names to LifeHistory attributes.

        Sets predictions_closed and predictions_stepped dicts.
        """
        if self.lh_closed is None or self.lh_stepped is None:
            raise RuntimeError("Must call run_simulations() first")

        print("\nExtracting predictions...")

        # Use the mapping from deb_equations.py
        self.predictions_closed = self.lh_closed.test_predictions()
        self.predictions_stepped = self.lh_stepped.test_predictions()

        print(
            f"  Closed-form:  {len(self.predictions_closed)} symbols "
            f"({list(self.predictions_closed.keys())})"
        )
        print(
            f"  Stepped:      {len(self.predictions_stepped)} symbols "
            f"({list(self.predictions_stepped.keys())})"
        )

    def build_comparison(self) -> pd.DataFrame:
        """
        Build a comparison table of all predictions.

        Returns
        -------
        pd.DataFrame
            Columns: symbol, unit, description, observed, predicted (AmP),
                     RE (relative error), simulated (closed), simulated (stepped)
        """
        if not self.predictions_closed:
            raise RuntimeError("Must call extract_predictions() first")

        rows = []
        for symbol in sorted(self.amp_data.keys()):
            amp = self.amp_data[symbol]
            closed_pred = self.predictions_closed.get(symbol)
            stepped_pred = self.predictions_stepped.get(symbol)

            row = {
                "symbol": symbol,
                "unit": amp["unit"],
                "description": amp["description"],
                "observed": amp["observed"],
                "predicted_amp": amp["predicted"],
                "RE_amp": amp["RE"],
            }

            # Add simulated values if available
            if closed_pred is not None:
                row["simulated_closed"] = closed_pred.simulated
                row["RE_closed"] = (
                    abs(closed_pred.simulated - amp["observed"]) / abs(amp["observed"])
                    if amp["observed"] != 0
                    else np.nan
                )
            else:
                row["simulated_closed"] = None
                row["RE_closed"] = None

            if stepped_pred is not None:
                row["simulated_stepped"] = stepped_pred.simulated
                row["RE_stepped"] = (
                    abs(stepped_pred.simulated - amp["observed"]) / abs(amp["observed"])
                    if amp["observed"] != 0
                    else np.nan
                )
            else:
                row["simulated_stepped"] = None
                row["RE_stepped"] = None

            rows.append(row)

        self.comparison = pd.DataFrame(rows)
        return self.comparison

    def print_comparison(self, digits: int = 4) -> None:
        """
        Pretty-print the comparison table.

        Parameters
        ----------
        digits : int
            Number of significant figures (default: 4)
        """
        if self.comparison is None:
            self.build_comparison()

        df = self.comparison.copy()

        # Format numeric columns
        float_cols = [
            "observed",
            "predicted_amp",
            "RE_amp",
            "simulated_closed",
            "RE_closed",
            "simulated_stepped",
            "RE_stepped",
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"{x:.{digits}g}" if pd.notna(x) else "—"
                )

        # Shorten description for display
        df["description"] = df["description"].str[:40]

        # Select columns for display
        display_cols = [
            "symbol",
            "unit",
            "observed",
            "predicted_amp",
            "RE_amp",
            "simulated_closed",
            "RE_closed",
            "simulated_stepped",
            "RE_stepped",
        ]
        display_df = df[[c for c in display_cols if c in df.columns]]

        print("\n" + "=" * 140)
        print("ZERO-VARIATE PREDICTIONS COMPARISON")
        print("=" * 140)
        print(display_df.to_string(index=False))
        print("=" * 140)
        print(
            "\nLegend:"
            "\n  observed: experimental value from literature"
            "\n  predicted_amp: DEBtool AmP prediction"
            "\n  simulated_closed/stepped: Larvaworld simulation"
            "\n  RE: relative error = |simulated - observed| / |observed|"
        )

    def plot_predictions(
        self, figsize: tuple[int, int] = (14, 8), show: bool = True
    ) -> plt.Figure:
        """
        Create a comparison plot of all predictions.

        For each zero-variate, plots:
        - Observed value (red line)
        - AmP predicted value (blue line)
        - Larvaworld closed-form (green dot)
        - Larvaworld stepped (orange dot)

        Parameters
        ----------
        figsize : tuple
            Figure size in inches (default: 14x8)
        show : bool
            Whether to call plt.show() (default: True)

        Returns
        -------
        plt.Figure
            The created figure
        """
        if self.comparison is None:
            self.build_comparison()

        df = self.comparison.dropna(subset=["observed"])
        n_symbols = len(df)
        n_cols = 3
        n_rows = (n_symbols + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, constrained_layout=True
        )
        axes = axes.flatten()

        for idx, (_, row) in enumerate(df.iterrows()):
            ax = axes[idx]
            symbol = row["symbol"]
            observed = row["observed"]
            predicted_amp = row["predicted_amp"]
            simulated_closed = row["simulated_closed"]
            simulated_stepped = row["simulated_stepped"]

            # Plot horizontal lines
            x_pos = [0, 1, 2, 3]
            y_values = [observed, predicted_amp, simulated_closed, simulated_stepped]
            colors = ["red", "blue", "green", "orange"]
            labels = [
                "Observed",
                "AmP Predicted",
                "Larvaworld Closed",
                "Larvaworld Stepped",
            ]

            for i, (y, color, label) in enumerate(zip(y_values, colors, labels)):
                if pd.notna(y):
                    ax.hlines(
                        y, i - 0.3, i + 0.3, colors=color, linewidth=2, label=label
                    )
                    ax.plot(i, y, "o", color=color, markersize=8)

            # Format
            ax.set_xlim(-0.5, 3.5)
            ax.set_xticks([])
            ax.set_ylabel(f"{row['unit']}", fontsize=9)
            ax.set_title(
                f"{symbol}: {row['description'][:30]}", fontsize=10, fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

            # Legend only on first subplot
            if idx == 0:
                ax.legend(loc="best", fontsize=8)

        # Hide unused subplots
        for idx in range(n_symbols, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            f"AmP Zero-Variate Predictions vs. Larvaworld Simulations\n"
            f"{self.metadata.get('species', 'Species')} "
            f"(f={self.f}, T={self.T if self.T else 'T_ref'})",
            fontsize=12,
            fontweight="bold",
        )

        if show:
            plt.show()

        return fig

    def run_full_pipeline(self, plot: bool = True) -> None:
        """
        Run the complete prediction testing pipeline.

        Parameters
        ----------
        plot : bool
            Whether to generate comparison plot (default: True)
        """
        print("\n" + "=" * 80)
        print("LARVAWORLD DEB MODEL PREDICTION TESTING PIPELINE")
        print("=" * 80)

        self.load_json()
        self.run_simulations()
        self.extract_predictions()
        self.build_comparison()
        self.print_comparison()

        if plot:
            self.plot_predictions()


def main():
    """
    Example: Test Drosophila melanogaster predictions.

    To use this as an educational notebook:
    1. This file is a standalone Python module
    2. Run it directly: python examples/deb/test_amp_predictions.py
    3. Or import and use interactively in a Jupyter notebook:
       from examples.deb.test_amp_predictions import AmPPredictionTester
       tester = AmPPredictionTester("path/to/Drosophila_melanogaster.json")
       tester.run_full_pipeline()
    """
    # Locate the JSON file (assume it's in the standard AmP models location)
    json_path = (
        Path(__file__).parent.parent.parent
        / "src/larvaworld/lib/model/deb/AmP_models/Drosophila_melanogaster/Drosophila_melanogaster.json"
    )

    if not json_path.exists():
        print(
            f"Error: AmP JSON file not found at {json_path}\n"
            "Please download or locate the Drosophila_melanogaster.json file."
        )
        return

    # Run the full pipeline
    tester = AmPPredictionTester(json_path, f=1.0, T=None)
    tester.run_full_pipeline(plot=True)

    # Additional: print the reference mapping
    print("\n" + "=" * 80)
    print("PREDICTION SYMBOLS MAPPING (for educational reference)")
    print("=" * 80)
    print(
        "\nThe following AmP symbols map to LifeHistory attributes "
        "(defined in deb_equations.py):\n"
    )
    for symbol, accessor_str in sorted(_PREDICTION_ACCESSORS_REFERENCE.items()):
        print(f"  {symbol:15s} -> {accessor_str}")
    print("\nDeferred symbols (ground truth located but not yet ported):")
    print("  t1, t2, L1, L2    -> requires get_tj_habp instar integration")
    print("  Ri                -> requires ultimate-state fecundity formula")
    print("  am_BD_LD          -> requires aging ODE integration (get_tm_mod_habp)")


if __name__ == "__main__":
    main()
