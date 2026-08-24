r"""
Test AmP zero-variate predictions against Larvaworld simulations.

This module demonstrates the complete educational pipeline for validating DEB model
predictions and comparing to the generalized animal reference parameters:

1. Download/locate AmP parameter export and observed/predicted data
2. Load JSON and organize into free vs. non-free parameters
3. Compare free/non-free parameters against the generalized animal baseline
4. Simulate life cycle using two integration engines (closed-form ODE, stepped)
5. Extract and compare zero-variate predictions (age/length/fecundity at life events)
6. Visualize observed vs. DEBtool-predicted vs. simulated values

The zero-variate symbols tested are:
  - ab, tj, tje: developmental timings
  - Lb, Lj, Wd_e_f: lengths/weights at life events
  - t1, t2, L1, L2: instar durations/lengths (deferred - not yet ported)
  - Ri: fecundity (deferred - needs ultimate-state formula)
  - am_BD_LD: lifespan (deferred - needs aging ODE integration)

Parameter Fitting Strategy:
  The AmP database fits models by:
  - Designating some parameters as "free" (fitted to data)
  - Designating others as "non-free" (held at generalized_animal defaults)
  - The generalized_animal dict provides pseudo-data anchors for missing parameters
  By default, simulations use only the free parameters; non-free ones are ignored
  to allow focused optimization.

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
    from larvaworld.lib.model.deb.generalized_animal import deb_generalized_animal
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

        # Parameter organization (free vs. non-free)
        self.free_pars: dict[str, float] = {}
        self.nonfree_pars: dict[str, float] = {}
        self.param_status: Optional[pd.DataFrame] = None

    def load_json(self) -> None:
        """
        Load AmP parameter export from JSON.

        Extracts:
        - parameters: DEB model parameters (organized as free vs. non-free)
        - results: observed and predicted zero-variate values
        - metadata: species, author, date, etc.

        The 'free' field in each parameter (0 or 1) indicates whether it was
        fitted (1) or held at generalized_animal defaults (0).
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

        # Parse parameters, separating free (fitted) vs. non-free (defaults)
        params_list = data.get("parameters", [])
        param_dict = {}
        param_metadata = {}  # Track all params for later analysis (including non-DEBPars ones)

        for p in params_list:
            symbol = p["symbol"]
            value = p["value"]
            is_free = p.get("free", 1)
            param_metadata[symbol] = (value, is_free)

            # Only include params that DEBPars recognizes
            try:
                # Try to set it; if DEBPars rejects, skip it
                test_pars = de.DEBPars(**{symbol: value})
                param_dict[symbol] = value

                if is_free:
                    self.free_pars[symbol] = value
                else:
                    self.nonfree_pars[symbol] = value
            except TypeError:
                # Parameter not recognized by DEBPars (e.g., f_DR, f_JAZZ)
                # Still track it for reporting
                if is_free:
                    self.free_pars[symbol] = value
                else:
                    self.nonfree_pars[symbol] = value

        self.pars = de.DEBPars(**param_dict)
        print(f"  Parameters processed: {len(param_metadata)} total in JSON")
        print(f"    - Used by DEBPars:     {len(param_dict)}")
        print(
            f"    - Free (fitted):       {len([k for k in self.free_pars if k in param_dict])}"
        )
        print(
            f"    - Non-free (default):  {len([k for k in self.nonfree_pars if k in param_dict])}"
        )
        print(f"    - Extra (not used):    {len(param_metadata) - len(param_dict)}")

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

    def analyze_parameter_divergence(self) -> None:
        r"""
        Compare free and non-free parameters against generalized_animal baseline.

        Builds a summary report showing:
        - Non-free parameters that differ from generalized_animal
        - Free parameters that equal generalized_animal (indicating redundant fitting)
        - Parameters only in this species (not in generalized_animal)
        - Parameters only in generalized_animal (not fitted for this species)

        This analysis reveals the fitting strategy: which parameters were
        intentionally varied from defaults, and which free parameters converged
        back to defaults (suggesting they may be non-identifiable).

        Prints a structured summary to stdout.
        """
        if not self.free_pars and not self.nonfree_pars:
            raise RuntimeError("Must call load_json() first")

        gen_keys = set(deb_generalized_animal.keylist)
        all_local_keys = set(self.free_pars.keys()) | set(self.nonfree_pars.keys())

        # Organize by category
        nonfree_diverge = []  # Non-free params that differ from generalized
        free_match = []  # Free params that equal generalized (redundant fitting)
        common_keys = gen_keys & all_local_keys
        only_local = all_local_keys - gen_keys
        only_gen = gen_keys - all_local_keys

        for key in sorted(common_keys):
            gen_val = deb_generalized_animal[key]
            if key in self.nonfree_pars:
                if self.nonfree_pars[key] != gen_val:
                    nonfree_diverge.append(
                        (key, self.nonfree_pars[key], gen_val, "mismatch")
                    )
            elif key in self.free_pars:
                if self.free_pars[key] == gen_val:
                    free_match.append((key, self.free_pars[key], gen_val, "matches"))

        print("\n" + "=" * 100)
        print("PARAMETER DIVERGENCE ANALYSIS vs. GENERALIZED ANIMAL")
        print("=" * 100)

        if nonfree_diverge:
            print(
                f"\n[!] Non-free parameters with divergent values "
                f"({len(nonfree_diverge)} found):"
            )
            print(
                f"  {'Parameter':<15s} {'Species':<15s} {'Generalized':<15s} {'Note'}"
            )
            for k, sp_val, gen_val, note in nonfree_diverge:
                print(f"  {k:<15s} {sp_val:<15.6g} {gen_val:<15.6g} ({note})")
        else:
            print("\n[OK] All non-free parameters match generalized_animal defaults")

        if free_match:
            print(
                f"\n[!] Free parameters equal to generalized_animal "
                f"({len(free_match)} found):"
            )
            print(
                f"  These fitted parameters converged back to defaults, "
                f"suggesting they may not be identifiable:\n"
            )
            print(f"  {'Parameter':<15s} {'Value':<15s}")
            for k, val, _, _ in free_match:
                print(f"  {k:<15s} {val:<15.6g}")
        else:
            print(
                "\n[OK] No free parameters matched generalized_animal "
                "(all fitted values diverged)"
            )

        if only_local:
            print(f"\n  Parameters in this species only " f"({len(only_local)}):")
            print(f"    {', '.join(sorted(only_local))}")

        if only_gen:
            print(
                f"\n  Generalized_animal parameters not used in this species "
                f"({len(only_gen)}):"
            )
            print(f"    {', '.join(sorted(only_gen))}")

        print("=" * 100)

        # Build summary dataframe for reference
        rows = []
        for key in sorted(common_keys):
            free_val = self.free_pars.get(key)
            nonfree_val = self.nonfree_pars.get(key)
            gen_val = deb_generalized_animal.get(key)
            is_free = key in self.free_pars

            rows.append(
                {
                    "parameter": key,
                    "is_free": is_free,
                    "species_value": free_val if is_free else nonfree_val,
                    "generalized_value": gen_val,
                    "equals_default": (free_val if is_free else nonfree_val) == gen_val,
                }
            )

        self.param_status = pd.DataFrame(rows)

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

        Sets predictions_closed and predictions_stepped dicts with AttrDict objects
        containing: observed, predicted (AmP), RE (relative error), simulated.
        """
        if self.lh_closed is None or self.lh_stepped is None:
            raise RuntimeError("Must call run_simulations() first")

        print("\nExtracting predictions...")

        # Get the mapping from deb_equations
        from larvaworld.lib.model.deb.deb_equations import _PREDICTION_ACCESSORS

        # Extract predictions from both engines
        for engine_name, lh in [
            ("closed", self.lh_closed),
            ("stepped", self.lh_stepped),
        ]:
            predictions = AttrDict({})
            for symbol in self.amp_data.keys():
                accessor = _PREDICTION_ACCESSORS.get(symbol)
                if accessor is not None:
                    try:
                        simulated_val = accessor(lh)
                        if simulated_val is not None:
                            predictions[symbol] = AttrDict(
                                {
                                    "observed": self.amp_data[symbol]["observed"],
                                    "predicted": self.amp_data[symbol]["predicted"],
                                    "RE": self.amp_data[symbol]["RE"],
                                    "simulated": simulated_val,
                                }
                            )
                    except Exception:
                        pass  # Skip symbols that fail to extract

            if engine_name == "closed":
                self.predictions_closed = predictions
            else:
                self.predictions_stepped = predictions

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

    def compute_error_statistics(self) -> dict[str, float]:
        """
        Compute summary error statistics for both engines.

        Returns
        -------
        dict
            Keys: "closed_median_RE", "closed_mean_RE", "stepped_median_RE",
                  "stepped_mean_RE", "n_predictions"
            Values: relative error metrics
        """
        if self.comparison is None:
            self.build_comparison()

        df = self.comparison
        stats = {
            "n_predictions": len(df.dropna(subset=["simulated_closed"])),
            "closed_median_RE": df["RE_closed"].median(),
            "closed_mean_RE": df["RE_closed"].mean(),
            "closed_max_RE": df["RE_closed"].max(),
            "stepped_median_RE": df["RE_stepped"].median(),
            "stepped_mean_RE": df["RE_stepped"].mean(),
            "stepped_max_RE": df["RE_stepped"].max(),
        }
        return stats

    def print_error_summary(self) -> None:
        """Print a summary of prediction accuracy statistics."""
        stats = self.compute_error_statistics()
        print("\n" + "=" * 80)
        print("PREDICTION ACCURACY SUMMARY")
        print("=" * 80)
        print(f"Number of zero-variate predictions: {stats['n_predictions']}")
        print(
            f"\nClosed-form ODE engine:"
            f"\n  Median relative error: {stats['closed_median_RE']:.4f}"
            f"\n  Mean relative error:   {stats['closed_mean_RE']:.4f}"
            f"\n  Max relative error:    {stats['closed_max_RE']:.4f}"
        )
        print(
            f"\nStepped Euler engine:"
            f"\n  Median relative error: {stats['stepped_median_RE']:.4f}"
            f"\n  Mean relative error:   {stats['stepped_mean_RE']:.4f}"
            f"\n  Max relative error:    {stats['stepped_max_RE']:.4f}"
        )
        print("=" * 80)

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
        self.analyze_parameter_divergence()
        self.run_simulations()
        self.extract_predictions()
        self.build_comparison()
        self.print_comparison()
        self.print_error_summary()

        if plot:
            self.plot_predictions()


def _find_json_for_species(species: str) -> Optional[Path]:
    """
    Locate or suggest the AmP JSON file for a given species.

    Searches in the standard AmP_models directory structure.
    If not found, prints download instructions for the AmP results page.

    Parameters
    ----------
    species : str
        Species name (e.g., "Drosophila_melanogaster")

    Returns
    -------
    Path or None
        Path to the JSON file if found, None otherwise.
    """
    amph_models_dir = (
        Path(__file__).parent.parent.parent / "src/larvaworld/lib/model/deb/AmP_models"
    )
    json_path = amph_models_dir / species / f"{species}.json"

    if json_path.exists():
        return json_path

    # Check if _res.html exists (indicates where to find the data)
    html_path = amph_models_dir / species / f"{species}_res.html"
    if html_path.exists():
        print(
            f"\n[!] Found AmP results page ({html_path.name}) "
            f"but JSON has not been generated yet."
        )
        print(f"    To generate the JSON, run:")
        print(
            f"      python -m larvaworld.lib.model.deb.amp_import {html_path} "
            f"--out {json_path}"
        )
        return None

    print(f"\n[!] AmP JSON file not found at {json_path}")
    print(
        f"    To download from AmP database, visit: https://www.bio.vu.nl/thb/deb/deblab/"
    )
    print(f"    1. Search for '{species}'")
    print(f"    2. Right-click the results page link -> 'Save page as...'")
    print(f"    3. Save as: {html_path}")
    print(f"    4. Run: python -m larvaworld.lib.model.deb.amp_import {html_path}")
    return None


def main():
    r"""
    Example: Test AmP predictions for multiple species.

    Demonstrates the complete educational pipeline:
    1. Locate or download AmP species data
    2. Parse JSON and organize parameters (free vs. non-free)
    3. Compare against generalized_animal baseline
    4. Run simulations with both integration engines
    5. Compare predictions and visualize accuracy

    To use this as an educational notebook:
    1. This file is a standalone Python module
    2. Run it directly: python examples/deb/test_amp_predictions.py
    3. Or import and use interactively in a Jupyter notebook:
       from examples.deb.test_amp_predictions import AmPPredictionTester
       tester = AmPPredictionTester("path/to/Species.json")
       tester.run_full_pipeline()

    Species available (if JSON files exist):
    - Drosophila_melanogaster (fruit fly) — [currently has JSON]
    - Bactrocera_oleae (olive fly) — [example: can be added if HTML page is available]
    """
    # Try to locate JSON files for species to test
    species_list = [
        "Drosophila_melanogaster",
        "Bactrocera_oleae",
    ]

    print("\n" + "=" * 80)
    print("AVAILABLE SPECIES FOR TESTING")
    print("=" * 80)

    available_species = []
    for species in species_list:
        json_path = _find_json_for_species(species)
        if json_path:
            available_species.append((species, json_path))
            print(f"[OK] {species:<30s} (found)")
        else:
            print(f"[--] {species:<30s} (not found)")

    if not available_species:
        print("\nNo species data available. Please download from AmP database.")
        return

    # Run pipeline for each available species
    for species, json_path in available_species:
        print("\n" + "=" * 80)
        print(f"TESTING: {species}")
        print("=" * 80)

        tester = AmPPredictionTester(json_path, f=1.0, T=None)
        tester.run_full_pipeline(plot=False)  # Set plot=True for interactive use

    # Print educational reference
    print("\n" + "=" * 80)
    print("PARAMETER FITTING STRATEGY (for educational context)")
    print("=" * 80)
    print(
        r"""
The AmP database uses a two-tier parameter strategy:

1. GENERALIZED_ANIMAL (baseline):
   - Pseudo-data anchor values that capture the "typical" animal
   - Located in: src/larvaworld/lib/model/deb/generalized_animal.py
   - Provides missing-data imputation and optimization initialization

2. SPECIES-SPECIFIC FITTING:
   - Each species' JSON lists parameters marked "free" (fitted) or not
   - Free parameters are optimized to match observed data
   - Non-free parameters are held at generalized_animal values

3. SIMULATIONS:
   - By default, only free parameters are used (non-free are ignored)
   - This focuses the model on fitted behaviors for the species
   - Non-free parameters provide theoretical structure without overfitting

4. PARAMETER DIVERGENCE ANALYSIS:
   - Non-free parameters that differ from generalized: suggests model
     discrepancies or generalized_animal miscalibration
   - Free parameters that equal generalized: suggests the parameter was
     not identifiable from data (fitted but converged to default)
"""
    )

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
