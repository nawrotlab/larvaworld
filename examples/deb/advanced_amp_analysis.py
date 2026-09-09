r"""
Advanced AmP prediction analysis: moults, multiple f values, k_X variants, engine comparison.

This module extends the basic prediction testing to include:
1. Instar moult calculations (t1, t2, L1, L2) from MATLAB DEBtool
2. Simulations with alternate feeding levels (f_DR, f_HS, f_F424, f_JAZZ)
3. Sensitivity to digestion efficiency (k_X: rovers higher efficiency, sitters lower)
4. Side-by-side comparison of closed-form ODE vs. stepped Euler engines
5. Comprehensive results table matching AmP HTML format

Educational value:
- Demonstrates how different feeding and metabolic parameters affect DEB predictions
- Shows numerical differences between integration methods
- Provides framework for behavioral phenotype modeling (rovers vs. sitters)

Author: Claude Haiku 4.5
License: Same as Larvaworld
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
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


@dataclass
class MoultAnalysis:
    """
    Compute instar moult timings and lengths from a life history trajectory.

    Uses the MATLAB DEBtool formulas from predict_Drosophila_melanogaster.m:
        L_b = L_m * l_b
        L_1 = L_b * s_1^0.5
        L_2 = L_1 * s_2^0.5
        r_j = g * k_M * (f/l_b - 1) / (f + g)  # specific growth rate
        t_1 = log(L_1/L_b) * 3 / r_j / TC_t1
        t_2 = log(L_2/L_1) * 3 / r_j / TC_t2
    """

    lh: de.LifeHistory
    f: float = 1.0
    T: Optional[float] = None

    # Computed outputs
    t_1: Optional[float] = None  # time to end of instar 1 (since birth)
    t_2: Optional[float] = None  # time to end of instar 2 (since end of instar 1)
    L_1: Optional[float] = None  # structural length at end of instar 1 (cm)
    L_2: Optional[float] = None  # structural length at end of instar 2 (cm)
    Lw_1: Optional[float] = None  # physical length at end of instar 1 (cm)
    Lw_2: Optional[float] = None  # physical length at end of instar 2 (cm)

    def __post_init__(self) -> None:
        """Compute moult events if trajectory is available."""
        if self.lh.trajectory is None:
            return

        pars = self.lh.pars

        # Reference values from simulation
        L_b = self.lh.L_b
        if L_b is None or not np.isfinite(L_b):
            return

        l_b = L_b / pars.L_m
        L_m = pars.L_m
        s_1 = pars.s_1
        s_2 = pars.s_2
        g = pars.g
        k_M = pars.k_M
        del_M = pars.del_M

        # Specific growth rate (1/d, unscaled)
        if l_b > 0 and (f := self.f) > 0:
            r_j = g * k_M * (f / l_b - 1.0) / (f + g)
            if r_j <= 0:
                return  # No exponential growth

            # Moult lengths (structural)
            self.L_1 = L_b * np.sqrt(s_1)
            self.L_2 = self.L_1 * np.sqrt(s_2)

            # Physical lengths
            self.Lw_1 = self.L_1 / del_M
            self.Lw_2 = self.L_2 / del_M

            # Moult times (log scale, accounting for temperature)
            TC = 1.0 if self.T is None else de.tempcorr(self.T, pars.T_ref, pars.T_A)

            try:
                self.t_1 = np.log(self.L_1 / L_b) * 3.0 / r_j / TC
                self.t_2 = np.log(self.L_2 / self.L_1) * 3.0 / r_j / TC
            except (ValueError, ZeroDivisionError):
                pass  # Log/zero error; skip


class AdvancedAmPAnalyzer:
    """
    Comprehensive analysis framework for AmP predictions with multiple scenarios.

    Extends AmPPredictionTester to include:
    - Instar moult calculations
    - Multiple functional response (f) values
    - Parameter sensitivity analysis (k_X variants)
    - Engine comparison (closed vs. stepped)
    - Comprehensive results table generation
    """

    def __init__(
        self,
        json_path: str | Path,
        T: Optional[float] = None,
        dt: float = 1.0 / (24.0 * 60.0),
    ):
        """
        Initialize advanced analyzer.

        Parameters
        ----------
        json_path : str or Path
            Path to AmP JSON export
        T : float, optional
            Temperature in Kelvin (default: T_ref)
        dt : float
            Step size for stepped engine (default: 1 minute)
        """
        self.json_path = Path(json_path)
        self.T = T
        self.dt = dt

        self.pars: Optional[de.DEBPars] = None
        self.metadata: dict = {}
        self.amp_data: dict = {}

        # Storage for multi-scenario results
        self.results: dict[str, dict] = {}  # scenario_name -> predictions dict

    def load_json(self) -> None:
        """Load AmP JSON file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        params_list = data.get("parameters", [])
        param_dict = {}

        for p in params_list:
            symbol = p["symbol"]
            value = p["value"]
            try:
                test_pars = de.DEBPars(**{symbol: value})
                param_dict[symbol] = value
            except TypeError:
                pass  # Skip unrecognized parameters

        self.pars = de.DEBPars(**param_dict)

        # Load zero-variate reference data
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

        print(
            f"Loaded {self.metadata.get('species', 'unknown')} ({len(param_dict)} pars, {len(self.amp_data)} zero-variates)"
        )

    def run_scenario(self, scenario_name: str, f: float, k_X: float) -> dict:
        r"""
        Run a single scenario (f, k_X combination) with both engines.

        Parameters
        ----------
        scenario_name : str
            Label for this scenario (e.g., "f=1.0_k_X=0.8")
        f : float
            Scaled functional response
        k_X : float
            Digestion efficiency (food to reserve)

        Returns
        -------
        dict
            Results dict with closed and stepped predictions
        """
        if self.pars is None:
            raise RuntimeError("Must call load_json() first")

        # Modify parameters for this scenario
        pars = replace(self.pars, f=f, kap_X=k_X)

        results = {
            "scenario": scenario_name,
            "f": f,
            "k_X": k_X,
            "closed": {},
            "stepped": {},
        }

        # Run simulations
        for engine in ["closed", "stepped"]:
            try:
                lh = de.run_life_cycle(pars, f=f, T=self.T, engine=engine, dt=self.dt)

                # Extract basic predictions
                predictions = {}
                for symbol in ["ab", "tj", "tje", "Lb", "Lj", "Wd_e_f"]:
                    accessor = de._PREDICTION_ACCESSORS.get(symbol)
                    if accessor:
                        try:
                            predictions[symbol] = accessor(lh)
                        except Exception:
                            pass

                # Compute moults
                moults = MoultAnalysis(lh, f=f, T=self.T)
                for key in ["t_1", "t_2", "L_1", "L_2", "Lw_1", "Lw_2"]:
                    val = getattr(moults, key)
                    if val is not None:
                        predictions[key] = val

                results[engine] = predictions

            except Exception as e:
                print(f"  Warning: {engine} engine failed for {scenario_name}: {e}")
                results[engine] = {}

        return results

    def run_multi_scenario(
        self, f_values: Optional[list] = None, k_X_values: Optional[list] = None
    ) -> None:
        """
        Run multiple scenarios and store results.

        Parameters
        ----------
        f_values : list, optional
            Functional response values to test (default: [1.0, 0.99, 0.90, 1.36, 1.56])
        k_X_values : list, optional
            Digestion efficiencies to test (default: [0.8])
        """
        if f_values is None:
            f_values = [
                1.0,  # standard
                0.98936,  # f_DR
                0.89751,  # f_HS
                1.3563,  # f_F424
                1.5645,  # f_JAZZ
            ]
        if k_X_values is None:
            k_X_values = [0.8]  # Generalized

        labels_f = {
            1.0: "standard",
            0.98936: "DR",
            0.89751: "HS",
            1.3563: "F424",
            1.5645: "JAZZ",
        }
        labels_kX = {
            0.8: "gen",
            0.5: "rover",
            0.15: "sitter",
        }

        print(f"\nRunning {len(f_values)} f x {len(k_X_values)} k_X scenarios...")

        for f in f_values:
            for k_X in k_X_values:
                label_f = labels_f.get(f, f"f={f:.3f}")
                label_kX = labels_kX.get(k_X, f"k_X={k_X:.3f}")
                scenario_name = f"{label_f}_{label_kX}"

                result = self.run_scenario(scenario_name, f, k_X)
                self.results[scenario_name] = result
                print(
                    f"  {scenario_name:20s} — closed: {len(result['closed'])} preds, stepped: {len(result['stepped'])} preds"
                )

    def build_results_table(self) -> pd.DataFrame:
        """
        Build comprehensive results table comparing all scenarios.

        Columns: symbol | unit | observed | AmP_pred | closed_base | stepped_base | closed_DR | stepped_DR | ...

        Returns
        -------
        pd.DataFrame
            Results table with one row per symbol, one column pair (closed, stepped) per scenario
        """
        rows = []

        for symbol in sorted(self.amp_data.keys()):
            amp_info = self.amp_data[symbol]
            row = {
                "symbol": symbol,
                "unit": amp_info["unit"],
                "description": amp_info["description"][:40],
                "observed": amp_info["observed"],
                "predicted_amp": amp_info["predicted"],
                "RE_amp": amp_info["RE"],
            }

            # Add predictions from each scenario
            for scenario_name in sorted(self.results.keys()):
                result = self.results[scenario_name]
                for engine in ["closed", "stepped"]:
                    predictions = result[engine]
                    val = predictions.get(symbol)
                    if val is not None:
                        col_name = f"{scenario_name}_{engine}"
                        row[col_name] = val

            rows.append(row)

        return pd.DataFrame(rows)

    def print_results_table(self, digits: int = 4) -> None:
        """Print formatted results table."""
        df = self.build_results_table()

        # Format numeric columns
        for col in df.columns:
            if col not in ["symbol", "unit", "description"]:
                try:
                    df[col] = df[col].apply(
                        lambda x: f"{x:.{digits}g}"
                        if pd.notna(x) and isinstance(x, (int, float))
                        else "—"
                    )
                except Exception:
                    pass

        print("\n" + "=" * 140)
        print("COMPREHENSIVE PREDICTIONS TABLE")
        print("=" * 140)
        print(df.to_string(index=False))
        print("=" * 140)

    def summary(self) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Species: {self.metadata.get('species', 'unknown')}")
        print(f"Scenarios analyzed: {len(self.results)}")
        print(f"Predictions per scenario: closed engine, stepped engine")
        for scenario_name in sorted(self.results.keys()):
            result = self.results[scenario_name]
            nc = len(result["closed"])
            ns = len(result["stepped"])
            print(f"  {scenario_name:20s}  closed: {nc:2d}, stepped: {ns:2d}")
        print("=" * 80)


def main():
    r"""
    Example: comprehensive analysis of Drosophila melanogaster.

    Demonstrates:
    1. Loading AmP JSON
    2. Testing multiple f values (standard + alternate diets)
    3. (Optional) Testing k_X variants
    4. Comparing closed vs. stepped engines
    5. Generating comprehensive results table
    6. Computing moult events
    """
    json_path = (
        Path(__file__).parent.parent.parent
        / "src/larvaworld/lib/model/deb/AmP_models/Drosophila_melanogaster/Drosophila_melanogaster.json"
    )

    if not json_path.exists():
        print(f"Error: AmP JSON not found at {json_path}")
        return

    analyzer = AdvancedAmPAnalyzer(json_path, T=None)
    analyzer.load_json()

    # Run multi-scenario analysis
    analyzer.run_multi_scenario(
        f_values=[1.0, 0.98936, 0.89751],  # standard, DR, HS
        k_X_values=[0.8],  # Generalized only (k_X variations can be tested later)
    )

    # Generate and print results
    analyzer.print_results_table()
    analyzer.summary()


if __name__ == "__main__":
    main()
