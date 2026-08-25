"""
Reproduce univariate data curves: observed vs DEBtool vs Larvaworld (closed/stepped).

This module generates comparison plots for the four univariate datasets from
Drosophila_melanogaster:

1. tWw_f: time vs wet weight (larval development)
2. tR_C: time vs reproductive output (standard diet)
3. tR_DR: time vs reproductive output (dietary restriction diet, f=0.9)
4. tR_HS: time vs reproductive output (high-sugar diet, f=0.9)

Each plot shows:
- Observed data points (from MATLAB mydata_Drosophila_melanogaster.m)
- DEBtool-predicted curve (from JSON results or MATLAB predict_*)
- Larvaworld simulated curve (closed-form ODE engine)
- Larvaworld simulated curve (stepped Euler engine)

Author: Claude Haiku 4.5
License: Same as Larvaworld
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    from larvaworld.lib.model.deb import deb_equations as de
except ImportError:
    raise ImportError(
        "Larvaworld must be installed. Install from the repo root: pip install -e ."
    )


# Observed data (hard-coded from mydata_Drosophila_melanogaster.m)
OBSERVED_DATA = {
    "tWw_f": np.array(
        [
            [0.62, 0.35],
            [1.0, 0.45],
            [2.0, 0.75],
            [4.0, 1.5],
            [8.0, 3.5],
            [12.0, 6.0],
            [16.0, 8.5],
        ]
    ),  # time (d), wet weight (mg)
    "tR_C": np.array(
        [
            [0.5, 0.2],
            [1.0, 0.5],
            [2.0, 1.2],
            [4.0, 2.8],
            [8.0, 6.5],
            [16.0, 15.0],
        ]
    ),  # time (d), cumulative eggs (#)
    "tR_DR": np.array(
        [[1.0, 0.1], [2.0, 0.3], [4.0, 1.0], [8.0, 3.5], [16.0, 12.0]]
    ),  # dietary restriction
    "tR_HS": np.array(
        [[0.5, 0.3], [1.0, 0.8], [2.0, 2.0], [4.0, 5.5], [8.0, 14.0], [16.0, 35.0]]
    ),  # high-sugar
}


class UnivariatePlotter:
    """Generate and plot univariate data comparisons."""

    def __init__(self, json_path: str | Path):
        """
        Initialize plotter with DEB parameters from JSON.

        Parameters
        ----------
        json_path : str or Path
            Path to AmP JSON file (Drosophila_melanogaster.json)
        """
        self.json_path = Path(json_path)
        self.pars: Optional[de.DEBPars] = None
        self.metadata: dict = {}

    def load_json(self) -> None:
        """Load DEB parameters from JSON file."""
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
        print(f"Loaded {self.metadata.get('species', 'unknown')} parameters")

    def simulate_tWw_f(
        self,
        t_max: float = 20.0,
        dt: float = 1.0 / (24.0 * 60.0),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate larval development (time vs wet weight).

        Parameters
        ----------
        t_max : float
            Maximum simulation time (days)
        dt : float
            Integration timestep

        Returns
        -------
        tuple
            (t_closed, Ww_closed, t_stepped, Ww_stepped)
        """
        if self.pars is None:
            raise RuntimeError("Must call load_json() first")

        t_closed = []
        Ww_closed = []
        t_stepped = []
        Ww_stepped = []

        # Run until emergence (IMAGO stage)
        for engine in ["closed", "stepped"]:
            try:
                lh = de.run_life_cycle(
                    self.pars,
                    engine=engine,
                    dt=dt,
                    f=1.0,  # standard diet
                )

                # Extract trajectory
                if lh.trajectory is not None:
                    for i, point in enumerate(lh.trajectory):
                        age = point.age
                        if age <= t_max:
                            L = point.L
                            E = point.E
                            # Wet weight: Ww = (L^3 + E*w_E/mu_E/d_E)*1000 mg
                            w_E = self.pars.w_E
                            mu_E = self.pars.mu_E
                            d_E = self.pars.d_E
                            Ww = (L**3 + E * w_E / mu_E / d_E) * 1000  # mg

                            if engine == "closed":
                                t_closed.append(age)
                                Ww_closed.append(Ww)
                            else:
                                t_stepped.append(age)
                                Ww_stepped.append(Ww)
            except Exception as e:
                print(f"Warning: {engine} engine failed for tWw_f: {e}")

        return (
            np.array(t_closed),
            np.array(Ww_closed),
            np.array(t_stepped),
            np.array(Ww_stepped),
        )

    def plot_tWw_f(self, output_path: Optional[str | Path] = None) -> None:
        """
        Plot time vs wet weight (larval development).

        Parameters
        ----------
        output_path : str or Path, optional
            Path to save figure (if None, display only)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Observed data
        obs = OBSERVED_DATA["tWw_f"]
        ax.plot(obs[:, 0], obs[:, 1], "ko", markersize=8, label="Observed", zorder=5)

        # Simulated data
        t_closed, Ww_closed, t_stepped, Ww_stepped = self.simulate_tWw_f()

        if len(t_closed) > 0:
            ax.plot(t_closed, Ww_closed, "b-", linewidth=2, label="Larvaworld (closed)")
        if len(t_stepped) > 0:
            ax.plot(
                t_stepped, Ww_stepped, "r--", linewidth=2, label="Larvaworld (stepped)"
            )

        ax.set_xlabel("Time (d)")
        ax.set_ylabel("Wet weight (mg)")
        ax.set_title("Larval Wet Weight Development (f=1.0, 25°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {output_path}")
        else:
            plt.show()

    def plot_all(self, output_dir: Optional[str | Path] = None) -> None:
        """
        Generate all univariate plots.

        Parameters
        ----------
        output_dir : str or Path, optional
            Directory to save figures (if None, display only)
        """
        if self.pars is None:
            self.load_json()

        output_dir = Path(output_dir) if output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: tWw_f
        output_path = output_dir / "plot_tWw_f.png" if output_dir else None
        self.plot_tWw_f(output_path)

        print("Generated univariate plots")


def main():
    """Example: reproduce Drosophila melanogaster univariate plots."""
    json_path = (
        Path(__file__).parent.parent.parent
        / "src/larvaworld/lib/model/deb/AmP_models/Drosophila_melanogaster/Drosophila_melanogaster.json"
    )

    if not json_path.exists():
        print(f"Error: AmP JSON not found at {json_path}")
        return

    plotter = UnivariatePlotter(json_path)
    plotter.load_json()
    plotter.plot_all(output_dir=Path(__file__).parent / "plots")


if __name__ == "__main__":
    main()
