import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from larvaworld.lib import reg, sim

RUN_WINDOWS_HEAVY = os.environ.get("LW_ENABLE_WINDOWS_HEAVY") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.heavy,
    pytest.mark.skipif(
        sys.platform.startswith("win") and not RUN_WINDOWS_HEAVY,
        reason="Slow on Windows CI (enabled weekly)",
    ),
]


def test_genetic_algorithm_no_video():
    """Run a genetic algorithm experiment without visualization on screen."""
    exp = "realism"
    ga1 = sim.GAlauncher(experiment=exp, duration=0.5)
    ga1.selector.Ngenerations = 3
    ga1.selector.Nagents = 20
    best1 = ga1.simulate()
    print(best1)
    assert best1 is not None


def test_genetic_algorithm_with_video():
    """Run a genetic algorithm experiment with visualization on screen."""
    exp = "realism"
    p = reg.conf.Ga.expand(exp)
    p.ga_select_kws.Ngenerations = 3
    p.ga_select_kws.Nagents = 20
    ga2 = sim.GAlauncher(
        parameters=p,
        duration=0.5,
        screen_kws={"show_display": True, "vis_mode": "video"},
    )
    best2 = ga2.simulate()
    print(best2)
    assert best2 is not None


def test_genetic_algorithm_stores_per_generation_results():
    """
    store_genomes() previously silently swallowed real failures (a bare
    except: pass around both corr_df and diff_df) -- diff_df itself crashed
    with a KeyError whenever a differing model field (e.g. life_history, only
    present on a live-instantiated agent's mConf) had no entry in
    ModuleColorDict. Assert the per-generation CSV is written with one row
    per genome per generation, and that both corr_df/diff_df are actually
    computed, not silently dropped.
    """
    bestConfID = "test_ga_storage_optimized_turner"
    p = reg.conf.Ga.expand("exploration")
    p.ga_select_kws.base_model = "explorer"
    p.ga_select_kws.bestConfID = bestConfID
    p.ga_select_kws.space_mkeys = ["turner"]
    p.ga_select_kws.include_effector_params = True
    p.ga_select_kws.space_pkeys = ["input_noise", "output_noise"]
    p.ga_select_kws.Nagents = 6
    p.ga_select_kws.Nelits = 2
    p.ga_select_kws.Ngenerations = 2

    ga = sim.GAlauncher(
        parameters=p,
        duration=0.2,
        screen_kws={"show_display": False},
        store_data=True,
    )
    ga.simulate()

    csv_path = Path(ga.data_dir) / f"{bestConfID}.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 6 * 2
    for col in ["generation", "input_noise", "output_noise", "fitness"]:
        assert col in df.columns

    assert hasattr(ga, "corr_df")
    assert hasattr(ga, "diff_df")

    # GAselector.new_genome() injects a fixed, non-optimized life_history
    # constant into every live genome's mConf (needed to instantiate an
    # agent during the run). The static base model never had it, so
    # registering best_genome.mConf verbatim used to make every
    # base-vs-best diff spuriously flag life_history as "different" and
    # blank out the base model's column on those rows.
    stored = reg.conf.Model.getID(bestConfID)
    assert "life_history" not in stored.flatten()
    assert "life_history" not in ga.diff_df["parameter"].values
