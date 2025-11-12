import pandas as pd

import pytest

import larvaworld
import larvaworld.lib.process.dataset
import larvaworld.lib.reg as reg

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skip(reason="legacy integration tests kept for manual execution"),
]


def test_import_Schleyer():
    g = reg.conf.LabFormat.get("Schleyer")
    kws0 = {
        # 'labID': 'Schleyer',
        "group_id": "exploration",
    }

    # Merged case
    N = 40
    kws1 = {
        "parent_dir": "exploration",
        "merged": True,
        "color": "green",
        "max_Nagents": N,
        "min_duration_in_sec": 60,
        "id": f"{N}controls",
        "refID": f"exploration.{N}controls",
        **kws0,
    }

    # Single dish case
    folder = "dish02"
    kws2 = {
        "parent_dir": f"exploration/{folder}",
        "merged": False,
        "color": "red",
        "min_duration_in_sec": 90,
        "id": folder,
        "refID": f"exploration.{folder}",
        **kws0,
    }

    for kws in [kws1, kws2]:
        d = g.import_dataset(**kws)
        assert isinstance(d, larvaworld.lib.process.dataset.LarvaDataset)
        d.process(is_last=False)
        d.annotate(is_last=True)
        # d.save()
        assert isinstance(d.s, pd.DataFrame)


def xx_test_import_Jovanic():
    g = reg.conf.LabFormat.get("Jovanic")

    kws0 = {
        #   'labID': 'Jovanic',
        "merged": False
    }

    kws1 = {
        "parent_dir": "ProteinDeprivation",
        "source_ids": ["Fed", "Pd"],
        "colors": ["green", "red"],
        **kws0,
    }

    for kws in [kws1]:
        ds = g.import_datasets(**kws)
        for d in ds:
            assert isinstance(d, larvaworld.lib.LarvaDataset)
