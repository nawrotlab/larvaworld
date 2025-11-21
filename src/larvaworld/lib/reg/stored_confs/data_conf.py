from __future__ import annotations

import json
import os
from typing import Any

from .... import DATA_DIR
from ... import reg, util, funcs
from ...param import Filesystem, PreprocessConf, TrackerOps
from ...util import nam

__all__: list[str] = [
    "LabFormat_dict",
    "Ref_dict",
]


@funcs.stored_conf("LabFormat")
def LabFormat_dict() -> util.AttrDict:
    d = {
        "Schleyer": {
            "tracker": TrackerOps(
                XY_unit="mm",
                fr=16.0,
                Npoints=12,
                Ncontour=22,
                front_vector=(2, 6),
                rear_vector=(7, 11),
                point_idx=9,
            ),
            "filesystem": Filesystem(
                **{
                    "read_sequence": ["Step"]
                    + nam.midline_xy(
                        12, reverse=True, flat=True
                    )  # As stated in the dataset description midline order is reversed from rear to front. See https://doi.gin.g-node.org/10.12751/g-node.5e1ifd/
                    + nam.contour_xy(22, flat=True)
                    + nam.centroid_xy
                    + [
                        "blob_orientation",
                        "area",
                        "grey_value",
                        "raw_spinelength",
                        "width",
                        "perimeter",
                        "collision_flag",
                    ],
                    "read_metadata": True,
                    "file_suf": ".csv",
                    "folder_pref": "box",
                }
            ),
            "env_params": reg.gen.Env(
                arena=reg.gen.Arena(dims=(0.15, 0.15), geometry="circular")
            ),
            "preprocess": PreprocessConf(
                filter_f=2.0, rescale_by=0.001, drop_collisions=True
            ),
        },
        "Jovanic": {
            "tracker": TrackerOps(
                XY_unit="mm",
                fr=1 / 0.07,
                constant_framerate=False,
                Npoints=11,
                Ncontour=0,
                front_vector=(2, 6),
                rear_vector=(6, 10),
                point_idx=9,
            ),
            "filesystem": Filesystem(
                **{
                    "structure": "per_parameter",
                    "file_suf": "larvaid.txt",
                    "file_sep": "_",
                }
            ),
            "env_params": reg.gen.Env(
                arena=reg.gen.Arena(dims=(0.193, 0.193), geometry="rectangular")
            ),
            "preprocess": PreprocessConf(
                filter_f=2.0, rescale_by=0.001, transposition="arena"
            ),
        },
        "Berni": {
            "tracker": TrackerOps(
                fr=2.0, Npoints=1, front_vector=(1, 1), rear_vector=(1, 1), point_idx=1
            ),
            "filesystem": Filesystem(
                **{"read_sequence": ["Date"] + nam.traj_xy, "file_sep": "_-_"}
            ),
            "env_params": reg.gen.Env(
                arena=reg.gen.Arena(dims=(0.24, 0.24), geometry="rectangular")
            ),
            "preprocess": PreprocessConf(filter_f=0.1, transposition="arena"),
        },
        "Arguello": {
            "tracker": TrackerOps(
                fr=10.0,
                Npoints=5,
                front_vector=(1, 3),
                rear_vector=(3, 5),
                point_idx=-1,
            ),
            "filesystem": Filesystem(
                **{
                    "read_sequence": ["Date"]
                    + nam.midline_xy(5, flat=True)
                    + nam.centroid_xy,
                    "file_sep": "_-_",
                }
            ),
            "env_params": reg.gen.Env(
                arena=reg.gen.Arena(dims=(0.17, 0.17), geometry="rectangular")
            ),
            "preprocess": PreprocessConf(filter_f=0.1, transposition="arena"),
        },
    }

    return util.AttrDict(
        {k: reg.generators.LabFormat(labID=k, **kws).nestedConf for k, kws in d.items()}
    )


@funcs.stored_conf("Ref")
def Ref_dict() -> util.AttrDict:
    dds = [
        [f"{DATA_DIR}/JovanicGroup/processed/AttP{g}/{c}" for g in ["2", "240"]]
        for c in ["Fed", "Deprived", "Starved"]
    ]
    dds = util.flatten_list(dds)
    dds.append(f"{DATA_DIR}/SchleyerGroup/processed/exploration/dish")
    dds.append(f"{DATA_DIR}/SchleyerGroup/processed/exploration/40controls")
    dds.append(f"{DATA_DIR}/SchleyerGroup/processed/no_odor/150controls")
    entries = {}
    for dr in dds:
        f = f"{dr}/data/conf.txt"
        if os.path.isfile(f):
            try:
                with open(f) as tfp:
                    c = json.load(tfp)
                c = util.AttrDict(c)
                entries[c.refID] = c.dir
            except:
                pass
    return util.AttrDict(entries)
