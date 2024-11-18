import os

import numpy as np
import param

from .. import util
from .custom import (
    IntegerRangeOrdered,
    OptionalPositiveInteger,
    OptionalPositiveNumber,
    PositiveInteger,
    PositiveIntegerRangeOrdered,
    PositiveNumber,
)
from .nested_parameter_group import NestedConf

__all__ = [
    "FramerateOps",
    "XYops",
    "SimTimeOps",
    "SimSpatialOps",
    "SimOps",
    "RuntimeGeneralOps",
    "RuntimeDataOps",
    "RuntimeOps",
    "Filesystem",
    "TrackedPointIdx",
    "SimMetricOps",
    "TrackerOps",
    # 'OptimizationOps',
]

__displayname__ = "Configuration parameter groups"


class FramerateOps(NestedConf):
    fr = PositiveNumber(
        10,
        softmax=100,
        step=0.1,
        label="framerate",
        doc="The tracking/simulation framerate (inverse of timestep) in Hz.",
    )
    dt = PositiveNumber(
        0.1,
        softmax=1.0,
        step=0.01,
        label="timestep",
        doc="The tracking/simulation timestep (inverse of tracking framerate) in seconds.",
    )
    constant_framerate = param.Boolean(
        True, doc="Whether the tracking framerate is constant."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "dt" in kwargs:
            self.update_framerate()
        elif "fr" in kwargs:
            self.update_timestep()

    @param.depends("dt", watch=True)
    def update_framerate(self):
        self.fr = 1 / self.dt

    @param.depends("fr", watch=True)
    def update_timestep(self):
        self.dt = 1 / self.fr
        # raise


class XYops(NestedConf):
    XY_unit = param.Selector(
        default="m",
        objects=["m", "mm"],
        doc="The spatial unit of the XY coordinate data",
    )
    Npoints = PositiveInteger(
        3,
        softmax=20,
        label="# midline 2D points",
        doc="The number of points tracked along the larva midline.",
    )
    Ncontour = PositiveInteger(
        0,
        softmax=100,
        label="# contour 2D points",
        doc="The number of points tracked around the larva contour.",
    )

    @property
    def Nangles(self):
        return np.clip(self.Npoints - 2, a_min=0, a_max=None)

    @property
    def Nsegs(self):
        return np.clip(self.Npoints - 1, a_min=0, a_max=None)

    @property
    def angles(self):
        return util.SuperList([f"angle{i}" for i in range(self.Nangles)])

    @property
    def midline_points(self):
        return util.nam.midline(self.Npoints, type="point")

    @property
    def midline_segs(self):
        return util.nam.midline(self.Nsegs, type="seg")

    @property
    def midline_seg_xy(self, flat=True):
        return util.nam.xy(self.midline_segs, flat=flat)

    @property
    def seg_orientations(self):
        return util.nam.orient(self.midline_segs)

    @property
    def midline_xy(self, flat=True):
        return util.nam.xy(self.midline_points, flat=flat)

    @property
    def contour_points(self):
        return util.nam.contour(self.Ncontour)

    @property
    def contour_xy(self, flat=True):
        return util.nam.xy(self.contour_points, flat=flat)

    @property
    def centroid_xy(self):
        return util.nam.xy("centroid")

    @property
    def traj_xy(self):
        return util.nam.xy("")

    @property
    def all_xy(self, flat=True):
        return util.nam.xy(
            self.midline_points + self.contour_points + ["centroid", ""], flat=flat
        )

    def get_track_point(self, idx):
        if idx == -1:
            return "centroid"
        else:
            return self.midline_points[idx - 1]

    def get_midline_xy_data(self, s):
        xy = self.midline_xy
        assert xy.exist_in(s)
        return s[xy].values.reshape([-1, self.Npoints, 2])

    def get_contour_xy_data(self, s):
        xy = self.contour_xy
        assert xy.exist_in(s)
        return s[xy].values.reshape([-1, self.Ncontour, 2])


class SimTimeOps(FramerateOps):
    duration = OptionalPositiveNumber(
        softmax=100.0, step=0.1, doc="The duration of the simulation in minutes."
    )
    Nsteps = OptionalPositiveInteger(
        label="# simulation timesteps", doc="The number of simulation timesteps."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_Nsteps()

    @param.depends("duration", "dt", watch=True)
    def update_Nsteps(self):
        if self.duration is not None:
            self.Nsteps = int(self.duration * 60 / self.dt)
        else:
            self.Nsteps = None

    @param.depends("Nsteps", watch=True)
    def update_duration(self):
        if self.Nsteps is not None:
            self.duration = self.Nsteps * self.dt / 60
        else:
            self.duration = None


class SimSpatialOps(NestedConf):
    Box2D = param.Boolean(False, doc="Whether to use the Box2D physics engine or not.")
    larva_collisions = param.Boolean(
        True, doc="Whether to allow overlap between larva bodies."
    )

    @property
    def scaling_factor(self):
        return 1000.0 if self.Box2D else 1.0


class SimOps(SimTimeOps, SimSpatialOps):
    pass


# How to load existing


# How to launch


class RuntimeGeneralOps(NestedConf):
    offline = param.Boolean(
        False, doc="Whether to launch a full Larvaworld environment"
    )
    multicore = param.Boolean(False, doc="Whether to use multiple cores")
    store_data = param.Boolean(True, doc="Whether to store the simulation data")


class RuntimeDataOps(NestedConf):
    id = param.Parameter(
        None,
        doc="ID of the simulation. If not specified,set according to runtype and experiment.",
    )
    dir = param.String(
        default=None, label="storage folder", doc="The directory to store data"
    )

    @property
    def data_dir(self):
        if self.dir is not None:
            f = f"{self.dir}/data"
            os.makedirs(f, exist_ok=True)
            return f

    @property
    def plot_dir(self):
        if self.dir is not None:
            f = f"{self.dir}/plots"
            os.makedirs(f, exist_ok=True)
            return f


# What minimum to store, of course will be used to launch as well


class RuntimeOps(RuntimeGeneralOps, RuntimeDataOps):
    pass


class Filesystem(NestedConf):
    read_sequence = param.List(
        label="data columns",
        doc="The sequence of columns in the tracker-exported files.",
    )
    read_metadata = param.Boolean(
        False,
        doc="Whether metadata files are available for the tracker-exported files/folders.",
    )
    folder_pref = param.String(doc="A prefix for detecting a raw-data folder.")
    folder_suff = param.String(doc="A suffix for detecting a raw-data folder.")
    file_pref = param.String(default="", doc="A prefix for detecting a raw-data file.")
    file_suf = param.String(default="", doc="A suffix for detecting a raw-data file.")
    file_sep = param.String(doc="A separator for detecting a raw-data file.")
    structure = param.Selector(
        objects=["per_larva", "per_parameter"],
        doc="Whether each raw file corresponds to all parameters of a single larva or to a single parameter over all larvae.",
    )

    def valid_files_in_folder(self, dir):
        return [
            os.path.join(dir, n)
            for n in os.listdir(dir)
            if (n.endswith(self.file_suf) and n.startswith(self.file_pref))
        ]


class TrackedPointIdx(XYops):
    point_idx = param.Integer(
        default=-1,
        softbounds=(None, 20),
        bounds=(-1, None),
        doc="Index of midline point to use as the larva spatial position. Default is None meaning use the centroid.",
    )
    point = param.String(
        doc="Midline point to use as the larva spatial position. Default is centroid."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_tracked_point()

    @param.depends("Npoints", "point_idx", watch=True)
    def update_tracked_point(self):
        self.param.point_idx.bounds = (hardmin, hardmax) = (-1, self.Npoints)
        self.point_idx = self.param.point_idx.crop_to_bounds(self.point_idx)
        self.point = self.get_track_point(self.point_idx)

    @property
    def point_xy(self):
        return util.nam.xy(self.point)


class SimMetricOps(TrackedPointIdx):
    bend = param.Selector(
        objects=["from_vectors", "from_angles"],
        doc="Whether bending angle is computed as a sum of sequential segmental angles or as the angle between front and rear body vectors.",
    )
    front_vector = PositiveIntegerRangeOrdered(
        (1, 2), softmax=12, doc="The initial & final segment of the front body vector."
    )
    rear_vector = IntegerRangeOrdered(
        (-2, -1),
        softbounds=(-12, 12),
        doc="The initial & final segment of the rear body vector.",
    )
    front_body_ratio = param.Magnitude(
        0.5,
        doc="The fraction of the body considered front, relevant for bend computation from angles.",
    )

    use_component_vel = param.Boolean(
        False,
        doc="Whether to use the component velocity ralative to the axis of forward motion.",
    )

    @param.depends("Npoints", watch=True)
    def update_vectors(self):
        N = self.Npoints
        # self.param.params('front_vector').softbounds = (-N,N)
        # self.param.front_vector.bounds = (-N,N)
        self.param.params("front_vector").bounds = (0, N)

        self.param.params("front_vector")._validate(self.front_vector)
        self.param.params("rear_vector").bounds = (-N, N)
        self.param.params("rear_vector")._validate(self.rear_vector)

    @property
    def Nbend_angles(self):
        return int(np.round(self.front_body_ratio * self.Nangles))

    @property
    def vector_dict(self):
        f1, f2 = self.front_vector
        r1, r2 = self.rear_vector
        return util.AttrDict(
            {
                "front": (f2 - 1, f1 - 1),
                "rear": (r2 - 1, r1 - 1),
                "head": (1, 0),
                "tail": (-1, -2),
            }
        )


class TrackerOps(SimMetricOps, FramerateOps):
    pass
