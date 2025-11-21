from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

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

__all__: list[str] = [
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
]

__displayname__ = "Configuration parameter groups"


class FramerateOps(NestedConf):
    """
    Framerate and timestep configuration parameter group.

    Manages bidirectional synchronization between framerate (Hz) and
    timestep (seconds), ensuring dt = 1/fr relationship is maintained.

    Attributes:
        fr: Framerate in Hz (default: 10 Hz)
        dt: Timestep in seconds (default: 0.1 s)
        constant_framerate: Whether framerate is constant (default: True)

    Example:
        >>> fr_ops = FramerateOps(fr=20)  # dt auto-updates to 0.05
        >>> fr_ops.dt = 0.1  # fr auto-updates to 10
    """

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

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if "dt" in kwargs:
            self.update_framerate()
        elif "fr" in kwargs:
            self.update_timestep()

    @param.depends("dt", watch=True)
    def update_framerate(self) -> None:
        self.fr = 1 / self.dt

    @param.depends("fr", watch=True)
    def update_timestep(self) -> None:
        self.dt = 1 / self.fr
        # raise


class XYops(NestedConf):
    """
    XY coordinate tracking configuration parameter group.

    Manages spatial coordinate tracking parameters for larva body shape,
    including midline points, contour points, and derived quantities
    (segments, angles, orientations).

    Attributes:
        XY_unit: Spatial unit ('m' or 'mm', default: 'm')
        Npoints: Number of midline tracking points (default: 3)
        Ncontour: Number of contour tracking points (default: 0)
        Nangles: Computed number of bend angles (Npoints - 2)
        Nsegs: Computed number of body segments (Npoints - 1)

    Example:
        >>> xy_ops = XYops(Npoints=11, Ncontour=20, XY_unit='mm')
        >>> xy_ops.Nangles  # 9 bend angles
        >>> xy_ops.midline_xy  # ['point0_x', 'point0_y', ...]
    """

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
    def Nangles(self) -> int:
        return np.clip(self.Npoints - 2, a_min=0, a_max=None)

    @property
    def Nsegs(self) -> int:
        return np.clip(self.Npoints - 1, a_min=0, a_max=None)

    @property
    def angles(self) -> util.SuperList:
        return util.SuperList([f"angle{i}" for i in range(self.Nangles)])

    @property
    def midline_points(self) -> util.SuperList:
        return util.nam.midline(self.Npoints, type="point")

    @property
    def midline_segs(self) -> util.SuperList:
        return util.nam.midline(self.Nsegs, type="seg")

    @property
    def midline_seg_xy(self, flat: bool = True) -> util.SuperList:
        return util.nam.xy(self.midline_segs, flat=flat)

    @property
    def seg_orientations(self) -> util.SuperList:
        return util.nam.orient(self.midline_segs)

    @property
    def midline_xy(self, flat: bool = True) -> util.SuperList:
        return util.nam.xy(self.midline_points, flat=flat)

    @property
    def contour_points(self) -> util.SuperList:
        return util.nam.contour(self.Ncontour)

    @property
    def contour_xy(self, flat: bool = True) -> util.SuperList:
        return util.nam.xy(self.contour_points, flat=flat)

    @property
    def centroid_xy(self) -> util.SuperList:
        return util.nam.xy("centroid")

    @property
    def traj_xy(self) -> util.SuperList:
        return util.nam.xy("")

    @property
    def all_xy(self, flat: bool = True) -> util.SuperList:
        return util.nam.xy(
            self.midline_points + self.contour_points + ["centroid", ""], flat=flat
        )

    def get_track_point(self, idx: int) -> str:
        if idx == -1:
            return "centroid"
        else:
            return self.midline_points[idx - 1]

    def get_midline_xy_data(self, s: Any) -> np.ndarray:
        xy = self.midline_xy
        assert xy.exist_in(s)
        return s[xy].values.reshape([-1, self.Npoints, 2])

    def get_contour_xy_data(self, s: Any) -> np.ndarray:
        xy = self.contour_xy
        assert xy.exist_in(s)
        return s[xy].values.reshape([-1, self.Ncontour, 2])


class SimTimeOps(FramerateOps):
    """
    Simulation time configuration parameter group.

    Extends FramerateOps with duration and step count, maintaining
    bidirectional synchronization: Nsteps = duration*60/dt.

    Attributes:
        duration: Simulation duration in minutes (optional)
        Nsteps: Number of simulation timesteps (optional)
        fr: Inherited framerate in Hz
        dt: Inherited timestep in seconds

    Example:
        >>> time_ops = SimTimeOps(duration=5.0, dt=0.1)  # Nsteps auto-updates to 3000
        >>> time_ops.Nsteps = 6000  # duration auto-updates to 10.0
    """

    duration = OptionalPositiveNumber(
        softmax=100.0, step=0.1, doc="The duration of the simulation in minutes."
    )
    Nsteps = OptionalPositiveInteger(
        label="# simulation timesteps", doc="The number of simulation timesteps."
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.update_Nsteps()

    @param.depends("duration", "dt", watch=True)
    def update_Nsteps(self) -> None:
        if self.duration is not None:
            self.Nsteps = int(self.duration * 60 / self.dt)
        else:
            self.Nsteps = None

    @param.depends("Nsteps", watch=True)
    def update_duration(self) -> None:
        if self.Nsteps is not None:
            self.duration = self.Nsteps * self.dt / 60
        else:
            self.duration = None


class SimSpatialOps(NestedConf):
    """
    Simulation spatial operations configuration parameter group.

    Manages physics engine settings and collision detection parameters
    for spatial simulation dynamics.

    Attributes:
        Box2D: Use Box2D physics engine (default: False)
        larva_collisions: Allow body overlap (default: True)
        scaling_factor: Spatial scaling coefficient (always 1.0)

    Example:
        >>> spatial_ops = SimSpatialOps(Box2D=True, larva_collisions=False)
    """

    Box2D = param.Boolean(False, doc="Whether to use the Box2D physics engine or not.")
    larva_collisions = param.Boolean(
        True, doc="Whether to allow overlap between larva bodies."
    )

    @property
    def scaling_factor(self) -> float:
        return 1000.0 if self.Box2D else 1.0


class SimOps(SimTimeOps, SimSpatialOps):
    """
    Combined simulation operations parameter group.

    Merges temporal (SimTimeOps) and spatial (SimSpatialOps) configuration
    into unified simulation control parameters.

    Example:
        >>> sim_ops = SimOps(duration=10.0, dt=0.1, Box2D=True)
    """

    pass


# How to load existing


# How to launch


class RuntimeGeneralOps(NestedConf):
    """
    General runtime operations parameter group.

    Controls basic runtime execution modes (offline, multicore, storage).

    Attributes:
        offline: Offline mode without full environment (default: False)
        multicore: Use parallel execution (default: False)
        store_data: Store simulation data (default: True)

    Example:
        >>> runtime_gen = RuntimeGeneralOps(multicore=True, store_data=False)
    """

    offline = param.Boolean(
        False, doc="Whether to launch a full Larvaworld environment"
    )
    multicore = param.Boolean(False, doc="Whether to use multiple cores")
    store_data = param.Boolean(True, doc="Whether to store the simulation data")


class RuntimeDataOps(NestedConf):
    """
    Runtime data directory operations parameter group.

    Manages data and plot directory paths, auto-creating subdirectories
    when base dir is specified.

    Attributes:
        id: Simulation run ID (optional)
        dir: Base storage directory path (optional)
        data_dir: Auto-created data subdirectory ({dir}/data)
        plot_dir: Auto-created plots subdirectory ({dir}/plots)

    Example:
        >>> data_ops = RuntimeDataOps(id='exp001', dir='/path/to/output')
        >>> data_ops.data_dir  # '/path/to/output/data' (auto-created)
    """

    id = param.Parameter(
        None,
        doc="ID of the simulation. If not specified,set according to runtype and experiment.",
    )
    dir = param.String(
        default=None, label="storage folder", doc="The directory to store data"
    )

    @property
    def data_dir(self) -> Optional[str]:
        if self.dir is not None:
            f = f"{self.dir}/data"
            os.makedirs(f, exist_ok=True)
            return f

    @property
    def plot_dir(self) -> Optional[str]:
        if self.dir is not None:
            f = f"{self.dir}/plots"
            os.makedirs(f, exist_ok=True)
            return f


# What minimum to store, of course will be used to launch as well


class RuntimeOps(RuntimeGeneralOps, RuntimeDataOps):
    """
    Combined runtime operations parameter group.

    Merges general and data runtime operations into unified runtime control.

    Example:
        >>> runtime = RuntimeOps(show_display=True, store_data=True, save_video=False)
    """

    pass


class Filesystem(NestedConf):
    """
    Filesystem configuration for raw tracker data import.

    Defines file/folder naming conventions, data format, and structure
    for importing experimental tracker datasets into Larvaworld.

    Attributes:
        read_sequence: Column sequence in tracker files
        read_metadata: Metadata files available (default: False)
        folder_pref: Raw-data folder prefix pattern
        folder_suff: Raw-data folder suffix pattern
        file_pref: Raw-data file prefix (default: "")
        file_suf: Raw-data file suffix (default: "")
        file_sep: File name separator pattern
        structure: File organization ('per_larva' or 'per_parameter')

    Example:
        >>> fs = Filesystem(
        ...     file_pref='trial_',
        ...     file_suf='.csv',
        ...     structure='per_larva'
        ... )
        >>> fs.valid_files_in_folder('/path/to/data')
    """

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

    def valid_files_in_folder(self, dir) -> list[str]:
        return [
            os.path.join(dir, n)
            for n in os.listdir(dir)
            if (n.endswith(self.file_suf) and n.startswith(self.file_pref))
        ]


class TrackedPointIdx(XYops):
    """
    Tracked point index configuration parameter group.

    Extends XYops to specify which midline point serves as larva position
    reference, with automatic point name synchronization.

    Attributes:
        point_idx: Midline point index (-1=centroid, 0 to Npoints-1, default: -1)
        point: Point name string (auto-synced with point_idx)

    Example:
        >>> tracked = TrackedPointIdx(Npoints=11, point_idx=5)
        >>> tracked.point  # 'point4' (auto-updated)
    """

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
    def update_tracked_point(self) -> None:
        self.param.point_idx.bounds = (hardmin, hardmax) = (-1, self.Npoints)
        self.point_idx = self.param.point_idx.crop_to_bounds(self.point_idx)
        self.point = self.get_track_point(self.point_idx)

    @property
    def point_xy(self) -> util.SuperList:
        return util.nam.xy(self.point)


class SimMetricOps(TrackedPointIdx):
    """
    Simulation metrics computation configuration parameter group.

    Extends TrackedPointIdx with bend angle computation settings,
    front/rear body vector definitions, and velocity component options.

    Attributes:
        bend: Bend computation method ('from_vectors' or 'from_angles')
        front_vector: Front body segment range (1-indexed, ordered)
        rear_vector: Rear body segment range (can be negative, ordered)
        front_body_ratio: Front body fraction for angle-based bend (0-1, default: 0.5)
        use_component_vel: Use velocity component along forward axis (default: False)
        Nbend_angles: Computed front bend angles count (property)
        vector_dict: Body vector definitions dict (property)

    Example:
        >>> metrics = SimMetricOps(
        ...     Npoints=11,
        ...     bend='from_vectors',
        ...     front_vector=(1, 5),
        ...     rear_vector=(-5, -1),
        ...     front_body_ratio=0.6
        ... )
        >>> metrics.vector_dict  # {'front': (4, 0), 'rear': (-6, -2), ...}
    """

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
    def update_vectors(self) -> None:
        N = self.Npoints
        # self.param.params('front_vector').softbounds = (-N,N)
        # self.param.front_vector.bounds = (-N,N)
        self.param.params("front_vector").bounds = (0, N)

        self.param.params("front_vector")._validate(self.front_vector)
        self.param.params("rear_vector").bounds = (-N, N)
        self.param.params("rear_vector")._validate(self.rear_vector)

    @property
    def Nbend_angles(self) -> int:
        return int(np.round(self.front_body_ratio * self.Nangles))

    @property
    def vector_dict(self) -> util.AttrDict:
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
    """
    Combined tracker operations parameter group.

    Merges metrics computation (SimMetricOps) and framerate (FramerateOps)
    into unified tracker configuration for experimental data import.

    Example:
        >>> tracker_ops = TrackerOps(
        ...     Npoints=11,
        ...     fr=10,
        ...     bend='from_angles'
        ... )
    """

    pass
