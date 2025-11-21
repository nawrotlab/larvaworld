from __future__ import annotations
from typing import Any, Optional, Sequence

import param

from .custom import ClassAttr, OptionalPositiveNumber, OptionalSelector
from .nested_parameter_group import NestedConf

__all__: list[str] = [
    "PreprocessConf",
    "ProcessConf",
    "EnrichConf",
]

__displayname__ = "Enrichment configuration"


class PreprocessConf(NestedConf):
    """
    Preprocessing configuration for raw tracker data.

    Defines spatial coordinate transformations, filtering, and
    data cleaning operations applied before analysis.

    Attributes:
        rescale_by: Spatial rescaling factor in meters (optional)
        filter_f: Low-pass filter cutoff frequency in Hz (optional)
        transposition: Coordinate transposition mode ('origin', 'arena', 'center', or None)
        interpolate_nans: Interpolate missing values (default: False)
        drop_collisions: Remove collision timepoints (default: False)

    Example:
        >>> preproc = PreprocessConf(rescale_by=0.001, filter_f=1.0, transposition='center')
    """

    rescale_by = OptionalPositiveNumber(
        softmax=1000.0,
        step=0.001,
        doc="Whether to rescale spatial coordinates by a scalar in meters.",
    )
    filter_f = OptionalPositiveNumber(
        softmax=5.0,
        step=0.01,
        doc="Whether to filter spatial coordinates by a grade-1 low-pass filter of the given cut-off frequency.",
    )
    transposition = OptionalSelector(
        ["origin", "arena", "center"], doc="Whether to transpose spatial coordinates."
    )
    interpolate_nans = param.Boolean(
        False, doc="Whether to interpolate missing values."
    )
    drop_collisions = param.Boolean(
        False, doc="Whether to drop timepoints where larva collisions are detected."
    )


class ProcessConf(NestedConf):
    """
    Processing configuration for derived metrics computation.

    Defines processing pipelines and parameters for computing spatial,
    angular, source-related, and behavioral metrics from trajectories.

    Attributes:
        proc_keys: Active processing pipelines (default: ['angular', 'spatial'])
        dsp_starts: Dispersal computation start times in seconds (default: [0.0])
        dsp_stops: Dispersal computation stop times in seconds (default: [40.0, 60.0])
        tor_durs: Tortuosity time windows in seconds (default: [5, 10, 20])

    Example:
        >>> proc = ProcessConf(proc_keys=['spatial', 'angular', 'source'], tor_durs=[10, 30])
    """

    proc_keys = param.ListSelector(
        default=["angular", "spatial"],
        objects=["angular", "spatial", "source", "PI", "wind"],
        doc="The processing pipelines",
    )
    dsp_starts = param.List(
        default=[0.0],
        item_type=float,
        doc="The starting times for dispersal computation.",
    )
    dsp_stops = param.List(
        default=[40.0, 60.0],
        item_type=float,
        doc="The stopping times for dispersal computation.",
    )
    tor_durs = param.List(
        default=[5, 10, 20],
        item_type=int,
        doc="The time windows for tortuosity computation.",
    )


class EnrichConf(ProcessConf):
    """
    Complete enrichment configuration for dataset processing.

    Extends ProcessConf with preprocessing settings and annotation pipelines,
    providing full dataset enrichment workflow configuration.

    Attributes:
        pre_kws: Preprocessing configuration (PreprocessConf instance)
        anot_keys: Active annotation pipelines (default: bout_detection, bout_distribution, interference)
        recompute: Force recomputation of existing results (default: False)
        mode: Processing mode ('minimal' or 'full')

    Example:
        >>> enrich = EnrichConf(
        ...     pre_kws={'rescale_by': 0.001},
        ...     proc_keys=['spatial', 'angular'],
        ...     anot_keys=['bout_detection'],
        ...     mode='full'
        ... )
        >>> enrich_simple = EnrichConf.spatial_proc()  # Preset config
    """

    pre_kws = ClassAttr(PreprocessConf, doc="The preprocessing pipelines")
    anot_keys = param.ListSelector(
        default=["bout_detection", "bout_distribution", "interference"],
        objects=[
            "bout_detection",
            "bout_distribution",
            "interference",
            "source_attraction",
            "patch_residency",
        ],
        doc="The annotation pipelines",
    )
    recompute = param.Boolean(False, doc="Whether to recompute")
    mode = param.Selector(objects=["minimal", "full"], doc="The processing mode")

    @classmethod
    def no_tor_dsp(cls, **kwargs):
        return cls(tor_durs=[], dsp_starts=[], dsp_stops=[], **kwargs)

    @classmethod
    def single_proc(cls, k, **kwargs):
        return cls.no_tor_dsp(proc_keys=[k], anot_keys=[], **kwargs)

    @classmethod
    def PI_proc(cls, **kwargs):
        return cls.single_proc(k="PI", **kwargs)

    @classmethod
    def spatial_proc(cls, **kwargs):
        return cls.single_proc(k="spatial", **kwargs)

    @classmethod
    def source_proc(cls, anot_keys=[], **kwargs):
        return cls.no_tor_dsp(
            proc_keys=["spatial", "angular", "source"], anot_keys=anot_keys, **kwargs
        )

    @classmethod
    def wind_proc(cls, anot_keys=[], **kwargs):
        return cls.no_tor_dsp(
            proc_keys=["spatial", "angular", "wind"], anot_keys=anot_keys, **kwargs
        )

    @classmethod
    def sourcewind_proc(cls, anot_keys=[], **kwargs):
        return cls.no_tor_dsp(
            proc_keys=["spatial", "angular", "source", "wind"],
            anot_keys=anot_keys,
            **kwargs,
        )

    @classmethod
    def patch_proc(cls, **kwargs):
        return cls.no_tor_dsp(
            proc_keys=["spatial", "angular", "source"],
            anot_keys=[
                "bout_detection",
                "bout_distribution",
                "interference",
                "patch_residency",
            ],
            **kwargs,
        )
