import param

from .custom import ClassAttr, OptionalPositiveNumber, OptionalSelector
from .nested_parameter_group import NestedConf

__all__ = [
    "PreprocessConf",
    "ProcessConf",
    "EnrichConf",
]

__displayname__ = "Enrichment configuration"


class PreprocessConf(NestedConf):
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
