from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from shapely.geometry import Point

from larvaworld.lib.param.spatial import BoundedArea

__all__ = [
    "CompatibilityIssue",
    "CompatibilityReport",
    "validate_experiment_environment_compatibility",
]


_BOUNDARY_TOLERANCE = 1e-9
_DETERMINISTIC_MODES = {"uniform", "periphery", "grid"}
_RECT_SHAPES = {"rect", "rectangular"}
_CIRCLE_SHAPES = {"circle"}
_OVAL_SHAPES = {"oval"}
_LEGACY_REGISTRY_ENVELOPE_WARNING_EXPERIMENTS = {"tactile_detection"}


@dataclass(frozen=True)
class CompatibilityIssue:
    severity: str
    path: str
    message: str


@dataclass(frozen=True)
class CompatibilityReport:
    issues: tuple[CompatibilityIssue, ...]

    @property
    def errors(self) -> tuple[CompatibilityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[CompatibilityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


def validate_experiment_environment_compatibility(
    parameters: Any,
    *,
    allow_registry_legacy: bool = False,
    experiment_id: str | None = None,
) -> CompatibilityReport:
    issues: list[CompatibilityIssue] = []
    legacy_envelope_warning = (
        allow_registry_legacy
        and experiment_id in _LEGACY_REGISTRY_ENVELOPE_WARNING_EXPERIMENTS
    )

    env_params = _get(parameters, "env_params", None)
    if env_params is None:
        issues.append(
            CompatibilityIssue(
                severity="error",
                path="env_params",
                message="Missing environment parameters.",
            )
        )
        return CompatibilityReport(issues=tuple(issues))

    arena_conf = _get(env_params, "arena", None)
    if arena_conf is None:
        issues.append(
            CompatibilityIssue(
                severity="error",
                path="env_params.arena",
                message="Missing arena configuration.",
            )
        )
        return CompatibilityReport(issues=tuple(issues))

    dims = _pair(_get(arena_conf, "dims", None))
    if dims is None:
        issues.append(
            CompatibilityIssue(
                severity="error",
                path="env_params.arena.dims",
                message="Arena dims must contain two numeric values.",
            )
        )
        return CompatibilityReport(issues=tuple(issues))
    if dims[0] <= 0 or dims[1] <= 0:
        issues.append(
            CompatibilityIssue(
                severity="error",
                path="env_params.arena.dims",
                message="Arena dims must be strictly positive.",
            )
        )
        return CompatibilityReport(issues=tuple(issues))

    geometry = str(_get(arena_conf, "geometry", "rectangular") or "rectangular").lower()
    if geometry == "circular" and abs(float(dims[0]) - float(dims[1])) > 1e-12:
        issues.append(
            CompatibilityIssue(
                severity="warning",
                path="env_params.arena.dims",
                message=(
                    "Circular arena has unequal dims; this may be ambiguous for "
                    "legacy configurations."
                ),
            )
        )

    if geometry not in {"rectangular", "circular"}:
        issues.append(
            CompatibilityIssue(
                severity="warning",
                path="env_params.arena.geometry",
                message=(
                    f'Unsupported arena geometry "{geometry}" for compatibility checks.'
                ),
            )
        )
        return CompatibilityReport(issues=tuple(issues))

    arena = BoundedArea(dims=dims, geometry=geometry)

    food_params = _as_mapping(_get(env_params, "food_params", None))
    if food_params is not None:
        source_units = _as_mapping(_get(food_params, "source_units", {}))
        if source_units is not None:
            for source_id, source_payload in source_units.items():
                source_path = f"env_params.food_params.source_units.{source_id}"
                source_mapping = _as_mapping(source_payload)
                pos = _pair(_get(source_mapping, "pos", None))
                if pos is None:
                    issues.append(
                        CompatibilityIssue(
                            severity="error",
                            path=f"{source_path}.pos",
                            message="Source position must contain two numeric values.",
                        )
                    )
                    continue

                if not _point_in_arena(arena, pos):
                    issues.append(
                        CompatibilityIssue(
                            severity="error",
                            path=f"{source_path}.pos",
                            message=f"Source center {pos} lies outside the arena boundary.",
                        )
                    )
                    continue

                raw_radius = _get(source_mapping, "radius", None)
                radius = _number(raw_radius)
                if raw_radius is None:
                    continue
                if radius is None:
                    issues.append(
                        CompatibilityIssue(
                            severity="warning",
                            path=f"{source_path}.radius",
                            message=(
                                "Source radius must be numeric; skipped radius envelope check."
                            ),
                        )
                    )
                    continue
                if radius < 0.0:
                    issues.append(
                        CompatibilityIssue(
                            severity="warning",
                            path=f"{source_path}.radius",
                            message=(
                                "Source radius must be non-negative; skipped radius envelope check."
                            ),
                        )
                    )
                    continue
                radius_points = _source_radius_boundary_points(
                    center=pos, radius=radius
                )
                if any(not _point_in_arena(arena, point) for point in radius_points):
                    issues.append(
                        CompatibilityIssue(
                            severity="warning",
                            path=source_path,
                            message=(
                                "Source radius envelope extends outside the arena boundary."
                            ),
                        )
                    )

        source_groups = _as_mapping(_get(food_params, "source_groups", {}))
        if source_groups is not None:
            for group_id, group_payload in source_groups.items():
                distribution = _as_mapping(_get(group_payload, "distribution", None))
                _validate_distribution(
                    issues=issues,
                    arena=arena,
                    distribution=distribution,
                    path=f"env_params.food_params.source_groups.{group_id}.distribution",
                    legacy_envelope_warning=legacy_envelope_warning,
                )

    larva_groups = _as_mapping(_get(parameters, "larva_groups", {}))
    if larva_groups is None:
        return CompatibilityReport(issues=tuple(issues))

    for group_id, group_payload in larva_groups.items():
        distribution = _as_mapping(_get(group_payload, "distribution", None))
        _validate_distribution(
            issues=issues,
            arena=arena,
            distribution=distribution,
            path=f"larva_groups.{group_id}.distribution",
            legacy_envelope_warning=legacy_envelope_warning,
        )

    return CompatibilityReport(issues=tuple(issues))


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "items"):
        return value
    return None


def _get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    mapping = _as_mapping(value)
    if mapping is not None:
        try:
            return mapping.get(key, default)
        except Exception:
            pass
    return getattr(value, key, default)


def _pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if hasattr(value, "__array__"):
        try:
            value = list(value)
        except Exception:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _point_in_arena(arena: BoundedArea, point: tuple[float, float]) -> bool:
    polygon = arena.polygon.buffer(_BOUNDARY_TOLERANCE)
    return polygon.covers(Point(float(point[0]), float(point[1])))


def _source_radius_boundary_points(
    *, center: tuple[float, float], radius: float
) -> list[tuple[float, float]]:
    x0, y0 = float(center[0]), float(center[1])
    r = abs(float(radius))
    return [
        (x0 + r, y0),
        (x0 - r, y0),
        (x0, y0 + r),
        (x0, y0 - r),
    ]


def _validate_distribution(
    *,
    issues: list[CompatibilityIssue],
    arena: BoundedArea,
    distribution: Mapping[str, Any] | None,
    path: str,
    legacy_envelope_warning: bool = False,
) -> None:
    if distribution is None:
        issues.append(
            CompatibilityIssue(
                severity="warning",
                path=path,
                message="Missing distribution; skipped compatibility envelope checks.",
            )
        )
        return

    loc = _pair(_get(distribution, "loc", None))
    if loc is None:
        issues.append(
            CompatibilityIssue(
                severity="warning",
                path=f"{path}.loc",
                message="Distribution center must contain two numeric values.",
            )
        )
        return
    if not _point_in_arena(arena, loc):
        issues.append(
            CompatibilityIssue(
                severity="error",
                path=f"{path}.loc",
                message=f"Distribution center {loc} lies outside the arena boundary.",
            )
        )

    mode = str(_get(distribution, "mode", "") or "").lower()
    shape = str(_get(distribution, "shape", "") or "").lower()
    scale = _pair(_get(distribution, "scale", None))
    if scale is None:
        issues.append(
            CompatibilityIssue(
                severity="warning",
                path=f"{path}.scale",
                message=(
                    "Distribution scale must contain two numeric values; skipped envelope check."
                ),
            )
        )
        return

    if mode == "normal":
        points = _distribution_boundary_points(
            loc=loc,
            scale=(scale[0] * 1.5, scale[1] * 1.5),
            shape=shape,
            mode=mode,
        )
        if points is None:
            issues.append(
                CompatibilityIssue(
                    severity="warning",
                    path=path,
                    message=(
                        f'Unsupported normal distribution shape "{shape}" '
                        "for envelope check."
                    ),
                )
            )
            return
        if any(not _point_in_arena(arena, point) for point in points):
            issues.append(
                CompatibilityIssue(
                    severity="warning",
                    path=path,
                    message="Normal distribution 3-sigma envelope extends outside the arena.",
                )
            )
        return

    if mode not in _DETERMINISTIC_MODES:
        issues.append(
            CompatibilityIssue(
                severity="warning",
                path=f"{path}.mode",
                message=(
                    f'Unsupported distribution mode "{mode}" for deterministic '
                    "envelope checks."
                ),
            )
        )
        return

    points = _distribution_boundary_points(
        loc=loc,
        scale=scale,
        shape=shape,
        mode=mode,
    )
    if points is None:
        issues.append(
            CompatibilityIssue(
                severity="warning",
                path=path,
                message=(
                    "Ambiguous distribution shape/mode combination; "
                    "skipped strict envelope check."
                ),
            )
        )
        return
    if any(not _point_in_arena(arena, point) for point in points):
        issues.append(
            CompatibilityIssue(
                severity="warning" if legacy_envelope_warning else "error",
                path=path,
                message="Distribution envelope extends outside the arena.",
            )
        )


def _distribution_boundary_points(
    *,
    loc: tuple[float, float],
    scale: tuple[float, float],
    shape: str,
    mode: str,
) -> list[tuple[float, float]] | None:
    x0, y0 = float(loc[0]), float(loc[1])
    sx, sy = abs(float(scale[0])), abs(float(scale[1]))

    if mode == "grid":
        # `grid` uses area-like scale semantics; use conservative half extents.
        hx, hy = sx / 2.0, sy / 2.0
        return [
            (x0 - hx, y0 - hy),
            (x0 - hx, y0 + hy),
            (x0 + hx, y0 - hy),
            (x0 + hx, y0 + hy),
        ]

    if shape in _RECT_SHAPES:
        return [
            (x0 - sx, y0 - sy),
            (x0 - sx, y0 + sy),
            (x0 + sx, y0 - sy),
            (x0 + sx, y0 + sy),
        ]
    if shape in _CIRCLE_SHAPES:
        return [
            (x0 + sx, y0),
            (x0 - sx, y0),
            (x0, y0 + sy),
            (x0, y0 - sy),
        ]
    if shape in _OVAL_SHAPES:
        return [
            (x0 + sx, y0),
            (x0 - sx, y0),
            (x0, y0 + sy),
            (x0, y0 - sy),
        ]
    return None
