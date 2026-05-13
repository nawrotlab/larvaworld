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
) -> CompatibilityReport:
    issues: list[CompatibilityIssue] = []

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
    larva_groups = _as_mapping(_get(parameters, "larva_groups", {}))
    if larva_groups is None:
        return CompatibilityReport(issues=tuple(issues))

    for group_id, group_payload in larva_groups.items():
        group_path = f"larva_groups.{group_id}"
        distribution = _as_mapping(_get(group_payload, "distribution", None))
        if distribution is None:
            issues.append(
                CompatibilityIssue(
                    severity="warning",
                    path=f"{group_path}.distribution",
                    message="Missing distribution; skipped compatibility envelope checks.",
                )
            )
            continue

        loc = _pair(_get(distribution, "loc", None))
        if loc is None:
            issues.append(
                CompatibilityIssue(
                    severity="warning",
                    path=f"{group_path}.distribution.loc",
                    message="Distribution center is missing or non-numeric.",
                )
            )
            continue
        if not _point_in_arena(arena, loc):
            issues.append(
                CompatibilityIssue(
                    severity="error",
                    path=f"{group_path}.distribution.loc",
                    message=(
                        f"Distribution center {loc} lies outside the arena boundary."
                    ),
                )
            )

        mode = str(_get(distribution, "mode", "") or "").lower()
        shape = str(_get(distribution, "shape", "") or "").lower()
        scale = _pair(_get(distribution, "scale", None))
        if scale is None:
            issues.append(
                CompatibilityIssue(
                    severity="warning",
                    path=f"{group_path}.distribution.scale",
                    message=(
                        "Distribution scale is missing or non-numeric; skipped envelope check."
                    ),
                )
            )
            continue

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
                        path=f"{group_path}.distribution",
                        message=(
                            f'Unsupported normal distribution shape "{shape}" '
                            "for envelope check."
                        ),
                    )
                )
                continue
            if any(not _point_in_arena(arena, point) for point in points):
                issues.append(
                    CompatibilityIssue(
                        severity="warning",
                        path=f"{group_path}.distribution",
                        message=(
                            "Normal distribution 3-sigma envelope extends "
                            "outside the arena."
                        ),
                    )
                )
            continue

        if mode not in _DETERMINISTIC_MODES:
            issues.append(
                CompatibilityIssue(
                    severity="warning",
                    path=f"{group_path}.distribution.mode",
                    message=(
                        f'Unsupported distribution mode "{mode}" for deterministic '
                        "envelope checks."
                    ),
                )
            )
            continue

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
                    path=f"{group_path}.distribution",
                    message=(
                        "Ambiguous distribution shape/mode combination; "
                        "skipped strict envelope check."
                    ),
                )
            )
            continue
        if any(not _point_in_arena(arena, point) for point in points):
            issues.append(
                CompatibilityIssue(
                    severity="error",
                    path=f"{group_path}.distribution",
                    message="Distribution envelope extends outside the arena.",
                )
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
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _point_in_arena(arena: BoundedArea, point: tuple[float, float]) -> bool:
    polygon = arena.polygon.buffer(_BOUNDARY_TOLERANCE)
    return polygon.covers(Point(float(point[0]), float(point[1])))


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
        radius = max(sx, sy)
        return [
            (x0 + radius, y0),
            (x0 - radius, y0),
            (x0, y0 + radius),
            (x0, y0 - radius),
        ]
    if shape in _OVAL_SHAPES:
        return [
            (x0 + sx, y0),
            (x0 - sx, y0),
            (x0, y0 + sy),
            (x0, y0 - sy),
        ]
    return None
