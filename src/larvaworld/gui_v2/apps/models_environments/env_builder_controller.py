from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable

from larvaworld.lib import reg, util
from larvaworld.lib.reg.generators import EnvConf
from larvaworld.portal.models_architecture.environment_builder_common import (
    BORDER_SEGMENT,
    DEFAULT_BORDER_WIDTH,
    DEFAULT_GROUP_N,
    DEFAULT_GROUP_SCALE,
    DEFAULT_SOURCE_GROUP_RADIUS,
    DEFAULT_SOURCE_UNIT_RADIUS,
    EnvBuilderObjectRow,
    SOURCE_GROUP,
    SOURCE_UNIT,
    add_border_segment,
    add_source_group,
    add_source_unit,
    build_canvas_state,
    delete_border_segment,
    delete_source_group,
    delete_source_unit,
    normalize_group_shape,
    normalize_preset_filename,
    object_rows_from_payload,
    payload_with_object_rows,
    translate_environment_payload,
    update_border_segment,
    update_source_group,
    update_source_unit,
)
from larvaworld.portal.workspace import (
    WorkspaceError,
    get_active_workspace,
    get_workspace_dir,
)


class EnvBuilderController:
    def __init__(self) -> None:
        self._listeners: list[Callable[[], None]] = []
        self._payload = translate_environment_payload(EnvConf().nestedConf)
        self.object_rows = object_rows_from_payload(self._payload)
        self.canvas_state = build_canvas_state(self._payload)
        self.selected_object_id: str | None = None
        self.preset_name = "environment_builder_config"
        self.preset_source = "default"
        self.dirty = False
        self.status_message = "Ready."
        self.interaction_mode = "select"
        self.pending_border_start: tuple[float, float] | None = None
        self.active_workspace = get_active_workspace()
        self.current_export_json = self.export_json()

    @property
    def payload(self) -> dict[str, Any]:
        return self._payload

    def add_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in list(self._listeners):
            callback()

    def set_status(self, message: str, *, dirty: bool | None = None) -> None:
        self.status_message = message
        if dirty is not None:
            self.dirty = dirty
        self._notify()

    def set_interaction_mode(self, mode: str) -> None:
        allowed = {
            "select",
            "move",
            "erase",
            "add_unit",
            "add_group",
            "add_border",
        }
        if mode not in allowed:
            raise ValueError(f"Unsupported interaction mode: {mode}")
        self.interaction_mode = mode
        self.pending_border_start = None
        self.set_status(f"Interaction mode set to {mode}.")

    def set_preset_name(self, name: str) -> None:
        self.preset_name = str(name).strip() or "environment_builder_config"
        self._notify()

    def set_arena(
        self,
        *,
        geometry: str | None = None,
        width: float | None = None,
        height: float | None = None,
        torus: bool | None = None,
    ) -> None:
        payload = translate_environment_payload(self._payload)
        arena = dict(payload.get("arena") or {})
        current_geometry = str(arena.get("geometry") or "rectangular")
        current_dims = arena.get("dims") or (0.2, 0.2)
        try:
            current_width = float(current_dims[0])
        except Exception:
            current_width = 0.2
        try:
            current_height = float(current_dims[1])
        except Exception:
            current_height = current_width

        new_geometry = str(geometry or current_geometry).strip() or "rectangular"
        if new_geometry == "circular":
            radius = (
                width if width is not None else min(current_width, current_height) / 2.0
            )
            dims = (float(radius) * 2.0, float(radius) * 2.0)
        else:
            dims = (
                float(width if width is not None else current_width),
                float(height if height is not None else current_height),
            )

        arena["geometry"] = new_geometry
        arena["dims"] = dims
        if torus is not None:
            arena["torus"] = bool(torus)

        payload["arena"] = arena
        self._apply_payload(payload, "edit", selected_object_id=self.selected_object_id)
        self.set_status("Arena updated.", dirty=True)

    def current_row(self) -> EnvBuilderObjectRow | None:
        if not self.selected_object_id:
            return None
        for row in self.object_rows:
            if row.object_id == self.selected_object_id:
                return row
        return None

    def select_object(self, object_id: str | None) -> None:
        if object_id is None:
            self.selected_object_id = None
            self._notify()
            return
        if any(row.object_id == object_id for row in self.object_rows):
            self.selected_object_id = object_id
        else:
            self.selected_object_id = None
        self._notify()

    def load_default(self) -> None:
        self._apply_payload(
            translate_environment_payload(EnvConf().nestedConf), "default"
        )

    def load_registry(self, name: str) -> None:
        payload = reg.conf.Env.getID(name)
        self._apply_payload(payload, f"registry:{name}", preset_name=name)
        self.set_status(f'Loaded registry environment "{name}".', dirty=False)

    def load_workspace_json(self, name_or_path: str) -> None:
        path = self._resolve_workspace_path(name_or_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise WorkspaceError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise WorkspaceError(f"Failed to decode JSON: {exc}") from exc
        self._apply_payload(payload, f"workspace:{path.stem}", preset_name=path.stem)
        self.set_status(f'Loaded workspace preset "{path.stem}".', dirty=False)

    def save_workspace(self, name: str | None = None) -> Path:
        target_name = (
            str(name or self.preset_name).strip() or "environment_builder_config"
        )
        filename = normalize_preset_filename(target_name)
        try:
            preset_dir = get_workspace_dir("environments")
        except WorkspaceError as exc:
            self.set_status(
                f"Cannot save workspace preset without an active workspace: {exc}"
            )
            raise
        preset_dir.mkdir(parents=True, exist_ok=True)
        path = preset_dir / filename
        validation_error = self.validate_for_save()
        if validation_error is not None:
            self.set_status(f"{validation_error} Preset was not saved.")
            raise WorkspaceError(validation_error)
        path.write_text(self.export_json(), encoding="utf-8")
        self.preset_name = Path(filename).stem
        self.preset_source = f"workspace:{self.preset_name}"
        self.dirty = False
        self.current_export_json = self.export_json()
        self.set_status(
            f'Saved environment preset "{self.preset_name}" to the workspace.'
        )
        self._notify()
        return path

    def export_payload(self) -> dict[str, Any]:
        return self._payload

    def export_json(self) -> str:
        return json.dumps(self._payload, indent=2) + "\n"

    def reset_to_default(self) -> None:
        self._apply_payload(
            EnvConf().nestedConf, "default", preset_name="environment_builder_config"
        )
        self.set_status("Reset to default environment.", dirty=False)

    def add_source_unit(
        self,
        *,
        x: float,
        y: float,
        object_id: str | None = None,
    ) -> EnvBuilderObjectRow:
        row = EnvBuilderObjectRow(
            object_id=object_id or self._next_object_id("food"),
            object_type=SOURCE_UNIT,
            x=float(x),
            y=float(y),
            radius=DEFAULT_SOURCE_UNIT_RADIUS,
            color="#4caf50",
            amount=0.0,
            substrate_type="standard",
            substrate_quality=1.0,
        )
        self._apply_payload(
            add_source_unit(self._payload, row),
            "edit",
            selected_object_id=row.object_id,
        )
        self.set_status(f'Source unit "{row.object_id}" added.', dirty=True)
        return row

    def add_source_group(
        self,
        *,
        x: float = 0.0,
        y: float = 0.0,
        object_id: str | None = None,
    ) -> EnvBuilderObjectRow:
        row = EnvBuilderObjectRow(
            object_id=object_id or self._next_object_id("group"),
            object_type=SOURCE_GROUP,
            x=float(x),
            y=float(y),
            radius=DEFAULT_SOURCE_GROUP_RADIUS,
            color="#6688aa",
            amount=0.0,
            substrate_type="standard",
            substrate_quality=1.0,
            distribution_mode="uniform",
            distribution_shape="circle",
            distribution_n=DEFAULT_GROUP_N,
            distribution_scale_x=DEFAULT_GROUP_SCALE,
            distribution_scale_y=DEFAULT_GROUP_SCALE,
            distribution_show_shape=True,
        )
        self._apply_payload(
            add_source_group(self._payload, row),
            "edit",
            selected_object_id=row.object_id,
        )
        self.set_status(f'Source group "{row.object_id}" added.', dirty=True)
        return row

    def add_border_segment(
        self,
        *,
        start: tuple[float, float],
        end: tuple[float, float],
        object_id: str | None = None,
    ) -> EnvBuilderObjectRow:
        row = EnvBuilderObjectRow(
            object_id=object_id or self._next_object_id("border"),
            object_type=BORDER_SEGMENT,
            x=float(start[0]),
            y=float(start[1]),
            x2=float(end[0]),
            y2=float(end[1]),
            width=DEFAULT_BORDER_WIDTH,
            color="#333333",
        )
        self._apply_payload(
            add_border_segment(self._payload, row),
            "edit",
            selected_object_id=row.object_id,
        )
        self.set_status(f'Border segment "{row.object_id}" added.', dirty=True)
        return row

    def delete_object(self, object_id: str) -> None:
        row = self._row_by_id(object_id)
        if row is None:
            return
        if row.object_type == SOURCE_UNIT:
            payload = delete_source_unit(self._payload, row.object_id)
        elif row.object_type == SOURCE_GROUP:
            payload = delete_source_group(self._payload, row.object_id)
        else:
            payload = delete_border_segment(self._payload, row.object_id)
        self._apply_payload(payload, "edit")
        self.set_status(f'Object "{object_id}" deleted.', dirty=True)

    def update_selected_object(self, **changes: Any) -> None:
        row = self.current_row()
        if row is None:
            return
        updated = replace(row, **changes)
        if updated.object_type == SOURCE_UNIT:
            payload = update_source_unit(self._payload, updated)
        elif updated.object_type == SOURCE_GROUP:
            payload = update_source_group(self._payload, updated)
        else:
            payload = update_border_segment(self._payload, updated)
        self._apply_payload(payload, "edit", selected_object_id=updated.object_id)
        self.set_status(f'Object "{updated.object_id}" updated.', dirty=True)

    def sync_scene_items(self, items: list[Any]) -> None:
        updated_rows = list(self.object_rows)
        changed = False
        for index, row in enumerate(updated_rows):
            item = self._scene_item_by_id(items, row.object_id)
            if item is None:
                continue
            current = self._row_from_scene_item(row, item)
            if current != row:
                updated_rows[index] = current
                changed = True
        if not changed:
            return
        self._apply_payload(
            payload_with_object_rows(self._payload, updated_rows),
            "edit",
            selected_object_id=self.selected_object_id,
        )
        self.set_status("Scene changes applied.", dirty=True)

    def canvas_click(
        self,
        *,
        world_x: float,
        world_y: float,
        clicked_object_id: str | None = None,
    ) -> None:
        if self.interaction_mode == "select":
            self.select_object(clicked_object_id)
            return
        if self.interaction_mode == "erase":
            if clicked_object_id:
                self.delete_object(clicked_object_id)
            return
        if self.interaction_mode == "move":
            if clicked_object_id:
                self.select_object(clicked_object_id)
            return
        if self.interaction_mode == "add_unit":
            self.add_source_unit(x=world_x, y=world_y)
            return
        if self.interaction_mode == "add_group":
            self.add_source_group(x=world_x, y=world_y)
            return
        if self.interaction_mode == "add_border":
            if self.pending_border_start is None:
                self.pending_border_start = (world_x, world_y)
                self.set_status(
                    "Border start point placed. Click a second point to finish the segment."
                )
                return
            start = self.pending_border_start
            self.pending_border_start = None
            self.add_border_segment(start=start, end=(world_x, world_y))
            return

    def validate_for_save(self) -> str | None:
        arena = self._payload.get("arena") or {}
        dims = arena.get("dims") or (0.2, 0.2)
        try:
            width = float(dims[0])
            height = float(dims[1])
        except Exception:
            width = 0.2
            height = 0.2
        half_w = width / 2.0
        half_h = height / 2.0
        for row in self.object_rows:
            points = [(row.x, row.y)]
            if row.object_type == BORDER_SEGMENT:
                points.append((row.x2, row.y2))
            for point in points:
                if point[0] is None or point[1] is None:
                    continue
                if abs(float(point[0])) > half_w or abs(float(point[1])) > half_h:
                    return (
                        f'Object "{row.object_id}" has coordinates outside the arena.'
                    )
        return None

    def _apply_payload(
        self,
        payload: Any,
        source: str,
        *,
        preset_name: str | None = None,
        selected_object_id: str | None = None,
    ) -> None:
        self._payload = translate_environment_payload(payload)
        self.object_rows = object_rows_from_payload(self._payload)
        self.canvas_state = build_canvas_state(self._payload)
        if preset_name is not None:
            self.preset_name = preset_name
        self.preset_source = source
        self.dirty = False if source != "edit" else self.dirty
        if selected_object_id is not None and any(
            row.object_id == selected_object_id for row in self.object_rows
        ):
            self.selected_object_id = selected_object_id
        elif self.selected_object_id and not any(
            row.object_id == self.selected_object_id for row in self.object_rows
        ):
            self.selected_object_id = None
        self.current_export_json = self.export_json()
        self._notify()

    def _row_by_id(self, object_id: str) -> EnvBuilderObjectRow | None:
        for row in self.object_rows:
            if row.object_id == object_id:
                return row
        return None

    def _next_object_id(self, prefix: str) -> str:
        highest = 0
        for row in self.object_rows:
            base = row.object_id.split(":", 1)[0]
            if not base.startswith(prefix + "_"):
                continue
            suffix = base[len(prefix) + 1 :]
            try:
                highest = max(highest, int(suffix))
            except ValueError:
                continue
        return f"{prefix}_{highest + 1:03d}"

    def _scene_item_by_id(self, items: list[Any], object_id: str) -> Any | None:
        for item in items:
            try:
                if item.data(0) == object_id:
                    return item
            except Exception:
                continue
        return None

    def _row_from_scene_item(
        self, row: EnvBuilderObjectRow, item: Any
    ) -> EnvBuilderObjectRow:
        try:
            if row.object_type == BORDER_SEGMENT:
                line = item.line()
                pos = item.pos()
                x1 = float(line.x1() + pos.x())
                y1 = float(line.y1() + pos.y())
                x2 = float(line.x2() + pos.x())
                y2 = float(line.y2() + pos.y())
                return replace(row, x=x1, y=y1, x2=x2, y2=y2)
            rect = item.rect()
            pos = item.pos()
            center_x = float(rect.center().x() + pos.x())
            center_y = float(rect.center().y() + pos.y())
            return replace(row, x=center_x, y=center_y)
        except Exception:
            return row

    def _resolve_workspace_path(self, name_or_path: str) -> Path:
        candidate = Path(name_or_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        if candidate.suffix.lower() == ".json" and candidate.exists():
            return candidate.resolve()
        if candidate.is_absolute() and candidate.exists():
            return candidate.resolve()
        try:
            return (
                get_workspace_dir("environments")
                / normalize_preset_filename(candidate.stem or candidate.name)
            ).resolve()
        except WorkspaceError as exc:
            raise WorkspaceError(str(exc)) from exc
