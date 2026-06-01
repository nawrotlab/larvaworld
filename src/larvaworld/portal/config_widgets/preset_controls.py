from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import panel as pn
import param

from larvaworld.lib import reg

__all__ = [
    "PresetSource",
    "PresetRef",
    "PresetCatalog",
    "PresetActionPolicy",
    "USER_PRESET_POLICY",
    "ADVANCED_PRESET_POLICY",
    "RegistryPresetStore",
    "WorkspacePresetStore",
    "PresetControlsController",
    "build_user_preset_controls",
    "build_advanced_preset_controls",
]


_PRESET_NAME_INVALID = re.compile(r"[^a-zA-Z0-9._-]+")


class PresetSource:
    REGISTRY = "registry"
    WORKSPACE = "workspace"


@dataclass(frozen=True)
class PresetRef:
    source: str
    name: str
    display_label: str
    token: str
    conftype: str | None = None
    workspace_filename: str | None = None
    workspace_path: Path | None = None


@dataclass(frozen=True)
class PresetCatalog:
    refs: tuple[PresetRef, ...]
    by_token: dict[str, PresetRef]

    def resolve(self, token: str | None) -> PresetRef | None:
        if not token:
            return None
        return self.by_token.get(str(token))


@dataclass(frozen=True)
class PresetActionPolicy:
    can_load_registry: bool
    can_load_workspace: bool
    can_save_registry: bool
    can_save_workspace: bool
    can_delete_registry: bool
    can_delete_workspace: bool
    can_reset_registry: bool

    def can_load(self, source: str) -> bool:
        if source == PresetSource.REGISTRY:
            return self.can_load_registry
        return self.can_load_workspace

    def can_save(self, source: str) -> bool:
        if source == PresetSource.REGISTRY:
            return self.can_save_registry
        return self.can_save_workspace

    def can_delete(self, source: str) -> bool:
        if source == PresetSource.REGISTRY:
            return self.can_delete_registry
        return self.can_delete_workspace


USER_PRESET_POLICY = PresetActionPolicy(
    can_load_registry=True,
    can_load_workspace=True,
    can_save_registry=False,
    can_save_workspace=True,
    can_delete_registry=False,
    can_delete_workspace=True,
    can_reset_registry=False,
)

ADVANCED_PRESET_POLICY = PresetActionPolicy(
    can_load_registry=True,
    can_load_workspace=True,
    can_save_registry=True,
    can_save_workspace=True,
    can_delete_registry=True,
    can_delete_workspace=True,
    can_reset_registry=True,
)


@dataclass(frozen=True)
class WorkspacePresetRecord:
    name: str
    filename: str
    path: Path


class RegistryPresetStore:
    def __init__(self, conftype: str) -> None:
        if conftype not in reg.conf:
            raise ValueError(f'Unknown conftype "{conftype}".')
        self.conftype = str(conftype)

    @property
    def _conf(self) -> Any:
        return reg.conf[self.conftype]

    @property
    def source_path(self) -> str:
        return str(self._conf.path_to_dict)

    def list_ids(self) -> list[str]:
        self._conf.load()
        return sorted(str(name) for name in self._conf.confIDs)

    def exists(self, name: str) -> bool:
        return str(name) in set(self.list_ids())

    def load(self, name: str) -> Any:
        self._conf.load()
        payload = self._conf.getID(str(name))
        return copy.deepcopy(payload)

    def save(self, name: str, payload: Any) -> None:
        self._conf.setID(str(name), payload)
        self._conf.load()

    def delete(self, name: str) -> None:
        target = str(name)
        if target not in self._conf.confIDs:
            raise FileNotFoundError(f'Registry preset "{target}" does not exist.')
        self._conf.delete(target)
        self._conf.load()

    def reset_defaults(self) -> None:
        self._conf.reset(recreate=True)
        self._conf.load()


class WorkspacePresetStore:
    def __init__(self, directory: str | Path, *, directory_key: str) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory_key = str(directory_key)

    @property
    def source_path(self) -> str:
        return str(self.directory)

    @staticmethod
    def normalize_name(name: str) -> str:
        raw = str(name or "").strip()
        if not raw:
            raise ValueError("Preset name cannot be empty.")
        candidate = Path(raw)
        if candidate.is_absolute():
            raise ValueError("Absolute preset names are not allowed.")
        if "/" in raw or "\\" in raw:
            raise ValueError("Preset names cannot contain directory separators.")
        if ".." in candidate.parts:
            raise ValueError("Preset names cannot traverse directories.")
        cleaned = _PRESET_NAME_INVALID.sub("_", raw).strip("._-")
        if not cleaned:
            raise ValueError("Preset name does not contain valid characters.")
        return cleaned

    def _resolve_workspace_file(self, filename: str) -> Path:
        raw = str(filename or "").strip()
        if not raw:
            raise ValueError("Workspace filename cannot be empty.")
        candidate = Path(raw)
        if candidate.is_absolute():
            raise ValueError("Workspace filename must be relative.")
        if "/" in raw or "\\" in raw:
            raise ValueError("Workspace filename cannot contain directory separators.")
        if ".." in candidate.parts:
            raise ValueError("Workspace filename cannot traverse directories.")
        if candidate.suffix.lower() != ".json":
            raise ValueError("Workspace filename must end with .json.")
        resolved = (self.directory / candidate.name).resolve()
        resolved.relative_to(self.directory)
        return resolved

    def list_presets(self) -> list[WorkspacePresetRecord]:
        if not self.directory.is_dir():
            return []
        records: list[WorkspacePresetRecord] = []
        for path in sorted(self.directory.glob("*.json"), key=lambda p: p.stem.lower()):
            resolved = path.resolve()
            resolved.relative_to(self.directory)
            records.append(
                WorkspacePresetRecord(
                    name=resolved.stem,
                    filename=resolved.name,
                    path=resolved,
                )
            )
        return records

    def exists_name(self, name: str) -> bool:
        safe = self.normalize_name(name)
        filename = f"{safe}.json"
        target = self._resolve_workspace_file(filename)
        return target.is_file()

    def load(self, filename: str) -> Any:
        target = self._resolve_workspace_file(filename)
        if not target.is_file():
            raise FileNotFoundError(f'Workspace preset "{filename}" does not exist.')
        return json.loads(target.read_text(encoding="utf-8"))

    def save(self, name: str, payload: Any) -> Path:
        safe = self.normalize_name(name)
        filename = f"{safe}.json"
        target = self._resolve_workspace_file(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(_payload_to_jsonable(payload), indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    def delete(self, filename: str) -> None:
        target = self._resolve_workspace_file(filename)
        if not target.is_file():
            raise FileNotFoundError(f'Workspace preset "{filename}" does not exist.')
        target.unlink()


@dataclass
class _PendingConfirmation:
    message: str
    execute: Callable[[], bool]


class PresetControlsController:
    def __init__(
        self,
        *,
        conftype: str,
        workspace_store: WorkspacePresetStore,
        policy: PresetActionPolicy,
        build_workspace_payload: Callable[[str], Any],
        build_registry_payload: Callable[[str], Any] | None = None,
        before_save: Callable[[str, str], None] | None = None,
        on_load: Callable[[PresetRef, Any], None] | None = None,
        on_save: Callable[[PresetRef, Any], None] | None = None,
        on_status: Callable[..., None] | None = None,
        title: str | None = "Stored Configurations",
        preset_name_after_refresh: bool = False,
        confirm_destructive: bool = True,
    ) -> None:
        self.conftype = str(conftype)
        self.workspace_store = workspace_store
        self.registry_store = RegistryPresetStore(conftype)
        self.policy = policy
        self.build_workspace_payload = build_workspace_payload
        self.build_registry_payload = build_registry_payload or build_workspace_payload
        self.before_save = before_save
        self.on_load = on_load
        self.on_save = on_save
        self.on_status = on_status
        self.preset_name_after_refresh = bool(preset_name_after_refresh)
        self.confirm_destructive = bool(confirm_destructive)
        self.catalog = PresetCatalog(refs=tuple(), by_token={})
        self._pending_confirmation: _PendingConfirmation | None = None

        self.title = pn.pane.Markdown(f"### {title}") if title else None
        self.preset_name = pn.widgets.TextInput(
            name=(
                "Workspace preset name"
                if not self.policy.can_save_registry
                else "Preset name"
            ),
            placeholder="my_preset",
            sizing_mode="stretch_width",
        )
        self.preset_select = pn.widgets.Select(
            name="Preset to load", options={}, sizing_mode="stretch_width"
        )
        self.refresh_button = pn.widgets.Button(
            name="Refresh list", button_type="default", sizing_mode="stretch_width"
        )
        self.load_button = pn.widgets.Button(
            name="Load", button_type="primary", sizing_mode="stretch_width"
        )
        self.save_button = pn.widgets.Button(
            name="Save",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.delete_button = pn.widgets.Button(
            name="Delete",
            button_type="warning",
            sizing_mode="stretch_width",
        )
        self.save_target = (
            pn.widgets.RadioButtonGroup(
                name="Save target",
                options=["Workspace", "Registry"],
                value="Workspace",
                button_type="default",
            )
            if self.policy.can_save_registry
            else None
        )
        self.reset_button = (
            pn.widgets.Button(
                name="Reset registry defaults",
                button_type="danger",
            )
            if self.policy.can_reset_registry
            else None
        )
        self.status = pn.pane.HTML("", sizing_mode="stretch_width")
        self.status.visible = False
        self.storage_info = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.confirmation_host = pn.Column(sizing_mode="stretch_width")

        self.refresh_button.on_click(lambda _event: self.refresh_list())
        self.load_button.on_click(lambda _event: self.load_selected())
        self.save_button.on_click(lambda _event: self.save_current())
        self.delete_button.on_click(lambda _event: self.delete_selected())
        if self.reset_button is not None:
            self.reset_button.on_click(lambda _event: self.request_reset_registry())

        controls: list[Any] = []
        if self.title is not None:
            controls.append(self.title)
        if self.preset_name_after_refresh:
            controls.extend(
                [
                    self.preset_select,
                    self.refresh_button,
                    self.preset_name,
                ]
            )
        else:
            controls.extend(
                [
                    self.preset_name,
                    self.preset_select,
                    self.refresh_button,
                ]
            )
        if self.save_target is not None:
            controls.append(self.save_target)
        controls.extend(
            [
                pn.Row(
                    self.save_button,
                    self.load_button,
                    self.delete_button,
                    sizing_mode="stretch_width",
                ),
            ]
        )
        if self.reset_button is not None:
            controls.append(self.reset_button)
        controls.extend([self.confirmation_host, self.status])
        self.view = pn.Column(*controls, sizing_mode="stretch_width")

        self.refresh_list()

    def _set_status(self, message: str, *, tone: str = "neutral") -> None:
        self.status.object = f"<div>{message}</div>"
        self.status.visible = tone in {"warning", "danger"}
        if self.on_status is not None:
            self.on_status(message, tone=tone)

    def _set_storage_info(self) -> None:
        self.storage_info.object = (
            "Storage info:\n"
            f"Workspace preset directory:\n  `{self.workspace_store.source_path}`\n\n"
            f"Registry presets from:\n  `{self.registry_store.source_path}`"
        )

    def _token_for_registry(self, name: str) -> str:
        return f"registry:{self.conftype}:{name}"

    def _token_for_workspace(self, filename: str) -> str:
        return f"workspace:{self.workspace_store.directory_key}:{filename}"

    def _build_catalog(self) -> PresetCatalog:
        refs: list[PresetRef] = []
        for name in self.registry_store.list_ids():
            refs.append(
                PresetRef(
                    source=PresetSource.REGISTRY,
                    name=name,
                    display_label=f"Registry / {name}",
                    token=self._token_for_registry(name),
                    conftype=self.conftype,
                )
            )
        for record in self.workspace_store.list_presets():
            refs.append(
                PresetRef(
                    source=PresetSource.WORKSPACE,
                    name=record.name,
                    display_label=f"Workspace / {record.name}",
                    token=self._token_for_workspace(record.filename),
                    conftype=self.conftype,
                    workspace_filename=record.filename,
                    workspace_path=record.path,
                )
            )
        return PresetCatalog(
            refs=tuple(refs), by_token={ref.token: ref for ref in refs}
        )

    def refresh_list(self) -> bool:
        current = str(self.preset_select.value or "")
        try:
            self.catalog = self._build_catalog()
        except Exception as exc:
            self._set_status(f"Failed to refresh preset list: {exc}", tone="danger")
            return False

        options = {ref.display_label: ref.token for ref in self.catalog.refs}
        self.preset_select.options = options
        if current in self.catalog.by_token:
            self.preset_select.value = current
        elif options:
            self.preset_select.value = next(iter(options.values()))
        else:
            self.preset_select.value = None

        self._set_storage_info()
        self._set_status("Refreshed preset list.", tone="success")
        return True

    def _selected_ref(self) -> PresetRef | None:
        return self.catalog.resolve(str(self.preset_select.value or ""))

    def _resolve_saved_ref(self, *, source: str, token: str) -> PresetRef | None:
        ref = self.catalog.resolve(token)
        if ref is None or ref.source != source:
            return None
        return ref

    def _run_on_save_callback(self, ref: PresetRef, payload: Any) -> bool:
        if self.on_save is None:
            return True
        try:
            self.on_save(ref, payload)
        except Exception as exc:
            self._set_status(
                f"Saved preset, but post-save update failed: {exc}",
                tone="warning",
            )
            return False
        return True

    def _request_confirmation(self, message: str, execute: Callable[[], bool]) -> None:
        self._pending_confirmation = _PendingConfirmation(
            message=message, execute=execute
        )

        confirm_button = pn.widgets.Button(
            name="Confirm", button_type="danger", sizing_mode="stretch_width"
        )
        cancel_button = pn.widgets.Button(
            name="Cancel", button_type="default", sizing_mode="stretch_width"
        )
        confirm_button.on_click(lambda _event: self.confirm_pending_action())
        cancel_button.on_click(lambda _event: self.cancel_pending_action())
        self.confirmation_host.objects = [
            pn.pane.HTML(f"<div>{message}</div>", margin=(0, 0, 6, 0)),
            pn.Row(confirm_button, cancel_button, sizing_mode="stretch_width"),
        ]

    def confirm_pending_action(self) -> bool:
        pending = self._pending_confirmation
        if pending is None:
            return False
        self._pending_confirmation = None
        self.confirmation_host.objects = []
        return pending.execute()

    def cancel_pending_action(self) -> bool:
        if self._pending_confirmation is None:
            return False
        self._pending_confirmation = None
        self.confirmation_host.objects = []
        self._set_status("Action cancelled.", tone="warning")
        return True

    def load_selected(self) -> bool:
        ref = self._selected_ref()
        if ref is None:
            self._set_status("Select a preset to load.", tone="warning")
            return False
        if not self.policy.can_load(ref.source):
            self._set_status("Loading from this source is not allowed.", tone="warning")
            return False

        try:
            if ref.source == PresetSource.REGISTRY:
                payload = self.registry_store.load(ref.name)
            else:
                assert ref.workspace_filename is not None
                payload = self.workspace_store.load(ref.workspace_filename)
        except Exception as exc:
            self._set_status(f"Load failed: {exc}", tone="danger")
            return False

        if self.on_load is not None:
            try:
                self.on_load(ref, payload)
            except Exception as exc:
                self._set_status(f"Load failed: {exc}", tone="danger")
                return False
        self._set_status(f"Loaded {ref.display_label}.", tone="success")
        return True

    def _normalized_name(self) -> str:
        return WorkspacePresetStore.normalize_name(self.preset_name.value or "")

    def save_current(self) -> bool:
        try:
            target_name = self._normalized_name()
        except ValueError as exc:
            self._set_status(str(exc), tone="warning")
            return False

        target_source = PresetSource.WORKSPACE
        if self.save_target is not None and self.save_target.value == "Registry":
            target_source = PresetSource.REGISTRY

        if not self.policy.can_save(target_source):
            self._set_status("Saving to this source is not allowed.", tone="warning")
            return False

        if self.before_save is not None:
            try:
                self.before_save(target_name, target_source)
            except Exception as exc:
                self._set_status(str(exc), tone="warning")
                return False

        if target_source == PresetSource.WORKSPACE:
            return self._save_workspace(target_name)
        return self._save_registry(target_name)

    def _save_workspace(self, target_name: str) -> bool:
        exists = self.workspace_store.exists_name(target_name)

        def _execute() -> bool:
            try:
                payload = self.build_workspace_payload(target_name)
                target = self.workspace_store.save(target_name, payload)
            except Exception as exc:
                self._set_status(f"Save failed: {exc}", tone="danger")
                return False
            self.refresh_list()
            token = self._token_for_workspace(target.name)
            if token in self.catalog.by_token:
                self.preset_select.value = token
            ref = self._resolve_saved_ref(source=PresetSource.WORKSPACE, token=token)
            if ref is None:
                self._set_status(
                    "Workspace preset saved, but the refreshed catalog could not resolve it.",
                    tone="warning",
                )
                return True
            if not self._run_on_save_callback(ref, payload):
                return True
            self._set_status(f'Saved workspace preset "{target.stem}".', tone="success")
            return True

        if exists and self.confirm_destructive:
            self._request_confirmation(
                f'Workspace preset "{target_name}" already exists. Overwrite?',
                _execute,
            )
            self._set_status("Overwrite confirmation required.", tone="warning")
            return False
        return _execute()

    def _save_registry(self, target_name: str) -> bool:
        exists = self.registry_store.exists(target_name)

        def _execute() -> bool:
            try:
                payload = self.build_registry_payload(target_name)
                self.registry_store.save(target_name, payload)
            except Exception as exc:
                self._set_status(f"Save failed: {exc}", tone="danger")
                return False
            self.refresh_list()
            token = self._token_for_registry(target_name)
            if token in self.catalog.by_token:
                self.preset_select.value = token
            ref = self._resolve_saved_ref(source=PresetSource.REGISTRY, token=token)
            if ref is None:
                self._set_status(
                    "Registry preset saved, but the refreshed catalog could not resolve it.",
                    tone="warning",
                )
                return True
            if not self._run_on_save_callback(ref, payload):
                return True
            self._set_status(f'Saved registry preset "{target_name}".', tone="success")
            return True

        if exists and self.confirm_destructive:
            self._request_confirmation(
                (
                    f'Registry preset "{target_name}" already exists. '
                    "Overwrite global/default configuration?"
                ),
                _execute,
            )
            self._set_status("Overwrite confirmation required.", tone="warning")
            return False
        return _execute()

    def delete_selected(self) -> bool:
        ref = self._selected_ref()
        if ref is None:
            self._set_status("Select a preset to delete.", tone="warning")
            return False

        if ref.source == PresetSource.REGISTRY and not self.policy.can_delete_registry:
            self._set_status(
                "Registry presets are read-only in this workflow.", tone="warning"
            )
            return False
        if (
            ref.source == PresetSource.WORKSPACE
            and not self.policy.can_delete_workspace
        ):
            self._set_status(
                "Workspace deletion is not allowed in this workflow.", tone="warning"
            )
            return False

        def _execute() -> bool:
            try:
                if ref.source == PresetSource.REGISTRY:
                    self.registry_store.delete(ref.name)
                else:
                    assert ref.workspace_filename is not None
                    self.workspace_store.delete(ref.workspace_filename)
            except Exception as exc:
                self._set_status(f"Delete failed: {exc}", tone="danger")
                return False
            self.refresh_list()
            self._set_status(f"Deleted {ref.display_label}.", tone="success")
            return True

        if self.confirm_destructive:
            self._request_confirmation(f"Delete {ref.display_label}?", _execute)
            self._set_status("Delete confirmation required.", tone="warning")
            return False
        return _execute()

    def request_reset_registry(self) -> bool:
        if not self.policy.can_reset_registry:
            self._set_status(
                "Registry reset is not allowed in this workflow.", tone="warning"
            )
            return False

        def _execute() -> bool:
            try:
                self.registry_store.reset_defaults()
            except Exception as exc:
                self._set_status(f"Reset failed: {exc}", tone="danger")
                return False
            self.refresh_list()
            self._set_status(
                "Registry defaults reset. Workspace presets were not modified.",
                tone="success",
            )
            return True

        if self.confirm_destructive:
            self._request_confirmation(
                "This resets registry defaults only. Workspace presets will not be modified.",
                _execute,
            )
            self._set_status("Reset confirmation required.", tone="warning")
            return False
        return _execute()


def _payload_to_jsonable(payload: Any) -> Any:
    if isinstance(payload, param.Parameterized):
        nested = getattr(payload, "nestedConf", None)
        if nested is not None:
            return copy.deepcopy(nested)
    nested_attr = getattr(payload, "nestedConf", None)
    if nested_attr is not None:
        return copy.deepcopy(nested_attr)
    return copy.deepcopy(payload)


def build_user_preset_controls(
    *,
    conftype: str,
    workspace_directory: str | Path,
    directory_key: str,
    build_workspace_payload: Callable[[str], Any],
    on_load: Callable[[PresetRef, Any], None] | None = None,
    on_save: Callable[[PresetRef, Any], None] | None = None,
    on_status: Callable[..., None] | None = None,
) -> pn.Column:
    controller = PresetControlsController(
        conftype=conftype,
        workspace_store=WorkspacePresetStore(
            workspace_directory,
            directory_key=directory_key,
        ),
        policy=USER_PRESET_POLICY,
        build_workspace_payload=build_workspace_payload,
        on_load=on_load,
        on_save=on_save,
        on_status=on_status,
        title="Stored Configurations",
    )
    return controller.view


def build_advanced_preset_controls(
    *,
    conftype: str,
    workspace_directory: str | Path,
    directory_key: str,
    build_workspace_payload: Callable[[str], Any],
    build_registry_payload: Callable[[str], Any] | None = None,
    on_load: Callable[[PresetRef, Any], None] | None = None,
    on_save: Callable[[PresetRef, Any], None] | None = None,
    on_status: Callable[..., None] | None = None,
) -> pn.Column:
    controller = PresetControlsController(
        conftype=conftype,
        workspace_store=WorkspacePresetStore(
            workspace_directory,
            directory_key=directory_key,
        ),
        policy=ADVANCED_PRESET_POLICY,
        build_workspace_payload=build_workspace_payload,
        build_registry_payload=build_registry_payload,
        on_load=on_load,
        on_save=on_save,
        on_status=on_status,
        title="Advanced Stored Configurations",
    )
    return controller.view
