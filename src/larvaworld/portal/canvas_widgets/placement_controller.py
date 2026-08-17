"""
Generic, data-model-agnostic canvas object placement helpers.

Extracted from the click/select/hit-test/table loop originally built for
`environment_builder_app.py`'s environment-object placement (food sources,
borders), so it can be reused for other click-to-place canvas interactions
(e.g. Single Experiment's larva-group placement) without depending on any
particular domain object shape.

Nothing in this module imports Bokeh, `_ObjectRow`, `LarvaGroup`, or any
app-specific type -- every function/class here operates purely on object
IDs, (x, y) coordinates, and caller-supplied callables. Callers own their
own domain data, hit-test tolerance rules, and rendering; this module only
owns the generic *shape* of the interaction: "which object (if any) is
nearest a click", "keep a dropdown/table/highlight in sync with one
selected id", and "is this tap a select or an insert".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

__all__: list[str] = [
    "HitCandidate",
    "pick_nearest",
    "SelectionSync",
    "TapDispatcher",
    "ObjectTable",
]

T = TypeVar("T")


@dataclass(frozen=True)
class HitCandidate(Generic[T]):
    """
    One candidate result of hit-testing a click against a single object.

    `ref` is opaque to this module -- callers put whatever they need to
    recover the hit object there (an id, a row, a tuple, ...) and read it
    back from whichever `HitCandidate` `pick_nearest` returns.

    Attributes:
        object_id: Identifier of the candidate object.
        distance: Click distance to the candidate, in the caller's own
            units (e.g. simulation-space meters). Only ever compared
            against other `HitCandidate.distance` values from the same
            caller, so the unit doesn't matter as long as it's consistent.
        ref: Caller-owned payload, returned as-is by `pick_nearest`.
    """

    object_id: str
    distance: float
    ref: T


def pick_nearest(candidates: Sequence[HitCandidate[T]]) -> Optional[HitCandidate[T]]:
    """
    Return the candidate with the smallest `distance`, or None if empty.

    Extracted from the "nearest wins across all candidate lists" merge in
    `environment_builder_app.py`'s `_pick_object_at`: that method builds
    several per-type candidate lists (member-glyph proximity, border
    segment distance, point-radius distance) with different tolerance
    rules, then keeps whichever single candidate is closest overall. This
    function is just that final merge step -- callers build their own
    candidate lists (already filtered to "within tolerance") and pass them
    all in together.

    Args:
        candidates: Hit candidates already filtered to within tolerance.

    Returns:
        The nearest `HitCandidate`, or None if `candidates` is empty.

    Example:
        >>> pick_nearest([HitCandidate("a", 0.5, None), HitCandidate("b", 0.2, None)])
        HitCandidate(object_id='b', distance=0.2, ref=None)
    """
    nearest: Optional[HitCandidate[T]] = None
    for candidate in candidates:
        if nearest is None or candidate.distance < nearest.distance:
            nearest = candidate
    return nearest


class SelectionSync:
    """
    Keep one "selected object id" in sync across multiple views, with a
    re-entrancy guard so that this class's own writes to those views don't
    loop back into themselves.

    Extracted from `environment_builder_app.py`'s `_set_selected_object` /
    `_syncing_selection` pattern: a dropdown, a table's row selection, and
    a canvas highlight all need to agree on one selected id, regardless of
    whether the selection changed via a canvas tap, a table row click, or
    the dropdown itself. This class owns only the id and the guard; it has
    no knowledge of dropdowns/tables/canvases at all -- `on_change` is
    called with the new id (or None) and the caller does every view update
    itself inside that callback.

    Callers must wrap their own external widget-change handlers (a
    dropdown's "value" watcher, a table's "selection" watcher) with a
    check against `syncing` before calling `set_selected` again, mirroring
    `_on_selected_object_change`/`_on_table_selection_change`'s own
    `if self._syncing_selection: return` guards -- otherwise a
    programmatic update inside `on_change` (e.g. `dropdown.value = id`)
    would re-trigger that dropdown's own watcher and call back in.

    Attributes:
        selected_id: The currently selected object id, or None.
        syncing: True while `on_change` is executing -- external watchers
            should no-op while this is True.

    Example:
        >>> def on_change(object_id):
        ...     dropdown.value = object_id  # dropdown's own watcher checks `sync.syncing`
        >>> sync = SelectionSync(on_change=on_change)
        >>> sync.set_selected("food_001")
    """

    def __init__(self, *, on_change: Callable[[Optional[str]], None]) -> None:
        """
        Args:
            on_change: Called with the new selected id (or None) every
                time `set_selected` runs, while `syncing` is True.
        """
        self._on_change = on_change
        self.selected_id: Optional[str] = None
        self.syncing: bool = False

    def set_selected(self, object_id: Optional[str]) -> None:
        """Update the selected id and notify `on_change`, guarded against re-entrancy."""
        self.syncing = True
        try:
            self.selected_id = object_id
            self._on_change(object_id)
        finally:
            self.syncing = False


class TapDispatcher:
    """
    Route a canvas tap event to either "select an existing object" or
    "insert a new object", based on a caller-supplied select-mode flag.

    Extracted from the top of `environment_builder_app.py`'s `_on_tap`:
    the select-vs-insert branch is generic (it only needs the click
    coordinates and a mode flag), but everything each branch does with
    those coordinates -- hit-testing, validating arena bounds, building a
    new object -- is domain-specific and stays entirely in the caller's
    `on_select`/`on_insert` hooks.

    Example:
        >>> dispatcher = TapDispatcher(
        ...     select_mode=lambda: select_toggle.value,
        ...     on_select=lambda x, y: ...,
        ...     on_insert=lambda x, y: ...,
        ... )
        >>> fig.on_event(Tap, dispatcher.on_tap)
    """

    def __init__(
        self,
        *,
        select_mode: Callable[[], bool],
        on_select: Callable[[float, float], None],
        on_insert: Callable[[float, float], None],
    ) -> None:
        """
        Args:
            select_mode: Returns True when the canvas is currently in
                "select an existing object" mode rather than "insert a
                new object" mode.
            on_select: Called with the tap's (x, y) when `select_mode()`
                is True. Expected to hit-test and update selection itself.
            on_insert: Called with the tap's (x, y) when `select_mode()`
                is False. Expected to validate bounds and build/insert a
                new object itself.
        """
        self._select_mode = select_mode
        self._on_select = on_select
        self._on_insert = on_insert

    def on_tap(self, event: Any) -> None:
        """Dispatch a Bokeh `Tap` event (or anything with `.x`/`.y`) to the right hook."""
        x = float(event.x)
        y = float(event.y)
        if self._select_mode():
            self._on_select(x, y)
        else:
            self._on_insert(x, y)


class ObjectTable(Generic[T]):
    """
    Generic id-keyed table scaffold: rebuild a flat table from a list of
    objects, and translate row-index selection back to an object id.

    Extracted (shape only) from `environment_builder_app.py`'s Placed
    Objects `Tabulator` + `_refresh_table`/`_on_table_selection_change`:
    the table itself is always fully rebuilt (not diffed) from the
    current object list, and row selection is positional, translated to
    an id via the same list the table was built from. This class doesn't
    construct a `pn.widgets.Tabulator` itself (so it stays Panel-agnostic
    and independently testable) -- it only maintains the id-ordering a
    caller's own Tabulator needs to translate `table.selection` (a list of
    row indices) into an object id.

    Not currently wired into any app -- kept available for a future
    Placed-Objects-style table (e.g. for larva groups), a decision left
    open for now.

    Example:
        >>> table = ObjectTable(to_row=lambda obj: {"id": obj.id, "x": obj.x})
        >>> rows = table.rebuild(objects, id_of=lambda obj: obj.id)
        >>> table.id_at_row(0)
        'food_001'
    """

    def __init__(self, *, to_row: Callable[[T], dict[str, Any]]) -> None:
        """
        Args:
            to_row: Converts one domain object into a flat dict of column
                values for display.
        """
        self._to_row = to_row
        self._ids: list[str] = []

    def rebuild(
        self, objects: Sequence[T], *, id_of: Callable[[T], str]
    ) -> list[dict[str, Any]]:
        """
        Recompute the row list (and the row-index -> id ordering) from
        the current object list.

        Args:
            objects: Current domain objects, in display order.
            id_of: Extracts an object's id.

        Returns:
            One row dict per object, in the same order as `objects`.
        """
        self._ids = [id_of(obj) for obj in objects]
        return [self._to_row(obj) for obj in objects]

    def id_at_row(self, row_index: int) -> Optional[str]:
        """Return the object id at `row_index` (as of the last `rebuild`), or None if out of range."""
        if row_index < 0 or row_index >= len(self._ids):
            return None
        return self._ids[row_index]
