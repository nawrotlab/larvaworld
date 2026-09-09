"""Unit tests for the generic canvas placement helpers.

Pure Python, no Panel/Bokeh dependency -- these operate purely on object
ids, coordinates, and caller-supplied callables.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from larvaworld.portal.canvas_widgets.placement_controller import (
    HitCandidate,
    ObjectTable,
    SelectionSync,
    TapDispatcher,
    pick_nearest,
)


@pytest.mark.fast
class TestPickNearest:
    def test_empty_returns_none(self) -> None:
        assert pick_nearest([]) is None

    def test_single_candidate_returned(self) -> None:
        candidate = HitCandidate("a", 0.5, None)
        assert pick_nearest([candidate]) is candidate

    def test_returns_smallest_distance_regardless_of_order(self) -> None:
        far = HitCandidate("far", 0.9, None)
        near = HitCandidate("near", 0.1, None)
        mid = HitCandidate("mid", 0.5, None)
        assert pick_nearest([far, near, mid]).object_id == "near"
        assert pick_nearest([near, mid, far]).object_id == "near"

    def test_tie_keeps_first_seen(self) -> None:
        first = HitCandidate("first", 0.3, None)
        second = HitCandidate("second", 0.3, None)
        assert pick_nearest([first, second]) is first

    def test_ref_is_returned_unmodified(self) -> None:
        ref = {"anything": "the caller wants"}
        result = pick_nearest([HitCandidate("a", 0.1, ref)])
        assert result.ref is ref


@pytest.mark.fast
class TestSelectionSync:
    def test_set_selected_calls_on_change_with_new_id(self) -> None:
        seen = []
        sync = SelectionSync(on_change=seen.append)
        sync.set_selected("obj_1")
        assert seen == ["obj_1"]
        assert sync.selected_id == "obj_1"

    def test_set_selected_none_clears_selection(self) -> None:
        seen = []
        sync = SelectionSync(on_change=seen.append)
        sync.set_selected("obj_1")
        sync.set_selected(None)
        assert seen == ["obj_1", None]
        assert sync.selected_id is None

    def test_syncing_flag_true_only_during_on_change(self) -> None:
        observed_during = []

        def on_change(object_id):
            observed_during.append(sync.syncing)

        sync = SelectionSync(on_change=on_change)
        assert sync.syncing is False
        sync.set_selected("obj_1")
        assert observed_during == [True]
        assert sync.syncing is False

    def test_reentrant_set_selected_inside_on_change_does_not_loop_forever(
        self,
    ) -> None:
        # A caller's own widget watcher, wired to fire set_selected again,
        # must check `sync.syncing` before doing so -- this test proves the
        # flag is available and accurate for that guard, not that
        # SelectionSync itself prevents re-entrant calls (it doesn't; the
        # guard is the caller's responsibility, matching
        # _on_selected_object_change's own `if self._syncing_selection: return`).
        calls = []

        def external_widget_watcher(new_value):
            if sync.syncing:
                return
            sync.set_selected(new_value)

        def on_change(object_id):
            calls.append(object_id)
            external_widget_watcher(object_id)  # simulates a dropdown.value write

        sync = SelectionSync(on_change=on_change)
        sync.set_selected("obj_1")
        assert calls == ["obj_1"]


@pytest.mark.fast
class TestTapDispatcher:
    def test_select_mode_true_calls_on_select(self) -> None:
        select_calls = []
        insert_calls = []
        dispatcher = TapDispatcher(
            select_mode=lambda: True,
            on_select=lambda x, y: select_calls.append((x, y)),
            on_insert=lambda x, y: insert_calls.append((x, y)),
        )
        dispatcher.on_tap(SimpleNamespace(x=0.01, y=0.02))
        assert select_calls == [(0.01, 0.02)]
        assert insert_calls == []

    def test_select_mode_false_calls_on_insert(self) -> None:
        select_calls = []
        insert_calls = []
        dispatcher = TapDispatcher(
            select_mode=lambda: False,
            on_select=lambda x, y: select_calls.append((x, y)),
            on_insert=lambda x, y: insert_calls.append((x, y)),
        )
        dispatcher.on_tap(SimpleNamespace(x=0.03, y=-0.01))
        assert insert_calls == [(0.03, -0.01)]
        assert select_calls == []

    def test_select_mode_is_read_fresh_on_every_tap(self) -> None:
        mode = {"select": False}
        select_calls = []
        insert_calls = []
        dispatcher = TapDispatcher(
            select_mode=lambda: mode["select"],
            on_select=lambda x, y: select_calls.append((x, y)),
            on_insert=lambda x, y: insert_calls.append((x, y)),
        )
        dispatcher.on_tap(SimpleNamespace(x=0.0, y=0.0))
        mode["select"] = True
        dispatcher.on_tap(SimpleNamespace(x=0.0, y=0.0))
        assert len(insert_calls) == 1
        assert len(select_calls) == 1


@pytest.mark.fast
class TestObjectTable:
    def test_rebuild_returns_rows_in_object_order(self) -> None:
        objects = [
            SimpleNamespace(id="a", x=1),
            SimpleNamespace(id="b", x=2),
        ]
        table = ObjectTable(to_row=lambda obj: {"id": obj.id, "x": obj.x})
        rows = table.rebuild(objects, id_of=lambda obj: obj.id)
        assert rows == [{"id": "a", "x": 1}, {"id": "b", "x": 2}]

    def test_id_at_row_matches_last_rebuild_order(self) -> None:
        objects = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
        table = ObjectTable(to_row=lambda obj: {"id": obj.id})
        table.rebuild(objects, id_of=lambda obj: obj.id)
        assert table.id_at_row(0) == "a"
        assert table.id_at_row(1) == "b"

    def test_id_at_row_out_of_range_returns_none(self) -> None:
        table = ObjectTable(to_row=lambda obj: {})
        table.rebuild([], id_of=lambda obj: obj)
        assert table.id_at_row(0) is None
        assert table.id_at_row(-1) is None

    def test_rebuild_reflects_deletions(self) -> None:
        table = ObjectTable(to_row=lambda obj: {"id": obj.id})
        table.rebuild(
            [SimpleNamespace(id="a"), SimpleNamespace(id="b")],
            id_of=lambda obj: obj.id,
        )
        table.rebuild([SimpleNamespace(id="b")], id_of=lambda obj: obj.id)
        assert table.id_at_row(0) == "b"
        assert table.id_at_row(1) is None
