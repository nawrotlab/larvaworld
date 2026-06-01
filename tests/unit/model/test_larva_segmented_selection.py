from __future__ import annotations

from larvaworld.lib.model.agents._larva import LarvaSegmented


def test_segmented_larva_selection_draws_segments_without_union_shape() -> None:
    class _Segment:
        def __init__(self, vertices):
            self.vertices = vertices

    class _Larva:
        pass

    class _Viewer:
        selection_color = "red"

        def __init__(self) -> None:
            self.polygons = []

        def draw_polygon(self, **kwargs) -> None:
            self.polygons.append(kwargs)

    def _boom(*args, **kwargs):
        raise AssertionError("draw_selected must not call get_shape")

    larva = _Larva()
    larva.segs = [
        _Segment([(0.0, 0.0), (1.0, 0.0), (1.0, 0.2)]),
        _Segment([(1.0, 0.0), (2.0, 0.0), (2.0, 0.2)]),
    ]
    larva.get_shape = _boom
    viewer = _Viewer()

    LarvaSegmented.draw_selected(larva, viewer)

    assert len(viewer.polygons) == 2
    assert [polygon["vertices"] for polygon in viewer.polygons] == [
        larva.segs[0].vertices,
        larva.segs[1].vertices,
    ]
    assert all(polygon["filled"] is False for polygon in viewer.polygons)
    assert all(polygon["color"] == "red" for polygon in viewer.polygons)
