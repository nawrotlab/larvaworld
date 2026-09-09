"""
Body geometry of the simulated larva.

Defines the stored body plans, the contour they generate, and the segmented
body whose joints the locomotor drives. Also provides the sensor mounting
points used by the touch modality.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence

import numpy as np
import param
from shapely import affinity, geometry, ops

from .. import util
from .custom import ItemListParam, PositiveInteger, PositiveNumber, XYLine
from .drawable import Viewable
from .spatial import LineClosed, MobileVector

__all__: list[str] = [
    "body_plans",
    "BodyContour",
    "ShapeMobile",
    "ShapeViewable",
    "SegmentedBody",
    "SegmentedBodySensored",
]

__displayname__ = "Virtual body"

body_plans = util.AttrDict(
    {
        "drosophila_larva": [(0.9, 0.1), (0.05, 0.1)],
        "zebrafish_larva": [(0.9, 0.25), (0.7, 0.25), (0.6, 0.005), (0.05, 0.005)],
    }
)


class BodyContour(LineClosed):
    """
    Body contour defined by guide points and symmetry.

    Generates body vertices from guide points using bilateral or radial
    symmetry. Supports predefined body plans (drosophila/zebrafish larvae).

    Attributes:
        symmetry: Body symmetry type ('bilateral' or 'radial')
        guide_points: List of 2D points outside midline to generate vertices
        base_vertices: Generated list of vertices forming the contour
        body_plan: Predefined body plan ('drosophila_larva' or 'zebrafish_larva')

    Example:
        >>> contour = BodyContour(body_plan='drosophila_larva', symmetry='bilateral')
        >>> ratio = contour.width_to_length_ratio
    """

    symmetry = param.Selector(objects=["bilateral", "radial"], doc="The body symmetry.")
    guide_points = XYLine(
        doc="A list of 2d points outside the midline in order to generate the vertices"
    )
    base_vertices = XYLine(doc="The list of 2d points")
    body_plan = param.Selector(
        objects=["drosophila_larva", "zebrafish_larva"], doc="The body plan."
    )

    def __init__(self, **kwargs):
        """Build the contour and its base vertices.

        Args:
            **kwargs: Attributes, forwarded to the parent class.
        """
        super().__init__(**kwargs)

        self.generate_base_vertices()

    def generate_base_vertices(self) -> None:
        """Build the outline vertices from the body plan's guide points.

        The guide points give half the outline; it is mirrored to close the
        shape.

        Returns:
            The contour vertices.
        """
        if not self.guide_points:
            self.guide_points = body_plans[self.body_plan]
        if self.symmetry == "bilateral":
            symmetric_ps = [(x, -y) for (x, y) in self.guide_points]
            symmetric_ps.reverse()
            self.base_vertices = (
                [(1.0, 0.0)] + self.guide_points + [(0.0, 0.0)] + symmetric_ps
            )

    # TODO make this more explicit
    @property
    def width_to_length_ratio(self) -> float:
        """The body's mean width, relative to its length."""
        return np.mean(np.array(self.guide_points)[:, 1])


class ShapeMobile(LineClosed, MobileVector):
    """
    Mobile shape with dynamic vertex updates based on position and orientation.

    Combines closed line shape with mobile vector capabilities, automatically
    updating vertices when position, orientation, or length changes.

    Attributes:
        length: Physical length of the shape in meters
        base_vertices: Template vertices scaled and transformed to actual position

    Example:
        >>> shape = ShapeMobile(length=0.005, base_vertices=[(1,0), (0,0.1), (0,-0.1)])
        >>> shape.set_position((0.01, 0.01))
    """

    length = PositiveNumber(0.005)
    base_vertices = XYLine(doc="The list of 2d points")

    def __init__(self, **kwargs):
        """Build the movable shape and cache its length ratio.

        Args:
            **kwargs: Attributes, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        self.length_ratio = self.get_length_ratio()
        self.update_vertices()

    def get_length_ratio(self) -> float:
        """Return the outline's extent along its long axis."""
        xs = np.array(self.base_vertices)[:, 0]
        return -np.min(xs) + np.max(xs)

    @param.depends("pos", "orientation", "length", watch=True)
    def update_vertices(self) -> None:
        """Re-place the outline vertices at the current pose."""
        self.vertices = self.translate(
            self.length / self.length_ratio * np.array(self.base_vertices)
        )
        # self.vertices = [self.translate(self.length / self.length_ratio * np.array(p)) for p in self.base_vertices]
        # self.vertices = self.pos + self.length*self.base_vertices @ self.rotationMatrix


class ShapeViewable(ShapeMobile, Viewable):
    """
    Mobile shape with rendering capabilities.

    Extends ShapeMobile with visualization methods for drawing the shape
    as a filled polygon using a viewer object.

    Example:
        >>> shape = ShapeViewable(length=0.005, color=(255, 0, 0))
        >>> shape.draw(viewer)
    """

    def draw(self, v, **kwargs) -> None:
        # self.update_vertices()
        """Render the shape as a filled polygon.

        Args:
            v: The viewer to draw into.
            **kwargs: Accepted for signature compatibility; unused.
        """
        v.draw_polygon(self.vertices, filled=True, color=self.color)


class BodyMobile(ShapeMobile, BodyContour):
    """A body whose length and mass stay consistent with each other.

    Adds the density-based relation between body length and mass to the
    movable shape, so that growth updates both together.
    """

    density = PositiveNumber(
        300.0, softmax=10000.0, step=1.0, doc="The density of the larva body in kg/m**2"
    )

    def __init__(self, **kwargs):
        """Build the body and record its initial length.

        Args:
            **kwargs: Attributes, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        self.initial_length = self.length
        self.body_bend = 0

    """We make the following assumptions :
        1. Larvae are V1-morphs, meaning mass is proportional to L**2 (Llandres&al)
        2. ratio of width to length constant :0.2. So area A = L*L*0.2.
        3. For this to give realistic values for both l1 (L=1.3mm) and l3 (L=5.2mm) we set
        density = 300 kg/m**2=0.3 mg/mm*2. (It is totally fortunate that box2d calculates mass as density*area)
        This yields m3=0.3 * 5.2*5.2*0.2  = 1.6224 mg and m1=0.3 * 1.3*1.3*0.2 = 0.1014 mg
        It follows that mass=density*width_to_length_ratio*length**2 for both real and simulated mass
        So, when using a scaling factor sf where sim_length=sf*length ==> sim_mass=sf**2 * mass"""

    def compute_mass_from_length(self):
        """Set the mass implied by the current length and density."""
        self.mass = self.density * self.length**2 * self.width_to_length_ratio

    def adjust_shape_to_mass(self):
        """Set the length implied by the current mass and density."""
        self.length = np.sqrt(self.mass / (self.density * self.width_to_length_ratio))


class SegmentedBody(BodyMobile):
    """
    Multi-segment body with articulation and mass distribution.

    Divides body into configurable number of segments with independent
    positions and orientations. Supports custom segment length ratios
    and provides utilities for shape manipulation and rendering.

    Attributes:
        Nsegs: Number of segments comprising the body
        segment_ratio: Ratio of each segment's length to total body length
        segs: List of body segment shape objects

    Example:
        >>> body = SegmentedBody(Nsegs=12, length=0.005)
        >>> body.compute_body_bend()
        >>> head_pos = body.head.get_position()
    """

    Nsegs = PositiveInteger(
        2, softmax=20, doc="The number of segments comprising the segmented larva body."
    )
    segment_ratio = param.Parameter(
        None, doc="The number of segments comprising the segmented larva body."
    )
    segs = ItemListParam(item_type=ShapeViewable, doc="The body segments.")

    def __init__(self, **kwargs):
        """Build the segmented body and place its segments.

        Segment lengths default to an even split of the body length.

        Args:
            **kwargs: Attributes, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        # If segment ratio is not provided, generate equal-length segments
        if self.segment_ratio is None:
            self.segment_ratio = np.array([1 / self.Nsegs] * self.Nsegs)
        self.base_seg_vertices = self.segmentize()
        self.seg_positions = self.generate_seg_positions()
        self.generate_segs()
        self.update_seg_lengths()

    @property
    def Nangles(self) -> int:
        """The number of joint angles, one fewer than the segments."""
        return self.Nsegs - 1

    def segmentize(self, centered: bool = True, closed: bool = False) -> np.ndarray:
        """
        Segments a body into equal-length or given-length segments via vertical lines.

        Args:
        - N: Number of segments to divide the body into.
        - points: Array with shape (M,2) representing the contour of the body to be segmented.
        - ratio: List of N floats specifying the ratio of the length of each segment to the length of the body.
                    Defaults to None, in which case equal-length segments will be generated.
        - centered: If True, centers the segments around the origin. Defaults to True.
        - closed: If True, the last point of each segment is connected to the first point. Defaults to False.

        Returns:
        - ps: Numpy array with shape (Nsegs,L,2), where L is the number of vertices of each segment.
              The first segment in the list is the front-most segment.

        """
        # N=self.Nsegs
        R = self.segment_ratio

        # Create a polygon from the given body contour
        p = geometry.Polygon(np.array(self.base_vertices))
        # Get maximum y value of contour
        y0 = np.max(p.exterior.coords.xy[1])

        # Segment body via vertical lines
        multiline = geometry.MultiLineString(
            [
                geometry.LineString([(1 - cum_r, y0), (1 - cum_r, -y0)])
                for cum_r in np.cumsum(R)
            ]
        )

        # Divide multipolygon by lines
        for line in multiline.geoms:
            p = geometry.MultiPolygon(ops.split(p, line).geoms)

        ps = list(p.geoms)

        # Sort segments so that front segments come first
        ps.sort(key=lambda x: x.exterior.xy[0], reverse=True)

        # Transform to 2D array of coords
        ps = [p.exterior.coords.xy for p in ps]
        ps = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]

        # Center segments around 0,0
        if centered:
            for i, (r, cum_r) in enumerate(zip(R, np.cumsum(R))):
                ps[i] -= [(1 - cum_r) + r / 2, 0]

        # Put front point at the start of segment vertices. Drop duplicate rows
        for i in range(len(ps)):
            if i == 0:
                ind = np.argmax(ps[i][:, 0])
                ps[i] = np.roll(ps[i], -ind - 1, axis=0)
            else:
                ps[i] = np.roll(ps[i], 1, axis=0)
            ps[i] = np.flip(ps[i], axis=0)
            _, idx = np.unique(ps[i], axis=0, return_index=True)
            ps[i] = ps[i][np.sort(idx)]
            if closed:
                ps[i] = np.concatenate([ps[i], [ps[i][0]]])
        ps = [util.np2Dtotuples(pp) for pp in ps]
        return ps

    def generate_seg_positions(self) -> list[tuple[float, float]]:
        """Lay the segments out head to tail along the body axis.

        Returns:
            The centre position of each segment.
        """
        N = self.Nsegs
        ls_x = np.cos(self.orientation) * self.length * self.segment_ratio
        ls_y = np.sin(self.orientation) * self.length / N
        return [
            (self.x + (-i + (N - 1) / 2) * ls_x[i], self.y + (-i + (N - 1) / 2) * ls_y)
            for i in range(N)
        ]

    def generate_segs(self) -> None:
        """Construct the body segments at their computed positions."""
        self.segs = util.ItemList(
            objs=self.Nsegs,
            cls=self.param.segs.item_type,
            pos=self.seg_positions,
            orientation=self.orientation,
            base_vertices=self.base_seg_vertices,
            length=(self.length * self.segment_ratio).tolist(),
        )

    def compute_body_bend(self) -> None:
        """Set the body bend from the leading joint angles.

        Only the angles counted towards the bend are summed, so that the
        measure reflects the front of the body rather than its whole length.
        """
        angles = [
            util.angle_dif(
                self.segs[i].get_orientation(),
                self.segs[i + 1].get_orientation(),
                in_deg=False,
            )
            for i in range(int(self.Nangles + 1 / 2))
        ]
        self.body_bend = util.wrap_angle_to_0(sum(angles))

    @property
    def head(self) -> ShapeViewable:
        """The leading body segment."""
        return self.segs[0]

    @property
    def tail(self) -> ShapeViewable:
        """The trailing body segment."""
        return self.segs[-1]

    @property
    def direction(self) -> float:
        """The heading of the leading segment."""
        return self.head.get_orientation()

    def get_shape(self, scale=1):
        """Return the body outline as one merged polygon.

        Args:
            scale: Factor applied to the segment polygons.

        Returns:
            The union of the segment shapes.
        """
        ps = [geometry.Polygon(seg.vertices) for seg in self.segs]
        if scale != 1:
            ps = [affinity.scale(p, xfact=scale, yfact=scale) for p in ps]
        return ops.unary_union(ps).boundary.coords

    @property
    def global_midspine_of_body(self) -> tuple[float, float]:
        """The midpoint of the body axis, in world coordinates."""
        if self.Nsegs == 1:
            return self.head.get_position()
        elif self.Nsegs == 2:
            return self.head.rear_end
        if (self.Nsegs % 2) == 0:
            seg_idx = int(self.Nsegs / 2)
            global_pos = self.segs[seg_idx].front_end
        else:
            seg_idx = int((self.Nsegs + 1) / 2)
            global_pos = self.segs[seg_idx].get_position()
        return global_pos

    @param.depends("length", watch=True)
    def update_seg_lengths(self) -> None:
        """Rescale the segments after the body length changed."""
        for i in range(self.Nsegs):
            self.segs[i].length = self.length * self.segment_ratio[i]

    def move_body(self, dx, dy) -> None:
        """Translate the whole body.

        Args:
            dx: The x displacement.
            dy: The y displacement.
        """
        x0, y0 = self.get_position()
        self.set_position((x0 + dx, y0 + dy))
        for i, seg in enumerate(self.segs):
            x, y = seg.get_position()
            seg.set_position((x + dx, y + dy))

    def valid_Dbend_range(self, idx=0) -> tuple[float, float]:
        """Return the bend change admissible at one joint.

        Args:
            idx: The joint index.
            ang: The proposed bend change.

        Returns:
            The change clipped to the joint's angular limits.
        """
        if self.Nsegs > idx + 1:
            dang = util.wrap_angle_to_0(
                self.segs[idx + 1].get_orientation() - self.segs[idx].get_orientation()
            )
        else:
            dang = 0
        return (-np.pi + dang), (np.pi + dang)

    def set_color(self, colors) -> None:
        """Colour the segments.

        Args:
            colors: One colour per segment, or a single colour applied
                to all of them.
        """
        if len(colors) != self.Nsegs:
            colors = [tuple(colors)] * self.Nsegs
        for seg, col in zip(self.segs, colors):
            seg.color = col

    @property
    def midline_xy(self) -> list[tuple]:
        """The midline points, from the head end to the tail end."""
        return [seg.front_end for seg in self.segs] + [self.tail.rear_end]

    @property
    def front_orientation(self) -> float:
        """The leading segment's heading, wrapped to [0, 2pi)."""
        return self.head.get_orientation() % (2 * np.pi)

    @property
    def rear_orientation(self) -> float:
        """The trailing segment's heading, wrapped to [0, 2pi)."""
        return self.tail.get_orientation() % (2 * np.pi)

    def draw_segs(self, v, **kwargs) -> None:
        """Render every body segment.

        Args:
            v: The viewer to draw into.
            **kwargs: Forwarded to the segment drawing.
        """
        self.segs.draw(v, **kwargs)


class SegmentedBodySensored(SegmentedBody):
    """
    Segmented body with configurable sensory organs.

    Extends SegmentedBody with support for positioning sensors (olfactory,
    touch, etc.) on specific body segments with local coordinates.

    Attributes:
        sensors: Dictionary of sensor definitions with positions and modalities

    Example:
        >>> body = SegmentedBodySensored(Nsegs=12)
        >>> body.add_touch_sensors([5, 10])
        >>> olf_pos = body.olfactor_pos
    """

    def __init__(self, **kwargs):
        """Build the body with an empty sensor set.

        Args:
            **kwargs: Attributes, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        self.sensors = util.AttrDict()
        self.define_sensor("olfactor", (1, 0), modality="olfaction")

    @property
    def olfactor_pos(self) -> tuple:
        """The position of the olfactory sensor, at the head tip."""
        return self.head.front_end

    @property
    def olfactor_point(self) -> geometry.Point:
        """The olfactory sensor position, as a geometry point."""
        return geometry.Point(self.olfactor_pos[0], self.olfactor_pos[1])

    def define_sensor(self, sensor, pos_on_body, modality) -> None:
        """Mount a sensor at a position along the body.

        The body-relative position is resolved to the segment containing it,
        so the sensor follows that segment as the body bends.

        Args:
            sensor: The sensor name.
            pos_on_body: Its position, as a fraction of body length from the
                head, and an offset across the body axis.
            modality: The sensory modality it belongs to.
        """
        x, y = pos_on_body
        for i, (r, cum_r) in enumerate(
            zip(self.segment_ratio, np.cumsum(self.segment_ratio))
        ):
            if x >= 1 - cum_r:
                seg_idx = i
                local_pos = np.array([x - 1 + cum_r - r / 2, y])
                break
        self.sensors[sensor] = util.AttrDict(
            {
                "seg_idx": seg_idx,
                "local_pos": local_pos,
                "modality": modality,
            }
        )

    def get_sensor_position(self, sensor) -> tuple:
        """Return a sensor's current world position.

        Args:
            sensor: The sensor name.

        Returns:
            Its position.
        """
        d = self.sensors[sensor]
        return self.segs[d.seg_idx].translate(tuple(d.local_pos * self.length))

    def add_touch_sensors(self, idx) -> None:
        """Mount touch sensors at the given body positions.

        Args:
            idx: The indices of the standard mounting points to use.
        """
        for i in idx:
            self.define_sensor(
                f"touch_sensor_{i}", self.base_vertices[i], modality="touch"
            )

    def draw_sensors(self, v, **kwargs) -> None:
        """Render the mounted sensors.

        Args:
            v: The viewer to draw into.
            **kwargs: Accepted for signature compatibility; unused.
        """
        for s in self.sensors:
            pos = self.get_sensor_position(s)
            v.draw_circle(
                radius=self.length / 10,
                position=pos,
                filled=True,
                color=(255, 0, 0),
                width=0.1,
            )

    def get_sensors_by_modality(self, modality) -> list[str]:
        """List the sensors of one modality.

        Args:
            modality: The modality to select.

        Returns:
            The matching sensor names.
        """
        return [s for s, dic in self.sensors.items() if dic["modality"] == modality]

    @property
    def touch_sensorIDs(self) -> list[str]:
        """The names of the mounted touch sensors."""
        return self.get_sensors_by_modality("touch")
