"""
Mixins that make an object renderable in the simulation viewer.

Provides the base viewable, its toggleable and line-shaped variants, and the
contour primitive, each carrying the colour and visibility state the renderer
reads.
"""

from __future__ import annotations
from typing import Any

import param

from .. import util
from .custom import RandomizedColor
from .nested_parameter_group import NestedConf
from .spatial import LineClosed, LineExtended

__all__: list[str] = [
    "Viewable",
    "ViewableToggleable",
    "ViewableLine",
    "Contour",
]

__displayname__ = "Viewable elements"


class Viewable(NestedConf):
    """
    Base class for all visible objects in simulation.

    Provides color management, visibility toggling, and drawing infrastructure
    for visual entities. Subclasses implement specific drawing methods.

    Attributes:
        color: Color of the entity (string name or RGB tuple)
        visible: Whether the entity is currently visible
        selected: Whether the entity is selected for highlighting

    Example:
        >>> obj = Viewable(color='red', visible=True)
        >>> obj.set_color('blue')
        >>> obj.toggle_vis()
    """

    color = RandomizedColor(default="black", doc="The default color of the entity")
    visible = param.Boolean(True, doc="Whether the entity is visible or not")
    selected = param.Boolean(False, doc="Whether the entity is selected or not")

    def __init__(self, **kwargs):
        """Build the viewable object.

        Args:
            **kwargs: Attributes, forwarded to the parent class. A colour
                given as a tuple is normalized first.
        """
        if "color" in kwargs:
            if isinstance(kwargs["color"], tuple):
                kwargs["color"] = util.colortuple2str(kwargs["color"])
        super().__init__(**kwargs)

    @property
    def default_color(self) -> str:
        """Set the colour the object reverts to.

        Args:
            new_color: The new default colour.
        """
        return self.param.color.default

    @default_color.setter
    def default_color(self, new_color) -> None:
        """The colour the object reverts to."""
        self.param.color.default = new_color

    def set_color(self, color) -> None:
        """Set the current display colour.

        Args:
            color: The new colour.
        """
        self.color = color

    def set_default_color(self, color) -> None:
        """Set both the default and the current colour.

        Args:
            color: The new colour.
        """
        self.default_color = color
        self.color = color

    def invert_default_color(self) -> None:
        """Invert the default colour, for drawing on a dark background."""
        c00, c01 = util.invert_color(self.default_color)
        self.set_default_color(c01)

    def _draw(self, v, **kwargs) -> None:
        """Render the object if it is visible.

        Args:
            v: The viewer to draw into.
            **kwargs: Forwarded to :meth:`draw`.
        """
        if self.visible:
            self.draw(v, **kwargs)
            if self.selected:
                self.draw_selected(v, **kwargs)
            if hasattr(self, "id_box"):
                self.id_box._draw(v, **kwargs)

    def draw_selected(self, v, **kwargs) -> None:
        """Render the object's selection highlight. Does nothing by default.

        Args:
            v: The viewer to draw into.
            **kwargs: Subclass-specific arguments.
        """
        pass

    def draw(self, v, **kwargs) -> None:
        """Render the object. Subclasses override this.

        Args:
            v: The viewer to draw into.
            **kwargs: Subclass-specific arguments.
        """
        pass

    # @property
    def toggle_vis(self) -> bool:
        """Flip the object's visibility.

        Returns:
            The visibility after the change.
        """
        self.visible = not self.visible
        return self.visible


class ViewableToggleable(Viewable):
    """
    Viewable object with active/inactive state and color switching.

    Extends Viewable with an active state that automatically switches
    between active and inactive colors.

    Attributes:
        active: Whether the entity is currently active
        active_color: Color when entity is active
        inactive_color: Color when entity is inactive

    Example:
        >>> obj = ViewableToggleable(active_color='red', inactive_color='blue')
        >>> obj.toggle()  # Switches active state and color
    """

    active = param.Boolean(False, doc="Whether entity is active")
    active_color = param.Color("lightblue", doc="The color of the entity when active")
    inactive_color = param.Color(
        "lightgreen", doc="The color of the entity when inactive"
    )

    def __init__(self, **kwargs):
        """Build the object and resolve its active and inactive colours.

        Args:
            **kwargs: Attributes, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        if self.active_color is None:
            self.active_color = self.color
        if self.inactive_color is None:
            self.inactive_color = self.color
        self.update_color()

    @param.depends("active", watch=True)
    def update_color(self) -> None:
        """Apply the colour matching the current active state."""
        self.color = self.active_color if self.active else self.inactive_color

    def toggle(self) -> None:
        """Flip the active state and update the colour."""
        self.active = not self.active


class ViewableLine(Viewable, LineExtended):
    """
    Viewable line or polyline with rendering capabilities.

    Combines Viewable and LineExtended to create drawable lines/polylines
    with configurable width, color, and closure.

    Example:
        >>> line = ViewableLine(vertices=[(0,0), (1,0), (1,1)], color='red')
        >>> line.draw(viewer)
    """

    def draw(self, v, **kwargs) -> None:
        """Render the object as a polyline.

        Args:
            v: The viewer to draw into.
            **kwargs: Accepted for signature compatibility; unused.
        """
        try:
            v.draw_polyline(
                vertices=self.vertices,
                color=self.color,
                width=self.width,
                closed=self.closed,
            )
        except:
            for ver in self.vertices:
                v.draw_polyline(
                    ver, color=self.color, width=self.width, closed=self.closed
                )


class Contour(Viewable, LineClosed):
    """
    Viewable closed contour (filled polygon).

    Combines Viewable and LineClosed to create drawable filled polygons
    with configurable color.

    Example:
        >>> contour = Contour(vertices=[(0,0), (1,0), (1,1), (0,1)], color='green')
        >>> contour.draw(viewer)
    """

    def draw(self, v, **kwargs) -> None:
        """Render the contour as a filled polygon.

        Args:
            v: The viewer to draw into.
            **kwargs: Accepted for signature compatibility; unused.
        """
        v.draw_polygon(self.vertices, filled=True, color=self.color)
