from __future__ import annotations
from typing import Any
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import LarvaReplay, LarvaReplayContoured, LarvaReplaySegmented'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import LarvaReplay, LarvaReplayContoured, LarvaReplaySegmented'",
        DeprecationWarning,
        stacklevel=2,
    )
import numpy as np

from ... import util
from . import Larva, LarvaContoured, LarvaSegmented

__all__: list[str] = [
    "LarvaReplay",
    "LarvaReplayContoured",
    "LarvaReplaySegmented",
]

__displayname__ = "Experimental replay larva"


class LarvaReplay(Larva):
    """
    Larva agent for replaying experimental trajectory data.

    Extends Larva to replay pre-recorded positional and orientation data
    from real experiments, enabling visualization and analysis of
    experimental trajectories within the simulation environment.

    Attributes:
        data: ReplayData instance with recorded trajectories
        midline_xy: Midline points at current timestep (property)
        front_orientation: Front orientation at current timestep (property)
        rear_orientation: Rear orientation at current timestep (property)

    Args:
        data: Recorded experimental data (positions, orientations, midlines)
        **kwargs: Additional larva configuration

    Example:
        >>> replay_data = load_experiment_data('SchleyerGroup')
        >>> larva = LarvaReplay(data=replay_data['larva_0'])
        >>> larva.step()  # Update to next recorded frame
    """

    __displayname__ = "Replay larva"

    def __init__(self, data: Any, **kwargs: Any) -> None:
        self.data = data
        fo0 = self.data.front_orientation[0]
        if np.isnan(fo0):
            fo0 = 0

        super().__init__(pos=self.data.pos[0], orientation=fo0, **kwargs)

    def step(self) -> None:
        """
        Update the replay larva's position and orientation based on recorded data.

        Notes
        -----
        This method updates the replay larva's position and orientation for each time step based on recorded data.
        It also updates the trajectory and orientation_trajectory attributes.

        """
        self.pos = self.data.pos[self.model.t]
        self.trajectory.append(self.pos)
        self.orientation_trajectory.append(self.front_orientation)
        if not np.isnan(self.pos).any():
            self.model.space.move_to(self, np.array(self.pos))

    @property
    def midline_xy(self) -> list[tuple[float, float]]:
        """
        Get the xy coordinates of the midline points based on recorded data.

        Returns
        -------
        list of tuple
            A list of tuples representing the xy coordinates of the midline points.

        """
        return util.np2Dtotuples(self.data.midline[self.model.t])

    @property
    def front_orientation(self) -> float:
        """
        Get the front orientation of the replay larva based on recorded data.

        Returns
        -------
        float
            The front orientation in radians.

        """
        return self.data.front_orientation[self.model.t]

    @property
    def rear_orientation(self) -> float:
        """
        Get the rear orientation of the replay larva based on recorded data.

        Returns
        -------
        float
            The rear orientation in radians.

        """
        return self.data.rear_orientation[self.model.t]


class LarvaReplayContoured(LarvaReplay, LarvaContoured):
    """
    Replay larva with contour body representation from experimental data.

    Combines LarvaReplay trajectory playback with LarvaContoured geometry,
    displaying recorded contour vertices at each timestep for realistic
    body shape visualization from experimental recordings.

    Attributes:
        contour_xy: Contour vertices at current timestep (property)

    Example:
        >>> replay_data = load_experiment_data('SchleyerGroup')
        >>> larva = LarvaReplayContoured(data=replay_data['larva_0'])
        >>> larva.step()  # Update to next frame with contour
    """

    __displayname__ = "Contoured replay larva"

    def step(self) -> None:
        """
        Update the replay larva's position, orientation, and contour based on recorded data.

        Notes
        -----
        This method updates the replay larva's position and orientation using the base class `LarvaReplay` and adds
        contour data to the replay larva.

        """
        super().step()
        self.vertices = self.contour_xy

    @property
    def contour_xy(self) -> list[tuple[float, float]]:
        """
        Get the xy coordinates of the contour points based on recorded data.

        Returns
        -------
        list of tuple
            A list of tuples representing the xy coordinates of the contour points.

        """
        a = self.data.contour[self.model.t]
        a = a[~np.isnan(a)].reshape(-1, 2)
        return util.np2Dtotuples(a)


class LarvaReplaySegmented(LarvaReplay, LarvaSegmented):
    """
    Replay larva with segmented body from experimental data.

    Combines LarvaReplay trajectory playback with LarvaSegmented multi-segment
    body, positioning and orienting each segment based on recorded midline
    data for high-fidelity biomechanical visualization.

    Attributes:
        segs: Body segments positioned from recorded midline data

    Example:
        >>> replay_data = load_experiment_data('SchleyerGroup')
        >>> larva = LarvaReplaySegmented(data=replay_data['larva_0'], Nsegs=11)
        >>> larva.step()  # Update all segments to recorded positions
    """

    __displayname__ = "Segmented replay larva"

    def step(self) -> None:
        """
        Update the replay larva's position, orientation, and body segments based on recorded data.

        Notes
        -----
        This method updates the replay larva's position and orientation using the base class `LarvaReplay` and creates
        and positions multiple body segments based on recorded data.

        """
        super().step()
        mid = self.midline_xy
        ors = self.data.seg_orientations[self.model.t]
        for i, seg in enumerate(self.segs):
            seg.set_position(mid[i])
            try:
                seg.set_orientation(ors[i])
            except:
                pass
