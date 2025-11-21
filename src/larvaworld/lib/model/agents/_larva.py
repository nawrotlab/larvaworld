from __future__ import annotations
from typing import Any
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import Larva, LarvaContoured, LarvaSegmented, LarvaMotile'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import Larva, LarvaContoured, LarvaSegmented, LarvaMotile'",
        DeprecationWarning,
        stacklevel=2,
    )
import math
from copy import deepcopy

import numpy as np

from ... import util
from ...param import Contour, SegmentedBodySensored
from . import MobileAgent, Source

__all__: list[str] = [
    "Larva",
    "LarvaContoured",
    "LarvaSegmented",
    "LarvaMotile",
]

__displayname__ = "Larva agent"


class Larva(MobileAgent):
    """
    Base larva agent with trajectory tracking and visualization.

    Extends MobileAgent to provide larva-specific behaviors including
    trajectory recording, orientation tracking, and comprehensive
    drawing capabilities for visualization.

    Attributes:
        trajectory: List of (x, y) positions recorded over time
        orientation_trajectory: List of orientations (radians) over time
        cum_dur: Cumulative duration of simulation in seconds
        model: The ABM model containing this agent
        unique_id: Unique identifier string for this agent

    Example:
        >>> larva = Larva(model=sim_model, unique_id="larva_001")
        >>> larva.step()  # Execute one simulation timestep
    """

    def __init__(
        self, model: Any | None = None, unique_id: str | None = None, **kwargs: Any
    ) -> None:
        if unique_id is None and model:
            unique_id = model.next_id(type="Larva")
        super().__init__(unique_id=unique_id, model=model, **kwargs)
        self.trajectory = [self.initial_pos]
        self.orientation_trajectory = [self.initial_orientation]
        self.cum_dur = 0
        # self.cum_t = 0

    def draw(self, v: Any, **kwargs: Any) -> None:
        """
        Draws the larva on the screen using the provided ScreenManager instance.

        Parameters:
        v (ScreenManager): The screen manager instance used for drawing.
        **kwargs: Additional keyword arguments.

        The drawing includes:
        - Centroid if `v.draw_centroid` is True.
        - Midline if `v.draw_midline` is True.
        - Head if `v.draw_head` is True.
        - Trails if `v.visible_trails` is True.
        - Orientations if `v.draw_orientations` is True.

        The method handles NaN values in the position and trajectory data gracefully.
        """
        p, c, r, l = self.get_position(), self.color, self.radius, self.length
        mid = self.midline_xy
        if np.isnan(p).all():
            return
        if v.draw_centroid:
            v.draw_circle(p, r / 4, c, True, r / 10)

        if v.draw_midline:
            if not any(np.isnan(np.array(mid).flatten())):
                Nmid = len(mid)
                v.draw_polyline(mid, color=(0, 0, 255), closed=False, width=l / 20)
                for i, xy in enumerate(mid):
                    c = 255 * i / (Nmid - 1)
                    v.draw_circle(xy, l / 30, color=(c, 255 - c, 0), width=l / 40)

        if v.draw_head:
            v.draw_circle(mid[0], l / 4, color=(255, 0, 0), width=l / 12)

        if v.visible_trails:
            Nfade = int(v.trail_dt / self.model.dt)
            traj = self.trajectory[-Nfade:]
            or_traj = self.orientation_trajectory[-Nfade:]
            if not np.isnan(traj).any():
                parsed_traj = [traj]
                parsed_or_traj = [or_traj]
            elif np.isnan(traj).all():
                return
            # This is the case for larva trajectories derived from experiments where some values are np.nan
            else:
                ds, de = util.parse_array_at_nans(np.array(traj)[:, 0])
                parsed_traj = [traj[s:e] for s, e in zip(ds, de)]
                parsed_or_traj = [or_traj[s:e] for s, e in zip(ds, de)]
            Npars = len(parsed_traj)
            for i in range(Npars):
                t = parsed_traj[i]
                or_t = parsed_or_traj[i]
                # If trajectory has one point, skip
                if len(t) < 2:
                    pass
                else:
                    if v.trail_color == "normal":
                        color = self.color
                    elif v.trail_color == "linear":
                        color = util.scaled_velocity_to_col(
                            util.eudist(np.array(t)) / self.length / self.model.dt
                        )
                    elif v.trail_color == "angular":
                        color = util.angular_velocity_to_col(
                            np.diff(np.array(or_t)) / self.model.dt
                        )
                    else:
                        raise

                    try:
                        v.draw_polyline(t, color=color, width=0.0005)
                    except:
                        pass

        if v.draw_orientations:
            p02 = [
                p[0] + math.cos(self.front_orientation) * l,
                p[1] + math.sin(self.front_orientation) * l,
            ]
            v.draw_line(p, p02, color="green", width=l / 10)
            p12 = [
                p[0] - math.cos(self.rear_orientation) * l,
                p[1] - math.sin(self.rear_orientation) * l,
            ]
            v.draw_line(p, p12, color="red", width=l / 10)
        super().draw(v, **kwargs)


class LarvaContoured(Larva, Contour):
    """
    Larva agent with contour-based body representation.

    Combines base Larva behavior with Contour geometry, providing
    visual representation via polygon vertices around the body perimeter.
    Useful for rendering realistic larva shapes and collision detection.

    Attributes:
        vertices: List of (x, y) points defining the body contour

    Example:
        >>> larva = LarvaContoured(model=sim_model, num_vertices=20)
        >>> larva.draw(screen_manager)  # Draws contour outline
    """

    __displayname__ = "Contoured larva"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def draw(self, v: Any, **kwargs: Any) -> None:
        """
        Draws the larva agent on the screen.

        Parameters:
        v (ScreenManager): The screen manager responsible for rendering.
        **kwargs: Additional keyword arguments to pass to the drawing functions.

        Returns:
        None
        """
        if v.draw_contour:
            Contour.draw(self, v, **kwargs)
        super().draw(v, **kwargs)

    def draw_selected(self, v: Any, **kwargs: Any) -> None:
        """
        Draws the selected larva on the screen.

        Parameters:
        v (ScreenManager): The screen manager instance responsible for drawing.
        **kwargs: Additional keyword arguments (not used in this method).

        Returns:
        None
        """
        v.draw_polygon(
            vertices=self.vertices, color=v.selection_color, filled=False, width=0.0002
        )


class LarvaSegmented(Larva, SegmentedBodySensored):
    """
    Larva agent with segmented body and sensor capabilities.

    Combines base Larva with SegmentedBodySensored to provide realistic
    biomechanical modeling via discrete body segments and sensor placement
    for olfaction, touch, and other modalities.

    Attributes:
        segs: Collection of body segments with individual colors
        sensors: Dictionary of sensory modules by modality

    Example:
        >>> larva = LarvaSegmented(model=sim_model, Nsegs=11)
        >>> larva.define_sensor('olfactor', pos=(0.5, 0))
        >>> larva.draw_segs(screen_manager)
    """

    __displayname__ = "Segmented larva"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_default_color(self.default_color)

    def draw(self, v: Any, **kwargs: Any) -> None:
        """
        Draws the larva agent on the screen using the provided ScreenManager.

        Parameters:
        v (ScreenManager): The screen manager responsible for rendering.
        **kwargs: Additional keyword arguments to pass to the drawing methods.

        If the ScreenManager's draw_sensors attribute is True, the larva's sensors
        will be drawn using the draw_sensors method. If the draw_contour attribute
        is True, the larva's contour segments will be drawn using the draw_segs method.
        Finally, the superclass's draw method is called to complete the drawing process.
        """
        if v.draw_sensors:
            self.draw_sensors(v, **kwargs)
        if v.draw_contour:
            self.draw_segs(v, **kwargs)
        super().draw(v, **kwargs)

    def set_default_color(self, color: str | tuple[int, int, int]) -> None:
        """
        Sets the default color for the larva and its segments.

        Args:
            color: Color as string name ('red', 'blue') or RGB tuple (r, g, b)
        """
        super().set_default_color(color)
        self.segs.set_default_color(color)

    def invert_default_color(self) -> None:
        """
        Inverts the default color of the larva agent.

        This method calls the parent class's invert_default_color method to
        perform the inversion and then updates the segments' default color
        to match the new default color of the larva agent.
        """
        super().invert_default_color()
        self.segs.set_default_color(self.default_color)

    def draw_selected(self, v: Any, **kwargs: Any) -> None:
        """
        Draws the selected larva on the screen.

        This method uses the ScreenManager instance to draw the outline of the larva
        with a specific selection color. The shape of the larva is determined by the
        `get_shape` method.

        Parameters:
        v (ScreenManager): The screen manager instance responsible for drawing.
        **kwargs: Additional keyword arguments (not used in this method).

        Returns:
        None
        """
        v.draw_polygon(
            vertices=self.get_shape(),
            color=v.selection_color,
            filled=False,
            width=0.0002,
        )


class LarvaMotile(LarvaSegmented):
    """
    Complete larva agent with behavior, energetics, and growth.

    Extends LarvaSegmented with brain-driven behavior, DEB-based energetics,
    and life history dynamics. Represents a fully autonomous larva capable
    of sensing, decision-making, feeding, and growth over developmental stages.

    Attributes:
        brain: Behavioral control system (DefaultBrain or NengoBrain)
        deb: Dynamic Energy Budget model for metabolism and growth
        food_detected: Currently detected food source (if any)
        feeder_motion: Whether currently executing feeding behavior
        cum_food_detected: Cumulative timesteps on food
        amount_eaten: Total food consumed (in mg)
        carried_objects: List of objects being carried

    Args:
        brain: Brain configuration dict or instance
        energetics: DEB energetics parameters (optional)
        life_history: Life stages and substrate info (optional)
        body: Body shape parameters dict
        **kwargs: Additional agent configuration

    Example:
        >>> larva = LarvaMotile(
        ...     brain={'olfactor': {'gain': 2.0}},
        ...     energetics={'X_substrate': 0.8},
        ...     life_history={'age': 96.0},
        ...     body={'length': 0.003}
        ... )
        >>> larva.step()  # Sense, decide, act, feed, grow
    """

    __displayname__ = "Behaving & growing larva"

    def __init__(
        self,
        brain: Any,
        energetics: Any,
        life_history: Any,
        body: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(**body, **kwargs)
        self.carried_objects = []
        self.brain = self.build_brain(brain)
        self.build_energetics(energetics, life_history=life_history)
        self.food_detected, self.feeder_motion = None, False
        self.cum_food_detected, self.amount_eaten = 0, 0

    def build_brain(self, conf: Any) -> Any:
        """Build the brain for the larva agent."""
        if conf.nengo:
            from ..modules.nengobrain import NengoBrain

            return NengoBrain(agent=self, conf=conf, dt=self.model.dt)
        else:
            from ..modules.brain import DefaultBrain

            return DefaultBrain(agent=self, conf=conf, dt=self.model.dt)

    def feed(self, source: Any | None, motion: bool) -> float:
        """
        Feeds the larva from a given source based on its motion.

        Parameters:
        source (object): The source from which the larva will feed. It can be an object that has a method `subtract_amount`.
        motion (bool): A boolean indicating whether the larva is in motion or not.

        Returns:
        float: The volume of food consumed by the larva. Returns 0 if the source is None or if the larva is not in motion.
        """

        def get_max_V_bite():
            return self.brain.locomotor.feeder.V_bite * self.V * 1000  # ** (2 / 3)

        if motion:
            a_max = get_max_V_bite()
            if source is not None:
                grid = self.model.food_grid
                if grid:
                    V = -grid.add_cell_value(source, -a_max)
                else:
                    V = source.subtract_amount(a_max)

                return V
            else:
                return 0
        else:
            return 0

    def build_energetics(
        self, energetic_pars: Any | None, life_history: Any | None
    ) -> None:
        """
        Initializes and builds the energetics model for the larva.

        Parameters:
        -----------
        energetic_pars : AttrDict or None
            Parameters related to the energetics of the larva, including gut parameters and DEB (Dynamic Energy Budget) parameters.
        life_history : AttrDict or None
            Life history information of the larva, including epochs and age.

        Attributes:
        -----------
        deb : DEB or None
            An instance of the DEB class representing the energetics model of the larva.
        length : float
            The length of the larva in meters.
        mass : float or None
            The mass of the larva.
        V : float
            The volume of the larva.

        Notes:
        ------
        If `energetic_pars` is None, the energetics model is not initialized, and the volume (`V`) is calculated as the cube of the length.
        """
        from ...model.deb.deb import DEB

        if energetic_pars is not None:
            try:
                im = self.brain.locomotor.intermitter
            except:
                im = None
            if life_history is None:
                life_history = util.AttrDict({"epochs": {}, "age": None})

            self.deb = DEB(
                id=self.unique_id,
                intermitter=im,
                gut_params=energetic_pars.gut,
                **energetic_pars.DEB,
            )
            self.deb.grow_larva(epochs=life_history.epochs)
            self.length = self.deb.Lw * 10 / 1000
            self.mass = self.deb.Ww
            self.V = self.deb.V

        else:
            self.deb = None
            self.V = self.length**3
            self.mass = None
            # self.length = None

    def run_energetics(self, V_eaten: float) -> None:
        """
        Update the energetics of the larva based on the volume of food eaten.

        This method updates the larva's state by running a DEB (Dynamic Energy Budget) model check
        with the given volume of food eaten and the model's time step. It then updates the larva's
        length, mass, and volume based on the DEB model's outputs.

        Parameters:
        V_eaten (float): The volume of food eaten by the larva.

        Returns:
        None
        """
        self.deb.run_check(dt=self.model.dt, X_V=V_eaten)
        self.length = self.deb.Lw * 10 / 1000
        self.mass = self.deb.Ww
        self.V = self.deb.V
        # TODO add this again
        # self.adjust_body_vertices()

    def get_feed_success(self, t: float) -> int:
        if self.feeder_motion:
            if self.on_food:
                return 1
            else:
                return -1
        else:
            return 0

    @property
    def on_food_dur_ratio(self) -> float:
        return (
            self.cum_food_detected * self.model.dt / self.cum_dur
            if self.cum_dur != 0
            else 0
        )

    @property
    def on_food(self) -> bool:
        return self.food_detected is not None

    def get_on_food(self, t: float) -> bool:
        return self.on_food

    @property
    def scaled_amount_eaten(self) -> float:
        return self.amount_eaten / self.mass

    def resolve_carrying(self, food: Any | None) -> None:
        """
        Handles the logic for an agent to carry a food object.

        Parameters:
        food (Source): The food object to be carried by the agent.

        Returns:
        None

        The method performs the following steps:
        1. Checks if the food object is valid and can be carried.
        2. If the food is already being carried by another agent, it removes it from the previous carrier.
        3. Assigns the food to the current agent and updates the carried objects list.
        4. Depending on the experiment type ("capture_the_flag" or "keep_the_flag"), it adjusts the olfactory gains for the agent and its group members.
        5. Updates the position of all carried objects to match the agent's position.

        Raises:
        ValueError: If the agent's group is neither "Left" nor "Right" in the "keep_the_flag" experiment.
        """
        gain_for_base_odor = 100.0
        if food is None or not isinstance(food, Source):
            return
        if food.can_be_carried and food not in self.carried_objects:
            if food.is_carried_by is not None:
                prev_carrier = food.is_carried_by
                if prev_carrier == self:
                    return
                prev_carrier.carried_objects.remove(food)
                prev_carrier.brain.olfactor.reset_all_gains()
            food.is_carried_by = self
            self.carried_objects.append(food)
            if self.model.experiment == "capture_the_flag":
                self.brain.olfactor.set_gain(
                    gain_for_base_odor, f"{self.group}_base_odor"
                )
            elif self.model.experiment == "keep_the_flag":
                carrier_group = self.group
                carrier_group_odor_id = self.odor.id
                if carrier_group == "Left":
                    opponent_group = "Right"
                elif carrier_group == "Right":
                    opponent_group = "Left"
                else:
                    raise ValueError(
                        f"Argument {carrier_group} is neither Left nor Right"
                    )

                opponent_group_odor_id = f"{opponent_group}_odor"
                for f in self.model.agents:
                    if f.group == carrier_group:
                        f.brain.olfactor.set_gain(
                            gain_for_base_odor, opponent_group_odor_id
                        )
                    else:
                        f.brain.olfactor.set_gain(0.0, carrier_group_odor_id)
                self.brain.olfactor.set_gain(
                    -gain_for_base_odor, opponent_group_odor_id
                )

        for o in self.carried_objects:
            o.pos = self.pos

    def update_behavior_dict(self, mode: str = "lin") -> None:
        """
        Updates the behavior dictionary and sets the color based on the current mode and state.

        Parameters:
        mode (str): The mode of behavior update. It can be "lin" for linear or "ang" for angular. Default is "lin".

        Raises:
        Exception: If the mode is "lin" and no active bouts are found.
        Exception: If the mode is not recognized.

        Behavior:
        - In "lin" mode, the color is set based on the active bouts from the intermitter:
            - Green ([0, 150, 0]) if 's' or 'r' is active.
            - Red ([255, 0, 0]) if 'p' is active.
            - Blue ([0, 0, 255]) if 'f' is active.
        - In "ang" mode, the color is set based on the front orientation velocity:
            - Blue component is set to 150 if the velocity is positive.
            - Blue component is set to 50 if the velocity is negative.

        The method uses the `set_color` function to apply the determined color.
        """
        inter = self.brain.locomotor.intermitter
        if mode == "lin" and inter is not None:
            s, f, p, r = inter.active_bouts
            if s or r:
                color = np.array([0, 150, 0])
            elif p:
                color = np.array([255, 0, 0])
            elif f:
                color = np.array([0, 0, 255])
            else:
                raise
        elif mode == "ang":
            color = deepcopy(self.default_color)
            orvel = self.front_orientation_vel
            if orvel > 0:
                color[2] = 150
            elif orvel < 0:
                color[2] = 50
        else:
            raise
        self.set_color(color)

    def sense(self) -> None:
        pass

    def step(self) -> None:
        """
        Perform a single step in the simulation for the larva agent.

        This method updates the larva's state by:
        - Incrementing the cumulative duration by the model's time step.
        - Sensing the environment to update the larva's position.
        - Detecting food sources either from accessible sources or by using the brain's sensors.
        - Resolving the carrying state based on detected food.
        - Calculating linear and angular motion using the brain's step method.
        - Preparing the larva for motion based on calculated values.
        - Feeding the larva if food is detected and updating the amount eaten.
        - Updating the cumulative food detected count.
        - Running energetics calculations if the DEB model is present.
        - Updating the larva's color based on behavior if the screen manager's color behavior is enabled.

        Exceptions are caught and ignored if there is an issue updating the color behavior.
        """
        m = self.model
        self.cum_dur += m.dt
        self.sense()
        pos = self.olfactor_pos

        if m.space.accessible_sources:
            self.food_detected = m.space.accessible_sources[self]

        elif self.brain.locomotor.feeder or self.brain.toucher:
            self.food_detected = util.sense_food(
                pos, sources=m.sources, grid=m.food_grid, radius=self.radius
            )
        self.resolve_carrying(self.food_detected)

        lin, ang, self.feeder_motion = self.brain.step(
            pos, length=self.length, on_food=self.on_food
        )
        self.prepare_motion(lin=lin, ang=ang)

        V = self.feed(self.food_detected, self.feeder_motion)
        self.amount_eaten += V * 1000
        self.cum_food_detected += int(self.on_food)
        if self.deb is not None:
            self.run_energetics(V)

        try:
            if self.model.screen_manager.color_behavior:
                self.update_behavior_dict()
            else:
                self.color = self.default_color
        except:
            pass

    def prepare_motion(self, lin: float, ang: float) -> None:
        pass
        # Overriden by subclasses
