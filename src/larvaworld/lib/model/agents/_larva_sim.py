from __future__ import annotations
from typing import Any, Tuple
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import BaseController, LarvaSim'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import BaseController, LarvaSim'",
        DeprecationWarning,
        stacklevel=2,
    )
import math

import numpy as np
import param
from shapely import geometry

from ... import util
from ...param import PositiveNumber
from . import LarvaMotile

__all__: list[str] = [
    "BaseController",
    "LarvaSim",
]

__displayname__ = "Simulation larva"


class BaseController(param.Parameterized):
    """
    Physics controller for larva kinematic simulation.

    Provides motion generation modes (velocity/force/torque), damping
    coefficients, and body mechanics (torsional spring, bend correction)
    for realistic larva movement simulation.

    Attributes:
        lin_vel_coef: Translational velocity scaling coefficient (default: 1.0)
        ang_vel_coef: Angular velocity scaling coefficient (default: 1.0)
        lin_force_coef: Force scaling coefficient (default: 1.0)
        torque_coef: Torque scaling coefficient (default: 0.5)
        body_spring_k: Torsional spring constant for body bending (default: 1.0)
        bend_correction_coef: Bend angle correction factor (default: 1.0)
        lin_damping: Translational damping coefficient (default: 1.0)
        ang_damping: Angular damping coefficient (default: 1.0)
        lin_mode: Motion mode ('velocity', 'force', or 'impulse')
        ang_mode: Rotation mode ('torque' or 'velocity')

    Example:
        >>> controller = BaseController(torque_coef=0.8, body_spring_k=1.5)
        >>> delta_angle = controller.compute_delta_rear_angle(0.2, 0.001, 0.003)
    """

    lin_vel_coef = PositiveNumber(1.0, doc="Coefficient for translational velocity")
    ang_vel_coef = PositiveNumber(1.0, doc="Coefficient for angular velocity")
    lin_force_coef = PositiveNumber(1.0, doc="Coefficient for force")
    torque_coef = PositiveNumber(0.5, doc="Coefficient for torque")
    body_spring_k = PositiveNumber(
        1.0, doc="Torsional spring constant for body bending"
    )
    bend_correction_coef = PositiveNumber(1.0, doc="Bend correction coefficient")
    lin_damping = PositiveNumber(1.0, doc="Translational damping coefficient")
    ang_damping = PositiveNumber(1.0, doc="Angular damping coefficient")
    lin_mode = param.Selector(
        objects=["velocity", "force", "impulse"],
        doc="Mode of translational motion generation",
    )
    ang_mode = param.Selector(
        objects=["torque", "velocity"], doc="Mode of angular motion generation"
    )

    def compute_delta_rear_angle(self, bend: float, dst: float, length: float) -> float:
        """
        Compute the change in rear angle based on bend, distance, and length.

        Parameters
        ----------
        bend : float
            Bend angle.
        dst : float
            Distance.
        length : float
            Length of the larva.

        Returns
        -------
        float
            Change in rear angle.

        """
        k0 = 2 * dst * self.bend_correction_coef / length
        if 0 <= k0 < 1:
            return bend * k0
        elif 1 <= k0:
            return bend
        elif k0 < 0:
            return 0


class LarvaSim(LarvaMotile, BaseController):
    """
    Physically-simulated larva agent with realistic biomechanics.

    Combines LarvaMotile behavioral capabilities with BaseController
    physics to provide realistic simulation including body mechanics,
    collision detection, and arena boundary handling.

    Attributes:
        collision_with_object: Whether currently colliding with obstacle
        body_bend: Current body bend angle (radians)
        dst: Distance traveled in last timestep
        cum_dst: Cumulative distance traveled
        border_collision: Border collision detection (property)
        larva_collision: Inter-larva collision detection (property)

    Args:
        physics: Physics parameters dict (velocities, damping, spring constants)
        Box2D: Box2D physics engine parameters (optional)
        sensorimotor: Sensorimotor coupling parameters (optional)
        **kwargs: Additional larva configuration

    Example:
        >>> larva = LarvaSim(
        ...     physics={'torque_coef': 0.5, 'body_spring_k': 1.0},
        ...     brain={'olfactor': {'gain': 2.0}},
        ...     body={'length': 0.003}
        ... )
        >>> larva.step()  # Physics-based simulation step
    """

    __displayname__ = "Simulated larva"

    def __init__(
        self,
        physics: dict[str, Any] = {},
        Box2D: dict[str, Any] = {},
        sensorimotor: Any = None,
        **kwargs: Any,
    ) -> None:
        BaseController.__init__(self, **physics)
        LarvaMotile.__init__(self, **kwargs)

        self.collision_with_object = False

    def compute_ang_vel(self, amp: float) -> float:
        """
        Compute angular velocity based on torque amplitude.

        Parameters
        ----------
        amp : float
            Torque amplitude.

        Returns
        -------
        float
            Angular velocity.

        """
        torque = amp * self.torque_coef
        ang_vel = self.get_angularvelocity()
        return (
            ang_vel
            + (
                -self.ang_damping * ang_vel
                - self.body_spring_k * self.body_bend
                + torque
            )
            * self.dt
        )

    def prepare_motion(self, lin: float, ang: float) -> None:
        """
        Prepare translational and angular motion.

        Parameters
        ----------
        lin : float
            Linear motion parameter.
        ang : float
            Angular motion parameter.

        """
        lin_vel = lin * self.lin_vel_coef
        if self.ang_mode == "torque":
            ang_vel = self.compute_ang_vel(ang)
        elif self.ang_mode == "velocity":
            ang_vel = ang * self.ang_vel_coef
        else:
            raise
        if self.border_collision or self.larva_collision:
            lin_vel = 0
            ang_vel += np.sign(ang_vel) * np.pi / 10
        self.position_body(lin_vel, ang_vel)

    @property
    def border_collision(self) -> bool:
        """
        Check for collisions with borders.

        Returns
        -------
        bool
            True if there is a border collision, False otherwise.

        """
        if len(self.model.borders) == 0:
            return False
        else:
            x, y = self.pos
            p0 = geometry.Point(self.pos)
            d0 = self.length / 4
            oM = self.get_orientation()
            p1 = geometry.Point(x + math.cos(oM) * d0, y + math.sin(oM) * d0)

            sensor_ray = p0, p1

            min_dst, nearest_obstacle = util.detect_nearest_obstacle(
                self.model.borders, sensor_ray, p0
            )

            if min_dst is None:
                return False
            else:
                return True

    @property
    def larva_collision(self) -> bool:
        """
        Check for collisions with other larvae.

        Returns
        -------
        bool
            True if there is a collision, False otherwise.

        """
        if not self.model.larva_collisions:
            ids = self.model.detect_collisions(self.unique_id)
            return False if len(ids) == 0 else True
        else:
            return False

    def position_head_in_tank(
        self,
        hr0: Tuple[float, float],
        ho0: float,
        l0: float,
        fov0: float,
        fov1: float,
        ang_vel: float,
        lin_vel: float,
    ) -> tuple[float, float]:
        """
        Position the larva's head in the simulated tank.

        Parameters
        ----------
        hr0 : tuple
            Initial position of the rear end of the larva.
        ho0 : float
            Initial orientation of the larva.
        l0 : float
            Length of the larva.
        fov0 : float
            Minimum allowed angular velocity.
        fov1 : float
            Maximum allowed angular velocity.
        ang_vel : float
            Angular velocity.
        lin_vel : float
            Linear velocity.

        Returns
        -------
        tuple
            New angular velocity and linear velocity.

        """
        dt = self.model.dt
        sf = self.model.scaling_factor

        def get_hf0(ang_vel):
            return tuple(
                np.array(hr0)
                + np.array([l0, 0]) @ util.rotationMatrix(-ho0 - ang_vel * dt)
            )

        def fov(ang_vel):
            dv = 8 * np.pi / 90
            idx = 0
            while not self.model.space.in_area(get_hf0(ang_vel)):
                if idx == 0:
                    dv *= np.sign(ang_vel)
                ang_vel -= dv
                if ang_vel < fov0:
                    ang_vel = fov0
                    dv = np.abs(dv)
                elif ang_vel > fov1:
                    ang_vel = fov1
                    dv -= np.abs(dv)
                idx += 1
                if np.isnan(ang_vel) or idx > 100:
                    ang_vel = 0
                    break
            return ang_vel

        ang_vel = fov(ang_vel)
        ho1 = ho0 + ang_vel * dt
        k = np.array([math.cos(ho1), math.sin(ho1)])
        hf01 = get_hf0(ang_vel)

        def get_hf1(lin_vel):
            return hf01 + dt * sf * k * lin_vel

        def lv(lin_vel):
            dv = 0.00011
            idx = 0
            while not self.model.space.in_area(get_hf1(lin_vel)):
                idx += 1
                lin_vel -= dv
                if np.isnan(lin_vel) or lin_vel < 0 or idx > 100:
                    lin_vel = 0
                    break
            return lin_vel

        lin_vel = lv(lin_vel)
        return ang_vel, lin_vel

    def position_body(self, lin_vel: float, ang_vel: float) -> None:
        """
        Position the larva's body based on translational and angular motion.

        Parameters
        ----------
        lin_vel : float
            Linear velocity.
        ang_vel : float
            Angular velocity.

        """
        dt = self.model.dt
        hp0, ho0 = self.head.get_pose()
        hr0 = self.head.rear_end
        l0 = self.head.length
        A0, A1 = self.valid_Dbend_range(0)
        ang_vel = np.clip(ang_vel, a_min=A0 / dt, a_max=A1 / dt)

        fov0, fov1 = A0 / dt, A1 / dt

        if not self.model.space.torus:
            ang_vel, lin_vel = self.position_head_in_tank(
                hr0, ho0, l0, fov0, fov1, ang_vel, lin_vel
            )

        # else:
        ho1 = ho0 + ang_vel * dt
        k = np.array([np.cos(ho1), np.sin(ho1)])
        d = lin_vel * dt
        hp1 = hr0 + k * (d * self.model.scaling_factor + l0 / 2)
        self.head.update_all(hp1, ho1, lin_vel, ang_vel)
        self.dst = d
        self.cum_dst += self.dst

        if self.Nsegs > 1:
            d_or = self.compute_delta_rear_angle(
                self.body_bend, self.dst, self.length
            ) / (self.Nsegs - 1)
            for i, seg in enumerate(self.segs[1:]):
                seg.drag_to_front(fp=self.segs[i].rear_end, d_or=d_or)

        pos = tuple(self.global_midspine_of_body)
        self.update_larva_pose(pos, ho1, lin_vel, ang_vel)

    def update_larva_pose(
        self,
        position: tuple[float, float],
        orientation: float,
        lin_vel: float = 0,
        ang_vel: float = 0,
    ) -> None:
        """
        Update the larva's pose and trajectories' log.

        Parameters
        ----------
        position : float
            2D position.
        orientation : float
            Head orientation.
        lin_vel : float
            Linear velocity.
        ang_vel : float
            Angular velocity.

        """
        self.update_all(position, orientation, lin_vel=lin_vel, ang_vel=ang_vel)
        self.trajectory.append(position)
        self.orientation_trajectory.append(orientation)
        self.model.space.move_to(self, np.array(position))
        self.compute_body_bend()

    def reset_larva_pose(self, reset_trajectories: bool = False) -> None:
        """
        Reset the larva's pose to the initial position and orientation.

        Parameters
        ----------
        reset_trajectories : bool
            Reset the trajectories's log

        """
        if reset_trajectories:
            self.trajectory = []
            self.orientation_trajectory = []
        self.update_larva_pose(self.initial_pos, self.initial_orientation)
