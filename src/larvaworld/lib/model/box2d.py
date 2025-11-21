"""
A Box2D extension for larvaworld. This requires the box2d-py package, specified as an optional dependency during larvaworld installation.
If not already installed, run : "pip install box2d-py"
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, List

# import Box2D
import numpy as np
import param
from Box2D.b2 import dynamicBody, polygonShape, staticBody, vec2, world
from shapely import geometry

from .. import util
from ..param import ShapeMobile
from .agents import LarvaSim
from .envs import Arena

__all__: list[str] = ["SegmentBox2D", "LarvaBox2D", "ArenaBox2D"]

__displayname__ = "Box2D extension"


class SegmentBox2D(ShapeMobile):
    """
    Box2D-based segment of a larva.

    Args:
        space (Box2D.b2World): The Box2D world space.
        physics_pars (dict): Parameters related to the physics simulation.
        **kwargs (dict): Additional keyword arguments.


    Methods:
        get_position():
            Get the world position of the segment.

        set_position(position):
            Set the world position of the segment.

        get_orientation():
            Get the orientation of the segment.

        set_orientation(orientation):
            Set the orientation of the segment.

        get_pose():
            Get the pose (position and orientation) of the segment.

        set_linearvelocity(lin_vel, local=False):
            Set the linear velocity of the segment.

        get_angularvelocity():
            Get the angular velocity of the segment.

        set_angularvelocity(ang_vel):
            Set the angular velocity of the segment.

        set_mass(mass):
            Set the mass of the segment.

        get_mass():
            Get the mass of the segment.

        get_world_point(point):
            Transform a local point to world coordinates.

        get_world_facing_axis():
            Get the world-facing axis of the segment.

    """

    __displayname__ = "Box2D body segment"

    def __init__(
        self, space: world, physics_pars: dict[str, Any], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._body: dynamicBody = space.CreateDynamicBody(
            position=self.pos,
            angle=self.orientation,
            # linearVelocity=Box2D.b2Vec2(*[.0, .0]),
            # angularVelocity=.0,
            bullet=True,
            linearDamping=physics_pars["lin_damping"],
            angularDamping=physics_pars["ang_damping"],
        )

        for v in self.vertices:
            self._body.CreatePolygonFixture(
                shape=polygonShape(vertices=v.tolist()),
                density=physics_pars["density"],
                friction=10,
                restitution=0,
            )

        self._fixtures = self._body.fixtures

    def get_position(self) -> np.ndarray:
        """
        Get the world position of the segment.

        Returns
        -------
        numpy.ndarray
            The world position of the segment.

        """
        pos = self._body.worldCenter
        return np.asarray(pos)

    def set_position(self, position: Tuple[float, float]) -> None:
        """
        Set the world position of the segment.

        Parameters
        ----------
        position : tuple
            The new world position.

        """
        self._body.position = position

    def get_orientation(self) -> float:
        """
        Get the orientation of the segment.

        Returns
        -------
        float
            The orientation of the segment.

        """
        return self._body.angle

    def set_orientation(self, orientation: float) -> None:
        """
        Set the orientation of the segment.

        Parameters
        ----------
        orientation : float
            The new orientation of the segment.

        """
        self._body.angle = orientation % (np.pi * 2)

    def get_pose(self) -> Tuple[float, float, float]:
        """
        Get the pose (position and orientation) of the segment.

        Returns
        -------
        tuple
            The pose of the segment, including position and orientation.

        """
        pos = np.asarray(self._body.position)
        return tuple((*pos, self._body.angle))

    def set_linearvelocity(
        self, lin_vel: Tuple[float, float], local: bool = False
    ) -> None:
        """
        Set the linear velocity of the segment.

        Parameters
        ----------
        lin_vel : tuple
            The new linear velocity.
        local : bool, optional
            Whether the linear velocity is in local coordinates. Defaults to False.

        """
        if local:
            lin_vel = self._body.GetWorldVector(np.asarray(lin_vel))
        self._body.linearVelocity = vec2(lin_vel)

    def get_angularvelocity(self) -> float:
        """
        Get the angular velocity of the segment.

        Returns
        -------
        float
            The angular velocity of the segment.

        """
        return self._body.angularVelocity

    def set_angularvelocity(self, ang_vel: float) -> None:
        """
        Set the angular velocity of the segment.

        Parameters
        ----------
        ang_vel : float
            The new angular velocity of the segment.

        """
        self._body.angularVelocity = ang_vel

    def set_mass(self, mass: float) -> None:
        """
        Set the mass of the segment.

        Parameters
        ----------
        mass : float
            The new mass of the segment.

        """
        self._body.mass = mass

    def get_mass(self) -> float:
        """
        Get the mass of the segment.

        Returns
        -------
        float
            The mass of the segment.

        """
        return self._body.mass

    def get_world_point(self, point: Tuple[float, float]) -> np.ndarray:
        """
        Transform a local point to world coordinates.

        Parameters
        ----------
        point : tuple
            The local point coordinates.

        Returns
        -------
        numpy.ndarray
            The world coordinates of the point.

        """
        return self._body.GetWorldPoint(np.asarray(point))

    def get_world_facing_axis(self) -> np.ndarray:
        """
        Get the world-facing axis of the segment.

        Returns
        -------
        numpy.ndarray
            The world-facing axis of the segment.

        """
        return np.asarray(self._body.GetWorldVector(vec2(1.0, 0.0)))


class LarvaBox2D(LarvaSim):
    """
    Box2D-based larva simulation.
    """

    segs = param.List(item_type=SegmentBox2D, doc="The body segments.")

    def __init__(self, Box2D: dict[str, Any], **kwargs: Any) -> None:
        self.Box2D_params = Box2D
        super().__init__(**kwargs)

    def generate_segs(self) -> None:
        """
        Generate the segments of the larva.
        """
        kws = {
            "physics_pars": {
                "density": self.density,
                "lin_damping": self.lin_damping,
                "ang_damping": self.ang_damping,
                "inertia": 0.0,
            },
            "space": self.model.space,
        }

        self.segs = util.ItemList(
            objs=self.Nsegs,
            cls=self.param.segs.item_type,
            pos=self.seg_positions,
            orientation=self.orientation,
            base_vertices=self.base_seg_vertices,
            length=(self.length * self.segment_ratio).tolist(),
            **kws,
        )

        if self.model.larva_collisions:
            for seg in self.segs:
                for fixture in seg._fixtures:
                    fixture.filterData.groupIndex = -1

        self.joints = []

        if self.Nsegs > 1:
            self.create_joints(
                self.Nsegs, self.segs, joint_types=self.Box2D_params["joint_types"]
            )

    def prepare_motion(self, lin: float, ang: float) -> None:
        """
        Prepare the larva for motion with given linear and angular velocities.

        Parameters
        ----------
        lin : float
            Linear velocity.
        ang : float
            Angular velocity.

        """
        if self.ang_mode == "velocity":
            ang_vel = ang * self.ang_vel_coef
            self.segs[0].set_angularvelocity(ang_vel)
            if self.Nsegs > 1:
                for i in np.arange(1, int(self.Nsegs / 2), 1):
                    self.segs[i].set_angularvelocity(ang_vel / i)
        elif self.ang_mode == "torque":
            self.segs[0]._body.ApplyTorque(ang * self.torque_coef, wake=True)

        # Linear component
        # Option : Apply to single body segment
        # We get the orientation of the front segment and compute the linear vector
        # target_segment = self.get_head()
        # lin_vec = self.compute_new_lin_vel_vector(target_segment)
        #
        # # From web : Impulse = Force x 1 Sec in Box2D
        # if self.mode == 'impulse':
        #     imp = lin_vec / target_segment.get_Box2D_mass()
        #     target_segment._body.ApplyLinearImpulse(imp, target_segment._body.worldCenter, wake=True)
        # elif self.mode == 'force':
        #     target_segment._body.ApplyForceToCenter(lin_vec, wake=True)
        # elif self.mode == 'velocity':
        #     # lin_vec = lin_vec * target_segment.get_Box2D_mass()
        #     # Use this with gaussian crawler
        #     # target_segment.set_lin_vel(lin_vec * self.lin_coef, local=False)
        #     # Use this with square crawler
        #     target_segment.set_lin_vel(lin_vec, local=False)
        #     # pass

        # Option : Apply to all body segments. This allows to control velocity for any Npoints. But it has the same shitty visualization as all options
        # for seg in [self.segs[0]]:
        for seg in self.segs:
            l = lin * seg.get_world_facing_axis()
            if self.lin_mode == "impulse":
                seg._body.ApplyLinearImpulse(
                    l * self.lin_vel_coef / seg.get_mass(),
                    seg._body.worldCenter,
                    wake=True,
                )
            elif self.lin_mode == "force":
                seg._body.ApplyForceToCenter(l * self.lin_force_coef, wake=True)
            elif self.lin_mode == "velocity":
                seg.set_linearvelocity(l * self.lin_vel_coef, local=False)

    def updated_by_Box2D(self) -> None:
        """
        Update the larva simulation based on Box2D physics.
        """
        self.set_position(tuple(self.global_midspine_of_body))
        self.trajectory.append(self.get_position())

        self.dst = geometry.Point(self.pos).distance(
            geometry.Point(self.trajectory[-1])
        )
        # self.dst = aux.eudis5(self.pos, self.trajectory[-1])
        self.cum_dst += self.dst
        self.compute_body_bend()

    # To make peristalsis visible we try to leave some space between the segments.
    # We define an interval proportional to the length : int*l.
    # We subtract it from the front end of all segments except the first and from the rear end of all segments except the last.
    # For Npoints=n, in total we subtract 2*(n-1)*int*l in length.
    # For width_to_length_ratio=w2l, the area of the body without intervals is A=w2l*l*l
    # Subtracting the intervals, this translates to A'= (l-2*(n-1)*int*l) * w2l*l = (1-2*(n-1)*int)*A
    # To get the same mass, we will raise the density=d to d' accordingly : mass=d*A = d'*A' ==> d'=d/(1-2*(n-1)*int)
    # def add_interval_between_segments(self, Nsegs, density, interval, seg_starts, seg_stops):
    #     for i in range(1, len(seg_starts)):
    #         seg_starts[i] -= interval
    #     for i in range(len(seg_stops) - 1):
    #         seg_stops[i] += interval
    #
    #     self.density = density / (1 - 2 * (Nsegs - 1) * interval)
    #
    #     return seg_starts, seg_stops

    def create_joints(
        self,
        Nsegs: int,
        segs: List[SegmentBox2D],
        joint_types: dict[str, Any] | None = None,
    ) -> None:
        """
        Create joints to connect the segments of the larva.

        Parameters
        ----------
        Nsegs : int
            The number of segments in the larva.
        segs : list of Box2DSegment
            The list of Box2DSegment objects representing the larva segments.
        joint_types : dict, optional
            A dictionary specifying the types of joints to create. The dictionary should contain keys for different
            joint types ('distance', 'revolute', 'friction') and values specifying the number of joints of each type
            to create ('N') and the joint parameters ('args') for each type.

        Notes
        -----
        The `joint_types` parameter is optional and, if not provided, it will use the joint types and parameters
        defined in the `Box2D` attribute of the larva simulation.

        This method creates various types of joints (distance, revolute, and friction) to connect the segments of the
        larva together in a physically realistic way.

        """
        if joint_types is None:
            joint_types = util.AttrDict(
                {
                    "friction": {
                        "N": 0,
                        "args": {
                            "maxForce": {"v": 1, "lim": (0.0, 100000)},
                            "maxTorque": {"v": 1, "lim": (0.0, 100000)},
                        },
                    },
                    "revolute": {
                        "N": 0,
                        "args": {
                            "enableMotor": True,
                            "maxMotorTorque": {"v": 0.0, "lim": (0.0, 100000)},
                            "motorSpeed": {"v": 0.0, "lim": (0.0, 100000)},
                        },
                    },
                    "distance": {
                        "N": 0,
                        "args": {
                            "frequencyHz": {"v": 5.0, "lim": (0.0, 100000)},
                            "dampingRatio": {"v": 1.0, "lim": (0.0, 100000)},
                        },
                    },
                }
            )
        space = self.model.space
        l0 = self.length / self.Nsegs

        # TODO Find compatible parameters.
        # Until now for the 12-seg body : density 30000 and maxForce 100000000  and torque_coef 3.5 seem to work for natural bend
        # Trying to implement friction joint
        # if joint_types is None :
        #     joint_types = {'distance': 0, 'revolute': 0, 'friction' : 0}
        for i in range(Nsegs):
            if i == 0:
                continue
            # if joint_types['friction']
            # friction_pars = {'maxForce': 10 ** 0, 'maxTorque': 10 ** 1}
            if joint_types["friction"]["N"] == 2:
                xAs = [-0.5, 0.5]
            elif joint_types["friction"]["N"] == 1:
                xAs = [0]
            else:
                xAs = []
            for xA in xAs:
                friction_joint = space.CreateFrictionJoint(
                    **joint_types["friction"]["args"],
                    bodyA=segs[i]._body,
                    bodyB=self.model.friction_body,
                    localAnchorA=(xA, 0),
                    localAnchorB=(0, 0),
                )

                self.joints.append(friction_joint)

        # For many segments, the front one(sigma) will be joint by points outside the body.
        # So we adopt a more conservative solution, bringing the attachment point more medially : No visible difference
        # lateral_attachment_dist = self.width_to_length_ratio * self.Npoints / 4

        dist_joint_def = {
            "collideConnected": False,
            # 'frequencyHz': 5,
            # 'dampingRatio': 1,
            "length": l0 * 0.01,
        }
        joint_types["distance"]["args"].update(dist_joint_def)
        w = self.width_to_length_ratio * Nsegs / 2

        for i in range(Nsegs - 1):
            weld_def = {"dampingRatio": 0.1, "referenceAngle": 0, "frequencyHz": 2000}
            A, B = segs[i]._body, segs[i + 1]._body
            # if joint_types['distance']['N'] == 2:

            space.CreateWeldJoint(
                **weld_def,
                bodyA=A,
                bodyB=B,
                localAnchorA=tuple(l0 * x for x in (-0.5, w)),
                localAnchorB=tuple(l0 * x for x in (0.5, w)),
            )
            # space.CreateWeldJoint(**weld_def,
            #                       bodyA=A, bodyB=B,
            #                       localAnchorA=tuple(l0 * x for x in (-0.5, -w)),
            #                       localAnchorB=tuple(l0 * x for x in (0.5, -w)))

        for i in range(Nsegs - 1):
            A, B = segs[i]._body, segs[i + 1]._body
            if joint_types["distance"]["N"] == 2:
                j_l = space.CreateDistanceJoint(
                    **joint_types["distance"]["args"],
                    bodyA=A,
                    bodyB=B,
                    localAnchorA=tuple(l0 * x for x in (-0.5, w)),
                    localAnchorB=tuple(l0 * x for x in (0.5, w)),
                )
                j_r = space.CreateDistanceJoint(
                    **dist_joint_def,
                    bodyA=A,
                    bodyB=B,
                    localAnchorA=tuple(l0 * x for x in (-0.5, -w)),
                    localAnchorB=tuple(l0 * x for x in (0.5, -w)),
                )
                self.joints.append([j_l, j_r])
            elif joint_types["distance"]["N"] == 1:
                j = space.CreateDistanceJoint(
                    **dist_joint_def,
                    bodyA=A,
                    bodyB=B,
                    localAnchorA=tuple(l0 * x for x in (-0.5, 0)),
                    localAnchorB=tuple(l0 * x for x in (0.5, 0)),
                )
                self.joints.append(j)

        if joint_types["revolute"]:
            rev_joint_def = {
                "collideConnected": False,
                "referenceAngle": 0,
                "enableLimit": True,
                "lowerAngle": -0.9 * (np.pi * 2) / (Nsegs - 1),
                "upperAngle": 0.9 * (np.pi * 2) / (Nsegs - 1),
                # 'enableMotor': True,  # )
                # 'maxMotorTorque': 1.0,
                # 'motorSpeed': 1
            }
            joint_types["revolute"]["args"].update(rev_joint_def)
            for i in range(Nsegs - 1):
                A, B = segs[i]._body, segs[i + 1]._body
                if joint_types["revolute"]["N"] == 2:
                    j_l = space.CreateRevoluteJoint(
                        **joint_types["revolute"]["args"],
                        bodyA=A,
                        bodyB=B,
                        localAnchorA=tuple(l0 * x for x in (-0.5, w)),
                        localAnchorB=tuple(l0 * x for x in (0.5, w)),
                    )
                    j_r = space.CreateRevoluteJoint(
                        **rev_joint_def,
                        bodyA=A,
                        bodyB=B,
                        localAnchorA=tuple(l0 * x for x in (-0.5, -w)),
                        localAnchorB=tuple(l0 * x for x in (0.5, -w)),
                    )
                    self.joints.append([j_l, j_r])
                elif joint_types["revolute"]["N"] == 1:
                    j = space.CreateRevoluteJoint(
                        **rev_joint_def,
                        bodyA=A,
                        bodyB=B,
                        localAnchorA=tuple(l0 * x for x in (-0.5, 0)),
                        localAnchorB=tuple(l0 * x for x in (0.5, 0)),
                    )
                    self.joints.append(j)


class ArenaBox2D(Arena, world):
    def __init__(self, **kwargs: Any) -> None:
        Arena.__init__(self, **kwargs)
        # --- pybox2d world setup ---
        # Create the world
        world.__init__(self, gravity=(0, -10), doSleep=True)

        # And a static body to hold the ground shape
        self.ground_body: staticBody = self.CreateStaticBody(
            position=(0, 0),
            shapes=polygonShape(self.edges),
        )
