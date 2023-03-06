import Box2D
import numpy as np

from larvaworld.lib import aux, reg
from larvaworld.lib.model.agents.segmented_body import LarvaBody, BaseController

class Box2DSegment:

    def __init__(self, pos,orientation,seg_vertices,color, space, physics_pars, facing_axis):

        self.color = color
        self.space = space
        self.physics_pars = physics_pars
        self._body: Box2D.b2Body = self.space.CreateDynamicBody(
            position=Box2D.b2Vec2(*pos),
            angle=orientation,

            # gravityScale=100,
            # fixedRotation=True if  self.idx!=0 else False,
            linearDamping=physics_pars['lin_damping'],
            angularDamping=physics_pars['ang_damping'])
        self._body.linearVelocity = Box2D.b2Vec2(*[.0, .0])
        self._body.angularVelocity = .0
        self._body.bullet = True

        # overriden by LarvaBody
        self.facing_axis = facing_axis

        # CAUTION
        # This sets the body'sigma origin (where pos, orientation is derived from)
        # self._body.localCenter = Box2D.b2Vec2(11.0, 10.0)
        # this sets the body' center of mass (where velocity is set etc)
        # self._body.massData.center = self._body.localCenter
        # self._body.massData.center= Box2D.b2Vec2(11.0, 10.0)
        # self._body.localCenter = self._body.massData.center

        # TODO: right now this assumes that all subpolygons have the same number of edges
        # TODO: rewrite such that arbitrary subpolygons can be used here
        # vertices = self.seg_vertices

        centroid = np.zeros(2)
        area = .0
        for vs in seg_vertices:
            # compute centroid of circle_to_polygon
            r0 = np.roll(vs[:, 0], 1)
            r1 = np.roll(vs[:, 1], 1)
            a = 0.5 * np.abs(np.dot(vs[:, 0], r1) - np.dot(vs[:, 1], r0))
            area += a
            # FIXME This changed in refactoring. It is wrong probably.
            # Find a way to use compute_centroid(points) function
            centroid += np.mean(vs, axis=0) * a

        centroid /= area
        self.__local_vertices = seg_vertices
        self.__local_vertices.setflags(write=False)
        for v in self.__local_vertices:
            self._body.CreatePolygonFixture(
                shape=Box2D.b2PolygonShape(vertices=v.tolist()),
                density=self.physics_pars['density'],
                friction=self.physics_pars['friction'],
                restitution=self.physics_pars['restitution'],
                # radius=.00000001
            )

        self._fixtures = self._body.fixtures

        # FIXME for some reason this produces error
        # self._body.inertia = self.physics_pars['inertia']

    @property
    def vertices(self):
        return np.array([[self.get_world_point(v) for v in vertices] for vertices in self.__local_vertices])

    def get_position(self):
        # CAUTION CAUTION This took me a whole day.
        # worldCenter gets the point where the torque is applied
        # pos gets a point (tried to identify whether it is center of mass or origin, no luck) unknown how
        pos = self._body.worldCenter
        return np.asarray(pos)

    def set_position(self, position):
        self._body.position = position

    def get_orientation(self):
        return self._body.angle



    def set_orientation(self, orientation):
        # orientation %= 2 * np.pi
        self._body.angle = orientation % (np.pi * 2)



    def get_pose(self):
        pos = np.asarray(self._body.position)
        return tuple((*pos, self._body.angle))

    def set_linearvelocity(self, lin_vel, local=False):
        if local:
            lin_vel = self._body.GetWorldVector(np.asarray(lin_vel))
        self._body.linearVelocity = Box2D.b2Vec2(lin_vel)

    def get_angularvelocity(self):
        return self._body.angularVelocity

    def set_angularvelocity(self, ang_vel):
        self._body.angularVelocity = ang_vel

    def set_mass(self, mass):
        self._body.mass = mass

    def get_mass(self):
        return self._body.mass

    def add_mass(self, added_mass):
        self._body.mass += added_mass

    def set_massdata(self, massdata):
        self._body.massData = massdata

    def get_local_point(self, point):
        return np.asarray(self._body.GetLocalPoint(np.asarray(point)))

    def get_local_vector(self, vector):
        return np.asarray(self._body.GetLocalVector(vector))

    def get_local_orientation(self, angle):
        return angle - self._body.angle

    def get_local_pose(self, pose):
        return tuple((*self.get_local_point(pose[:2]), self.get_local_orientation(pose[2])))

    def get_world_point(self, point):
        return self._body.GetWorldPoint(np.asarray(point))

    def get_world_vector(self, vector):
        return np.asarray(self._body.GetWorldVector(vector))

    def get_world_facing_axis(self):
        return np.asarray(self._body.GetWorldVector(self.facing_axis))

    def collides_with(self, other):
        for contact_edge in self._body.contacts_gen:
            if contact_edge.other == other and contact_edge.contact.touching:
                return True


class LarvaBox2D(LarvaBody,BaseController):
    def __init__(self, physics,Box2D_params,**kwargs):
        LarvaBody.__init__(self, **kwargs)

        BaseController.__init__(self, **physics)

        joint_types=Box2D_params['joint_types']

    # def __init__(self, joint_types,physics,**kwargs):
    #     super().__init__(**physics)


        self.joints = []
        self.segs = self.generate_segs()
        if self.Nsegs > 1:
            self.create_joints(self.Nsegs, self.segs, joint_types)


    def prepare_motion(self, lin, ang):

        if self.ang_mode == 'velocity':
            ang_vel = ang * self.ang_vel_coef
            self.segs[0]._body.angularVelocity = ang_vel
            if self.Nsegs > 1:
                for i in np.arange(1, self.mid_seg_index, 1):
                    self.segs[i]._body.angularVelocity = (ang_vel / i)
        elif self.ang_mode == 'torque':
            torque = ang * self.torque_coef
            self.segs[0]._body.ApplyTorque(torque, wake=True)

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
            if self.lin_mode == 'impulse':
                impulse = lin * self.lin_vel_coef * seg.get_world_facing_axis() / seg.get_mass()
                seg._body.ApplyLinearImpulse(impulse, seg._body.worldCenter, wake=True)
            elif self.lin_mode == 'force':
                force = lin * self.lin_force_coef * seg.get_world_facing_axis()
                seg._body.ApplyForceToCenter(force, wake=True)
            elif self.lin_mode == 'velocity':
                vel = lin * self.lin_vel_coef * seg.get_world_facing_axis()
                seg.set_linearvelocity(vel, local=False)


    def updated_by_Box2D(self):
        self.pos = self.global_midspine_of_body
        self.trajectory.append(self.pos)

        self.dst = aux.eudis5(self.pos, self.trajectory[-1])
        self.cum_dst += self.dst
        self.compute_body_bend()

    def generate_segs(self):
        segs = []
        physics_pars = {'density': self.density,
                        'friction': 10.0,
                        'restitution': 0.0,
                        'lin_damping': self.lin_damping,
                        'ang_damping': self.ang_damping,
                        'inertia': 0.0}

        fixtures = []
        for i in range(self.Nsegs):
            seg = Box2DSegment(space=self.model.space, pos=self.seg_positions[i], orientation=self.orientation,
                               physics_pars=physics_pars, facing_axis=Box2D.b2Vec2(1.0, 0.0),
                               seg_vertices=self.seg_vertices[i], color=self.seg_colors[i])
            fixtures.extend(seg._fixtures)
            segs.append(seg)

        # put all agents into same group (negative so that no collisions are detected)
        if self.model.larva_collisions:
            for fixture in fixtures:
                fixture.filterData.groupIndex = -1


            # self.create_rotator(segs, position, orientation, physics_pars)
        return segs

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

    def create_joints(self, Nsegs, segs, joint_types=None):
        if joint_types is None:
            joint_types = reg.get_null('Box2D_params').joint_types
        space = self.model.space
        l0 = self.sim_length/self.Nsegs

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
            if joint_types['friction']['N'] == 2:
                xAs = [-0.5, 0.5]
            elif joint_types['friction']['N'] == 1:
                xAs = [0]
            else:
                xAs = []
            for xA in xAs:
                friction_joint = space.CreateFrictionJoint(**joint_types['friction']['args'],
                                                           bodyA=segs[i]._body,
                                                           bodyB=self.model.friction_body,
                                                           localAnchorA=(xA, 0),
                                                           localAnchorB=(0, 0))

                self.joints.append(friction_joint)

        # For many segments, the front one(sigma) will be joint by points outside the body.
        # So we adopt a more conservative solution, bringing the attachment point more medially : No visible difference
        # lateral_attachment_dist = self.width_to_length_ratio * self.Npoints / 4

        dist_joint_def = {'collideConnected': False,
                          # 'frequencyHz': 5,
                          # 'dampingRatio': 1,
                          'length': l0 * 0.01}
        joint_types['distance']['args'].update(dist_joint_def)
        w = self.width_to_length_ratio * Nsegs / 2

        for i in range(Nsegs - 1):
            weld_def = {
                'dampingRatio': 0.1,
                'referenceAngle': 0,
                'frequencyHz': 2000
            }
            A, B = segs[i]._body, segs[i + 1]._body
            # if joint_types['distance']['N'] == 2:

            space.CreateWeldJoint(**weld_def,
                                  bodyA=A,
                                  bodyB=B,
                                  localAnchorA=tuple(l0 * x for x in (-0.5, w)),
                                  localAnchorB=tuple(l0 * x for x in (0.5, w))

                                  )
            # space.CreateWeldJoint(**weld_def,
            #                       bodyA=A, bodyB=B,
            #                       localAnchorA=tuple(l0 * x for x in (-0.5, -w)),
            #                       localAnchorB=tuple(l0 * x for x in (0.5, -w)))

        for i in range(Nsegs - 1):
            A, B = segs[i]._body, segs[i + 1]._body
            if joint_types['distance']['N'] == 2:
                j_l = space.CreateDistanceJoint(**joint_types['distance']['args'],
                                                bodyA=A,
                                                bodyB=B,
                                                localAnchorA=tuple(l0 * x for x in (-0.5, w)),
                                                localAnchorB=tuple(l0 * x for x in (0.5, w)))
                j_r = space.CreateDistanceJoint(**dist_joint_def,
                                                bodyA=A,
                                                bodyB=B,
                                                localAnchorA=tuple(l0 * x for x in (-0.5, -w)),
                                                localAnchorB=tuple(l0 * x for x in (0.5, -w)))
                self.joints.append([j_l, j_r])
            elif joint_types['distance']['N'] == 1:
                j = space.CreateDistanceJoint(**dist_joint_def,
                                              bodyA=A,
                                              bodyB=B,
                                              localAnchorA=tuple(l0 * x for x in (-0.5, 0)),
                                              localAnchorB=tuple(l0 * x for x in (0.5, 0)))
                self.joints.append(j)

        if joint_types['revolute']:

            rev_joint_def = {'collideConnected': False,
                             'referenceAngle': 0,
                             'enableLimit': True,
                             'lowerAngle': -0.9 * (np.pi * 2) / (Nsegs - 1),
                             'upperAngle': 0.9 * (np.pi * 2) / (Nsegs - 1),
                             # 'enableMotor': True,  # )
                             # 'maxMotorTorque': 1.0,
                             # 'motorSpeed': 1
                             }
            joint_types['revolute']['args'].update(rev_joint_def)
            for i in range(Nsegs - 1):
                A, B = segs[i]._body, segs[i + 1]._body
                if joint_types['revolute']['N'] == 2:
                    j_l = space.CreateRevoluteJoint(**joint_types['revolute']['args'],
                                                    bodyA=A,
                                                    bodyB=B,
                                                    localAnchorA=tuple(l0 * x for x in (-0.5, w)),
                                                    localAnchorB=tuple(l0 * x for x in (0.5, w)))
                    j_r = space.CreateRevoluteJoint(**rev_joint_def,
                                                    bodyA=A,
                                                    bodyB=B,
                                                    localAnchorA=tuple(l0 * x for x in (-0.5, -w)),
                                                    localAnchorB=tuple(l0 * x for x in (0.5, -w)))
                    self.joints.append([j_l, j_r])
                elif joint_types['revolute']['N'] == 1:
                    j = space.CreateRevoluteJoint(**rev_joint_def,
                                                  bodyA=A,
                                                  bodyB=B,
                                                  localAnchorA=tuple(l0 * x for x in (-0.5, 0)),
                                                  localAnchorB=tuple(l0 * x for x in (0.5, 0)))
                    self.joints.append(j)


# class LarvaBox2D(LarvaBody,Box2DController):
#     def __init__(self, physics,Box2D_params,**kwargs):
#         LarvaBody.__init__(self, **kwargs)
#
#         Box2DController.__init__(self, physics, **Box2D_params)