import Box2D
import numpy as np
import param
from shapely import geometry

from larvaworld.lib import aux, reg
from larvaworld.lib.model import LarvaSim

class BaseSegment:
    def __init__(self, pos, orientation,color, base_seg_vertices,base_seg_ratio, body_length):
        self.color = color
        self.pos = pos
        self.orientation = orientation % (np.pi * 2)
        self.base_seg_vertices = base_seg_vertices
        self.base_local_rear_end = np.array([np.min(self.base_seg_vertices[:, 0]), 0])
        self.base_local_front_end = np.array([np.max(self.base_seg_vertices[:, 0]), 0])
        self.base_seg_ratio = base_seg_ratio
        self.body_length=body_length

    @property
    def seg_vertices(self):
        return self.body_length*self.base_seg_vertices

class Box2DSegment(BaseSegment):

    def __init__(self, space, physics_pars, **kwargs):

        super().__init__(**kwargs)
        self._body: Box2D.b2Body = space.CreateDynamicBody(
            position=Box2D.b2Vec2(*self.pos),
            angle=self.orientation,
            linearVelocity=Box2D.b2Vec2(*[.0, .0]),
            angularVelocity=.0,
            bullet=True,
            linearDamping=physics_pars['lin_damping'],
            angularDamping=physics_pars['ang_damping'])

        for v in self.seg_vertices:
            self._body.CreatePolygonFixture(
                shape=Box2D.b2PolygonShape(vertices=v.tolist()),
                density=physics_pars['density'],
                friction=10,
                restitution=0,
                # radius=.00000001
            )

        self._fixtures = self._body.fixtures

        # FIXME for some reason this produces error
        # self._body.inertia = self.physics_pars['inertia']

    @property
    def vertices(self):
        return np.array([[self.get_world_point(v) for v in vertices] for vertices in self.seg_vertices])

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








    def get_world_point(self, point):
        return self._body.GetWorldPoint(np.asarray(point))



    def get_world_facing_axis(self):
        return np.asarray(self._body.GetWorldVector(Box2D.b2Vec2(1.0, 0.0)))





class LarvaBox2D(LarvaSim):
    segs = param.List(item_type=Box2DSegment, doc='The body segments.')


    def __init__(self, Box2D_params,**kwargs):
        self.Box2D_params=Box2D_params
        super().__init__(**kwargs)

        # BaseController.__init__(self, **physics)

        # joint_types=Box2D_params['joint_types']

    # def __init__(self, joint_types,physics,**kwargs):
    #     super().__init__(**physics)




    def generate_segs(self):
        # segs = []
        kws= {
            'physics_pars' : {'density': self.density,
                              'lin_damping': self.lin_damping,
                              'ang_damping': self.ang_damping,
                              'inertia': 0.0},
            'space' : self.model.space,
        }

        # segs=aux.generate_segs(self.Nsegs, self.pos, self.orientation,
        #                         self.sim_length, self.seg_ratio,self.default_color,self.body_plan,
        #                         segment_class=Box2DSegment, **kws)

        self.segs = [self.param.segs.item_type(pos=self.seg_positions[i], orientation=self.orientation,
                                                   base_vertices=self.base_seg_vertices[i],
                                                   length=self.length * self.segment_ratio[i], **kws) for i in
                         range(self.Nsegs)]


        # put all agents into same group (negative so that no collisions are detected)
        if self.model.larva_collisions:
            for seg in self.segs :
                for fixture in seg._fixtures:
                    fixture.filterData.groupIndex = -1

        self.joints = []
        # self.generate_segs()
        if self.Nsegs > 1:
            self.create_joints(self.Nsegs, self.segs, joint_types=self.Box2D_params['joint_types'])


    def prepare_motion(self, lin, ang):

        if self.ang_mode == 'velocity':
            ang_vel = ang * self.ang_vel_coef
            self.segs[0].set_angularvelocity(ang_vel)
            if self.Nsegs > 1:
                for i in np.arange(1, int(self.Nsegs / 2), 1):
                    self.segs[i].set_angularvelocity(ang_vel / i)
        elif self.ang_mode == 'torque':
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
            l=lin *  seg.get_world_facing_axis()
            if self.lin_mode == 'impulse':
                seg._body.ApplyLinearImpulse(l * self.lin_vel_coef / seg.get_mass(), seg._body.worldCenter, wake=True)
            elif self.lin_mode == 'force':
                seg._body.ApplyForceToCenter(l * self.lin_force_coef , wake=True)
            elif self.lin_mode == 'velocity':
                seg.set_linearvelocity(l * self.lin_vel_coef, local=False)


    def updated_by_Box2D(self):
        self.set_position(tuple(self.global_midspine_of_body))
        self.trajectory.append(self.get_position())

        self.dst = geometry.Point(self.pos).distance(geometry.Point(self.trajectory[-1]))
        # self.dst = aux.eudis5(self.pos, self.trajectory[-1])
        self.cum_dst += self.dst
        self.compute_body_bend()

    # def generate_segs(self):
    #     # segs = []
    #     kws= {
    #         'physics_pars' : {'density': self.density,
    #                           'lin_damping': self.lin_damping,
    #                           'ang_damping': self.ang_damping,
    #                           'inertia': 0.0},
    #         'space' : self.model.space,
    #     }
    #
    #     segs=aux.generate_segs(self.Nsegs, self.pos, self.orientation,
    #                             self.sim_length, self.seg_ratio,self.default_color,self.body_plan,
    #                             segment_class=Box2DSegment, **kws)
    #
    #
    #     # put all agents into same group (negative so that no collisions are detected)
    #     if self.model.larva_collisions:
    #         for seg in segs :
    #             for fixture in seg._fixtures:
    #                 fixture.filterData.groupIndex = -1
    #     return segs

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


