import abc
from random import sample, seed
import numpy as np
import Box2D
from Box2D import b2Vec2
from shapely import affinity
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import cascaded_union
# TODO Find a way to use this. Now if changed everything is scal except locomotion. It seems that
#  ApplyForceToCenter function does not scale
# _world_scale = np.int(100)
# from matplotlib.patches import Circle
# from shapely.geometry import Polygon, Point

import lib.aux.functions as fun
# from lib.aux.rendering import InputBox
# from lib.stor.paths import LarvaShape_path


class Box2DSegment:

    def __init__(self, space: Box2D.b2World, pos, orientation, physics_pars, facing_axis, color, **kwargs):
        if self.__class__ == Box2DSegment:
            raise NotImplementedError('Abstract class Box2DSegment cannot be instantiated.')
        self.physics_pars = physics_pars
        self._color = color
        self._body: Box2D.b2Body = space.CreateDynamicBody(
            position=Box2D.b2Vec2(*pos),
            angle=orientation,
            linearDamping=physics_pars['lin_damping'],
            angularDamping=physics_pars['ang_damping'])
        self._body.linearVelocity = Box2D.b2Vec2(*[.0, .0])
        self._body.angularVelocity = .0
        self._body.bullet = True

        # overriden by LarvaBody
        self.facing_axis = facing_axis

        # CAUTION
        # This sets the body's origin (where pos, orientation is derived from)
        # self._body.localCenter = b2Vec2(0.0, 0.0)
        # this sets the body' center of mass (where velocity is set etc)
        # self._body.massData.center= self._body.localCenter
        # self._body.massData.center= b2Vec2(0.0, 0.0)
        # self._body.localCenter = self._body.massData.center

    # @property
    # def width(self):
    #     raise NotImplementedError

    # @property
    # def height(self):
    #     raise NotImplementedError

    # @property
    def get_position(self):
        # CAUTION CAUTION This took me a whole day.
        # worldCenter gets the point where the torque is applied
        # pos gets a point (tried to identify whether it is center of mass or origin, no luck) unknown how
        pos = self._body.worldCenter
        # print(pos)
        return np.asarray(pos)

    def set_position(self, position):
        self._body.position = position

    def get_orientation(self):
        angle = self._body.angle
        return angle

    def get_normalized_orientation(self):
        angle = self.get_orientation()
        # I normalize the angle_to_x_axis in [-pi,pi]
        angle %= 2 * np.pi
        # if angle > np.pi:
        #     angle -= 2 * np.pi
        return angle

    def get_linearvelocity_vec(self):
        return self._body.linearVelocity

    def get_linearvelocity_amp(self):
        return np.linalg.norm(self._body.linearVelocity)

    def get_angularvelocity(self):
        return self._body.angularVelocity

    def set_orientation(self, orientation):
        # orientation %= 2 * np.pi
        self._body.angle = orientation

    def get_pose(self):
        pos = np.asarray(self._body.position)
        return tuple((*pos, self._body.angle))

    def set_pose(self, pose):
        self.set_position(pose[:2])
        self.set_orientation(pose[2])

    def set_lin_vel(self, lin_vel, local=False):
        if local:
            lin_vel = self._body.GetWorldVector(np.asarray(lin_vel))
        self._body.linearVelocity = Box2D.b2Vec2(lin_vel)

    def set_ang_vel(self, ang_vel):
        self._body.angularVelocity = ang_vel

    # Panos

    def set_mass(self, mass):
        self._body.mass = mass

    def get_mass(self):
        return self._body.mass

    def add_mass(self, added_mass):
        self._body.mass += added_mass

    def set_massdata(self, massdata):
        self._body.massData = massdata

    def get_state(self):
        return self.get_pose()
        # return tuple((*self._body.pos, self._body.angle_to_x_axis))

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
        # print(self._body.GetWorldVector(self.facing_axis))
        return np.asarray(self._body.GetWorldVector(self.facing_axis))

    def collides_with(self, other):
        for contact_edge in self._body.contacts_gen:
            if contact_edge.other == other and contact_edge.contact.touching:
                return True

    # @property
    def get_color(self):
        return self._color

    # @color.setter
    def set_color(self, color):
        color = np.asarray(color, dtype=np.int32)
        color = np.maximum(color, np.zeros_like(color, dtype=np.int32))
        color = np.minimum(color, np.full_like(color, 255, dtype=np.int32))
        self._color = color

    @property
    def highlight_color(self):
        return self._highlight_color

    @highlight_color.setter
    def highlight_color(self, color):
        color = np.asarray(color, dtype=np.int32)
        color = np.maximum(color, np.zeros_like(color, dtype=np.int32))
        color = np.minimum(color, np.full_like(color, 255, dtype=np.int32))
        self._highlight_color = color

    @abc.abstractmethod
    def draw(self, viewer):
        raise NotImplementedError('The draw method needs to be implemented by the subclass of Box2DSegment.')

    @abc.abstractmethod
    def plot(self, axes, **kwargs):
        raise NotImplementedError('The plot method needs to be implemented by the subclass of Box2DSegment.')


class Box2DPolygon(Box2DSegment):
    def __init__(self, seg_vertices=None, **kwargs):
        super().__init__(**kwargs)

        # TODO: right now this assumes that all subpolygons have the same number of edges
        # TODO: rewrite such that arbitrary subpolygons can be used here
        vertices = seg_vertices

        centroid = np.zeros(2)
        area = .0
        for vs in vertices:
            # compute centroid of circle_to_polygon
            r0 = np.roll(vs[:, 0], 1)
            r1 = np.roll(vs[:, 1], 1)
            a = 0.5 * np.abs(np.dot(vs[:, 0], r1) - np.dot(vs[:, 1], r0))
            area += a
            # FIXME This changed in refactoring. It is wrong probably.
            # Find a way to use compute_centroid(points) function
            centroid += np.mean(vs, axis=0) * a

        centroid /= area

        self.__local_vertices = vertices - centroid
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

    # @property
    # def width(self):
    #     return self._width
    #
    # @property
    # def height(self):
    #     return self._height

    @property
    def vertices(self):
        return np.array([[self.get_world_point(v) for v in vertices] for vertices in self.__local_vertices])

    @property
    def local_vertices(self):
        return self.__local_vertices

    @property
    def plot_vertices(self):
        raise NotImplementedError

    # @staticmethod
    # def _shape_vertices() -> np.ndarray:
    #     raise NotImplementedError

    def draw(self, viewer):
        for i, vertices in enumerate(self.vertices):
            viewer.draw_polygon(vertices, filled=True, color=self._color)

    # def plot(self, axes, **kwargs):
    #     from simulation.tools.plotting import plot_polygon
    #     return plot_polygon(axes, self, **kwargs)


class DefaultSegment:
    def __init__(self, pos, orientation, seg_vertices, color):
        self.pos = pos
        self.orientation = orientation
        self.seg_vertices = seg_vertices
        self.vertices = None
        self.update_vertices(self.pos, self.orientation)

        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self._color = color
        # print(self._color)
        # centroid = np.zeros(2)
        # area = .0
        # for vs in self.vertices:
        #     print(vs)
        #     # compute centroid of circle_to_polygon
        #     r0 = np.roll(vs[:, 0], 1)
        #     r1 = np.roll(vs[:, 1], 1)
        #     a = 0.5 * np.abs(np.dot(vs[:, 0], r1) - np.dot(vs[:, 1], r0))
        #     area += a
        #     centroid += vs.mean(axis=0) * a
        # centroid /= area

    def update_vertices(self, pos, orient):
        self.vertices = [pos + fun.rotate_around_center_multi(self.seg_vertices[0], -orient)]

    def draw(self, viewer):
        for vertices in self.vertices:
            viewer.draw_polygon(vertices, filled=True, color=self._color)

    def get_position(self):
        return np.array(self.pos)

    def set_position(self, pos):
        self.pos = pos

    def set_pose(self, pos, orientation, lin_vel, ang_vel):
        self.pos = pos
        self.orientation = orientation
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel

    def get_pose(self):
        return np.array(self.pos), self.orientation

    def get_world_point(self, local_point):
        return self.get_position() + fun.rotate_around_center(point=local_point, radians=-self.get_orientation())

    def get_orientation(self):
        return self.orientation

    def get_normalized_orientation(self):
        angle = self.get_orientation()
        # I normalize the angle_to_x_axis in [-pi,pi]
        angle %= 2 * np.pi
        # if angle > np.pi:
        #     angle -= 2 * np.pi
        return angle

    def set_orientation(self, orientation):
        self.orientation = orientation

    def get_linearvelocity_amp(self):
        return self.lin_vel

    def get_angularvelocity(self):
        return self.ang_vel

    def set_linear_velocity(self, lin_vel):
        self.lin_vel = lin_vel

    def set_ang_vel(self, ang_vel):
        self.ang_vel = ang_vel

    def set_color(self, color):
        color = np.asarray(color, dtype=np.int32)
        color = np.maximum(color, np.zeros_like(color, dtype=np.int32))
        color = np.minimum(color, np.full_like(color, 255, dtype=np.int32))
        self._color = color

    def get_color(self):
        return self._color

    def get_polygon(self, scale=1):
        p0=Polygon(self.vertices[0])
        p=affinity.scale(p0, xfact=scale, yfact=scale)
        return p


def generate_seg_colors(N, color):
    if N > 5:
        return [np.array((0, 255, 0))] + [np.copy(color) for i in range(N - 2)] + [np.array((255, 0, 0))]
    else:
        return [np.copy(color) for i in range(N)]


class LarvaBody:
    def __init__(self, model, pos=None, orientation=None, density=300.0,
                 initial_length=None, length_std=0, Nsegs=1, interval=0, joint_type={'distance': 2, 'revolute': 1},
                 seg_ratio=None, friction_pars={'maxForce': 10 ** 0, 'maxTorque': 10 ** -1}, **kwargs):

        self.model=model
        self.density = density
        self.friction_pars = friction_pars
        self.width_to_length_ratio = 0.2  # from [1] K. R. Kaun et al., “Natural variation in food acquisition mediated via a Drosophila cGMP-dependent protein kinase,” J. Exp. Biol., vol. 210, no. 20, pp. 3547–3558, 2007.
        if seg_ratio is None:
            seg_ratio = [1 / Nsegs] * Nsegs
        self.seg_ratio = seg_ratio
        self.interval = interval
        self.shape_scale = 1

        self.base_seg_vertices = self.generate_seg_shapes(Nsegs, self.width_to_length_ratio,
                                                          density=self.density, interval=self.interval,
                                                          seg_ratio=self.seg_ratio)

        self.Nsegs = Nsegs
        self.Nangles = Nsegs - 1
        self.angles = np.zeros(self.Nangles)
        self.seg_colors = generate_seg_colors(Nsegs, self.default_color)

        if not hasattr(self, 'real_length'):
            self.real_length = None
        if self.real_length is None:
            self.real_length = float(np.random.normal(loc=initial_length, scale=length_std, size=1))

        self.sim_length = self.real_length * model.scaling_factor
        self.seg_lengths = [self.sim_length * r for r in self.seg_ratio]
        self.seg_vertices = [v * self.sim_length for v in self.base_seg_vertices]

        self.set_head_edges()

        if not hasattr(self, 'real_mass'):
            self.real_mass = None
        if self.real_mass is None:
            self.compute_mass_from_length()

        if not hasattr(self, 'V'):
            self.V = None
        if self.V is None:
            self.V = self.get_real_length() ** 3

        self.segs = self.generate_segs(pos, orientation, joint_type=joint_type)

        self.contour = self.set_contour()

        self.sensors = []
        self.define_sensor('olfactor', (1, 0))
        if self.model.touch_sensors:
            self.add_touch_sensors()

    def get_real_length(self):
        return self.real_length

    def get_sim_length(self):
        return self.sim_length

    def get_real_mass(self):
        return self.real_mass

    '''We make the following assumptions :
        1. Larvae are V1-morphs, meaning mass is proportional to L**2 (Llandres&al)
        2. ratio of width to length constant :0.2. So area A = L*L*0.2.
        3. For this to give realistic values for both l1 (L=1.3mm) and l3 (L=5.2mm) we set
        density = 300 kg/m**2=0.3 mg/mm*2. (It is totally fortunate that box2d calculates mass as density*area)
        This yields m3=0.3 * 5.2*5.2*0.2  = 1.6224 mg and m1=0.3 * 1.3*1.3*0.2 = 0.1014 mg
        It follows that mass=density*width_to_length_ratio*length**2 for both real and simulated mass
        So, when using a scaling factor sf where sim_length=sf*real_length ==> sim_mass=sf**2 * real_mass'''

    def compute_mass_from_length(self):
        self.real_mass = self.density * self.real_length ** 2 * self.width_to_length_ratio
        # self.sim_mass = self.density * self.sim_length**2*self.width_to_length_ratio

    def adjust_shape_to_mass(self):
        self.real_length = np.sqrt(self.real_mass / (self.density * self.width_to_length_ratio))

    def adjust_body_vertices(self):
        self.sim_length = self.real_length * self.model.scaling_factor
        self.radius = self.sim_length / 2
        self.seg_lengths = [self.sim_length * r for r in self.seg_ratio]
        self.seg_vertices = [v * self.sim_length for v in self.base_seg_vertices]
        for vec, seg in zip(self.seg_vertices, self.segs):
            seg.seg_vertices = vec
        self.set_head_edges()
        self.update_sensor_position()

    '''
    seg_vertices of 2 segments example :
    [array([[[ 0.5 ,  0.  ],
            [ 0.26,  0.2 ],
            [-0.5 ,  0.2 ],
            [-0.5 , -0.2 ],
            [ 0.26, -0.2 ],
            [ 0.5 , -0.  ]]]), 
    array([[[ 0.5 ,  0.2 ],
            [-0.34,  0.2 ],
            [-0.5 ,  0.1 ],
            [-0.5 , -0.1 ],
            [-0.34, -0.2 ],
            [ 0.5 , -0.2 ]]])]
    So first index defines number of segment, second is 0 by default, then we take max or min and then we get the x to couple it with 0.
    Prerequirement : All segments are drawn horizontally with front to the right and midline on x axis.
    '''

    def update_sensor_position(self):
        for sensor_dict in self.sensors:
            sensor_dict['local_pos'] = sensor_dict['base_local_pos'] * self.sim_length

    def get_olfactor_position(self):
        return self.get_global_front_end_of_head()

    def define_sensor(self, sensor, pos_on_body):
        x, y = pos_on_body
        for i, (r, cum_r) in enumerate(zip(self.seg_ratio, np.cumsum(self.seg_ratio))):
            if x >= 1 - cum_r:
                seg_idx = i
                local_pos = np.array([x - 1 + cum_r - r / 2, y])
                break
        sensor_dict = {'sensor': sensor,
                       'seg_idx': seg_idx,
                       'base_local_pos': local_pos,
                       'local_pos': local_pos * self.sim_length}
        self.sensors.append(sensor_dict)

    def get_sensor(self, sensor):
        for sensor_dict in self.sensors:
            if sensor_dict['sensor'] == sensor:
                return sensor_dict

    def get_sensors(self):
        return [s['sensor'] for s in self.sensors]

    def get_sensor_position(self, sensor):
        d = self.get_sensor(sensor)
        return self.segs[d['seg_idx']].get_world_point(d['local_pos'])

    # def generate_seg_shapes2(self, Nsegs, width_to_length_proportion, density, interval, seg_ratio):
    #     N = Nsegs
    #     shape_length = 1
    #     w = width_to_length_proportion / 2
    #     x0 = 0.52
    #     w_max = 0.4
    #     l0 = x0 - w_max
    #     # rear_max = -0.4
    #     x2 = x0 - shape_length
    #     l = -w_max - x2
    #
    #     # generic larva shape with total lenth 1
    #     shape0 = [(x0, +0.0),
    #               (w_max, +w * 2 / 3),
    #               (w_max / 3, +w),
    #               (-w_max, +w * 2 / 3),
    #               (x2, 0.0),
    #               (-w_max, -w * 2 / 3),
    #               (w_max / 3, -w),
    #               (w_max, -w * 2 / 3)]
    #     # shape = self.get_larva_shape()
    #     generic_shape = np.array([shape0])
    #
    #     if N == 1:
    #         return [generic_shape]
    #     else:
    #         s0s = [x0 + r - cum_r for r, cum_r in zip(seg_ratio, np.cumsum(seg_ratio))]
    #         s1s = [x0 + r / 2 - cum_r for r, cum_r in zip(seg_ratio, np.cumsum(seg_ratio))]
    #         s2s = [x0 - cum_r for r, cum_r in zip(seg_ratio, np.cumsum(seg_ratio))]
    #         s0s, s2s = self.add_interval_between_segments(N, density, interval, s0s, s2s)
    #
    #         segment_vertices = []
    #         # for i, (s0, s1, s2) in enumerate(zip(s0s, s1s, s2s)):
    #         #     shape=[(np.clip(x, a_min=s2, a_max=s0),y) for x,y in shape0]
    #
    #         for i, (s0, s1, s2) in enumerate(zip(s0s, s1s, s2s)):
    #             if s0 > w_max and s2 >= w_max:
    #                 shape = [(s0 - s1, +(x0 - s0) / l0 * w),
    #                          (s2 - s1, +(x0 - s2) / l0 * w),
    #                          (s2 - s1, -(x0 - s2) / l0 * w),
    #                          (s0 - s1, -(x0 - s0) / l0 * w)]
    #             elif s0 > w_max > s2 >= -w_max:
    #                 shape = [(s0 - s1, +(x0 - s0) / l0 * w),
    #                          (w_max - s1, +w),
    #                          (s2 - s1, +w),
    #                          (s2 - s1, -w),
    #                          (w_max - s1, -w),
    #                          (s0 - s1, -(x0 - s0) / l0 * w)]
    #             elif -w_max < s0 <= w_max and -w_max <= s2 < w_max:
    #                 shape = [(s0 - s1, +w),
    #                          (s2 - s1, +w),
    #                          (s2 - s1, -w),
    #                          (s0 - s1, -w)]
    #             elif w_max >= s0 > -w_max >= s2:
    #                 shape = [(s0 - s1, +w),
    #                          (-w_max - s1, +w),
    #                          (s2 - s1, +((s2 - x2) / l + 1) * w / 2),
    #                          (s2 - s1, -((s2 - x2) / l + 1) * w / 2),
    #                          (-w_max - s1, -w),
    #                          (s0 - s1, -w)]
    #             elif -w_max >= s0:
    #                 shape = [(s0 - s1, +((s0 - x2) / l + 1) * w / 2),
    #                          (s2 - s1, +((s2 - x2) / l + 1) * w / 2),
    #                          (s2 - s1, -((s2 - x2) / l + 1) * w / 2),
    #                          (s0 - s1, -((s0 - x2) / l + 1) * w / 2)]
    #             segment_vertices.append(np.array([shape]) * N)
    #         return segment_vertices

    def generate_seg_shapes(self, Nsegs, width_to_length_proportion, density, interval, seg_ratio):
        self.density = density / (1 - 2 * (Nsegs - 1) * interval)
        w = width_to_length_proportion / 2
        points = np.array([[0.9, w], [0.05, w]])
        xy0 = fun.body(points)
        ps = fun.segment_body(Nsegs, xy0, seg_ratio=seg_ratio, centered=True)
        seg_vertices = [np.array([p]) * 1 for p in ps]
        return seg_vertices

    # def get_larva_shape(self, filepath=None):
    #     if filepath is None:
    #         filepath = LarvaShape_path
    #     return np.loadtxt(filepath, dtype=float, delimiter=",")

    def generate_segs(self, position, orientation, joint_type=None):
        N = self.Nsegs
        ls_x = [np.cos(orientation) * l for l in self.seg_lengths]
        ls_y = np.sin(orientation) * self.sim_length / N
        seg_positions = [[position[0] + (-i + (N - 1) / 2) * ls_x[i],
                          position[1] + (-i + (N - 1) / 2) * ls_y] for i in range(N)]

        segs = []
        if self.model.physics_engine:
            physics_pars = {'density': self.density,
                            'friction': 0.01,
                            'restitution': 0.0,
                            'lin_damping': self.lin_damping,
                            'ang_damping': self.ang_damping,
                            'inertia': 1.0}

            fixtures = []
            for i in range(N):
                seg = Box2DPolygon(space=self.model.space, pos=seg_positions[i], orientation=orientation,
                                   physics_pars=physics_pars, facing_axis=b2Vec2(1.0, 0.0),
                                   seg_vertices=self.seg_vertices[i], color=self.seg_colors[i])
                fixtures.extend(seg._fixtures)
                segs.append(seg)

            # put all agents into same group (negative so that no collisions are detected)
            if self.model.larva_collisions:
                for fixture in fixtures:
                    fixture.filterData.groupIndex = -1

            if joint_type is None:
                joint_type = {'distance': 2, 'revolute': 1}
            if N > 1:
                self.create_joints(N, segs, joint_type)
        else:
            for i in range(N):
                seg = DefaultSegment(pos=seg_positions[i], orientation=orientation, seg_vertices=self.seg_vertices[i],
                                     color=self.seg_colors[i])
                segs.append(seg)
            self.model.space.place_agent(self, position)
        return segs

    # To make peristalsis visible we try to leave some space between the segments.
    # We define an interval proportional to the length : int*l.
    # We subtract it from the front end of all segments except the first and from the rear end of all segments except the last.
    # For Npoints=n, in total we subtract 2*(n-1)*int*l in length.
    # For width_to_length_ratio=w2l, the area of the body without intervals is A=w2l*l*l
    # Subtracting the intervals, this translates to A'= (l-2*(n-1)*int*l) * w2l*l = (1-2*(n-1)*int)*A
    # To get the same mass, we will raise the density=d to d' accordingly : mass=d*A = d'*A' ==> d'=d/(1-2*(n-1)*int)
    def add_interval_between_segments(self, Nsegs, density, interval, seg_starts, seg_stops):
        for i in range(1, len(seg_starts)):
            seg_starts[i] -= interval
        for i in range(len(seg_stops) - 1):
            seg_stops[i] += interval

        self.density = density / (1 - 2 * (Nsegs - 1) * interval)

        return seg_starts, seg_stops

    def create_joints(self, Nsegs, segs, joint_type):
        space = self.model.space
        l0 = np.mean(self.seg_lengths)
        self.joints = []
        # TODO Find compatible parameters.
        # Until now for the 12-seg body : density 30000 and maxForce 100000000  and torque_coef 3.5 seem to work for natural bend
        # Trying to implement friction joint
        if self.friction_pars is not None:
            friction_joint_def = {'maxForce': self.friction_pars['maxForce'],
                                  # Good for one segment maybe : 100000000,
                                  'maxTorque': self.friction_pars['maxTorque']}  # Good for one segment maybe : 10,
            for i in range(Nsegs):
                friction_joint = space.CreateFrictionJoint(**friction_joint_def,
                                                           bodyA=segs[i]._body,
                                                           bodyB=self.model.friction_body,
                                                           localAnchorA=(0, 0),
                                                           localAnchorB=(0, 0))

                self.joints.append(friction_joint)
        w = self.width_to_length_ratio * Nsegs / 2
        # For many segments, the front one(s) will be joint by points outside the body.
        # So we adopt a more conservative solution, bringing the attachment point more medially : No visible difference
        # lateral_attachment_dist = self.width_to_length_ratio * self.Npoints / 4

        if joint_type['distance']:
            dist_joint_def = {'collideConnected': False,
                              'frequencyHz': 5,
                              'dampingRatio': 1,
                              'length': l0 * 0.01}

            for i in range(Nsegs - 1):
                A, B = segs[i]._body, segs[i + 1]._body
                if joint_type['distance'] == 2:
                    j_l = space.CreateDistanceJoint(**dist_joint_def,
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
                elif joint_type['distance'] == 1:
                    j = space.CreateDistanceJoint(**dist_joint_def,
                                                  bodyA=A,
                                                  bodyB=B,
                                                  localAnchorA=tuple(l0 * x for x in (-0.5, 0)),
                                                  localAnchorB=tuple(l0 * x for x in (0.5, 0)))
                    self.joints.append(j)

        if joint_type['revolute']:

            rev_joint_def = {'collideConnected': False,
                             'referenceAngle': 0,
                             'enableLimit': True,
                             'lowerAngle': -(np.pi / 2) / Nsegs,
                             'upperAngle': (np.pi / 2) / Nsegs,
                             'enableMotor': True,  # )
                             'maxMotorTorque': 0.1,
                             'motorSpeed': 0}

            for i in range(Nsegs - 1):
                A, B = segs[i]._body, segs[i + 1]._body
                if joint_type['revolute'] == 2:
                    j_l = space.CreateRevoluteJoint(**rev_joint_def,
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
                elif joint_type['revolute'] == 1:
                    j = space.CreateRevoluteJoint(**rev_joint_def,
                                                  bodyA=A,
                                                  bodyB=B,
                                                  localAnchorA=tuple(l0 * x for x in (-0.5, 0)),
                                                  localAnchorB=tuple(l0 * x for x in (0.5, 0)))
                    self.joints.append(j)

    def __del__(self):
        try:
            for seg in self.segs:
                self.space.DestroyBody(seg._body)
        except:
            try:
                self.space.remove_agent(self)
            except:
                pass

    def draw_sensor(self, viewer, sensor):
        viewer.draw_circle(radius=self.sim_length / 20,
                           # pos=self.get_olfactor_position(),
                           position=self.get_sensor_position(sensor),
                           filled=True, color=(255, 0, 0), width=.1)

    def draw(self, viewer):
        if not self.model.draw_contour:
            self.contour = self.set_contour()
            viewer.draw_polygon(self.contour, filled=True, color=self.get_head()._color)
        else:
            for seg in self.segs:
                seg.draw(viewer)
        if self.model.draw_head:
            viewer.draw_circle(radius=self.radius/2,
                               position=self.get_global_front_end_of_head(),
                               filled=True, color=(255, 0, 0), width=self.radius / 3)

        if self.model.draw_midline :
            # points=[self.segs[i].get_position() for i in range(self.Nsegs)]
            points=[self.get_global_front_end_of_seg(i) for i in range(self.Nsegs)] + [self.get_global_rear_end_of_body()]
            viewer.draw_polyline(points, color=(0, 0, 255), closed=False, width=self.radius / 10)
            for i, p in enumerate(points):
                c = 255 * i / (len(points) - 1)
                color = (c, 255 - c, 0)
                viewer.draw_circle(radius=self.radius / 10, position=p, filled=True, color=color, width=self.radius / 20)

        if self.model.draw_centroid:
            print('sss')
            viewer.draw_circle(radius=self.radius/2, position=self.get_position(), filled=True, color=self.default_color, width=self.radius / 3)

        if self.selected:
            r = self.seg_lengths[0] / 2
            try:
                for seg in self.segs:
                    for i, vertices in enumerate(seg.vertices):
                        viewer.draw_polygon(vertices, filled=False, color=self.model.selection_color, width=r / 5)
            except :
                viewer.draw_circle(radius=r,position=self.get_position(),
                               filled=False, color=self.model.selection_color, width=r / 5)

        # for s in self.get_sensors() :
        #     self.draw_sensor(viewer, s)

    def plot_vertices(self, axes, **kwargs):
        for seg in self.segs:
            seg.plot(axes, **kwargs)

    def get_Box2D_mass(self):
        mass = 0
        for seg in self.segs:
            mass += seg.get_mass()
        return mass

    def set_color(self, color):
        for seg, col in zip(self.segs, color):
            seg.set_color(col)

    def get_color(self):
        return [seg.get_color() for seg in self.segs]

    def get_segment(self, seg_index):
        return self.segs[seg_index]

    def get_head(self):
        return self.segs[0]

    def get_tail(self):
        return self.segs[-1]

    def get_centroid_position(self):
        seg_x_positions = []
        seg_y_positions = []
        for i, seg in enumerate(self.segs):
            x, y = seg.get_position().tolist()
            seg_x_positions.append(x)
            seg_y_positions.append(y)
        centroid = (sum(seg_x_positions) / len(self.segs), sum(seg_y_positions) / len(self.segs))

        return np.asarray(centroid)

    def get_local_front_end_of_seg(self, seg_index):
        front_local_x = np.max(self.seg_vertices[seg_index][0], axis=0)[0]
        return (front_local_x, 0)

    def get_local_rear_end_of_seg(self, seg_index):
        rear_local_x = np.min(self.seg_vertices[seg_index][0], axis=0)[0]
        return (rear_local_x, 0)

    def get_local_rear_end_of_head(self):
        return self.local_rear_end_of_head

    def get_local_front_end_of_head(self):
        return self.local_front_end_of_head

    def get_global_front_end_of_seg(self, seg_index):
        local_pos = self.get_local_front_end_of_seg(seg_index)
        global_pos = self.get_segment(seg_index).get_world_point(local_pos)
        return global_pos

    def get_global_rear_end_of_seg(self, seg_index):
        local_pos = self.get_local_rear_end_of_seg(seg_index)
        global_pos = self.get_segment(seg_index).get_world_point(local_pos)
        return global_pos

    def get_global_rear_end_of_head(self):
        return self.segs[0].get_world_point(self.local_rear_end_of_head)

    def get_global_front_end_of_head(self):
        return self.segs[0].get_world_point(self.local_front_end_of_head)

    def get_global_midspine_of_body(self):
        if self.Nsegs == 2:
            return self.get_global_rear_end_of_head()
        if (self.Nsegs % 2) == 0:
            seg_idx = int(self.Nsegs / 2)
            global_pos = self.get_global_front_end_of_seg(seg_idx)
        else:
            seg_idx = int((self.Nsegs + 1) / 2)
            global_pos = self.segs[seg_idx].get_world_point((0.0, 0.0))
        return global_pos

    def get_global_rear_end_of_body(self):
        local_pos = self.get_local_rear_end_of_seg(-1)
        global_pos = self.segs[-1].get_world_point(local_pos)
        return global_pos

    def get_contour(self):
        return self.contour

    def set_contour(self, Ncontour=22):
        vertices = [np.array(seg.vertices[0]) for seg in self.segs]
        l_side = fun.flatten_list([v[:int(len(v) / 2)] for v in vertices])
        r_side = fun.flatten_list([np.flip(v[int(len(v) / 2):], axis=0) for v in vertices])
        r_side.reverse()
        total_contour = l_side + r_side
        if len(total_contour) > Ncontour:
            seed(1)
            contour = [total_contour[i] for i in sorted(sample(range(len(total_contour)), Ncontour))]
        else:
            contour = total_contour
        # self.contour = contour[ConvexHull(contour).vertices].tolist()
        return contour

    def add_touch_sensors(self):
        y = 0.1
        x_f, x_m, x_r = 0.75, 0.5, 0.25
        self.define_sensor('M_front', (1.0, 0.0))
        self.define_sensor('L_front', (x_f, y))
        self.define_sensor('R_front', (x_f, -y))
        self.define_sensor('L_mid', (x_m, y))
        self.define_sensor('R_mid', (x_m, -y))
        self.define_sensor('L_rear', (x_r, y))
        self.define_sensor('R_rear', (x_r, -y))
        self.define_sensor('M_rear', (0.0, 0.0))

    def set_head_edges(self):
        self.local_rear_end_of_head = (np.min(self.seg_vertices[0][0], axis=0)[0], 0)
        self.local_front_end_of_head = (np.max(self.seg_vertices[0][0], axis=0)[0], 0)

    def get_polygon(self, scale=1):
        # p=self.segs[0].get_polygon()
        # for i in range(self.Nsegs) :
        #     if i!=0 :
        #         p=p.union(self.segs[i].get_polygon())
        # return p
        # mp=MultiPolygon([seg.get_polygon() for seg in self.segs])
        # p=Polygon(mp.bounds)
        p=cascaded_union([seg.get_polygon(scale=scale) for seg in self.segs])
        return p

    def move_body(self, dx, dy):
        for i, seg in enumerate(self.segs) :
            p, o = seg.get_pose()
            new_p = p + np.array([dx,dy])
            seg.set_position(tuple(new_p))
            seg.update_vertices(new_p, o)
        self.pos=self.get_global_midspine_of_body()


