from random import sample, seed
import numpy as np
from Box2D import b2Vec2
from shapely.ops import cascaded_union
# TODO Find a way to use this. Now if changed everything is scal except locomotion. It seems that
#  ApplyForceToCenter function does not scale
# _world_scale = np.int(100)

import lib.aux.functions as fun
from lib.model.body.segment import Box2DPolygon, DefaultSegment


class LarvaBody:
    def __init__(self, model, pos=None, orientation=None, density=300.0,
                 initial_length=None, length_std=0, Nsegs=1, interval=0, joint_type=None,
                 seg_ratio=None, friction_pars=None, **kwargs):

        if joint_type is None:
            joint_type = {'distance': 2, 'revolute': 1}
        if friction_pars is None:
            friction_pars = {'maxForce': 10 ** 0, 'maxTorque': 10 ** -1}
        self.model = model
        self.density = density
        self.friction_pars = friction_pars
        self.width_to_length_ratio = 0.2  # from [1] K. R. Kaun et al., “Natural variation in food acquisition mediated via a Drosophila cGMP-dependent protein kinase,” J. Exp. Biol., vol. 210, no. 20, pp. 3547–3558, 2007.
        if seg_ratio is None:
            seg_ratio = [1 / Nsegs] * Nsegs
        elif type(seg_ratio)==str :
            seg_ratio= seg_ratio.replace('(', '')
            seg_ratio= seg_ratio.replace(')', '')
            seg_ratio = [float(x) for x in seg_ratio.split(',')]
        self.seg_ratio = seg_ratio
        self.interval = interval
        self.shape_scale = 1

        self.base_seg_vertices = self.generate_seg_shapes(Nsegs, self.width_to_length_ratio,
                                                          density=self.density, interval=self.interval,
                                                          seg_ratio=self.seg_ratio)

        self.Nsegs = Nsegs
        self.Nangles = Nsegs - 1
        self.angles = np.zeros(self.Nangles)
        self.seg_colors = self.generate_seg_colors(Nsegs)

        if not hasattr(self, 'real_length'):
            self.real_length = None
        if self.real_length is None:
            self.real_length = float(np.random.normal(loc=initial_length, scale=length_std, size=1))

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
            self.V = self.real_length ** 3

        self.segs = self.generate_segs(pos, orientation, joint_type=joint_type)

        self.contour = self.set_contour()

        self.sensors = []
        self.define_sensor('olfactor', (1, 0))
        if self.model.touch_sensors:
            self.add_touch_sensors()

    @ property
    def sim_length(self):
        return self.real_length * self.model.scaling_factor

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
        self.radius = self.sim_length / 2
        # if not self.model.space_in_mm :
        #     self.radius*=1000
        self.seg_lengths = [self.sim_length * r for r in self.seg_ratio]
        self.seg_vertices = [v * self.sim_length for v in self.base_seg_vertices]
        for vec, seg in zip(self.seg_vertices, self.segs):
            seg.seg_vertices = vec
        self.set_head_edges()
        self.update_sensor_position()

    def generate_seg_colors(self, N):
        c=np.copy(self.default_color)
        return [np.array((0, 255, 0))] + [c]*(N - 2) + [np.array((255, 0, 0))] if N > 5 else [c]*N


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
                seg = DefaultSegment(space=self.model.space, pos=seg_positions[i], orientation=orientation,
                                     seg_vertices=self.seg_vertices[i],
                                     color=self.seg_colors[i])
                segs.append(seg)
            # print(position)
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
        # For many segments, the front one(sigma) will be joint by points outside the body.
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
        c, r = self.get_head().color, self.radius

        if not self.model.draw_contour:
            self.contour = self.set_contour()
            viewer.draw_polygon(self.contour, c, True, r / 5)
        else:
            # viewer.draw_polygon(self.get_shape().boundary.coords, c, True, r / 5)
            for seg in self.segs:
                seg.draw(viewer)
        if self.model.draw_head:
            viewer.draw_circle(self.get_global_front_end_of_head(), r / 2, (255, 0, 0), True, r / 6)

        if self.model.draw_midline:
            points = [self.get_global_front_end_of_seg(i) for i in range(self.Nsegs)] + [self.get_global_rear_end_of_body()]
            viewer.draw_polyline(points, color=(0, 0, 255), closed=False, width=r / 10)
            for i, p in enumerate(points):
                c = 255 * i / (len(points) - 1)
                color = (c, 255 - c, 0)
                viewer.draw_circle(p, r / 10, color, True, r / 20)

        if self.model.draw_centroid:
            viewer.draw_circle(self.get_position(), r / 2, self.default_color, True, r / 3)

        if self.selected:
            cc = self.model.selection_color
            try:
                viewer.draw_polygon(self.get_shape().boundary.coords, cc, False, r / 10)
                # for seg in self.segs:
                #     for i, vertices in enumerate(seg.vertices):
                #         viewer.draw_polygon(vertices, c, False, r)
            except:
                viewer.draw_circle(self.get_position(), r, cc, False, r/5)

        # for sigma in self.get_sensors() :
        #     self.draw_sensor(viewer, sigma)

    def plot_vertices(self, axes, **kwargs):
        for seg in self.segs:
            seg.plot(axes, **kwargs)

    def get_Box2D_mass(self):
        mass = 0
        for seg in self.segs:
            mass += seg.get_mass()
        return mass

    def set_color(self, colors):
        if len(colors) != self.Nsegs:
            colors = [tuple(colors)] * self.Nsegs
        for seg, col in zip(self.segs, colors):
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
        return np.array(self.segs[0].get_world_point(self.local_front_end_of_head))

    def get_global_midspine_of_body(self):
        if self.Nsegs == 1:
            return self.segs[0].get_position()
        elif self.Nsegs == 2:
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
        # contour = contour[ConvexHull(contour).vertices].tolist()
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

    def get_shape(self, scale=1):
        p = cascaded_union([seg.get_shape(scale=scale) for seg in self.segs])

        return p

    def move_body(self, dx, dy):
        for i, seg in enumerate(self.segs):
            p, o = seg.get_pose()
            new_p = p + np.array([dx, dy])
            seg.set_position(tuple(new_p))
            seg.update_vertices(new_p, o)
        self.pos = self.get_global_midspine_of_body()
