from random import sample, seed
import numpy as np

from shapely.ops import cascaded_union
# TODO Find a way to use this. Now if changed everything is scal except locomotion. It seems that
#  ApplyForceToCenter function does not scale
# _world_scale = np.int(100)
import lib.aux.dictsNlists
import lib.aux.sim_aux
from lib.model.body.segment import Box2DPolygon, DefaultSegment


class LarvaBody:
    def __init__(self, model, pos=None, orientation=None, density=300.0,
                 initial_length=None, length_std=0, Nsegs=1, interval=0,
                 seg_ratio=None, touch_sensors=False, **kwargs):
        # print(Nsegs, initial_length)
        # raise
        self.touch_sensors = touch_sensors
        self.model = model
        self.density = density
        self.width_to_length_ratio = 0.2  # from [1] K. R. Kaun et al., “Natural variation in food acquisition mediated via a Drosophila cGMP-dependent protein kinase,” J. Exp. Biol., vol. 210, no. 20, pp. 3547–3558, 2007.
        if seg_ratio is None:
            seg_ratio = [1 / Nsegs] * Nsegs
        elif type(seg_ratio) == str:
            seg_ratio = seg_ratio.replace('(', '')
            seg_ratio = seg_ratio.replace(')', '')
            seg_ratio = [float(x) for x in seg_ratio.split(',')]
        self.seg_ratio = np.array(seg_ratio)
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

        self.seg_lengths = self.sim_length * self.seg_ratio
        self.seg_vertices = [s* self.sim_length for s in self.base_seg_vertices ]
        # self.seg_vertices = self.base_seg_vertices * self.sim_length
        self.set_head_edges()

        if not hasattr(self, 'real_mass'):
            self.real_mass = None
        if self.real_mass is None:
            self.compute_mass_from_length()

        if not hasattr(self, 'V'):
            self.V = None
        if self.V is None:
            self.V = self.real_length ** 3

        self.segs = self.generate_segs(pos, orientation, **kwargs)

        self.contour = self.set_contour()

        self.sensors = []
        self.define_sensor('olfactor', (1, 0))
        if self.touch_sensors is not None:
            self.add_touch_sensors(self.touch_sensors)

        # print(self.sensors)

    @property
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
        self.seg_lengths = self.sim_length * self.seg_ratio
        self.seg_vertices = [s* self.sim_length for s in self.base_seg_vertices ]
        # self.seg_vertices = self.base_seg_vertices * self.sim_length
        for vec, seg in zip(self.seg_vertices, self.segs):
            seg.seg_vertices = vec
        self.set_head_edges()
        self.update_sensor_position()

    def generate_seg_colors(self, N):
        c = np.copy(self.default_color)
        return [np.array((0, 255, 0))] + [c] * (N - 2) + [np.array((255, 0, 0))] if N > 5 else [c] * N

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

    @property
    def olfactor_pos(self):
        return self.global_front_end_of_head

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
        # print(sensor)
        for sensor_dict in self.sensors:
            if sensor_dict['sensor'] == sensor:
                return sensor_dict

    def get_sensors(self):
        return [s['sensor'] for s in self.sensors]

    def get_sensor_position(self, sensor):
        d = self.get_sensor(sensor)
        # print(d)
        return self.segs[d['seg_idx']].get_world_point(d['local_pos'])

    def generate_seg_shapes(self, Nsegs, width_to_length_proportion, density, interval, seg_ratio):
        self.density = density / (1 - 2 * (Nsegs - 1) * interval)
        w = width_to_length_proportion / 2
        points = np.array([[0.9, w], [0.05, w]])
        xy0 = lib.aux.sim_aux.body(points)
        ps = lib.aux.sim_aux.segment_body(Nsegs, xy0, seg_ratio=seg_ratio, centered=True)
        seg_vertices = [np.array([p]) for p in ps]
        return seg_vertices

    # def get_larva_shape(self, filepath=None):
    #     if filepath is None:
    #         filepath = LarvaShape_path
    #     return np.loadtxt(filepath, dtype=float, delimiter=",")

    def generate_segs(self, position, orientation, **kwargs):
        from Box2D import b2Vec2
        N = self.Nsegs
        ls_x = [np.cos(orientation) * l for l in self.seg_lengths]
        ls_y = np.sin(orientation) * self.sim_length / N
        seg_positions = [[position[0] + (-i + (N - 1) / 2) * ls_x[i],
                          position[1] + (-i + (N - 1) / 2) * ls_y] for i in range(N)]

        segs = []
        if self.model.Box2D:
            physics_pars = {'density': self.density,
                            'friction': 10.0,
                            'restitution': 0.0,
                            'lin_damping': self.lin_damping,
                            'ang_damping': self.ang_damping,
                            'inertia': 0.0}

            fixtures = []
            for i in range(N):
                seg = Box2DPolygon(space=self.model.space, pos=seg_positions[i], orientation=orientation,
                                   physics_pars=physics_pars, facing_axis=b2Vec2(1.0, 0.0), idx=i,
                                   seg_vertices=self.seg_vertices[i], color=self.seg_colors[i])
                fixtures.extend(seg._fixtures)
                segs.append(seg)

            # put all agents into same group (negative so that no collisions are detected)
            if self.model.larva_collisions:
                for fixture in fixtures:
                    fixture.filterData.groupIndex = -1

            if N > 1:
                self.create_joints(N, segs, **kwargs)
                # self.create_rotator(segs, position, orientation, physics_pars)
        else:
            for i in range(N):
                seg = DefaultSegment(space=self.model.space, pos=seg_positions[i], orientation=orientation,
                                     seg_vertices=self.seg_vertices[i], idx=i, color=self.seg_colors[i])
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

    def create_joints(self, Nsegs, segs, joint_types):
        space = self.model.space
        l0 = np.mean(self.seg_lengths)
        self.joints = []
        # TODO Find compatible parameters.
        # Until now for the 12-seg body : density 30000 and maxForce 100000000  and torque_coef 3.5 seem to work for natural bend
        # Trying to implement friction joint
        # if joint_types is None :
        #     joint_types = {'distance': 0, 'revolute': 0, 'friction' : 0}
        print(joint_types['friction'])
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

    def __del__(self):
        try:
            for seg in self.segs:
                self.space.DestroyBody(seg._body)
        except:
            try:
                self.space.remove_agent(self)
            except:
                pass

    def draw_sensors(self, viewer, sensors=None):
        if sensors is None:
            sensors = [d['sensor'] for d in self.sensors]
        for s in sensors:
            viewer.draw_circle(radius=self.sim_length / 20,
                               position=self.get_sensor_position(s),
                               filled=True, color=(255, 0, 0), width=.1)

    def draw(self, viewer):
        m = self.model
        c, r = self.head.color, self.radius
        h_pos = self.global_front_end_of_head
        pos = self.get_position()
        mid = [self.get_global_front_end_of_seg(i) for i in range(self.Nsegs)] + [self.global_rear_end_of_body]

        if m.draw_contour:
            for seg in self.segs:
                seg.draw(viewer)
        else:
            self.contour = self.set_contour()
            viewer.draw_polygon(self.contour, c, True, r / 5)

        if m.draw_head:
            draw_body_head(viewer, h_pos, r)

        if m.draw_midline:
            draw_body_midline(viewer, mid, r)

        if m.draw_centroid:
            draw_body_centroid(viewer, pos, r, c)

        # if True:
        if m.draw_sensors:
            self.draw_sensors(viewer)

        if self.selected:
            draw_selected_body(viewer, pos, self.get_shape().boundary.coords, r, m.selection_color)
            # try:
            #     viewer.draw_polygon(self.get_shape().boundary.coords, cc, False, r / 10)
            #     # for seg in self.segs:
            #     #     for i, vertices in enumerate(seg.vertices):
            #     #         viewer.draw_polygon(vertices, c, False, r)
            # except:
            #     viewer.draw_circle(pos, r, cc, False, r/5)

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

    @property
    def head(self):
        return self.segs[0]

    @property
    def tail(self):
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

    @property
    def global_rear_end_of_head(self):
        return self.segs[0].get_world_point(self.local_rear_end_of_head)

    @property
    def global_front_end_of_head(self):
        return np.array(self.segs[0].get_world_point(self.local_front_end_of_head))

    @property
    def global_midspine_of_body(self):
        if self.Nsegs == 1:
            return self.segs[0].get_position()
        elif self.Nsegs == 2:
            return self.global_rear_end_of_head
        if (self.Nsegs % 2) == 0:
            seg_idx = int(self.Nsegs / 2)
            global_pos = self.get_global_front_end_of_seg(seg_idx)
        else:
            seg_idx = int((self.Nsegs + 1) / 2)
            global_pos = self.segs[seg_idx].get_world_point((0.0, 0.0))
        return global_pos

    @property
    def global_rear_end_of_body(self):
        local_pos = self.get_local_rear_end_of_seg(-1)
        global_pos = self.segs[-1].get_world_point(local_pos)
        return global_pos

    def get_contour(self):
        return self.contour

    def set_contour(self, Ncontour=22):
        vertices = [np.array(seg.vertices[0]) for seg in self.segs]
        l_side = lib.aux.dictsNlists.flatten_list([v[:int(len(v) / 2)] for v in vertices])
        r_side = lib.aux.dictsNlists.flatten_list([np.flip(v[int(len(v) / 2):], axis=0) for v in vertices])
        r_side.reverse()
        total_contour = l_side + r_side
        if len(total_contour) > Ncontour:
            seed(1)
            contour = [total_contour[i] for i in sorted(sample(range(len(total_contour)), Ncontour))]
        else:
            contour = total_contour
        # contour = contour[ConvexHull(contour).vertices].tolist()
        return contour

    def add_touch_sensors(self, N=8):
        y = 0.1
        x_f, x_m, x_r = 0.75, 0.5, 0.25
        if N == 8:
            self.define_sensor('M_front', (1.0, 0.0))
            self.define_sensor('L_front', (x_f, y))
            self.define_sensor('R_front', (x_f, -y))
            self.define_sensor('L_mid', (x_m, y))
            self.define_sensor('R_mid', (x_m, -y))
            self.define_sensor('L_rear', (x_r, y))
            self.define_sensor('R_rear', (x_r, -y))
            self.define_sensor('M_rear', (0.0, 0.0))
        elif N == 2:
            self.define_sensor('R_mid', (x_m, -y))
            self.define_sensor('M_rear', (0.0, 0.0))
        elif N == 0:
            pass

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
        self.pos = self.gglobal_midspine_of_body

    def create_rotator(self, segs, position, orientation, physics_pars):
        import Box2D
        # self.rotator = Box2DPolygon(space=self.model.space, pos=position, orientation=orientation,
        #                    physics_pars=physics_pars, facing_axis=Box2D.b2Vec2(1.0, 0.0), idx=None,
        #                    seg_vertices=self.seg_vertices[0], color=self.seg_colors[0])

        self.rotator: Box2D.b2Body = self.model.space.CreateDynamicBody(
            position=Box2D.b2Vec2(*position),
            angle=orientation,
            # gravityScale=100,
            # fixedRotation=True if  self.idx!=0 else False,
            # linearDamping=physics_pars['lin_damping'],
            # angularDamping=physics_pars['ang_damping']
        )
        self.rotator.linearVelocity = Box2D.b2Vec2(*[.0, .0])
        self.rotator.angularVelocity = .0
        self.rotator.bullet = True
        l0 = np.mean(self.seg_lengths)
        w = l0 / 10
        vs = lib.aux.sim_aux.circle_to_polygon(5, l0)
        rotator_shape = Box2D.b2ChainShape(vertices=vs.tolist())
        self.rotator.CreateFixture(shape=rotator_shape)
        # # print(l0,w)

        dist_kws = {'collideConnected': False, 'length': l0 * 0.01}
        rev_kws = {'collideConnected': False,
                   'referenceAngle': 0,
                   'enableLimit': False,
                   # 'lowerAngle': -0.9*(np.pi* 2) / (Nsegs-1),
                   # 'upperAngle': 0.9*(np.pi * 2) / (Nsegs-1),
                   # 'enableMotor': True,  # )
                   # 'maxMotorTorque': 1.0,
                   # 'motorSpeed': 1
                   }
        # for A,B in ([segs[0]._body, self.rotator], [self.rotator, segs[1]._body]):
        self.model.space.CreateDistanceJoint(**dist_kws,
                                             bodyA=segs[0]._body,
                                             bodyB=self.rotator,
                                             localAnchorA=tuple(l0 * x for x in (-0.5, w)),
                                             localAnchorB=tuple(l0 * x for x in (0.0, w)))
        self.model.space.CreateDistanceJoint(**dist_kws,
                                             bodyA=segs[0]._body,
                                             bodyB=self.rotator,
                                             localAnchorA=tuple(l0 * x for x in (-0.5, -w)),
                                             localAnchorB=tuple(l0 * x for x in (0.0, -w)))
        self.model.space.CreateDistanceJoint(**dist_kws,
                                             bodyA=self.rotator,
                                             bodyB=segs[1]._body,
                                             localAnchorA=tuple(l0 * x for x in (0.0, w)),
                                             localAnchorB=tuple(l0 * x for x in (0.5, w)))
        self.model.space.CreateDistanceJoint(**dist_kws,
                                             bodyA=self.rotator,
                                             bodyB=segs[1]._body,
                                             localAnchorA=tuple(l0 * x for x in (0.0, -w)),
                                             localAnchorB=tuple(l0 * x for x in (0.5, -w)))
        self.model.space.CreateRevoluteJoint(**rev_kws,
                                             bodyA=segs[0]._body,
                                             bodyB=self.rotator,
                                             localAnchorA=tuple(l0 * x for x in (-0.5, w)),
                                             localAnchorB=tuple(l0 * x for x in (0.0, w)))
        self.model.space.CreateRevoluteJoint(**rev_kws,
                                             bodyA=segs[0]._body,
                                             bodyB=self.rotator,
                                             localAnchorA=tuple(l0 * x for x in (-0.5, -w)),
                                             localAnchorB=tuple(l0 * x for x in (0.0, -w)))
        self.model.space.CreateRevoluteJoint(**rev_kws,
                                             bodyA=segs[0]._body,
                                             bodyB=self.rotator,
                                             localAnchorA=tuple(l0 * x for x in (-0.5, -w)),
                                             localAnchorB=tuple(l0 * x for x in (0.0, -w)))
        self.model.space.CreateRevoluteJoint(**rev_kws,
                                             bodyA=self.rotator,
                                             bodyB=segs[1]._body,
                                             localAnchorA=tuple(l0 * x for x in (0.0, w)),
                                             localAnchorB=tuple(l0 * x for x in (0.5, w)))


#


def draw_body_midline(viewer, midline_xy, radius):
    try:
        mid = midline_xy
        r = radius
        if not any(np.isnan(np.array(mid).flatten())):
            Nmid = len(mid)
            viewer.draw_polyline(mid, color=(0, 0, 255), closed=False, width=r / 10)
            for i, xy in enumerate(mid):
                c = 255 * i / (Nmid - 1)
                viewer.draw_circle(xy, r / 10, color=(c, 255 - c, 0), width=r / 20)
    except:
        pass


def draw_body_contour(viewer, contour_xy, radius):
    try:
        pass
    except:
        pass


def draw_body_centroid(viewer, pos, radius, color):
    try:
        viewer.draw_circle(pos, radius / 2, color=color, width=radius / 3)
    except:
        pass


def draw_body_head(viewer, midline_xy, radius):
    try:
        pos = midline_xy[0]
        viewer.draw_circle(pos, radius / 2, color=(255, 0, 0), width=radius / 6)
    except:
        pass


def draw_selected_body(viewer, pos, contour_xy, radius, color):
    try:
        if len(contour_xy) > 0 and not np.isnan(contour_xy).any():
            viewer.draw_polygon(contour_xy, filled=False, color=color, width=radius / 5)
        elif not np.isnan(pos).any():
            viewer.draw_circle(pos, radius=radius, filled=False, color=color, width=radius / 3)
    except:
        pass
