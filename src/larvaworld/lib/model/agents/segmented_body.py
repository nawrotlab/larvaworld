import random

import numpy as np
import param
from shapely import geometry, affinity, ops

from larvaworld.lib.model.agents.draw_body import draw_body
from larvaworld.lib.param import PositiveNumber, PositiveInteger
from larvaworld.lib.reg.stored.miscellaneous import body_shapes
from larvaworld.lib import aux

from larvaworld.lib.model.agents._larva import LarvaMotile

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

#
# class DefaultSegment(BaseSegment):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.base_local_rear_end = np.array([np.min(self.base_seg_vertices[:, 0]), 0])
#         self.base_local_front_end = np.array([np.max(self.base_seg_vertices[:, 0]), 0])
#
#         self.update_vertices(self.pos, self.orientation)
#
#         self.lin_vel = 0.0
#         self.ang_vel = 0.0
#         self.ang_acc = 0.0
#
#     def update_vertices(self, pos, orient):
#         self.vertices = pos + self.seg_vertices @ aux.rotationMatrix(-orient)
#
#     def update_poseNvertices(self, pos, orientation):
#         self.pos=pos
#         self.orientation=orientation % (np.pi * 2)
#         self.vertices = pos + self.seg_vertices @ aux.rotationMatrix(-orientation)
#
#     def update_all(self, pos, orientation, lin_vel, ang_vel):
#         self.pos=pos
#         self.orientation=orientation % (np.pi * 2)
#         self.vertices = pos + self.seg_vertices @ aux.rotationMatrix(-orientation)
#         self.lin_vel=lin_vel
#         self.ang_vel=ang_vel
#
#     def get_pose(self):
#         return np.array(self.pos), self.orientation
#
#     @property
#     def seg_length(self):
#         return self.base_seg_ratio*self.body_length
#
#
#     @property
#     def global_front_end(self):
#         return self.get_world_point(self.base_local_front_end*self.body_length)
#
#     @property
#     def global_rear_end(self):
#         return self.get_world_point(self.base_local_rear_end*self.body_length)
#
#
#     def get_world_point(self, local_point):
#         return self.get_position() + aux.rotate_point_around_point(point=local_point, radians=-self.get_orientation())
#
#     def get_angularvelocity(self):
#         return self.ang_vel
#
#     def get_linearvelocity(self):
#         return self.lin_vel
#
#
#     def get_position(self):
#         return np.array(self.pos)
#
#     def set_position(self, pos):
#         self.pos = pos
#
#         # self.rotation_pos = None
#
#     def set_orientation(self, orientation):
#         self.orientation = orientation % (np.pi * 2)
#
#
#     def get_orientation(self):
#         return self.orientation
#
#
#
#
# class LarvaBody(LarvaMotile):
#     Nsegs = PositiveInteger(2, softmax=20, doc='The number of segments comprising the segmented larva body.')
#     symmetry = param.Selector(objects=['bilateral', 'radial'], doc='The body symmetry.')
#     length = PositiveNumber(0.005, softmax=0.1, step=0.001, doc='The initial length of the body in meters')
#     density = PositiveNumber(300.0, softmax=10000.0, step=1.0, doc='The density of the larva body in kg/m**2')
#     body_plan=param.Selector(objects=['drosophila_larva', 'zebrafish'], doc='The body plan.')
#
#     def __init__(self, brain, energetics=None, life_history={}, segment_ratio=None,  **kwargs):
#         super().__init__(brain, energetics, life_history, **kwargs)
#         self.width_to_length_ratio = 0.2  # from [1] K. R. Kaun et al., “Natural variation in food acquisition mediated via a Drosophila cGMP-dependent protein kinase,” J. Exp. Biol., vol. 210, no. 20, pp. 3547–3558, 2007.
#         if segment_ratio is None:
#             segment_ratio = [1 / self.Nsegs] * self.Nsegs
#         self.seg_ratio = np.array(segment_ratio)
#         self.contour_points = body_shapes[self.body_plan]
#         self.body_bend = 0
#
#
#         self.Nangles = self.Nsegs - 1
#
#         self.Nangles_b = int(self.Nangles + 1 / 2)
#
#
#
#         self.initialize(self.length)
#         self.radius=self.sim_length/2
#
#
#         if not self.model.Box2D :
#             self.segs = aux.generate_segs(N=self.Nsegs, pos=self.pos, orient=self.orientation,
#                                             ratio=self.seg_ratio, l=self.sim_length,color=self.default_color,
#                                       body_plan=self.body_plan, segment_class=DefaultSegment)
#         self.sensors = aux.AttrDict()
#         self.define_sensor('olfactor', (1, 0))
#
#         # self.compute_body_bend()
#
#
#
#     def initialize(self, l):
#         if not hasattr(self, 'real_length') or self.real_length is None:
#             self.real_length = l
#
#         if not hasattr(self, 'real_mass') or self.real_mass is None:
#             self.compute_mass_from_length()
#
#         if not hasattr(self, 'V') or self.V is None:
#             self.V = self.real_length ** 3
#
#
#
#     '''We make the following assumptions :
#         1. Larvae are V1-morphs, meaning mass is proportional to L**2 (Llandres&al)
#         2. ratio of width to length constant :0.2. So area A = L*L*0.2.
#         3. For this to give realistic values for both l1 (L=1.3mm) and l3 (L=5.2mm) we set
#         density = 300 kg/m**2=0.3 mg/mm*2. (It is totally fortunate that box2d calculates mass as density*area)
#         This yields m3=0.3 * 5.2*5.2*0.2  = 1.6224 mg and m1=0.3 * 1.3*1.3*0.2 = 0.1014 mg
#         It follows that mass=density*width_to_length_ratio*length**2 for both real and simulated mass
#         So, when using a scaling factor sf where sim_length=sf*real_length ==> sim_mass=sf**2 * real_mass'''
#
#     def compute_mass_from_length(self):
#         self.real_mass = self.density * self.real_length ** 2 * self.width_to_length_ratio
#         # self.sim_mass = self.density * self.sim_length**2*self.width_to_length_ratio
#
#     def adjust_shape_to_mass(self):
#         self.real_length = np.sqrt(self.real_mass / (self.density * self.width_to_length_ratio))
#
#
#     @property
#     def sim_length(self):
#         return self.real_length * self.model.scaling_factor
#
#     def compute_body_bend(self):
#         angles = [
#             aux.angle_dif(self.segs[i].get_orientation(), self.segs[i + 1].get_orientation(), in_deg=False) for i in
#             range(int(self.Nangles + 1 / 2))]
#         self.body_bend = aux.wrap_angle_to_0(sum(angles))
#
#
#     def set_color(self, colors):
#         if len(colors) != self.Nsegs:
#             colors = [tuple(colors)] * self.Nsegs
#         for seg, col in zip(self.segs, colors):
#             seg.color=col
#
#     @property
#     def head(self):
#         return self.segs[0]
#
#     @property
#     def tail(self):
#         return self.segs[-1]
#
#     @property
#     def direction(self):
#         return self.head.get_orientation()
#
#
#     @property
#     def global_midspine_of_body(self):
#         if self.Nsegs == 1:
#             return self.head.get_position()
#         elif self.Nsegs == 2:
#             return self.head.rear_end
#         if (self.Nsegs % 2) == 0:
#             seg_idx = int(self.Nsegs / 2)
#             global_pos = self.segs[seg_idx].front_end
#         else:
#             seg_idx = int((self.Nsegs + 1) / 2)
#             global_pos = self.segs[seg_idx].get_position()
#         return global_pos
#
#
#
#     @property
#     def olfactor_pos(self):
#         return self.head.front_end
#
#     @property
#     def olfactor_point(self):
#         return geometry.Point(self.olfactor_pos[0], self.olfactor_pos[1])
#
#     @property
#     def midline_xy(self):
#         return [seg.front_end for seg in self.segs] + [self.tail.rear_end]
#
#
#
#     def adjust_body_vertices(self):
#         self.radius = self.sim_length / 2
#         for i in range(self.Nsegs) :
#             self.segs[i].body_length=self.sim_length
#
#
#     '''
#     seg_vertices of 2 segments example :
#     [array([[[ 0.5 ,  0.  ],
#             [ 0.26,  0.2 ],
#             [-0.5 ,  0.2 ],
#             [-0.5 , -0.2 ],
#             [ 0.26, -0.2 ],
#             [ 0.5 , -0.  ]]]),
#     array([[[ 0.5 ,  0.2 ],
#             [-0.34,  0.2 ],
#             [-0.5 ,  0.1 ],
#             [-0.5 , -0.1 ],
#             [-0.34, -0.2 ],
#             [ 0.5 , -0.2 ]]])]
#     So first index defines number of segment, second is 0 by default, then we take max or min and then we get the x to couple it with 0.
#     Prerequirement : All segments are drawn horizontally with front to the right and midline on x axis.
#     '''
#
#
#
#     def define_sensor(self, sensor, pos_on_body):
#         x, y = pos_on_body
#         for i, (r, cum_r) in enumerate(zip(self.seg_ratio, np.cumsum(self.seg_ratio))):
#             if x >= 1 - cum_r:
#                 seg_idx = i
#                 local_pos = np.array([x - 1 + cum_r - r / 2, y])
#                 break
#         self.sensors[sensor] = aux.AttrDict({
#             'seg_idx': seg_idx,
#             'local_pos': local_pos,
#             # 'local_pos': local_pos * self.sim_length
#         })
#
#
#     def get_sensor_position(self, sensor):
#         d=self.sensors[sensor]
#         return self.segs[d.seg_idx].get_world_point(d.local_pos* self.sim_length)
#
#
#
#     def __del__(self):
#         try:
#             for seg in self.segs:
#                 self.space.DestroyBody(seg._body)
#         except:
#             try:
#                 self.space.remove_agent(self)
#             except:
#                 pass
#
#
#     def draw(self, viewer, filled=True):
#         pos = tuple(self.pos)
#         draw_body(v=viewer, model=self.model, pos=pos, midline_xy=self.midline_xy, contour_xy=None,
#                   radius=self.radius, vertices=None, color=self.default_color,segs=self.segs,
#                   selected=self.selected, sensors=self.sensors, length=self.sim_length)
#
#
#
#
#
#     def add_touch_sensors(self, idx):
#         for i in idx:
#             self.define_sensor(f'touch_sensor_{i}', self.contour_points[i])
#
#     def get_shape(self, scale=1):
#         ps=[geometry.Polygon(seg.vertices) for seg in self.segs]
#         if scale!=1:
#             ps=[affinity.scale(p, xfact=scale, yfact=scale) for p in ps]
#         return ops.cascaded_union(ps).boundary.coords
#
#     # @property
#     # def velocity(self):
#     #     return self.head.get_linearvelocity()
#
#
#     @property
#     def front_orientation(self):
#         return self.head.get_orientation()%(2*np.pi)
#
#
#     @property
#     def rear_orientation(self):
#         return self.tail.get_orientation()%(2*np.pi)


class BaseController(param.Parameterized):
    lin_vel_coef = PositiveNumber(1.0, doc='Coefficient for translational velocity')
    ang_vel_coef = PositiveNumber(1.0, doc='Coefficient for angular velocity')
    lin_force_coef = PositiveNumber(1.0, doc='Coefficient for force')
    torque_coef = PositiveNumber(0.5, doc='Coefficient for torque')
    body_spring_k = PositiveNumber(1.0, doc='Torsional spring constant for body bending')
    bend_correction_coef = PositiveNumber(1.0, doc='Bend correction coefficient')
    lin_damping = PositiveNumber(1.0, doc='Translational damping coefficient')
    ang_damping = PositiveNumber(1.0, doc='Angular damping coefficient')
    lin_mode = param.Selector(objects=['velocity', 'force', 'impulse'], doc='Mode of translational motion generation')
    ang_mode = param.Selector(objects=['torque','velocity'], doc='Mode of angular motion generation')


    def compute_ang_vel(self, amp, ang_vel, dt, bend):
        torque = amp * self.torque_coef
        return ang_vel + (-self.ang_damping * ang_vel - self.body_spring_k * bend + torque) * dt

    def compute_delta_rear_angle(self, bend, dst, length):
        k0 = 2 * dst * self.bend_correction_coef / length
        if 0 <= k0 < 1:
            return bend * k0
        elif 1 <= k0:
            return bend
        elif k0 < 0:
            return 0

