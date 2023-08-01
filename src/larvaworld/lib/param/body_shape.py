import math
from typing import Optional

import numpy as np
import param
from shapely import geometry, ops, affinity

from larvaworld.lib import aux
from larvaworld.lib.param import LineClosed, XYLine, PositiveInteger, PositiveNumber, MobilePoint, Viewable, \
    MobileVector

body_plans=aux.AttrDict({
'drosophila_larva': [(0.9, 0.1), (0.05, 0.1)],
'zebrafish_larva': [(0.9, 0.25), (0.7, 0.25), (0.6, 0.005), (0.05, 0.005)]
})

class BodyContour(LineClosed):
    symmetry = param.Selector(objects=['bilateral', 'radial'], doc='The body symmetry.')
    guide_points=XYLine(doc='A list of 2d points outside the midline in order to generate the vertices')
    base_vertices = XYLine(doc='The list of 2d points')
    body_plan = param.Selector(objects=['drosophila_larva', 'zebrafish'], doc='The body plan.')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.generate_base_vertices()

    def generate_base_vertices(self):
        if not self.guide_points:
            self.guide_points=body_plans[self.body_plan]
        if self.symmetry=='bilateral' :
            symmetric_ps=[(x,-y) for (x,y) in self.guide_points]
            symmetric_ps.reverse()
            self.base_vertices=[(1.0,0.0)]+self.guide_points+ [(0.0,0.0)] + symmetric_ps

    # TODO make this more explicit
    @property
    def width_to_length_ratio(self):
        return np.mean(np.array(self.guide_points)[:,1])





class ShapeMobile(LineClosed, MobileVector):
    base_vertices = XYLine(doc='The list of 2d points')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.length_ratio=self.get_length_ratio()
        self.update_vertices()


    def get_length_ratio(self):
        xs=np.array(self.base_vertices)[:,0]
        return -np.min(xs)+np.max(xs)

    @param.depends('pos','orientation','length', watch=True)
    def update_vertices(self):
        self.vertices = self.translate(self.length/self.length_ratio*np.array(self.base_vertices))
        # self.vertices = [self.translate(self.length / self.length_ratio * np.array(p)) for p in self.base_vertices]
        # self.vertices = self.pos + self.length*self.base_vertices @ self.rotationMatrix

class ShapeViewable(ShapeMobile, Viewable):

    def draw(self, v, **kwargs):
        v.draw_polygon(self.vertices, filled=True, color=self.color)


class BodyMobile(ShapeMobile, BodyContour):
    density = PositiveNumber(300.0, softmax=10000.0, step=1.0, doc='The density of the larva body in kg/m**2')

    def __init__(self,**kwargs):

        super().__init__(**kwargs)
        self.initial_length = self.length
        self.body_bend = 0

    '''We make the following assumptions :
        1. Larvae are V1-morphs, meaning mass is proportional to L**2 (Llandres&al)
        2. ratio of width to length constant :0.2. So area A = L*L*0.2.
        3. For this to give realistic values for both l1 (L=1.3mm) and l3 (L=5.2mm) we set
        density = 300 kg/m**2=0.3 mg/mm*2. (It is totally fortunate that box2d calculates mass as density*area)
        This yields m3=0.3 * 5.2*5.2*0.2  = 1.6224 mg and m1=0.3 * 1.3*1.3*0.2 = 0.1014 mg
        It follows that mass=density*width_to_length_ratio*length**2 for both real and simulated mass
        So, when using a scaling factor sf where sim_length=sf*real_length ==> sim_mass=sf**2 * real_mass'''

    def compute_mass_from_length(self):
        self.real_mass = self.density * self.length ** 2 * self.width_to_length_ratio
        # self.sim_mass = self.density * self.sim_length**2*self.width_to_length_ratio

    def adjust_shape_to_mass(self):
        self.length = np.sqrt(self.real_mass / (self.density * self.width_to_length_ratio))


class SegmentedBody(BodyMobile):
    Nsegs = PositiveInteger(2, softmax=20, doc='The number of segments comprising the segmented larva body.')
    segment_ratio = param.Parameter(None, doc='The number of segments comprising the segmented larva body.')
    segs=param.List(item_type=ShapeViewable, doc='The body segments.')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # If segment ratio is not provided, generate equal-length segments
        if self.segment_ratio is None:
            self.segment_ratio = np.array([1 / self.Nsegs] * self.Nsegs)
        self.base_seg_vertices=self.segmentize()
        self.seg_positions = self.generate_seg_positions()
        self.generate_segs()
        self.update_seg_lengths()


    @property
    def Nangles(self):
        return self.Nsegs-1

    def segmentize(self, centered: bool = True, closed: bool = False) -> np.ndarray:
        """
        Segments a body into equal-length or given-length segments via vertical lines.

        Args:
        - N: Number of segments to divide the body into.
        - points: Array with shape (M,2) representing the contour of the body to be segmented.
        - ratio: List of N floats specifying the ratio of the length of each segment to the length of the body.
                    Defaults to None, in which case equal-length segments will be generated.
        - centered: If True, centers the segments around the origin. Defaults to True.
        - closed: If True, the last point of each segment is connected to the first point. Defaults to False.

        Returns:
        - ps: Numpy array with shape (Nsegs,L,2), where L is the number of vertices of each segment.
              The first segment in the list is the front-most segment.
        """


        # N=self.Nsegs
        R=self.segment_ratio

        # Create a polygon from the given body contour
        p = geometry.Polygon(np.array(self.base_vertices))
        # Get maximum y value of contour
        y0 = np.max(p.exterior.coords.xy[1])

        # Segment body via vertical lines
        ps = [p]
        for cum_r in np.cumsum(R):
            l = geometry.LineString([(1 - cum_r, y0), (1 - cum_r, -y0)])
            new_ps = []
            for p in ps:
                new_ps += list(ops.split(p, l).geoms)
            ps = new_ps

        # Sort segments so that front segments come first
        ps.sort(key=lambda x: x.exterior.xy[0], reverse=True)

        # Transform to 2D array of coords
        ps = [p.exterior.coords.xy for p in ps]
        ps = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]

        # Center segments around 0,0
        if centered:
            for i, (r, cum_r) in enumerate(zip(R, np.cumsum(R))):
                ps[i] -= [(1 - cum_r) + r / 2, 0]

        # Put front point at the start of segment vertices. Drop duplicate rows
        for i in range(len(ps)):
            if i == 0:
                ind = np.argmax(ps[i][:, 0])
                ps[i] = np.flip(np.roll(ps[i], -ind - 1, axis=0), axis=0)
            else:
                ps[i] = np.flip(np.roll(ps[i], 1, axis=0), axis=0)
            _, idx = np.unique(ps[i], axis=0, return_index=True)
            ps[i] = ps[i][np.sort(idx)]
            if closed:
                ps[i] = np.concatenate([ps[i], [ps[i][0]]])
            ps=[aux.np2Dtotuples(pp) for pp in ps]
        return ps

    def generate_seg_positions(self):
        N=self.Nsegs
        ls_x = np.cos(self.orientation) * self.length * self.segment_ratio
        ls_y = np.sin(self.orientation) * self.length /N
        return [(self.x + (-i + (N - 1) / 2) * ls_x[i],
                 self.y + (-i + (N - 1) / 2) * ls_y) for i in range(N)]

    def generate_segs(self):

        self.segs= [self.param.segs.item_type(pos=self.seg_positions[i], orientation=self.orientation,
                       base_vertices=self.base_seg_vertices[i], length=self.length*self.segment_ratio[i]) for i in range(self.Nsegs)]


    def compute_body_bend(self):
        angles = [
            aux.angle_dif(self.segs[i].get_orientation(), self.segs[i + 1].get_orientation(), in_deg=False) for i in
            range(int(self.Nangles + 1 / 2))]
        self.body_bend = aux.wrap_angle_to_0(sum(angles))

    @property
    def head(self):
        return self.segs[0]

    @property
    def tail(self):
        return self.segs[-1]

    @property
    def direction(self):
        return self.head.get_orientation()



    def get_shape(self, scale=1):
        ps=[geometry.Polygon(seg.vertices) for seg in self.segs]
        if scale!=1:
            ps=[affinity.scale(p, xfact=scale, yfact=scale) for p in ps]
        return ops.cascaded_union(ps).boundary.coords

    @property
    def global_midspine_of_body(self):
        if self.Nsegs == 1:
            return self.head.get_position()
        elif self.Nsegs == 2:
            return self.head.rear_end
        if (self.Nsegs % 2) == 0:
            seg_idx = int(self.Nsegs / 2)
            global_pos = self.segs[seg_idx].front_end
        else:
            seg_idx = int((self.Nsegs + 1) / 2)
            global_pos = self.segs[seg_idx].get_position()
        return global_pos

    @param.depends('length', watch=True)
    def update_seg_lengths(self):
        for i in range(self.Nsegs) :
            self.segs[i].length=self.length*self.segment_ratio[i]

    def move_body(self, dx, dy):
        x0, y0 = self.get_position()
        self.set_position((x0 + dx, y0 + dy))
        for i, seg in enumerate(self.segs):
            x, y = seg.get_position()
            seg.set_position((x+dx,y+dy))

    def valid_Dbend_range(self, idx=0):
        if self.Nsegs > idx + 1:
            dang = aux.wrap_angle_to_0(self.segs[idx + 1].get_orientation() - self.segs[idx].get_orientation())
        else:
            dang = 0
        return (-np.pi + dang), (np.pi + dang)

    def set_color(self, colors):
        if len(colors) != self.Nsegs:
            colors = [tuple(colors)] * self.Nsegs
        for seg, col in zip(self.segs, colors):
            seg.color=col





    @property
    def midline_xy(self):
        return [seg.front_end for seg in self.segs] + [self.tail.rear_end]

    @property
    def front_orientation(self):
        return self.head.get_orientation()%(2*np.pi)


    @property
    def rear_orientation(self):
        return self.tail.get_orientation()%(2*np.pi)

    def draw_segs(self, v, **kwargs):
        for seg in self.segs :
            seg.draw(v, **kwargs)


class SegmentedBodySensored(SegmentedBody):

    def __init__(self,**kwargs):

        super().__init__(**kwargs)
        self.sensors = aux.AttrDict()
        self.define_sensor('olfactor', (1, 0))

    @property
    def olfactor_pos(self):
        return self.head.front_end

    @property
    def olfactor_point(self):
        return geometry.Point(self.olfactor_pos[0], self.olfactor_pos[1])

    def define_sensor(self, sensor, pos_on_body):
        x, y = pos_on_body
        for i, (r, cum_r) in enumerate(zip(self.segment_ratio, np.cumsum(self.segment_ratio))):
            if x >= 1 - cum_r:
                seg_idx = i
                local_pos = np.array([x - 1 + cum_r - r / 2, y])
                break
        self.sensors[sensor] = aux.AttrDict({
            'seg_idx': seg_idx,
            'local_pos': local_pos,
            # 'local_pos': local_pos * self.sim_length
        })


    def get_sensor_position(self, sensor):
        d=self.sensors[sensor]
        return self.segs[d.seg_idx].translate(d.local_pos* self.length)

    def add_touch_sensors(self, idx):
        for i in idx:
            self.define_sensor(f'touch_sensor_{i}', self.contour_points[i])


    def draw_sensors(self, v, **kwargs):
        for s, d in self.sensors.items():
            pos = self.segs[d.seg_idx].translate(d.local_pos * self.length)
            v.draw_circle(radius=self.length / 10,
                               position=pos,
                               filled=True, color=(255, 0, 0), width=.1)


