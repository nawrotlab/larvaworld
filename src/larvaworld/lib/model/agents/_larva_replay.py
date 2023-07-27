from copy import deepcopy
import numpy as np

from larvaworld.lib import aux
from larvaworld.lib.aux import nam
from larvaworld.lib.model.agents._larva import Larva, LarvaContoured, LarvaSegmented
from larvaworld.lib.model.agents.draw_body import draw_body
from larvaworld.lib.param import SegmentedBody


class LarvaReplay(Larva):
    def __init__(self, model, data, length=0.005, **kwargs):

        c=model.config

        N = data.index.size
        cols=data.columns
        self.data=data
        # pos_pars = nam.xy(c.point)
        # if not set(pos_pars).issubset(cols):
        #     pos_pars = ['x', 'y']
        # self.pos_ar = data[pos_pars].values
        # self.pos_ar = list(zip(self.pos_ar[:,0], self.pos_ar[:,1]))

        # cen_pars = nam.xy('centroid')
        # if set(cen_pars).issubset(cols) :
        #     self.cen_ar = data[cen_pars].values
        # else:
        #     self.cen_ar = np.ones([N, 2]) * np.nan

        self.pos_ar = aux.np2Dtotuples(data[['x', 'y']].values)


        mid_pars = [xy for xy in nam.xy(nam.midline(c.Npoints, type='point')) if set(xy).issubset(cols)]
        con_pars = [xy for xy in nam.xy(nam.contour(c.Ncontour)) if set(xy).issubset(cols)]

        self.mid_ar = data[aux.flatten_list(mid_pars)].values.reshape([N, len(mid_pars), 2])
        self.con_ar = data[aux.flatten_list(con_pars)].values.reshape([N, len(con_pars), 2])


        self.bend_ar = np.deg2rad(data['bend'].values) if 'bend' in cols else np.ones(N) * np.nan


        self.front_or_ar = np.deg2rad(
            data['front_orientation'].values) if 'front_orientation' in cols else [None]*N
        self.rear_or_ar = np.deg2rad(
            data['rear_orientation'].values) if 'rear_orientation' in cols else [None]*N

        a=self.front_or_ar[0]
        kws={
            'model':model,
            'length':length,
            'pos':self.pos_ar[0],
            'orientation':a if not np.isnan(a) else 0.0,
            'radius':length / 2,
            **kwargs

        }
        super().__init__(**kws)

        # if self.Nsegs is not None:
        #     from larvaworld.lib.model.agents.segmented_body import DefaultSegment
        #     self.segs = aux.generate_segs(N=self.Nsegs, pos=self.pos, orient=self.orientation,
        #                                     ratio=None, l=length,color=self.default_color,
        #                               body_plan='drosophila_larva', segment_class=DefaultSegment)
        #
        #
        #
        #     or_pars = aux.nam.orient(aux.nam.midline(self.Nsegs, type='seg'))
        #     self.or_ar = np.ones([N, self.Nsegs]) * np.nan
        #     for i, p in enumerate(or_pars):
        #         if p in cols:
        #             self.or_ar[:, i] = np.deg2rad(data[p].values)
        # else :
        #
        #     self.segs=None

        # self.color = deepcopy(self.default_color)


        # self.data = data




    def step(self):
        # m = self.model
        # mid =self.midline = self.mid_ar[m.t].tolist()
        # self.contour = self.con_ar[m.t][~np.isnan(self.con_ar[m.t])].reshape(-1, 2)
        self.pos = self.pos_ar[self.model.t]
        # self.bend0 = self.bend_ar[m.t]
        self.trajectory.append(self.pos)
        # self.trajectory = self.pos_ar[:m.t]
        # for p in ['front_orientation_vel0']:
        #     setattr(self, p, self.data[p].values[m.t] if p in self.data.columns else np.nan)

        if not np.isnan(self.pos).any():
            self.model.space.move_to(self, np.array(self.pos))




    # def draw(self, v, filled=True):
    #
    #
    #     draw_body(v=v, pos=self.pos, midline_xy=self.midline, contour_xy=self.contour,segs=self.segs,
    #               radius=self.radius, vertices=None, color=self.color, selected=self.selected,
    #               front_or=self.front_orientation, rear_or=self.rear_orientation)

    # @property
    # def get_position(self):
    #     return self.pos_ar[self.model.t]

    @property
    def front_orientation(self):
        a=self.front_or_ar[self.model.t]
        return a if not np.isnan(a) else 0.0


    @property
    def rear_orientation(self):
        a = self.rear_or_ar[self.model.t]
        return a if not np.isnan(a) else 0.0

    @property
    def midline_xy(self):
        return aux.np2Dtotuples(self.mid_ar[self.model.t])

    @property
    def contour_xy(self):
        a = self.con_ar[self.model.t]
        a=a[~np.isnan(a)].reshape(-1, 2)
        # raise(a)
        return aux.np2Dtotuples(a)


class LarvaReplayContoured(LarvaReplay, LarvaContoured):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self):
        super().step()
        self.vertices=self.contour_xy



class LarvaReplaySegmented(LarvaReplay, LarvaSegmented):
    def __init__(self, model,**kwargs):
        # LarvaReplay.__init__(self, **kwargs)
    # def __init__(self, model, data, length=0.005, **kwargs):
        super().__init__(model=model,Nsegs=model.p.draw_Nsegs,**kwargs)
        or_pars = aux.nam.orient(aux.nam.midline(self.Nsegs, type='seg'))
        self.or_ar = np.ones([self.data.index.size, self.Nsegs]) * np.nan
        for i, p in enumerate(or_pars):
            if p in self.data.columns:
                self.or_ar[:, i] = np.deg2rad(self.data[p].values)

    def step(self):
        super().step()
        mid = self.midline_xy
        segs = self.segs
        if len(mid) == len(segs) + 1:
            for i, seg in enumerate(segs):
                pos = [np.nanmean([mid[i][j], mid[i + 1][j]]) for j in [0, 1]]
                o = self.or_ar[self.model.t, i]
                seg.update_poseNvertices(pos, o)
        elif len(segs) == 2:
            l1, l2 = self.real_length * self.seg_ratio
            x, y = self.pos
            h_or = self.front_orientation
            b_or = h_or - self.bend0
            p_head = aux.rotate_point_around_point(origin=[x, y], point=[l1 + x, y], radians=-h_or)
            p_tail = aux.rotate_point_around_point(origin=[x, y], point=[l2 + x, y], radians=np.pi - b_or)
            pos1 = [np.nanmean([p_head[j], [x, y][j]]) for j in [0, 1]]
            pos2 = [np.nanmean([p_tail[j], [x, y][j]]) for j in [0, 1]]
            segs[0].update_poseNvertices(pos1, h_or)
            segs[1].update_poseNvertices(pos2, b_or)
            self.midline = np.array([p_head, self.pos, p_tail])
