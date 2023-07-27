import numpy as np

from larvaworld.lib import aux
from larvaworld.lib.aux import nam
from larvaworld.lib.model.agents._larva import Larva, LarvaContoured, LarvaSegmented


class LarvaReplay(Larva):
    def __init__(self, midline_array,pos_array,front_orientation_array,rear_orientation_array, **kwargs):


        self.midline_array=midline_array
        self.pos_array=pos_array
        self.front_orientation_array=front_orientation_array
        self.rear_orientation_array=rear_orientation_array


        # self.pos_ar = aux.np2Dtotuples(data[['x', 'y']].values)
        #
        #
        # mid_pars = [xy for xy in nam.xy(nam.midline(c.Npoints, type='point')) if set(xy).issubset(cols)]
        # con_pars = [xy for xy in nam.xy(nam.contour(c.Ncontour)) if set(xy).issubset(cols)]
        #
        # self.mid_ar = data[aux.flatten_list(mid_pars)].values.reshape([N, len(mid_pars), 2])
        # self.con_ar = data[aux.flatten_list(con_pars)].values.reshape([N, len(con_pars), 2])
        #
        #
        # self.bend_ar = np.deg2rad(data['bend'].values) if 'bend' in cols else np.ones(N) * np.nan
        #
        #
        # self.front_or_ar = np.deg2rad(
        #     data['front_orientation'].values) if 'front_orientation' in cols else [None]*N
        # self.rear_or_ar = np.deg2rad(
        #     data['rear_orientation'].values) if 'rear_orientation' in cols else [None]*N
        #
        # a=self.front_or_ar[0]
        kws={
            # 'model':model,
            # 'length':length,
            'pos':self.pos_array[0],
            # 'orientation':a if not np.isnan(a) else 0.0,
            # 'radius':length / 2,
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
        self.pos = self.pos_array[self.model.t]


        if not np.isnan(self.pos).any():
            self.model.space.move_to(self, np.array(self.pos))



    # @property
    # def front_orientation(self):
    #     a=self.front_or_ar[self.model.t]
    #     return a if not np.isnan(a) else 0.0
    #
    #
    # @property
    # def rear_orientation(self):
    #     a = self.rear_or_ar[self.model.t]
    #     return a if not np.isnan(a) else 0.0

    @property
    def midline_xy(self):
        return aux.np2Dtotuples(self.midline_array[self.model.t])

    @property
    def front_orientation(self):
        return self.front_orientation_array[self.model.t]

    @property
    def rear_orientation(self):
        return self.rear_orientation_array[self.model.t]




class LarvaReplayContoured(LarvaReplay, LarvaContoured):
    def __init__(self, contour_array, **kwargs):
        super().__init__(orientation=0,**kwargs)
        self.contour_array = contour_array

    def step(self):
        super().step()
        self.vertices=self.contour_xy

    @property
    def contour_xy(self):
        a = self.contour_array[self.model.t]
        a = a[~np.isnan(a)].reshape(-1, 2)
        return aux.np2Dtotuples(a)



class LarvaReplaySegmented(LarvaReplay, LarvaSegmented):
    def __init__(self, orientation_array, **kwargs):
        super().__init__(orientation=0,**kwargs)
        self.orientation_array = orientation_array
        # N = self.data.index.size
        # or_pars = aux.nam.orient(aux.nam.midline(self.Nsegs, type='seg'))
        # self.or_ar = np.ones([N, self.Nsegs]) * np.nan
        # for i, p in enumerate(or_pars):
        #     if p in self.data.columns:
        #         self.or_ar[:, i] = np.deg2rad(self.data[p].values)
        '''
        if self.Nsegs==2:
            xy=self.data[aux.nam.xy('centroid')].values
            ho=np.deg2rad(self.data['front_orientation'].values)
            bo=np.deg2rad(self.data['bend'].values)
            l1, l2 = self.length * self.seg_ratio

            p1=xy+aux.rotationMatrix(-ho) @ (l1/2,0)
            p2=xy+aux.rotationMatrix(-ho+bo) @ (-l2/2,0)
        elif self.Nsegs==self.model.config.Npoints-1:
            or_pars = aux.nam.orient(aux.nam.midline(self.Nsegs, type='seg'))
            assert aux.cols_exist(or_pars, self.data)
            ors=np.deg2rad(self.data[or_pars].values)
            ps=self.mid_ar
        
        '''





    def step(self):
        super().step()
        mid = self.midline_xy
        ors = self.orientation_array[self.model.t]
        for i, seg in enumerate(self.segs):
            seg.set_position(mid[i])
            try:
                seg.set_orientation(ors[i])
            except:
                pass
        # segs = self.segs
        # if len(mid) == len(segs) + 1:
        #     for i, seg in enumerate(segs):
        #         pos = tuple([np.nanmean([mid[i][j], mid[i + 1][j]]) for j in [0, 1]])
        #         o = self.or_ar[self.model.t, i]
        #         seg.update_poseNvertices(pos, o)
        # elif len(segs) == 2:
        #     l1, l2 = self.length * self.segment_ratio
        #     x, y = self.pos
        #     h_or = self.front_orientation
        #     bb=self.bend_ar[self.model.t]
        #     if not np.isnan(bb):
        #         b_or = h_or - bb
        #
        #         p_head = aux.rotate_point_around_point(origin=[x, y], point=[l1 + x, y], radians=-h_or)
        #         p_tail = aux.rotate_point_around_point(origin=[x, y], point=[l2 + x, y], radians=np.pi - b_or)
        #         pos1 = tuple([np.nanmean([p_head[j], [x, y][j]]) for j in [0, 1]])
        #         pos2 = tuple([np.nanmean([p_tail[j], [x, y][j]]) for j in [0, 1]])
        #         segs[0].update_poseNvertices(pos1, h_or)
        #         segs[1].update_poseNvertices(pos2, b_or)
        #         self.midline = np.array([p_head, self.pos, p_tail])


