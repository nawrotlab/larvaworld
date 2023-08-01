import numpy as np

from larvaworld.lib import aux
from larvaworld.lib.aux import nam
from larvaworld.lib.model.agents._larva import Larva, LarvaContoured, LarvaSegmented


class LarvaReplay(Larva):
    def __init__(self, data, **kwargs):
        self.data=data
        fo0=self.data.front_orientation[0]
        if np.isnan(fo0):
            fo0=0

        super().__init__(pos=self.data.pos[0],orientation=fo0,**kwargs)





    def step(self):
        self.pos = self.data.pos[self.model.t]
        self.trajectory.append(self.pos)
        self.orientation_trajectory.append(self.front_orientation)
        if not np.isnan(self.pos).any():
            self.model.space.move_to(self, np.array(self.pos))


    @property
    def midline_xy(self):
        return aux.np2Dtotuples(self.data.midline[self.model.t])

    @property
    def front_orientation(self):
        return self.data.front_orientation[self.model.t]

    @property
    def rear_orientation(self):
        return self.data.rear_orientation[self.model.t]




class LarvaReplayContoured(LarvaReplay, LarvaContoured):


    def step(self):
        super().step()
        self.vertices=self.contour_xy

    @property
    def contour_xy(self):
        a = self.data.contour[self.model.t]
        a = a[~np.isnan(a)].reshape(-1, 2)
        return aux.np2Dtotuples(a)



class LarvaReplaySegmented(LarvaReplay, LarvaSegmented):

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
        ors = self.data.seg_orientations[self.model.t]
        for i, seg in enumerate(self.segs):
            seg.set_position(mid[i])
            try:
                seg.set_orientation(ors[i])
            except:
                pass