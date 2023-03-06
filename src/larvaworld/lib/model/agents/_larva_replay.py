from copy import deepcopy
import numpy as np

from larvaworld.lib import aux
from larvaworld.lib.aux import nam
from larvaworld.lib.model.agents._larva import Larva
from larvaworld.lib.model.agents.draw_body import draw_body


class LarvaReplay(Larva):
    def __init__(self, model, data, length=0.005, **kwargs):

        c=model.config

        N = data.index.size
        cols=data.columns

        pos_pars = nam.xy(c.point)
        if not set(pos_pars).issubset(cols):
            pos_pars = ['x', 'y']
        self.pos_ar = data[pos_pars].values

        cen_pars = nam.xy('centroid')
        if set(cen_pars).issubset(cols) :
            self.cen_ar = data[cen_pars].values
        else:
            self.cen_ar = np.ones([N, 2]) * np.nan


        mid_pars = [xy for xy in nam.xy(nam.midline(c.Npoints, type='point')) if set(xy).issubset(cols)]
        con_pars = [xy for xy in nam.xy(nam.contour(c.Ncontour)) if set(xy).issubset(cols)]

        self.mid_ar = data[aux.flatten_list(mid_pars)].values.reshape([N, len(mid_pars), 2])
        self.con_ar = data[aux.flatten_list(con_pars)].values.reshape([N, len(con_pars), 2])


        self.bend_ar = np.deg2rad(data['bend'].values) if 'bend' in cols else np.ones(N) * np.nan


        self.front_or_ar = np.deg2rad(
            data['front_orientation'].values) if 'front_orientation' in cols else np.ones(N) * np.nan
        self.rear_or_ar = np.deg2rad(
            data['rear_orientation'].values) if 'rear_orientation' in cols else np.ones(N) * np.nan

        self.chunk_ids = None
        self.real_length = length

        self.Nsegs = model.draw_Nsegs

        kws={
            'model':model,
            'pos':self.pos_ar[0],
            'orientation':self.front_or_ar[0],
            'radius':self.real_length / 2,
            **kwargs

        }
        super().__init__(**kws)

        if self.Nsegs is not None:
            self.segs=aux.generate_segs_offline(self.Nsegs , self.pos, self.orientation, length,
                shape='drosophila_larva', seg_ratio=None, color=self.default_color,
                 mode='default')

            or_pars = aux.nam.orient(aux.nam.midline(self.Nsegs, type='seg'))
            self.or_ar = np.ones([N, self.Nsegs]) * np.nan
            for i, p in enumerate(or_pars):
                if p in cols:
                    self.or_ar[:, i] = np.deg2rad(data[p].values)
        else :

            self.segs=None

        self.color = deepcopy(self.default_color)

        self.beh_ar = np.zeros([N, len(self.behavior_pars)], dtype=bool)
        for i, p in enumerate(self.behavior_pars):
            if p in cols:
                self.beh_ar[:, i] = np.array([not v for v in np.isnan(data[p].values).tolist()])
        self.data = data


    def update_behavior_dict(self):
        d = aux.AttrDict(dict(zip(self.behavior_pars, self.beh_ar[self.model.t, :].tolist())))
        self.color = self.update_color(self.default_color, d)

    def step(self):
        m = self.model
        mid =self.midline = self.mid_ar[m.t].tolist()
        self.contour = self.con_ar[m.t][~np.isnan(self.con_ar[m.t])].reshape(-1, 2)
        self.pos = self.pos_ar[m.t]
        self.cen_pos = self.cen_ar[m.t]
        self.front_or = self.front_or_ar[m.t]
        self.rear_or = self.rear_or_ar[m.t]
        self.bend0 = self.bend_ar[m.t]
        self.trajectory = self.pos_ar[:m.t, :].tolist()
        for p in ['front_orientation_vel0']:
            setattr(self, p, self.data[p].values[m.t] if p in self.data.columns else np.nan)

        if not np.isnan(self.pos).any():
            m.space.move_to(self, self.pos)
        if m.draw_Nsegs is not None:
            segs = self.segs
            if len(mid) == len(segs) + 1:
                for i, seg in enumerate(segs):
                    pos = [np.nanmean([mid[i][j], mid[i + 1][j]]) for j in [0, 1]]
                    o = self.or_ar[m.t, i]
                    seg.update_poseNvertices(pos, o)
            elif len(segs) == 2:
                l1, l2 = self.real_length * self.seg_ratio
                x, y = self.pos
                h_or = self.front_or
                b_or = self.front_or - self.bend0
                p_head = np.array(aux.rotate_point_around_point(origin=[x, y], point=[l1 + x, y], radians=-h_or))
                p_tail = np.array(aux.rotate_point_around_point(origin=[x, y], point=[l2 + x, y], radians=np.pi - b_or))
                pos1 = [np.nanmean([p_head[j], [x, y][j]]) for j in [0, 1]]
                pos2 = [np.nanmean([p_tail[j], [x, y][j]]) for j in [0, 1]]
                segs[0].update_poseNvertices(pos1, h_or)
                segs[1].update_poseNvertices(pos2, b_or)
                self.midline = np.array([p_head, self.pos, p_tail])

    def set_color(self, color):
        self.color = color

    def draw(self, viewer, filled=True):

        pos = self.cen_pos if not np.isnan(self.cen_pos).any() else self.pos

        draw_body(viewer=viewer, model=self.model, pos=pos, midline_xy=self.midline, contour_xy=self.contour,segs=self.segs,
                  radius=self.radius, vertices=None, color=self.color, selected=self.selected,
                  front_or=self.front_or, rear_or=self.rear_or)



