from copy import deepcopy

import numpy as np

from lib.aux import functions as fun
from lib.model.agents._larva import Larva
from lib.model.body.controller import BodyReplay


class LarvaReplay(Larva, BodyReplay):
    def __init__(self, unique_id, model, length=5, data=None, **kwargs):
        Larva.__init__(self, unique_id=unique_id, model=model, radius=length / 2, **kwargs)
        m=self.model
        N = m.Nsteps
        self.chunk_ids = None
        self.trajectory = []
        self.color = deepcopy(self.default_color)
        self.real_length = length
        self.pos_ar = data[m.pos_pars].values
        self.pos = self.pos_ar[0]
        if len(m.cen_pars) == 2 :
            self.cen_ar = data[m.cen_pars].values
            self.cen_pos=self.cen_ar[0]
        else :
            self.cen_ar=None
            self.cen_pos=(np.nan, np.nan)

        self.Nsegs = m.draw_Nsegs
        self.mid_ar = data[fun.flatten_list(m.mid_pars)].values.reshape([N, m.Npoints, 2])
        self.con_ar = data[fun.flatten_list(m.con_pars)].values.reshape([N, m.Ncontour, 2])



        vp_beh = [p for p in self.behavior_pars if p in m.chunk_pars]
        self.beh_ar = np.zeros([N, len(self.behavior_pars)], dtype=bool)
        for i, p in enumerate(self.behavior_pars):
            if p in vp_beh:
                self.beh_ar[:, i] = np.array([not v for v in np.isnan(data[p].values).tolist()])

        if self.Nsegs is not None:
            self.ang_ar = np.deg2rad(data[m.ang_pars].values) if m.Nangles > 0 else np.ones([N, m.Nangles]) * np.nan
            self.or_ar = np.deg2rad(data[m.or_pars].values) if m.Nors > 0 else np.ones([N, m.Nors]) * np.nan
            self.bend_ar = np.deg2rad(data['bend'].values) if 'bend' in data.columns else np.ones(N) * np.nan
            self.front_or_ar = np.deg2rad(
                data['front_orientation'].values) if 'front_orientation' in data.columns else np.ones(
                N) * np.nan
            # FIXME Here the sim_length is not divided by 1000 because all xy coords are in mm
            BodyReplay.__init__(self, model, pos=self.pos, orientation=self.or_ar[0][0],
                                initial_length=self.sim_length, length_std=0, Nsegs=self.Nsegs, interval=0)
        self.data = data

    def read_step(self, i):
        self.midline = self.mid_ar[i].tolist()
        self.vertices = self.con_ar[i][~np.isnan(self.con_ar[i])].reshape(-1,2)
        if self.cen_ar is not None :
            self.cen_pos = self.cen_ar[i]
        self.pos = self.pos_ar[i]
        self.trajectory = self.pos_ar[:i, :].tolist()
        self.beh_dict = dict(zip(self.behavior_pars, self.beh_ar[i, :].tolist()))
        if self.Nsegs is not None :
            self.angles = self.ang_ar[i]
            self.orients = self.or_ar[i]


            self.front_orientation = self.front_or_ar[i]
            self.bend = self.bend_ar[i]

            for p in ['front_orientation_vel']:
                setattr(self, p, self.data[p].values[i] if p in self.data.columns else np.nan)

    def step(self):
        m=self.model

        step = m.active_larva_schedule.steps
        self.read_step(step)
        mid = self.midline
        if not np.isnan(self.pos).any():
            m.space.move_agent(self, self.pos)
        if m.color_behavior:
            self.color = self.update_color(self.default_color, self.beh_dict)
        else:
            self.color = self.default_color
        if m.draw_Nsegs is not None:
            segs = self.segs
            if len(mid) == len(segs) + 1:
                for i, seg in enumerate(segs):
                    pos = [np.nanmean([mid[i][j], mid[i + 1][j]]) for j in [0, 1]]
                    o = self.orients[i]
                    seg.set_position(pos)
                    seg.set_orientation(o)
                    seg.update_vertices(pos, o)
            elif len(segs) == 2:
                l1, l2 = [self.sim_length * r for r in self.seg_ratio]
                x, y = self.pos
                h_or = self.front_orientation
                b_or = self.front_orientation - self.bend
                p_head = np.array(fun.rotate_around_point(origin=[x, y], point=[l1 + x, y], radians=-h_or))
                p_tail = np.array(fun.rotate_around_point(origin=[x, y], point=[l2 + x, y], radians=np.pi - b_or))
                pos1 = [np.nanmean([p_head[j], [x, y][j]]) for j in [0, 1]]
                pos2 = [np.nanmean([p_tail[j], [x, y][j]]) for j in [0, 1]]
                segs[0].set_position(pos1)
                segs[0].set_orientation(h_or)
                segs[0].update_vertices(pos1, h_or)
                segs[1].set_position(pos2)
                segs[1].set_orientation(b_or)
                segs[1].update_vertices(pos2, b_or)
                self.midline = np.array([p_head, self.pos, p_tail])

    def draw(self, viewer):
        r,c,m, v=self.radius,self.color,self.model, self.vertices
        if m.draw_contour:
            if self.Nsegs is not None:
                for seg in self.segs:
                    seg.set_color(c)
                    seg.draw(viewer)
            elif len(v) > 0:
                viewer.draw_polygon(v, color=c)

        if m.draw_centroid:
            if not np.isnan(self.cen_pos).any():
                viewer.draw_circle(radius=r / 2, position=self.cen_pos, color=c,width=r / 3)
        if m.draw_midline and m.Npoints > 1:
            try:
                viewer.draw_polyline(self.midline, color=(0, 0, 255), closed=False, width=.15)
                for i, seg_pos in enumerate(self.midline):
                    cc = 255 * i / (len(self.midline) - 1)
                    color = (cc, 255 - cc, 0)
                    viewer.draw_circle(radius=.1, position=seg_pos, color=color)
            except :
                pass
        if m.draw_head:
            try:
                viewer.draw_circle(self.midline[0], r / 2, color=(255, 0, 0), width= r / 6)
            except :
                pass
        if self.selected:
            if len(v) > 0 and not np.isnan(v).any():
                viewer.draw_polygon(v, filled=False, color=m.selection_color, width=r / 5)
            elif not np.isnan(self.pos).any():
                viewer.draw_circle(radius=r, position=self.pos,
                                   filled=False, color=m.selection_color, width=r / 3)

    def set_color(self, color):
        self.color = color