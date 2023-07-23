import math

import numpy as np
from larvaworld.lib import aux


def draw_body(v, model, pos,color,radius, midline_xy=None, contour_xy=None,  vertices=None,
              segs=None, selected=False,
              front_or=None, rear_or=None, sensors=None, length=None):
    if model.screen_manager.draw_centroid:
        draw_body_centroid(v, pos, radius, color)

    if model.screen_manager.draw_contour:
        if segs is not None:
            draw_body_segments(v, segs)
        elif contour_xy is not None :
            draw_body_contour(v, contour_xy, color, radius)

    if model.screen_manager.draw_midline:
        draw_body_midline(v, midline_xy, radius)

    if model.screen_manager.draw_head:
        draw_body_head(v, midline_xy, radius)

    if selected:
        if vertices is not None :
            draw_selected_body(v, pos, vertices, radius, model.screen_manager.selection_color)
        elif contour_xy is not None :

            draw_selected_body(v, pos, contour_xy, radius, model.screen_manager.selection_color)
        else :
            pass

    if model.screen_manager.draw_orientations:
        if not any(np.isnan(np.array(midline_xy).flatten())):
            Nmid=len(midline_xy)
            p0=midline_xy[int(Nmid/2)]
            p1=midline_xy[int(Nmid/2)+1]
            if front_or is None and rear_or is None:
                if segs is not None :
                    front_or=segs[0].get_orientation()
                    rear_or=segs[-1].get_orientation()
                else :
                    return
            # draw_body_orientation(viewer, self.midline[1], self.head_orientation, self.radius, 'green')
            # draw_body_orientation(viewer, self.midline[-2], self.tail_orientation, self.radius, 'red')
            draw_body_orientation(v, p0, front_or, radius, 'green')
            draw_body_orientation(v, p1, rear_or, radius, 'red')

    if model.screen_manager.draw_sensors:
        if sensors :
            draw_sensors(v,sensors, radius, segs, length)


def draw_sensors(viewer, sensors, radius, segs, length):
    for s, d in sensors.items():
        pos=segs[d.seg_idx].get_world_point(d.local_pos* length)
        viewer.draw_circle(radius=radius / 10,
                           position=pos,
                           filled=True, color=(255, 0, 0), width=.1)

def draw_body_midline(viewer, midline_xy, radius):
    try:
        mid = midline_xy
        r = radius
        if not any(np.isnan(np.array(mid).flatten())):
            Nmid = len(mid)
            viewer.draw_polyline(mid, color=(0, 0, 255), closed=False, width=r / 10)
            for i, xy in enumerate(mid):
                c = 255 * i / (Nmid - 1)
                viewer.draw_circle(xy, r / 15, color=(c, 255 - c, 0), width=r / 20)
    except:
        pass


def draw_body_contour(v, contour_xy, color, radius):
    try:
        v.draw_polygon(contour_xy, color=color, filled=True, width=radius / 5)
    except:
        pass

def draw_body_segments(v,segs):
    for seg in segs:
        v.draw_polygon(seg.vertices, filled=True, color=seg.color)


def draw_body_centroid(v, pos, radius, color):
    try:
        v.draw_circle(pos, radius / 2, color=color, width=radius / 3)
    except:
        pass


def draw_body_head(v, midline_xy, radius):
    try:
        pos = midline_xy[0]
        v.draw_circle(pos, radius / 2, color=(255, 0, 0), width=radius / 6)
    except:
        pass


def draw_selected_body(v, pos, xy_bounds, radius, color):
    try:
        if len(xy_bounds) > 0 and not np.isnan(xy_bounds).any():
            v.draw_polygon(xy_bounds, filled=False, color=color, width=radius / 5)
        elif not np.isnan(pos).any():
            v.draw_circle(pos, radius=radius, filled=False, color=color, width=radius / 3)
    except:
        pass


def draw_body_orientation(v, pos, orientation, radius, color):
    dst=radius * 3
    pos2=[pos[0] + math.cos(orientation) * dst,pos[1] + math.sin(orientation) * dst]

    v.draw_line(pos, pos2,color=color, width=radius / 10)
    # viewer.draw_line(self.midline[-1], xy_aux.xy_projection(self.midline[-1], self.rear_orientation, self.radius * 3),
    #                  color=self.color, width=self.radius / 10)



    # Using the forward Euler method to compute the next theta and theta'


