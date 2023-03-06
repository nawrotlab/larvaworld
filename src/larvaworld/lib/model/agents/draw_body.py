import numpy as np
from larvaworld.lib import aux


def draw_body(viewer, model, pos, midline_xy, contour_xy, radius, vertices, color,segs=None, selected=False,
              front_or=None, rear_or=None):
    if model.screen_manager.draw_centroid:
        draw_body_centroid(viewer, pos, radius, color)

    if model.screen_manager.draw_contour:
        if segs is not None:
            draw_body_segments(viewer, segs)
        elif contour_xy is not None :
            draw_body_contour(viewer, contour_xy, color, radius)

    if model.screen_manager.draw_midline:
        draw_body_midline(viewer, midline_xy, radius)

    if model.screen_manager.draw_head:
        draw_body_head(viewer, midline_xy, radius)

    if selected:
        if vertices is not None :
            draw_selected_body(viewer, pos, vertices, radius, model.screen_manager.selection_color)
        elif contour_xy is not None :

            draw_selected_body(viewer, pos, contour_xy, radius, model.screen_manager.selection_color)
        else :
            pass

    draw_orientations = False
    if draw_orientations:
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
            draw_body_orientation(viewer, p0, front_or, radius, 'green')
            draw_body_orientation(viewer, p1, rear_or, radius, 'red')


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


def draw_body_contour(viewer, contour_xy, color, radius):
    try:
        viewer.draw_polygon(contour_xy, color=color, filled=True, width=radius / 5)
    except:
        pass

def draw_body_segments(viewer,segs):
    for seg in segs:
        for vertices in seg.vertices:
            viewer.draw_polygon(vertices, filled=True, color=seg.color)


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


def draw_selected_body(viewer, pos, xy_bounds, radius, color):
    try:
        if len(xy_bounds) > 0 and not np.isnan(xy_bounds).any():
            viewer.draw_polygon(xy_bounds, filled=False, color=color, width=radius / 5)
        elif not np.isnan(pos).any():
            viewer.draw_circle(pos, radius=radius, filled=False, color=color, width=radius / 3)
    except:
        pass


def draw_body_orientation(viewer, pos, orientation, radius, color):
    viewer.draw_line(pos, aux.xy_projection(pos, orientation, radius * 3),
                     color=color, width=radius / 10)
    # viewer.draw_line(self.midline[-1], xy_aux.xy_projection(self.midline[-1], self.rear_orientation, self.radius * 3),
    #                  color=self.color, width=self.radius / 10)



    # Using the forward Euler method to compute the next theta and theta'

    '''Here we implement the lateral oscillator as described in Wystrach(2016) :
    We use a,b,c,d parameters to be able to generalize. In the paper a=1, b=2*z, c=k, d=0

    Quoting  : where z =n / (2* sqrt(k*g) defines the damping ratio, with n the damping force coefficient, 
    k the stiffness coefficient of a linear spring and g the muscle gain. We assume muscles on each side of the body 
    work against each other to change the heading and thus, in this two-dimensional model, the net torque produced is 
    taken to be the difference in spike rates of the premotor neurons E_L(t)-E_r(t) driving the muscles on each side. 

    Later : a level of damping, for which we have chosen an intermediate value z =0.5
    In the table of parameters  : k=1

    So a=1, b=1, c=n/4g=1, d=0 
    '''

