import math

import numpy as np
from scipy.signal import spectrogram
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import split


def LvsRtoggle(side):
    if side == 'Left':
        return 'Right'
    elif side == 'Right':
        return 'Left'
    else:
        raise ValueError(f'Argument {side} is neither Left nor Right')


def mutate_value(v, range, scale=0.01):
    r0, r1 = range
    return np.clip(np.random.normal(loc=v, scale=scale * np.abs(r1 - r0)), a_min=r0, a_max=r1).astype(float)


def circle_to_polygon(sides, radius, rotation=0, translation=None):
    one_segment = np.pi * 2 / sides

    points = [
        (math.sin(one_segment * i + rotation) * radius,
         math.cos(one_segment * i + rotation) * radius)
        for i in range(sides)]

    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]

    return np.array(points)


def inside_polygon(points, tank_polygon):
    return all([tank_polygon.contains(Point(x, y)) for x, y in points])
    # # // p is your point, p.x is the x coord, p.y is the y coord
    # Xmin, Xmax, Ymin, Ymax = space_edges_for_screen
    # # x, y = point
    #
    # if all([(x < Xmin or x > Xmax or y < Ymin or y > Ymax) for x, y in points]):
    #     # // Definitely not within the polygon!
    #     return False
    # else:
    #     # point = Point(x, y)
    #     # print(polygon.contains(point))
    #     return all([Polygon(vertices).contains(Point(x, y)) for x, y in points])


def body(points, start=[1, 0], stop=[0, 0]):
    xy = np.zeros([len(points) * 2+2, 2]) * np.nan
    xy[0, :] = start
    xy[len(points) + 1, :] = stop
    for i in range(len(points)):
        x, y = points[i]
        xy[1 + i, :] = x, y
        xy[-1 - i, :] = x, -y
    return xy

# Create a bilaterally symmetrical 2D contour with the long axis along x axis

# Arguments :
#   points : the points above x axis through which the contour passes
#   start : the front end of the body
#   stop : the rear end of the body


# Segment body in N segments of given ratios via vertical lines
def segment_body(N, xy0, seg_ratio=None, centered=True, closed=False):
    # If segment ratio is not provided, generate equal-length segments
    if seg_ratio is None:
        seg_ratio = [1 / N] * N

    # Create a polygon from the given body contour
    p = Polygon(xy0)
    # Get maximum y value of contour
    y0 = np.max(p.exterior.coords.xy[1])

    # Segment body via vertical lines
    ps = [p]
    for cum_r in np.cumsum(seg_ratio):
        l = LineString([(1 - cum_r, y0), (1 - cum_r, -y0)])
        new_ps = []
        for p in ps:
            new_p = [new_p for new_p in split(p, l)]
            new_ps += new_p
        ps = new_ps

    # Sort segments so that front segments come first
    ps.sort(key=lambda x: x.exterior.xy[0], reverse=True)

    # Transform to 2D array of coords
    ps = [p.exterior.coords.xy for p in ps]
    ps = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]

    # Center segments around 0,0
    if centered:
        for i, (r, cum_r) in enumerate(zip(seg_ratio, np.cumsum(seg_ratio))):
            ps[i] -= [(1 - cum_r) + r / 2, 0]
            # pass

    # Put front point at the start of segment vertices. Drop duplicate rows
    for i in range(len(ps)):
        if i == 0:
            ind = np.argmax(ps[i][:, 0])
            ps[i] = np.flip(np.roll(ps[i], -ind - 1, axis=0), axis=0)
        else:
            ps[i] = np.flip(np.roll(ps[i], 1, axis=0), axis=0)
        _, idx = np.unique(ps[i], axis=0, return_index=True)
        ps[i] = ps[i][np.sort(idx)]
        if closed :
            ps[i]=np.concatenate([ps[i], [ps[i][0]]])
        # ps[i][:,0]+=0.5
    # ps[0]=np.vstack([ps[0], np.array(ps[0][0,:])])

    # print(ps)
    return ps

def generate_seg_shapes(Nsegs,  points,seg_ratio=None, centered=True, closed=False, **kwargs):
    if seg_ratio is None :
        seg_ratio = np.array([1 / Nsegs] * Nsegs)
    ps = segment_body(Nsegs, np.array(points), seg_ratio=seg_ratio, centered=centered, closed=closed)
    seg_vertices = [np.array([p]) for p in ps]
    return seg_vertices

def rearrange_contour(ps0):
    ps_plus = [p for p in ps0 if p[1] >= 0]
    ps_plus.sort(key=lambda x: x[0], reverse=True)
    ps_minus = [p for p in ps0 if p[1] < 0]
    ps_minus.sort(key=lambda x: x[0], reverse=False)
    return ps_plus + ps_minus

def compute_dst(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def freq(d, dt, range=[0.7, 1.8]) :
    try:
        f, t, Sxx = spectrogram(d, fs=1 / dt)
        # keep only frequencies of interest
        f0, f1 = range
        valid = np.where((f >= f0) & (f <= f1))
        f = f[valid]
        Sxx = Sxx[valid, :][0]
        max_freq = f[np.argmax(np.nanmedian(Sxx, axis=1))]
    except:
        max_freq = np.nan
    return max_freq

def get_tank_polygon(c, k=0.97, return_polygon=True):
    X, Y = c.env_params.arena.arena_dims
    shape = c.env_params.arena.arena_shape
    if shape == 'circular':
        # This is a circle_to_polygon shape from the function
        tank_shape = circle_to_polygon(60, X / 2)
    elif shape == 'rectangular':
        # This is a rectangular shape
        tank_shape = np.array([(-X / 2, -Y / 2),
                               (-X / 2, Y / 2),
                               (X / 2, Y / 2),
                               (X / 2, -Y / 2)])
    if return_polygon :
        return Polygon(tank_shape * k)
    else :
        # tank_shape=np.insert(tank_shape,-1,tank_shape[0,:])
        return tank_shape