import math

import numpy as np
import pandas as pd


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


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def concat_datasets(ddic, key='end', unit='sec'):
    dfs = []
    for l, d in ddic.items():
        if key == 'end':
            try:
                df = d.endpoint_data
            except:
                df = d.read(key='end')
        elif key == 'step':
            try:
                df = d.step_data
            except:
                df = d.read(key='step')
        else :
            raise
        df['DatasetID'] = l
        df['GroupID'] = d.group_id
        dfs.append(df)
    df0 = pd.concat(dfs)
    if key == 'step':
        df0.reset_index(level='Step', drop=False, inplace=True)
        dts = np.unique([d.config['dt'] for l, d in ddic.items()])
        if len(dts) == 1:
            dt = dts[0]
            dic = {'sec': 1, 'min': 60, 'hour': 60 * 60, 'day': 24 * 60 * 60}
            df0['Step'] *= dt / dic[unit]
    return df0


def moving_average(a, n=3):
    return np.convolve(a, np.ones((n,)) / n, mode='same')


def mdict2df(mdict, columns=['symbol', 'value', 'description']):
    data = []
    for k, p in mdict.items():
        entry = [getattr(p, col) for col in columns]
        data.append(entry)
    df = pd.DataFrame(data, columns=columns)
    df.set_index(columns[0], inplace=True)
    return df


def body(points, start=None, stop=None):
    if start is None:
        start = [1, 0]
    if stop is None:
        stop = [0, 0]
    xy = np.zeros([len(points) * 2 + 2, 2]) * np.nan
    xy[0, :] = start
    xy[len(points) + 1, :] = stop
    for i in range(len(points)):
        x, y = points[i]
        xy[1 + i, :] = x, y
        xy[-1 - i, :] = x, -y
    return xy
