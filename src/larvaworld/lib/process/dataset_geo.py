'''
4 May 2023

Script that uses the movingpandas and geopandas libraries to facilitate spatial data processing.
Analysis starts from scratch meaning it only uses the primary tracked xy coordinates of the midline and contour.
The data used for this illustration is loaded from a stored reference dataset under the RefID.
Only the relevant columns of the double-index stepwise pandas dataframe are used.
See below for further explanation
'''
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
import shapely as shp
from datetime import datetime, timedelta

from pint_pandas import PintType
import copy
import warnings

from larvaworld.lib.process.dataset import BaseLarvaDataset

warnings.filterwarnings('ignore')


#mpd.show_versions()
# Initialize the larvaworld registry and load the sample dataset
from larvaworld.lib import reg, aux

class GeoLarvaDataset(BaseLarvaDataset,mpd.TrajectoryCollection):
    '''
    An alternative mode to maintain a dataset is this "trajectory" mode inheriting from "movingpandas.TrajectoryCollection"
     and adjusted to the needs of the larva-tracking data format

    It has the following powerful aspects :
     -  Straightforward computation of distance, velocity, acceleration and other spatial metrics.
        We also compute the scaled metrics by dividing by the agent's body length
     -  Both the timeseries and endpoint data are kept in pintpandas dataframes supporting SI units
     -  Shapely objects for easy drawing. These are stored as normal columns in the timeseries Geo dataframes.
        All columns except shapely objects that do not support units have a defined unit dtype.



    '''

    def __init__(self, step=None,dt=None,**kwargs):
        if step is not None:
            self.init_mpd(step, dt=dt)
        BaseLarvaDataset.__init__(self, **kwargs)

    # @property
    # def default_filename(self):
    #     return 'geodata'


    def init_gdf(self,step,dt):
        if len(step.index.names) != 1 or 'datetime' not in step.index.names:
            max_tick = step[['x', 'y']].dropna().index.unique('Step').max()
            step = step.query(f'Step<={max_tick}')

            step = step.reset_index()
            t = 't'
            if 'datetime' not in step.columns:
                if t not in step.columns:
                    if 'Step' in step.columns and dt is not None:
                        step[t] = step['Step'] * dt
                    else:
                        raise

                step['datetime'] = pd.to_datetime(step[t], unit='s')
                step[t] = step[t].astype({t: PintType('pint[s]')})
            step = step.set_index('datetime')

        if 'xy' not in step.columns:
            assert aux.cols_exist(['x', 'y'], step)
            step['xy'] = gpd.points_from_xy(step['x'], step['y'])
        gdf = gpd.GeoDataFrame(step)
        gdf = gdf.set_geometry('xy')
        return gdf

    def init_mpd(self, step,dt):
        gdf = self.init_gdf(step, dt)
        mpd.TrajectoryCollection.__init__(self, gdf, traj_id_col='AgentID')

        for tr in self:
            tr.df = reg.par.df_to_pint(tr.df)
            tr.df[['x', 'y']] = tr.df[['x', 'y']].astype({p: self.spatial_pint_unit for p in ['x', 'y']})
            # print(tr.df.xy.iloc[-1])
        # raise



    def set_data(self,step=None, end=None,**kwargs):
        '''
        Drop the Nan timesteps. This is extremely convenient as the geopandas handles unequal trajectories easily
        Build three shapely geometries :
         - The midline xy points are converted into a LineString
          - The contour xy points are converted into a Polygon
          - The position xy is converted into a Point
        All of these are timeseries over the entire dataset duration.
        Eventually the position xy timeseries is set to be the required geometry for the Trajectory instance of geopandas.
        By dropping the index levels to columns we can specify :
         - a column name "t" for timing. This is converted to a datetime type by using the timestep and the index ticks
        - a column name "AgentID" that is used to group the trajectories by agent

        Straightforward computation of the midline length at every timestep by taking the Linestring length.
        The mean length will be used for scaling
        '''



        c = self.config
        if step is not None:
            if not hasattr(self, 'trajectories'):
                self.init_mpd(step,c.dt)
                end = self.build_endpoint_data()
                self.step_data=self.get_step_data()

            else :
                self.step_data = step.sort_index(level=self.param.step_data.levels)

        if end is not None:
            self.endpoint_data=end.sort_index()



    @property
    def traj_dic(self):
        return aux.AttrDict({traj.id: traj for traj in self})


    @property
    def dt_mag(self):
        assert self.config.dt is not None
        _dt = self.config.dt * PintType.ureg.s
        return _dt.magnitude


    @ property
    def iter(self):
        for id in list(self.traj_dic.keys()) :
            yield self.traj_dic[id],self.endpoint_data.loc[id]


    def add_transposed_traj(self, name='xy_origin'):
        '''
        Computing also the transosed trajectory that starts from the center.
        Plotting the original and the transposed trajectories
        '''
        for tr in self:
            xy0 = tr.get_start_location()
            tr.df[name] = tr.df['xy'].apply(shp.affinity.translate, xoff=xy0.x, yoff=xy0.y)

    def add_speed(self,name=None,**kwargs):
        if name is None :
            name=reg.getPar('v')

        for tr in self:
            tr.add_speed(name=name,**kwargs)
            tr.df = tr.df.loc[tr.df['xy'] != None]
        self.set_dtype(name, self.spatial_unit / PintType.ureg.s)

    def add_distance(self,name=None,**kwargs):
        if name is None :
            name=reg.getPar('d')
        for tr in self:
            tr.add_distance(name=name,**kwargs)
            tr.df = tr.df.loc[tr.df['xy'] != None]
        self.set_dtype(name, self.spatial_unit)

    def add_acceleration(self,name=None,**kwargs):
        if name is None :
            name=reg.getPar('a')
        for tr in self:
            tr.add_acceleration(name=name,**kwargs)
            tr.df = tr.df.loc[tr.df['xy'] != None]
        self.set_dtype(name, self.spatial_unit/PintType.ureg.s**2)

    def scale_to_length(self, pars=None,ks=None):
        if 'length' not in self.endpoint_data.keys():
            return
        if pars is None and ks is not None:
            pars=reg.getPar(ks)
        valid=[p for p in pars if self.cols_exist_in_all_traj([p])]
        for tr,e in self.iter:
            l=e['length']
            for p in valid :
                tr.df[aux.nam.scal(p)]=tr.df[p]/l


    def get_means(self,pars=None,ks=None):
        if pars is None and ks is not None:
            pars=reg.getPar(ks)
        valid = [p for p in pars if self.cols_exist_in_all_traj([p])]
        for p in valid:
            p_mu = aux.nam.mean(p)
            self.endpoint_data[p_mu] = {tr.id: np.mean(tr.df[p].values) for tr in self}
            self.endpoint_data[p_mu] = self.endpoint_data[p_mu].astype({p_mu: self.dtypes[p]})

    @ property
    def dtypes(self):
        return pd.concat([traj.df.dtypes for traj in self]).drop_duplicates()

    def drop_xy_Nones(self):
        for tr in self:
            tr.df = tr.df.loc[tr.df['xy'] != None]

    def detect_pauses(self,max_scaled_diameter=0.3, min_duration=timedelta(seconds=1)):
        dic=aux.AttrDict({'times' : [], 'segments' : [], 'points' : []})
        for traj in self:
            l=self.endpoint_data['length'].loc[traj.id].magnitude
            kws={'max_diameter' : max_scaled_diameter*l, 'min_duration' : min_duration}
            D = mpd.TrajectoryStopDetector(traj)
            dic.times.append(D.get_stop_time_ranges(**kws))
            dic.segments.append(D.get_stop_segments(**kws))
            dic.points.append(D.get_stop_points(**kws))
        self.epoch_dict['pause']=dic

    @ property
    def spatial_unit(self):
        return self.spatial_pint_unit.units

    @property
    def spatial_pint_unit(self):
        return PintType(f'pint[m]')
        # return PintType(f'pint[{self.config.u}]')

    @property
    def temporal_pint_unit(self):
        return PintType(f'pint[s]')


    def cols_exist_in_all_traj(self,cols):
        return all([aux.cols_exist(cols,traj.df) for traj in self])



    def time_to_datetime(self, t):
        return pd.to_datetime(t,unit='s')

    def get_locations_at(self, t):
        if t not in ['start', 'end'] :
            if not isinstance(t,datetime):
                t=self.time_to_datetime(t)
        return super().get_locations_at(t)

    def get_locations_at_tick(self, tick):
        return self.get_locations_at(tick*self.dt_mag)

    def get_segments_between(self, t1, t2):
        if t1 not in ['start', 'end']:
            if not isinstance(t1,datetime):
                t1=self.time_to_datetime(t1)
        if t2 not in ['start', 'end']:
            if not isinstance(t2,datetime):
                t2=self.time_to_datetime(t2)
        return super().get_segments_between(t1, t2)

    def get_complete_segments_between(self, t1, t2):
        if t1 in ['start', 'end'] or t2 in ['start', 'end']:
            raise
        tt0, tt1 = reg.getPar(['t0', 't_fin'])
        return [tr.df[self.time_to_datetime(t1):self.time_to_datetime(t2)] for tr,e in self.iter if e[tt0].magnitude<t1 and e[tt1].magnitude>=t2]




    def get_segments_between_ticks(self, tick1, tick2):
        return self.get_segments_between(tick1*self.dt_mag,tick2*self.dt_mag)

    def get_length_from_traj_with_nans(self, traj):
        try :
            return traj.get_length()
        except :
            return None

    def build_endpoint_data(self, e=None):
        if e is None :
            e =pd.DataFrame(index=list(self.traj_dic.keys()))
        e.index.name = 'AgentID'
        cum_d,cum_t=reg.getPar(['cum_d','cum_t'])

        if cum_d not in e.columns:
            e[cum_d] = {traj.id: self.get_length_from_traj_with_nans(traj) for traj in self}
        if cum_t not in e.columns:
            e[cum_t] = {traj.id: traj.get_duration().total_seconds() for traj in self}

        t0, t1 = reg.getPar(['t0', 't_fin'])
        if t0 not in e.columns and t1 not in e.columns:
            e[t0] = {traj.id: mpd.trajectory.to_unixtime(traj.df.index.min()) for traj in self}
            e[t1] = {traj.id: mpd.trajectory.to_unixtime(traj.df.index.max()) for traj in self}
        x0, x1 = reg.getPar(['x0', 'x_fin'])
        y0, y1 = reg.getPar(['y0', 'y_fin'])

        # print([tr.df.index.max() for tr in self])
        if all([p not in e.columns for p in [x0,x1,y0,y1]]):
            e[x0] = {tr.id: tr.df.xy.iloc[0].x for tr in self}
            e[y0] = {tr.id: tr.df.xy.iloc[0].y for tr in self}
            # try:
            e[x1] = {tr.id: tr.df.xy.iloc[-1].x for tr in self}
            e[y1] = {tr.id: tr.df.xy.iloc[-1].y for tr in self}
            # except:
            #     e[x1] = {tr.id: tr.df.xy.iloc[-2].x for tr in self}
            #     e[y1] = {tr.id: tr.df.xy.iloc[-2].y for tr in self}


        spatial_ps=[x0,x1,y0,y1]+[cum_d]
        e[spatial_ps] = e[spatial_ps].astype({p: self.spatial_pint_unit for p in spatial_ps})
        temporal_ps = [t0, t1] + [cum_t]
        e[temporal_ps] = e[temporal_ps].astype({p: self.temporal_pint_unit for p in temporal_ps})

        e['group'] = self.config.group_id
        return e

    def load_midline(self, drop=True, keep_midline_LineString=False):
        l='length'
        xy_flat=self.config.midline_xy
        xy_pairs=xy_flat.in_pairs
        if self.cols_exist_in_all_traj(xy_flat):
            for tr in self:
                if keep_midline_LineString:
                    tr.df['midline'] = tr.df.apply(lambda r: shp.geometry.LineString( [(r[x], r[y]) for [x, y] in xy_pairs]), axis=1)
                    tr.df[l] = tr.df.apply(lambda r: r.midline.length, axis=1)
                else:
                    tr.df[l] = tr.df.apply(lambda r: shp.geometry.LineString( [(r[x], r[y]) for [x, y] in xy_pairs]).length, axis=1)
                if drop :
                    tr.df=tr.df.drop(columns=xy_flat)

            self.endpoint_data[l] = {traj.id: traj.df[l].values.mean() for traj in self}
            self.set_dtype(l, self.spatial_unit)

    def load_contour(self, drop=True):
        xy_flat = self.config.contour_xy
        xy_pairs = xy_flat.in_pairs
        if self.cols_exist_in_all_traj(xy_flat):
            for tr in self:
                tr.df['contour'] = tr.df.apply(lambda r: shp.geometry.Polygon([(r[x], r[y]) for [x, y] in xy_pairs]), axis=1)
                tr.df['area'] = tr.df.apply(lambda r: r.contour.area, axis=1)
                tr.df['centroid'] = tr.df.apply(lambda r: r.contour.centroid, axis=1)
                if drop :
                    tr.df=tr.df.drop(columns=xy_flat)
            self.set_dtype(cols='area',units=self.spatial_unit**2)

    def set_dtype(self,cols,units):
        #print(cols,units)
        if not isinstance(cols,list):
            cols=[cols]
        if not isinstance(units,list):
            units=[PintType(units)]*len(cols)
        elif len(units)!=len(cols):
            raise
        else :
            units=[PintType(u) for u in units]
        cols_end=aux.existing_cols(cols,self.endpoint_data)


        pint_dtypes=dict(zip(cols,units))
        if len(cols_end)>0 :
            self.endpoint_data[cols_end]=self.endpoint_data[cols_end].astype({col:pint_dtypes[col] for col in cols_end})
        for tr in self :
            cols_traj = aux.existing_cols(cols, tr.df)
            if len(cols_traj) > 0:
                tr.df[cols_traj] = tr.df[cols_traj].astype({col:pint_dtypes[col] for col in cols_traj})





    @classmethod
    def from_ID(cls, refID, **kwargs):
        # c = reg.getRef(refID)

        # cc=d.config.get_copy()
        # c.update(**kwargs)

        # inst = cls(load_data=False, config=reg.getRef(refID))

        d = reg.loadRef(refID)
        d.load(h5_ks=['midline', 'contour'])
        step = d.step_data


        inst = cls(step=step,dt=d.config.dt,load_data=False, refID=refID)
        inst.set_data(end=inst.build_endpoint_data(), step=inst.get_step_data())
        return inst

    def path_to_file(self, file='geostep'):
        return f'{self.config.data_dir}/{file}.txt'

    def save(self, refID=None):
        # print(self.config.dir)
        aux.save_dict(self.df, self.path_to_file('geodf'))
        aux.save_dict(self.get_step_data(), self.path_to_file('geostep'))

        if self.endpoint_data is not None:
            aux.save_dict(self.df, self.path_to_file('geoend'))
        self.save_config(refID=refID)
        reg.vprint(f'***** Dataset {self.config.id} stored.-----', 1)

    # def save(self, refID=None):
    #     aux.save_dict(self.endpoint_data, self.endpoint_data_path)
    #     aux.save_dict(self.df, self.data_path)
    #     self.save_config(refID=refID)
    #     reg.vprint(f'***** Dataset {self.config.id} stored.-----', 1)

    @property
    def df(self):
        return pd.concat([tr.df for tr in self])

    def get_step_data(self):
        df = self.df.reset_index()
        df.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True, verify_integrity=False)
        df.sort_index(level=['Step', 'AgentID'], inplace=True)
        return df


    @property
    def duration(self):
        idx=self.df.index
        return (idx.max()-idx.min()).total_seconds()

    # def set_interpolation_dt(self,dt):
    #     e=self.endpoint_data
    #     t0, t1 = reg.getPar(['t0', 't_fin'])
    #     tick0, tick1, Nticks = reg.getPar(['tick0', 'tick_fin', 'N_ticks'])
    #     e[tick1] = np.floor(e[t1] / dt).astype(int)
    #     e[tick0] = np.ceil(e[t0] / dt).astype(int)
    #     e[Nticks] = e[tick1] - e[tick0]


    def interpolate_traj(self, dt=0.1):
        dtu=dt*PintType.ureg.sec
        e = self.endpoint_data
        t0, t1 = reg.getPar(['t0', 't_fin'])
        tick0, tick1, Nticks = reg.getPar(['tick0', 'tick_fin', 'N_ticks'])
        e[tick1] = np.floor(e[t1] / dtu).astype(int)
        e[tick0] = np.ceil(e[t0] / dtu).astype(int)
        e[reg.getPar('N_ticks')] = e[tick1] - e[tick0]





        # c=self.config
        xy = ['x', 'y']
        Nticks=int(self.duration/dt)
        ticks = np.arange(0, Nticks, 1).astype(int)
        ts = np.array([self.time_to_datetime(i * dt) for i in ticks])
        ids=self.config.agent_ids



        my_index = pd.MultiIndex.from_product([ticks, ids], names=['Step', 'AgentID'])
        A = np.zeros([Nticks, ids.shape[0], 2]) * np.nan
        for j, id in enumerate(ids):
            tr = self.traj_dic[id]

            ee=e.loc[id]
            valid = ticks[ee[tick0]:ee[tick1]]
            points = [tr.get_position_at(ts[ii]) for ii in valid]
            A[valid, j, :] = np.array([[point.x, point.y] for point in points])

        A = A.reshape([-1, 2])
        df = pd.DataFrame(A, index=my_index, columns=xy)
        df = df.sort_index()
        df[xy] = df[xy].astype({p: float for p in xy})
        return df

    def load_traj(self):
        return self.interpolate_traj()





    def load(self, **kwargs):
        s = pd.DataFrame(aux.load_dict(self.path_to_file('geostep')))
        e = pd.DataFrame(aux.load_dict(self.path_to_file('geoend')))
        self.set_data(step=s, end=e)

    # @property
    # def data_path(self):
    #     return f'{self.data_dir}/data.txt'
    #
    # @property
    # def endpoint_data_path(self):
    #     return f'{self.data_dir}/end.txt'



    def match_ids(self):
        verbose=1
        reg.vprint(f'**--- Initializing matchIDs algorithm -----', verbose)
        Nids0=self.config.N
        e=self.endpoint_data


        x0, x1 = reg.getPar(['x0', 'x_fin'])
        y0, y1 = reg.getPar(['y0', 'y_fin'])
        t0, t1 = reg.getPar(['t0', 't_fin'])
        l = reg.getPar('l')

        ids = e.index.values
        pairs = {}
        for id1 in ids:
            for id2 in ids:
                if id1 == id2:
                    continue
                ee2, ee1 = e.loc[id2], e.loc[id1]
                dt = ee2[t0] - ee1[t1]
                if dt >= 0:
                    l_mu = (ee2[l] + ee1[l]) / 2
                    dx = ee2[x0] - ee1[x1]
                    dy = ee2[y0] - ee1[y1]
                    d = ((dx ** 2 + dy ** 2) ** 0.5) / l_mu
                    dl = np.abs(ee2[l] - ee1[l]) / l_mu

                    pairs[(id1, id2)] = {'dt': dt.magnitude, 'd': d.magnitude, 'dl': dl.magnitude}
        pairs = pd.DataFrame.from_records(pairs).T

        pairs['v'] = pairs['d'] / pairs['dt']
        pairs = pairs[pairs['v'] <= 0.4]
        pairs = pairs[pairs['dl'] <= 0.3]
        pairs['id0'] = [pair[0] for pair in pairs.index.values]
        pairs['id1'] = [pair[1] for pair in pairs.index.values]
        pairs.reset_index()
        pairs = pairs.set_index(['id0', 'id1'])
        pairs = pairs.sort_index()
        pairs = pairs.sort_values(['dt', 'd', 'v', 'dl'])
        valid = []
        while pairs.index.values.shape[0] > 1:
            p = pairs.index[0]
            valid.append(p)
            pairs.drop(p[0], level='id0', axis=0, inplace=True)
            pairs.drop(p[1], level='id1', axis=0, inplace=True)
        ids1 = np.array([id1 for id1, id2 in valid])
        ids2 = np.array([id2 for id1, id2 in valid])

        valid2 = np.array([ids1, ids2]).T
        valid2 = valid2[valid2[:, 0].argsort()]
        assert (valid2[:, 1].shape == np.unique(valid2[:, 1]).shape)
        assert (valid2[:, 0].shape == np.unique(valid2[:, 0]).shape)

        combos = []
        invalid = []
        for i, (id1, id2) in enumerate(valid2):
            if i in invalid:
                continue
            combo = [id1, id2]
            counter = 1
            while i + counter < valid2.shape[0]:
                if combo[-1] == valid2[i + counter, 0]:
                    combo.append(valid2[i + counter, 1])
                    invalid.append(i + counter)
                counter += 1
            done = False
            for j in range(len(combos)):
                if combo[0] == combos[j][-1]:
                    combos[j] += combo[1:]
                    done = True
                    break
            for j in range(len(combos)):
                if combo[-1] == combos[j][0]:
                    combos[j] = combo[:-1] + combos[j]
                    done = True
                    break
            if not done:
                combos.append(combo)


        f = aux.flatten_list(combos)
        uf=aux.unique_list(f)
        assert (len(f) == len(uf))
        traj_ids_to_drop=[]
        for combo in combos :
            traj_ids_to_drop+=combo[1:]
            trajs=[self.get_trajectory(traj_id) for traj_id in combo]
            trajs[0].df=pd.concat(tr.df for tr in trajs)
            trajs[0].df['AgentID']=combo[0]
        self.trajectories=[tr for tr in self.trajectories if tr.id not in traj_ids_to_drop]
        reg.vprint(f'**--- Recomputing endpoint metrics -----', verbose)
        end=self.build_endpoint_data(e=None)
        end[l] = {traj.id: np.mean(traj.df[l].values) for traj in self}
        end[l] = end[l].astype({l: self.spatial_pint_unit})

        self.endpoint_data = end.sort_index()
        reg.vprint(f'**--- Trajectories reduced from {Nids0} to {self.config.N} by the matchIDs algorithm -----',verbose)

    def comp_spatial(self):
        self.add_distance(overwrite=True)
        self.add_speed(overwrite=True)
        self.scale_to_length(ks=['d', 'v'])
        self.get_means(ks=['v', 'sv'])

if __name__ == "__main__":
    tpd = GeoLarvaDataset.from_ID(refID='exploration.40controls')
    # tpd.comp_spatial()
    # tpd.save()
    # tpd.load_midline()
    #tpd.load_contour(d.Ncontour)

#print(t.get_start_location().x)

#df.crs['units']