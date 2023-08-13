import copy
import numpy as np


from larvaworld.lib import reg, aux, util
from larvaworld.lib.aux import nam

from larvaworld.lib.model import agents, envs
from larvaworld.lib.param import XYops
# from larvaworld.lib.process.dataset import RefDataset
from larvaworld.lib.screen import ScreenManager
from larvaworld.lib.sim.base_run import BaseRun


class ReplayRun(BaseRun):
    def __init__(self,parameters,  dataset=None, screen_kws={},**kwargs):
        '''
        Simulation mode 'Replay' reconstructs a real or simulated experiment from stored data.

        Args:
            parameters: Dictionary of configuration parameters to be passed to the ABM model
            dataset: The stored dataset to replay. If not specified it is retrieved using either the storage path (parameters.dir) or the respective unique reference ID (parameters.RefID)
            experiment: The type of experiment. Defaults to 'replay'
            **kwargs: Arguments passed to parent class
        '''
        d = reg.conf.Ref.retrieve_dataset(dataset=dataset, id=parameters.refID, dir=parameters.refDir)

        self.refDataset=copy.deepcopy(d)

        # Configure the dataset to replay
        self.refDataset=self.smaller_dataset(p=parameters, d=self.refDataset)
        c=self.refDataset.config
        parameters.steps = c.Nsteps
        kwargs.update(**{'duration':c.duration,
                       'dt':c.dt,
                       'Nsteps':c.Nsteps})

        super().__init__(runtype='Replay', parameters=parameters,**kwargs)

    @property
    def configuration_text(self):
        c = self.p
        pref0 = '     '
        text = f"Dataset Replay configuration : \n" \
               f"{pref0}Reference Dataset : {c.refID}\n" \
               f"{pref0}Duration (min) : {c.duration}\n" \
               f"{pref0}Timestep (sec) : {c.dt}\n" \
               f"{pref0}Time range (sec) : {c.time_range}\n" \
               f"{pref0}Transposition : {c.transposition}\n" \
               f"{pref0}Tracked midline point : {c.point}"
        return text

    def setup(self):
        # s,e,c=self.step_data,self.endpoint_data,self.config


        self.draw_Nsegs = self.p.draw_Nsegs
        self.build_env(self.p.env_params)
        self.build_agents(d=self.refDataset)
        screen_kws = {
            'mode': 'video' if not self.p.overlap_mode else 'image',
            'show_display' : True,
            'image_mode':'overlap' if self.p.overlap_mode else None,
            'background_motion': self.background_motion,
        }
        self.screen_manager = ScreenManager(model=self, **screen_kws)



    def build_agents(self, d):
        s, e, c = d.data

        if 'length' in e.columns:
            ls = e['length'].values
        else:
            ls = np.ones(c.N) * 0.005

        ors=['front_orientation','rear_orientation']
        assert aux.cols_exist(ors, s)
        # mid_ps = c.midline_xy
        # con_ps = c.contour_xy
        if self.p.draw_Nsegs is not None:
            if self.p.draw_Nsegs == 2:
                pass
            elif self.p.draw_Nsegs == c.Npoints - 1:
                # or_ps = c.seg_orientations
                # assert or_ps.exist_in(s)
                # assert mid_ps.exist_in(s)
                seg_orientD=d.midline_seg_orients_data_byID
                midlineD=d.midline_seg_xy_data_byID
            else:
                raise
        else:
            contourD=d.contour_xy_data_byID
            midlineD=d.midline_xy_data_byID

        confs=[]
        for i, id in enumerate(c.agent_ids):
            conf = aux.AttrDict({'unique_id': id, 'length': ls[i]})
            data = aux.AttrDict()
            ss=s.xs(id, level='AgentID', drop_level=True)
            xy=ss[['x', 'y']].values
            data.pos = aux.np2Dtotuples(xy)
            fo,ro=ss['front_orientation'].values, ss['rear_orientation'].values
            data.front_orientation = fo
            data.rear_orientation = ro
            if self.p.draw_Nsegs is not None:
                conf.Nsegs=self.p.draw_Nsegs
                if conf.Nsegs == 2:
                    data.seg_orientations =np.vstack([fo,ro]).T
                    l1, l2 = conf.length/2,  conf.length/2
                    p1 = xy + aux.rotationMatrix(-fo).T @ (l1 / 2, 0)
                    p2 = xy - aux.rotationMatrix(-ro).T @ (l2 / 2, 0)
                    data.midline = np.hstack([p1,p2]).reshape([-1,2,2])
                elif conf.Nsegs == c.Npoints - 1:
                    # or_ps = aux.nam.orient(aux.nam.midline(conf.Nsegs, type='seg'))
                    # assert or_ps.exist_in(ss)
                    data.seg_orientations = seg_orientD[id]

                    # # assert mid_ps.exist_in(ss)
                    # mid = ss[mid_ps].values.reshape([-1, c.Npoints, 2])
                    # mid2=copy.deepcopy(mid)
                    # for i in range(conf.Nsegs):
                    #     mid2[:,i,:]=(mid[:,i,:]+mid[:,i+1:])/2
                    data.midline = midlineD[id]
                else:
                    raise
            else:

                # assert con_ps.exist_in(ss)
                data.contour = contourD[id]
                # assert mid_ps.exist_in(ss)
                data.midline = midlineD[id]
            conf.data=data
            confs.append(conf)
        self.place_agents(confs)




    def sim_step(self):
        """ Proceeds the simulation by one step, incrementing `Model.t` by 1
        and then calling :func:`Model.step` and :func:`Model.update`."""
        if not self.is_paused:
            self.step()
            self.update()
            self.t += 1
            if self.t >= self._steps :
                self.running = False

    def step(self):
        """ Defines the models' events per simulation step. """
        self.agents.step()
        self.screen_manager.step()

    def end(self):
        self.screen_manager.finalize()


    def smaller_dataset(self,p, d):
        d.load(h5_ks=['contour', 'midline', 'angular'])
        c=d.config

        assert isinstance(c, reg.DatasetConfig)
        # Group mode
        if p.track_point is not None:
            c.point_idx=p.track_point







        # Unit mode
        if p.fix_point is not None:
            c.fix_point = c.get_track_point(p.fix_point)
            if c.fix_point == 'centroid' or p.fix_segment is None:
                c.fix_point2 = None
            else:
                if p.fix_segment == 'rear':
                    P2_idx = p.fix_point + 1
                elif p.fix_segment == 'front':
                    P2_idx = p.fix_point - 1
                else:
                    raise
                c.fix_point2 = c.get_track_point(P2_idx)
        else:
            c.fix_point = None
        if p.agent_ids not in [None, []]:
            if isinstance(p.agent_ids, list) and all([type(i) == int for i in p.agent_ids]):
                p.agent_ids = [c.agent_ids[i] for i in p.agent_ids]
            elif isinstance(p.agent_ids, int):
                p.agent_ids = [c.agent_ids[p.agent_ids]]
            c.agent_ids = p.agent_ids
        if c.fix_point is not None:
            c.agent_ids = c.agent_ids[:1]
        d.update_ids_in_data()



        if p.env_params is None:
            p.env_params = c.env_params.nestedConf

        if p.close_view:
            p.env_params.arena = reg.gen.Arena(dims=(0.01, 0.01)).nestedConf

        s = d.step_data


        if p.time_range is not None:
            a, b = p.time_range
            s = s.query(f'{a}<=Step*{c.dt}<={b}')

        xy_pars = nam.xy(c.point)
        assert aux.cols_exist(xy_pars, s)
        s[['x', 'y']] = s[xy_pars]

        if c.fix_point is not None:
            s, bg = reg.funcs.preprocessing['fixation'](s, c, P1=c.fix_point,P2=c.fix_point2)
        else:
            bg = None
        self.background_motion=bg

        if p.transposition is not None:
            s = reg.funcs.preprocessing["transposition"](s, c=c, transposition=p.transposition,replace=True)
            xy_max=2*np.max(s[nam.xy(c.point)].dropna().abs().values.flatten())
            p.env_params.arena=reg.gen.Arena(dims=(xy_max, xy_max)).nestedConf

        d.set_data(step=s)

        return d