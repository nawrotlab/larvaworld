import copy

import numpy as np

from lib import reg
from lib.model.envs.world_replay import WorldReplay
from lib.aux import naming as nam, dictsNlists as dNl
from lib.process.spatial import fixate_larva

class ReplayRun:
    def __init__(self, dataset=None, refID=None, id=None, save_to=None, agent_ids=None,  time_range=None,
                  draw_Nsegs=None, env_params=None,close_view=False, track_point=None, dynamic_color=None,overlap_mode=False,
                  transposition=None, fix_point=None, fix_segment=None, show_output = True, **kwargs):

        if dataset is None and refID is not None :
            dataset=reg.loadRef(refID)
        self.dataset=dataset
        self.show_output=show_output



        s,e,c=smaller_dataset(d=self.dataset,ids=agent_ids, transposition=transposition, time_range=time_range, track_point=track_point,
                              env_params=env_params,close_view=close_view)

        c.env_params.windscape=None
        if id is None:
            if transposition is not None:
                n1 = f'aligned_to_{transposition}'
            elif fix_point is not None:
                n1 = f'fixed_at_{fix_point}'
                if overlap_mode :
                    n1=f'{n1}_overlap'
            else:
                n1 = 'normal'
            id = f'{self.dataset.config.id}_{n1}'
        self.id=id

        if not overlap_mode :
            vis_kwargs = reg.get_null(name='visualization', mode='video', video_speed=60, media_name=self.id)
        else :
            vis_kwargs = reg.get_null(name='visualization', mode='image', image_mode='overlap', media_name=self.id, draw_contour=False)


        if fix_point is not None:
            s, bg = fixate_larva(s, point=fix_point, fix_segment=fix_segment, c=c)
        else:
            bg = None

        if save_to is None:
            save_to = reg.datapath('visuals',c.dir)

        base_kws = {
            'step_data': s,
            'endpoint_data': e,
            'config': c,
            'draw_Nsegs': draw_Nsegs,
            'vis_kwargs': vis_kwargs,
            'id': self.id,
            'save_to': save_to,
            'background_motion': bg,
            'traj_color': s[dynamic_color] if dynamic_color is not None and dynamic_color in s.columns else None,
            **kwargs
        }
        self.env = WorldReplay(**base_kws)

    def run(self):
        reg.vprint()
        reg.vprint(f'---- Replay {self.id} ----')
        # Run the simulation
        completed = self.env.run()
        if not completed:
            print('Replay aborted!')
        else:
            print('Replay completed')



def smaller_dataset(d, track_point=None, ids=None, transposition=None, time_range=None, pars=None,env_params=None,close_view=False):


    c=d.config
    c0=dNl.copyDict(c)


    if track_point is None:
        track_point = c.point
    elif type(track_point) == int:
        track_point = 'centroid' if track_point == -1 else nam.midline(c.Npoints, type='point')[track_point]
    c0.point = track_point
    if ids is not None:
        if type(ids) == list and all([type(i) == int for i in ids]):
            ids = [c.agent_ids[i] for i in ids]
    else :
        ids = c.agent_ids
    c0.agent_ids = ids
    c0.N = len(ids)

    def get_data(d,ids) :
        if not hasattr(d, 'step_data'):
            d.load(h5_ks=['contour', 'midline'])
        s, e = d.step_data, d.endpoint_data
        e0=copy.deepcopy(e.loc[ids])
        s0=copy.deepcopy(s.loc[(slice(None), ids), :])
        return s0,e0

    s0,e0=get_data(d,ids)

    if pars is not None:
        s0 = s0.loc[(slice(None), slice(None)), pars]

    if env_params is not None:
        c0.env_params = env_params

    if transposition is not None:
        try:
            s_tr = d.load_traj(mode=transposition)
            s0.update(s_tr)

        except:
            from lib.process.spatial import align_trajectories
            s0 = align_trajectories(s0, c=c0, transposition=transposition,replace=True)

        xy_max=2*np.max(s0[nam.xy(c0.point)].dropna().abs().values.flatten())
        c0.env_params.arena = reg.get_null('arena', arena_dims=(xy_max, xy_max))

    if close_view:
        c0.env_params.arena = reg.get_null('arena', arena_dims=(0.01, 0.01))


    if time_range is not None:
        a, b = time_range
        a = int(a / c.dt)
        b = int(b / c.dt)
        s0 = s0.loc[(slice(a, b), slice(None)), :]

    c0.Nsteps = len(s0.index.unique('Step').values)
    return s0,e0, c0