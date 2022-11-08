import time


from lib.registry import reg
from lib.model.envs.world_replay import WorldReplay

from lib.process.spatial import fixate_larva
from lib.aux.dir_aux import smaller_dataset

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
        # print(c.Nsteps)
        # print(c.Nticks)
        # print(c.duration)
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