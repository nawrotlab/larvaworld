from lib.model.envs._larvaworld_replay import LarvaWorldReplay
from lib.registry.pars import preg
from lib.process.spatial import align_trajectories, fixate_larva
from lib.aux.dir_aux import smaller_dataset

class ReplayRun:
    def __init__(self, dataset=None, refID=None, id=None, save_to=None, vis_kwargs=None, agent_ids=None,  time_range=None,
                  draw_Nsegs=None, env_params=None,close_view=False, track_point=None, dynamic_color=None,overlap_mode=False,
                  transposition=None, fix_point=None, fix_segment=None, show_output = True, **kwargs):
        if dataset is None and refID is not None :
            dataset=preg.loadRef(refID)
        self.dataset=dataset
        self.show_output=show_output

        if id is None:
            if transposition is not None:
                n1 = 'transposed'
            elif fix_point is not None:
                n1 = f'fixed_at_{fix_point}'
                if overlap_mode :
                    n1=f'{n1}_overlap'
            else:
                n1 = 'normal'
            id = f'{self.dataset.config.id}_{n1}'
        self.id=id
        if vis_kwargs is None:
            if not overlap_mode :
                vis_kwargs = preg.get_null('visualization', mode='video', video_speed=60, media_name=self.id)
            else :
                vis_kwargs = preg.get_null('visualization', mode='image', image_mode='overlap', media_name=self.id, draw_contour=False)



        s,e,c=smaller_dataset(d=self.dataset,ids=agent_ids, time_range=time_range, track_point=track_point,
                              env_params=env_params,close_view=close_view)

        if transposition is not None:
            s = align_trajectories(s, mode=transposition, c=c)
            bg = None
        elif fix_point is not None:
            s, bg = fixate_larva(s, point=fix_point, fix_segment=fix_segment, c=c)
        else:
            bg = None

        if save_to is None:
            save_to = self.dataset.vis_dir

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
        self.env = LarvaWorldReplay(**base_kws)

    def run(self):
        if self.show_output:
            print()
            print(f'---- Replay {self.id} ----')
        # Run the simulation
        completed = self.env.run()
        if not completed:
            print('Replay aborted!')
        else:
            print('Replay completed')