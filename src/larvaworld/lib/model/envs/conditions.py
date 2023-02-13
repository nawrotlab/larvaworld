import random
import numpy as np

from larvaworld.lib import aux
from larvaworld.lib.process.spatial import comp_PI

def get_exp_condition(exp):
    d = {
        'PItrain_mini': PrefTrainCondition,
        'PItrain': PrefTrainCondition,
        'catch_me': CatchMeCondition,
        'keep_the_flag': KeepFlagCondition,
        'capture_the_flag': CaptureFlagCondition,
    }
    return d[exp] if exp in d.keys() else None


class PrefTrainCondition:
    def __init__(self, env):
        self.peak_intensity=2.0
        env.CS_counter = 0
        env.UCS_counter = 0
        env.CS_sources = [f for f in env.get_food() if f.odor_id=='CS']
        env.UCS_sources = [f for f in env.get_food() if f.odor_id=='UCS']


    def toggle_odors(self, env, CS_intensity=2.0, UCS_intensity=0.0):
        for f in env.CS_sources :
            f.set_odor_dist(intensity=CS_intensity)
            f.visible=True if CS_intensity>0 else False
        for f in env.UCS_sources :
            f.set_odor_dist(intensity=UCS_intensity)
            f.visible = True if UCS_intensity > 0 else False

    def init_test(self,env):
        for f in env.CS_sources:
            if f.unique_id=='CS_r':
                env.CS_sources.remove(f)
                env.delete_agent(f)
        for f in env.UCS_sources:
            if f.unique_id=='UCS_l':
                env.UCS_sources.remove(f)
                env.delete_agent(f)

    def start_trial(self,env, on_food=True):
        # m, s = env.sim_clock.minute, env.sim_clock.second
        c=self.peak_intensity
        if on_food :
            env.CS_counter += 1
            if env.CS_counter <= 3:
                text=f'Training trial {env.CS_counter}'
                self.toggle_odors(env, c,0.0)
                env.move_larvae_to_center()
            elif env.CS_counter == 4:
                text = f'Test trial on food'
                self.init_test(env)
                self.toggle_odors(env, c,c)
                env.move_larvae_to_center()


        else:
            env.UCS_counter += 1
            if env.UCS_counter <= 3:
                text = f'Starvation trial {env.UCS_counter}'
                self.toggle_odors(env, 0.0, c)
                env.move_larvae_to_center()
            elif env.UCS_counter == 4:
                PI = comp_PI(xs=[l.pos[0] for l in env.get_flies()], arena_xdim=env.space.dims[0])
                sec=int(env.Nticks*env.dt)
                m,s=int(sec/60), sec%60
                print()
                print(f'Test trial on food ended at {m}:{s} with PI={PI}')
                text = f'Test trial off food'
                self.toggle_odors(env, c, c)
                env.move_larvae_to_center()
        env.input_box.flash_text(text)

    def check(self, env):
        for i, ep in env.sim_epochs.items():
            if env.Nticks == ep['start']:
                q=ep['substrate']['quality']
                if q == 0.0:
                    env.food_grid.empty_grid()
                    self.start_trial(env, on_food=False)
                elif q == 1.0:
                    env.food_grid.reset()
                    self.start_trial(env, on_food=True)
            elif env.Nticks == ep['stop'] and i==len(env.sim_epochs)-1 :
                PI = comp_PI(xs=[l.pos[0] for l in env.get_flies()], arena_xdim=env.space.dims[0])
                print()
                print(f'Test trial off food ended with PI={PI}')
                text = f'Test trial off food PI={PI}'
                env.input_box.flash_text(text)
                return True
        return False



class CatchMeCondition:
    def __init__(self, env):
        env.target_group = 'Left' if random.uniform(0, 1) > 0.5 else 'Right'
        env.follower_group = 'Right' if env.target_group == 'Left' else 'Left'
        env.targets = [f for f in env.get_flies() if f.group == env.target_group]
        env.followers = [f for f in env.get_flies() if f.group == env.follower_group]
        for f in env.targets:
            f.brain.olfactor.gain = {id: -v for id, v in f.brain.olfactor.gain.items()}
        env.score = {env.target_group: 0.0,
                      env.follower_group: 0.0}

    def check(self, env):
        if env.Nticks == 0:
            env.input_box.flash_text('Catch me')
        def set_target_group(group):
            env.target_group = group
            env.follower_group = 'Right' if env.target_group == 'Left' else 'Left'
            env.targets=[f for f in env.get_flies() if f.group == env.target_group]
            env.followers = [f for f in env.get_flies() if f.group == env.follower_group]
            for f in env.get_flies():
                f.brain.olfactor.gain = {id: -v for id, v in f.brain.olfactor.gain.items()}
        targets_pos = [f.get_position() for f in env.targets]
        for f in env.followers:
            if any([aux.eudis5(f.get_position(),p)<f.radius for p in targets_pos]):
            # if any([aux.eudis5(f.get_position(),p)<f.radius for p in targets_pos]):
                set_target_group(f.group)
                break
        env.score[env.target_group] += env.dt
        for group, score in env.score.items():
            if score >= 20000.0:
                print(f'{group} group wins')
                return True
        try:
            env.sim_state.set_text(f'L:{np.round(env.score["Left"], 1)} vs R:{np.round(env.score["Right"], 1)}')
        except:
            pass
        return False

class KeepFlagCondition:
    def __init__(self, env):
        for f in env.get_food():
            if f.unique_id == 'Flag':
                env.flag = f
        env.l_t = 0
        env.r_t = 0

    def check(self, env):
        if env.Nticks == 0:
            env.input_box.flash_text('Keep the flag')
        dur = 180
        carrier = env.flag.is_carried_by
        if carrier is None:
            env.l_t = 0
            env.r_t = 0
        elif carrier.group == 'Left':
            env.l_t += env.dt
            env.r_t = 0
            if env.l_t - dur > 0:
                print('Left group wins')
                return True
        elif carrier.group == 'Right':
            env.r_t += env.dt
            env.l_t = 0
            if env.r_t - dur > 0:
                print('Right group wins')
                return True
        try :
            env.sim_state.set_text(f'L:{np.round(dur - env.l_t, 2)} vs R:{np.round(dur - env.r_t, 2)}')
        except:
            pass
        return False

class CaptureFlagCondition:
    def __init__(self, env):
        for f in env.get_food():
            if f.unique_id == 'Flag':
                env.flag = f
            elif f.unique_id == 'Left_base':
                env.l_base = f
            elif f.unique_id == 'Right_base':
                env.r_base = f
        env.l_base_p = env.l_base.get_position()
        env.r_base_p = env.r_base.get_position()
        env.l_dst0 = env.flag.radius * 2 + env.l_base.radius * 2
        env.r_dst0 = env.flag.radius * 2 + env.r_base.radius * 2

    def check(self, env):
        def compute_dst(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if env.Nticks == 0:
            env.input_box.flash_text('Capture the flag')
        flag_p = env.flag.get_position()
        l_dst = -env.l_dst0 + compute_dst(flag_p, env.l_base_p)
        r_dst = -env.r_dst0 + compute_dst(flag_p, env.r_base_p)
        l_dst = np.round(l_dst * 1000, 2)
        r_dst = np.round(r_dst * 1000, 2)
        if l_dst < 0:
            print('Left group wins')
            return True
        elif r_dst < 0:
            print('Right group wins')
            return True
        try:
            env.sim_state.set_text(f'L:{l_dst} vs R:{r_dst}')
        except:
            pass
        return False
