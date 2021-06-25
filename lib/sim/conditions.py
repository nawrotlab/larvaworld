import random
import numpy as np
from lib.anal.process.spatial import compute_preference_index
import lib.aux.functions as fun

def get_exp_condition(exp):
    exp_condition_dict = {
        'odor_pref_train': PrefTrainCondition,
        'catch_me': CatchMeCondition,
        'keep_the_flag': KeepFlagCondition,
        'capture_the_flag': CaptureFlagCondition,
        # 'odor_pref_train': PrefTrainCondition,
    }
    return exp_condition_dict[exp] if exp in exp_condition_dict.keys() else None


class PrefTrainCondition:
    def __init__(self, env):
        env.CS_counter = 1
        env.UCS_counter = 0
        print()
        print(f'Training trial {env.CS_counter} with CS started at {env.sim_clock.minute}:{env.sim_clock.second}')
        for f in env.get_food():
            if f.unique_id == 'CS':
                env.CS = f
            elif f.unique_id == 'UCS':
                env.UCS = f
        env.CS.set_odor_dist(intensity=2.0)
        env.UCS.set_odor_dist(intensity=0.0)
        # self.env=env

    def check(self, env):
        m, s = env.sim_clock.minute, env.sim_clock.second
        if env.sim_clock.timer_opened:
            env.UCS_counter += 1
            if env.UCS_counter <= 3:
                print()
                print(f'Starvation trial {env.UCS_counter} with UCS started at {m}:{s}')
                env.CS.set_odor_dist(intensity=0.0)
                env.UCS.set_odor_dist(intensity=2.0)
                env.move_larvae_to_center()
            else:
                PI = compute_preference_index(poses=[l.pos for l in env.get_flies()], arena_dims=env.arena_dims)
                print()
                print(f'Test trial on food ended at {m}:{s} with PI={PI}')
                print()
                print(f'Test trial without food started at {m}:{s}')
                env.move_larvae_to_center()
        if env.sim_clock.timer_closed:
            env.CS_counter += 1
            if env.CS_counter <= 3:
                print()
                print(f'Training trial {env.CS_counter} with CS started at {m}:{s}')
                env.CS.set_odor_dist(intensity=2.0)
                env.UCS.set_odor_dist(intensity=0.0)
                env.move_larvae_to_center()
            else:
                print()
                print(f'Test trial on food started at {m}:{s}')
                env.CS.set_odor_dist(intensity=2.0)
                env.UCS.set_odor_dist(intensity=2.0)
                env.move_larvae_to_center()
        if env.sim_clock.minute >= 40:
            PI = compute_preference_index(poses=[l.pos for l in env.get_flies()], arena_dims=env.arena_dims)
            print()
            print(f'Test trial without food ended at {m}:{s} with PI={PI}')
            env.end_condition_met = True

class CatchMeCondition:
    def __init__(self, env):
        env.target_group = 'Left' if random.uniform(0, 1) > 0.5 else 'Right'
        env.follower_group = 'Right' if env.target_group == 'Left' else 'Left'
        for f in env.get_flies():
            if f.group == env.target_group:
                f.brain.olfactor.gain = {id: -v for id, v in f.brain.olfactor.gain.items()}
        env.score = {env.target_group: 0.0,
                      env.follower_group: 0.0}

    def check(self, env):
        def set_target_group(group):
            env.target_group = group
            env.follower_group = 'Right' if env.target_group == 'Left' else 'Left'
            for f in env.get_flies():
                f.brain.olfactor.gain = {id: -v for id, v in f.brain.olfactor.gain.items()}

        targets = {f: f.get_position() for f in env.get_flies() if f.group == env.target_group}
        followers = [f for f in env.get_flies() if f.group == env.follower_group]
        for f in followers:
            if any([f.contained(p) for p in list(targets.values())]):
                set_target_group(f.group)
                break
        env.score[env.target_group] += env.dt
        for group, score in env.score.items():
            if score >= 1200.0:
                print(f'{group} group wins')
                env.end_condition_met = True
        env.sim_state.set_text(f'L:{np.round(env.score["Left"], 1)} vs R:{np.round(env.score["Right"], 1)}')

class KeepFlagCondition:
    def __init__(self, env):
        for f in env.get_food():
            if f.unique_id == 'Flag':
                env.flag = f
        env.l_t = 0
        env.r_t = 0

    def check(self, env):
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
                env.end_condition_met = True
        elif carrier.group == 'Right':
            env.r_t += env.dt
            env.l_t = 0
            if env.r_t - dur > 0:
                print('Right group wins')
                env.end_condition_met = True
        env.sim_state.set_text(f'L:{np.round(dur - env.l_t, 2)} vs R:{np.round(dur - env.r_t, 2)}')

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
        flag_p = env.flag.get_position()
        l_dst = -env.l_dst0 + fun.compute_dst(flag_p, env.l_base_p)
        r_dst = -env.r_dst0 + fun.compute_dst(flag_p, env.r_base_p)
        l_dst = np.round(l_dst * 1000, 2)
        r_dst = np.round(r_dst * 1000, 2)
        if l_dst < 0:
            print('Left group wins')
            env.end_condition_met = True
        elif r_dst < 0:
            print('Right group wins')
            env.end_condition_met = True
        env.sim_state.set_text(f'L:{l_dst} vs R:{r_dst}')
