import itertools
import random

import numpy as np

from lib.aux.dictsNlists import flatten_tuple
from lib.model.modules.basic import Effector


class RLmemory(Effector):
    def __init__(self, brain, gain, decay_coef, DeltadCon=0.02, state_spacePerOdorSide=3,
                 gain_space=[-500, -50, 50, 500],
                 decay_coef_space=None, update_dt=2, train_dur=30, alpha=0.05, gamma=0.6, epsilon=0.15, **kwargs):
        super().__init__(**kwargs)
        self.brain = brain
        self.effector = True
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.DeltadCon = DeltadCon
        self.gain_space = gain_space
        if decay_coef_space is None:
            decay_coef_space = [decay_coef]
        self.decay_coef_space = decay_coef_space
        # self.gain = gain
        self.odor_ids = list(gain.keys())
        self.Nodors = len(self.odor_ids)
        self.fit_pars = self.odor_ids + ['decay_coef']
        # self.Nfit_pars =self.Nodors + 1
        self.actions = [ii for ii in itertools.product(gain_space, repeat=self.Nodors)]
        self.actions = [flatten_tuple(ii) for ii in itertools.product(self.actions, list(self.decay_coef_space))]
        self.state_spacePerOdorSide = state_spacePerOdorSide
        self.state_space = np.array(
            [ii for ii in itertools.product(range(2 * self.state_spacePerOdorSide + 1), repeat=self.Nodors)])
        # self.q_table = [np.zeros(len(self.state_space), len(self.actions)) for ii in odor_ids]
        self.q_table = np.zeros((self.state_space.shape[0], len(self.actions)))
        self.lastAction = 0
        self.lastState = 0
        self.Niters = int(update_dt * 60 / self.dt)
        self.iterator = self.Niters
        self.train_dur = train_dur
        self.rewardSum = 0
        self.best_gain = gain
        # self.decay_coef = decay_coef
        self.best_decay_coef = decay_coef

        self.table = False

    def state_collapse(self, dCon):
        k=self.state_spacePerOdorSide
        if len(dCon) > 0:
            dCon = [dCon]
        stateV = []
        for index in range(len(dCon)):
            for i in dCon[index]:
                dConI = dCon[index][i]
                stateIntermitt = np.zeros(k)
                for ii in range(k):
                    stateIntermitt[ii] = np.abs(dConI) > (ii + 1) * self.DeltadCon
            stateV.append(int(np.sign(dConI) * (np.sum(stateIntermitt)) + k))
        state = np.where((self.state_space == stateV).all(axis=1))[0][0]
        return state

    def step(self, gain, dCon, reward, decay_coef):
        if self.table == False:
            temp = self.brain.agent.model.table_collector
            if temp is not None:
                self.table = temp.tables['best_gains'] if 'best_gains' in list(temp.tables.keys()) else None

        self.count_time()
        if self.effector and self.total_t > self.train_dur * 60:
            self.effector = False
            # print(f'Training stopped after {self.train_dur} minutes')
            print(f'Best gain : {self.best_gain}')
        if self.effector:
            self.rewardSum += int(reward) - 0.001
            if self.iterator >= self.Niters:
                self.iterator = 0
                state = self.state_collapse(dCon)
                if random.uniform(0, 1) < self.epsilon:
                    actionID = random.randrange(len(self.actions))
                else:
                    actionID = np.argmax(self.q_table[state])  # Exploit learned values
                old_value = self.q_table[self.lastState, self.lastAction]
                next_max = np.max(self.q_table[state])
                new_value = (1 - self.alpha) * old_value + self.alpha * (self.rewardSum + self.gamma * next_max)
                self.q_table[self.lastState, self.lastAction] = new_value
                self.lastAction = actionID
                self.lastState = state
                action = self.actions[actionID]
                for ii, id in enumerate(self.odor_ids):
                    gain[id] = action[ii]
                decay_coef = action[-1]
                best_combo = self.get_best_combo()
                self.best_gain = {id: best_combo[id] for id in self.odor_ids}
                self.best_decay_coef = best_combo['decay_coef']
                if self.table:
                    for col in list(self.table.keys()):
                        try:
                            self.table[col].append(getattr(self.brain.agent, col))
                        except:
                            self.table[col].append(np.nan)
                self.rewardSum = 0
            self.iterator += 1
            return gain, decay_coef
        else:
            return self.best_gain, self.best_decay_coef

    def get_best_combo(self):
        return dict(zip(self.fit_pars, self.actions[np.argmax(np.mean(self.q_table, axis=0))]))

class SimpleMemory(Effector):
    def __init__(self, brain, gain, decay_coef, DeltaGain=0.02, DeltadCon=0.02, train_dur=30, **kwargs):
        super().__init__(**kwargs)
        self.brain = brain
        self.effector = True
        self.DeltaGain = DeltaGain
        self.DeltadCon = DeltadCon
        self.odor_ids = list(gain.keys())
        self.Nodors = len(self.odor_ids)
        self.fit_pars = self.odor_ids + ['decay_coef']
        self.train_dur = train_dur
        self.rewardSum = 0
        self.best_gain = gain
        self.best_decay_coef = decay_coef
        self.table = False

    def step(self, gain, dCon, reward, decay_coef):
        if self.table == False:
            temp = self.brain.agent.model.table_collector
            if temp is not None:
                self.table = temp.tables['best_gains'] if 'best_gains' in list(temp.tables.keys()) else None

        self.count_time()
        if self.effector and self.total_t > self.train_dur * 60:
            self.effector = False
            # print(f'Training stopped after {self.train_dur} minutes')
            print(f'Best gain : {self.best_gain}')
        if self.effector:
            self.rewardSum += int(reward) - 0.001
            for ii, id in enumerate(self.odor_ids):
                gain[id] += int(reward)*np.abs(dCon[id])- 0.001
            # print(np.abs(list(dCon.values())))
            self.best_gain = gain
            self.best_decay_coef = decay_coef
            if self.table:
                for col in list(self.table.keys()):
                    try:
                        self.table[col].append(getattr(self.brain.agent, col))
                    except:
                        self.table[col].append(np.nan)
            return gain, decay_coef
        else:
            return self.best_gain, self.best_decay_coef

    # def get_best_combo(self):
    #     return dict(zip(self.fit_pars, self.actions[np.argmax(np.mean(self.q_table, axis=0))]))