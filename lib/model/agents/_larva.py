from copy import deepcopy
import numpy as np

from lib.model.agents._agent import LarvaworldAgent


class Larva(LarvaworldAgent):
    def __init__(self, unique_id, model, pos=None, radius=None, default_color=None, **kwargs):
        if default_color is None:
            default_color = model.generate_larva_color()
        super().__init__(unique_id=unique_id, model=model, default_color=default_color, pos=pos, radius=radius,
                         **kwargs)
        self.behavior_pars = ['stride_stop', 'stride_id', 'pause_id', 'feed_id', 'Lturn_id', 'Rturn_id']
        self.null_behavior_dict = dict(zip(self.behavior_pars, [False] * len(self.behavior_pars)))

    def update_color(self, default_color, behavior_dict, mode='lin'):
        color = deepcopy(default_color)
        if mode == 'lin':
            # if beh_dict['stride_stop'] :
            #     color=np.array([0, 255, 0])
            if behavior_dict['stride_id']:
                color = np.array([0, 150, 0])
            elif behavior_dict['pause_id']:
                color = np.array([255, 0, 0])
            elif behavior_dict['feed_id']:
                color = np.array([0, 0, 255])
        elif mode == 'ang':
            if behavior_dict['Lturn_id']:
                color[2] = 150
            elif behavior_dict['Rturn_id']:
                color[2] = 50
        return color

    @property
    def turner_activation(self):
        return self.brain.turner.activation

    @property
    def olfactory_activation(self):
        return self.brain.olfactory_activation

    @property
    def touch_activation(self):
        return self.brain.touch_activation

    @property
    def first_odor_concentration(self):
        return list(self.brain.olfactor.X.values())[0]

    @property
    def second_odor_concentration(self):
        return list(self.brain.olfactor.X.values())[1]

    @property
    def first_odor_best_gain(self):
        return list(self.brain.memory.best_gain.values())[0]

    @property
    def second_odor_best_gain(self):
        return list(self.brain.memory.best_gain.values())[1]

    @property
    def best_olfactor_decay(self):
        return self.brain.memory.best_decay_coef

    @property
    def cum_reward(self):
        try:
            return self.brain.memory.rewardSum
        except:
            return self.brain.touch_memory.rewardSum

    @property
    def dt(self):
        return self.model.dt

    @property
    def first_odor_concentration_change(self):
        return list(self.brain.olfactor.dX.values())[0]

    @property
    def second_odor_concentration_change(self):
        return list(self.brain.olfactor.dX.values())[1]

    @property
    def scaled_amount_eaten(self):
        return self.amount_eaten / self.get_real_mass()

    @property
    def x(self):
        return self.pos[0] / self.model.scaling_factor

    @property
    def y(self):
        return self.pos[1] / self.model.scaling_factor

    @property
    def x0(self):
        return self.initial_pos[0] / self.model.scaling_factor

    @property
    def y0(self):
        return self.initial_pos[1] / self.model.scaling_factor

    @property
    def deb_f(self):
        return self.deb.f

    @property
    def deb_f_mean(self):
        return np.mean(self.deb.dict['f'])

    @property
    def gut_occupancy(self):
        return self.deb.gut.occupancy

    @property
    def ingested_body_mass_ratio(self):
        return self.deb.gut.ingested_mass() / self.deb.Ww * 100

    @property
    def ingested_body_volume_ratio(self):
        return self.deb.gut.ingested_volume / self.deb.V * 100

    @property
    def ingested_gut_volume_ratio(self):
        return self.deb.gut.ingested_volume / (self.deb.V * self.deb.gut.V_gm) * 100

    @property
    def ingested_body_area_ratio(self):
        return (self.deb.gut.ingested_volume / self.deb.V) ** (1 / 2) * 100
        # return (self.deb.gut.ingested_volume()/self.deb.V)**(2/3)*100

    @property
    def amount_absorbed(self):
        return self.deb.gut.absorbed_mass('mg')

    @property
    def deb_f_deviation(self):
        return self.deb.f - 1

    @property
    def deb_f_deviation_mean(self):
        return np.mean(np.array(self.deb.dict['f']) - 1)

    @property
    def reserve(self):
        return self.deb.E

    @property
    def reserve_density(self):
        return self.deb.e

    @property
    def structural_length(self):
        return self.deb.L

    @property
    def maturity(self):
        return self.deb.E_H * 1000  # in mJ

    @property
    def reproduction(self):
        return self.deb.E_R * 1000  # in mJ

    @property
    def puppation_buffer(self):
        return self.deb.get_pupation_buffer()

    @property
    def structure(self):
        return self.deb.V * self.deb.E_V * 1000  # in mJ

    @property
    def age_in_hours(self):
        return self.deb.age * 24

    @property
    def hunger(self):
        return self.deb.hunger

    @property
    def death_time_in_hours(self):
        return self.deb.death_time_in_hours

    @property
    def pupation_time_in_hours(self):
        return self.deb.pupation_time_in_hours

    @property
    def birth_time_in_hours(self):
        return self.deb.birth_time_in_hours

    @property
    def hours_as_larva(self):
        return self.deb.hours_as_larva

    @property
    def exploitVSexplore_balance(self):
        return self.brain.intermitter.EEB
