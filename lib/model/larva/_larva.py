import random

import mesa
import nengo
import numpy as np
from copy import deepcopy

from lib.model.envs._space import agents_spatial_query
from lib.model.larva._effectors import Crawler, Feeder, Oscillator_coupling, Intermitter, Olfactor, Turner
from lib.model.larva._sensorimotor import VelocityAgent
from lib.model.larva._bodies import LarvaBody
from lib.aux import functions as fun, naming as nam
from lib.model.larva.deb import DEB
from lib.model.larva.nengo_effectors import FlyBrain, NengoManager


class Larva(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id=unique_id, model=model)
        self.default_color = self.model.generate_larva_color()
        self.behavior_pars = ['stride_stop', 'stride_id', 'pause_id', 'feed_id', 'Lturn_id', 'Rturn_id']
        self.null_behavior_dict = dict(zip(self.behavior_pars, [False]*len(self.behavior_pars)))

    def update_color(self, default_color, behavior_dict, mode='lin'):
        color = deepcopy(default_color)
        if mode=='lin' :
            if behavior_dict['stride_stop'] :
                color=np.array([0, 255, 0])
            elif behavior_dict['stride_id']:
                color = np.array([0, 150, 0])
            elif behavior_dict['pause_id'] :
                color=np.array([255, 0, 0])
            elif behavior_dict['feed_id'] :
                color=np.array([150, 150, 0])
        elif mode=='ang' :
            if behavior_dict['Lturn_id'] :
                color[2]=150
            elif behavior_dict['Rturn_id'] :
                color[2]=50
        return color

class LarvaReplay(Larva, LarvaBody):
    def __init__(self, unique_id, model,schedule,length=5,data=None):
        Larva.__init__(self, unique_id=unique_id, model=model)

        self.schedule = schedule
        self.data=data
        self.pars=self.data.columns.values
        self.Nticks = len(self.data.index.unique().values)
        self.t0 = self.data.index.unique().values[0]

        d=self.model.dataset
        self.spinepoint_xy_pars=[p for p in fun.flatten_list(d.points_xy) if p in self.pars]
        self.Npoints=int(len(self.spinepoint_xy_pars)/2)

        self.contour_xy_pars=[p for p in fun.flatten_list(d.contour_xy) if p in self.pars]
        self.Ncontour = int(len(self.contour_xy_pars) / 2)

        self.centroid_xy_pars=[p for p in d.cent_xy if p in self.pars]

        Nsegs=self.model.draw_Nsegs
        if Nsegs is not None :
            if Nsegs==self.Npoints-1 :
                self.orientation_pars = [p for p in nam.orient(d.segs) if p in self.pars]
                self.Nors = len(self.orientation_pars)
                self.Nangles = 0
                if self.Nors!=Nsegs :
                    raise ValueError (f'Orientation values are not present for all body segments : {self.Nors} of {Nsegs}')
            elif Nsegs==2 :
                self.orientation_pars = [p for p in ['front_orientation'] if p in self.pars]
                self.Nors = len(self.orientation_pars)
                self.angle_pars = [p for p in ['bend'] if p in self.pars]
                self.Nangles = len(self.angle_pars)
                if self.Nors != 1 or self.Nangles!=1:
                    raise ValueError(f'{self.Nors} orientation and {Nsegs} angle values are present and 1,1 are needed.')
        else :
            self.Nors, self.Nangles = 0, 0


        # self.angle_pars=[p for p in d.angles + ['bend'] if p in self.pars]
        # self.Nangles=len(self.angle_pars)
        #
        # self.orientation_pars=[p for p in nam.orient(d.segments) + ['front_orientation', 'rear_orientation'] if p in self.pars]
        # self.Nors = len(self.orientation_pars)

        self.chunk_ids = None
        self.trajectory = []
        self.color = deepcopy(self.default_color)
        self.length = length

        if self.Npoints > 0 :
            self.spinepoint_positions_ar = self.data[self.spinepoint_xy_pars].values
            self.spinepoint_positions_ar=self.spinepoint_positions_ar.reshape([self.Nticks,self.Npoints, 2])
        else :
            self.spinepoint_positions_ar = np.ones([self.Nticks, self.Npoints, 2])*np.nan

        if self.Ncontour > 0 :
            self.contourpoint_positions_ar = self.data[self.contour_xy_pars].values
            self.contourpoint_positions_ar=self.contourpoint_positions_ar.reshape([self.Nticks,self.Ncontour, 2])
        else :
            self.contourpoint_positions_ar = np.ones([self.Nticks, self.Ncontour, 2])*np.nan

        if len(self.centroid_xy_pars) == 2 :
            self.centroid_position_ar = self.data[self.centroid_xy_pars].values
        else :
            self.centroid_position_ar = np.ones([self.Nticks, 2])*np.nan

        if len(self.model.pos_xy_pars) == 2 :
            self.position_ar = self.data[self.model.pos_xy_pars].values
        else :
            self.position_ar = np.ones([self.Nticks, 2])*np.nan

        if self.Nangles > 0 :
            self.spineangles_ar = self.data[self.angle_pars].values
        else :
            self.spineangles_ar = np.ones([self.Nticks, self.Nangles])*np.nan

        if self.Nors > 0 :
            self.orientations_ar = self.data[self.orientation_pars].values
        else :
            self.orientations_ar = np.ones([self.Nticks, self.Nors])*np.nan

        vp_behavior=[p for p in self.behavior_pars if p in self.pars]
        self.behavior_ar=np.zeros([self.Nticks, len(self.behavior_pars)], dtype=bool)
        for i,p in enumerate(self.behavior_pars) :
            if p in vp_behavior :
                self.behavior_ar[:,i]=np.array([not v for v in np.isnan(self.data[p].values).tolist()])

        if self.model.draw_Nsegs is not None:
            LarvaBody.__init__(self, model, pos=self.position_ar[0], orientation=self.orientations_ar[0][0],
                               initial_length=self.length/1000, length_std=0, Nsegs=self.model.draw_Nsegs, interval=0)


    def step(self):
        step = self.schedule.steps
        self.spinepoint_positions = self.spinepoint_positions_ar[step].tolist()
        self.vertices = self.contourpoint_positions_ar[step]
        self.centroid_position = self.centroid_position_ar[step]
        self.position = self.position_ar[step]
        if not np.isnan(self.position).any():
            self.model.space.move_agent(self, self.position)
        self.trajectory=self.position_ar[ :step, :].tolist()
        self.spineangles = self.spineangles_ar[step]
        self.orientations = self.orientations_ar[step]
        if self.model.color_behavior :
            behavior_dict = dict(zip(self.behavior_pars, self.behavior_ar[step,:].tolist()))
            self.color=self.update_color(self.default_color, behavior_dict)
        else :
            self.color=self.default_color
        if self.model.draw_Nsegs is not None:
            segs=self.segs

            if len(self.spinepoint_positions) == len(segs) + 1:
                for i, seg in enumerate(segs):
                    pos = [np.nanmean([self.spinepoint_positions[i][j], self.spinepoint_positions[i + 1][j]]) for j in [0, 1]]
                    o=np.deg2rad(self.orientations[i])
                    seg.set_position(pos)
            # elif self.Nors == len(segs):
            #     for i, seg in enumerate(segs):
                    seg.set_orientation(o)
                    seg.update_vertices(pos,o)
            elif len(segs) == 2 and self.Nors == 1 and self.Nangles == 1:
                l1,l2=[self.length * r for r in self.seg_ratio]
                x, y = self.position
                h_or = np.deg2rad(self.orientations[0])
                b_or = np.deg2rad(self.orientations[0] - self.spineangles[0])
                p_head = np.array(fun.rotate_around_point(origin=[x, y], point=[l1 + x, y],radians=-h_or))
                p_tail = np.array(fun.rotate_around_point(origin=[x, y], point=[l2 + x, y],radians=np.pi - b_or))
                pos1 = [np.nanmean([p_head[j], [x, y][j]]) for j in [0, 1]]
                pos2 = [np.nanmean([p_tail[j], [x, y][j]]) for j in [0, 1]]
                segs[0].set_position(pos1)
                segs[0].set_orientation(h_or)
                segs[0].update_vertices(pos1, h_or)
                segs[1].set_position(pos2)
                segs[1].set_orientation(b_or)
                segs[1].update_vertices(pos2, b_or)
                self.spinepoint_positions = np.array([p_head, self.position, p_tail])

    def get_position(self):
        return np.array(self.position)

    def draw(self, viewer):
        if self.model.draw_contour :
            if self.model.draw_Nsegs is not None :
                for seg in self.segs:
                    seg.set_color(self.color)
                    seg.draw(viewer)
            elif len(self.vertices)>0:
                viewer.draw_polygon(self.vertices, filled=True, color=self.color)
        if self.model.draw_centroid :
            if not np.isnan(self.centroid_position).any():
                pos=self.centroid_position
            elif not np.isnan(self.position).any():
                pos=self.position
            else :
                pos =None
            if pos is not None :
                viewer.draw_circle(radius=.1, position=pos, filled=True, color=self.color, width=1)
        if self.model.draw_midline and self.Npoints>1:
            if not np.isnan(self.spinepoint_positions[0]).any() :
                viewer.draw_polyline(self.spinepoint_positions, color=(0, 0, 255), closed=False, width=.07)
                for i, seg_pos in enumerate(self.spinepoint_positions):
                    c = 255 * i / (len(self.spinepoint_positions)-1)
                    color = (c, 255-c, 0)
                    viewer.draw_circle(radius=.07, position=seg_pos, filled=True, color=color, width=.01)

class LarvaSim(VelocityAgent, Larva):
    def __init__(self,unique_id, model, fly_params, **kwargs):
        Larva.__init__(self, unique_id=unique_id, model=model)
        # State variables
        self.Nlayers = len(self.model.odor_layers)
        self.odor_concentrations = np.zeros(self.Nlayers)
        self.olfactory_activation = 0
        self.real_length = None
        self.real_mass = None
        self.__dict__.update(fly_params)
        self.modules = fly_params['neural_params']['component_params']

        if self.modules['energetics']:
            if fly_params['energetics_params']['deb'] :
                self.deb = DEB(species='default', steps_per_day=24 * 60, cv=0, aging=True)
                self.deb.reach_stage('larva')
                self.deb.steps_per_day = int(24 * 60 * 60 / self.model.dt)
                self.real_length = self.deb.get_real_L()
                self.real_mass = self.deb.get_W()

            else :
                self.deb = None
                self.food_to_biomass_ratio = fly_params['energetics_params']['food_to_biomass_ratio']

        VelocityAgent.__init__(self,**fly_params['sensorimotor_params'], **fly_params['body_params'], **kwargs)



        # self.__dict__.update(fly_params['neural_params']['component_params'])



        self.nengo_feeder_reset_to_be_deleted = False

        # Initialize oscillators



        self.nengo_olfactor, self.nengo_three_oscillators=self.modules['nengo_olfactor'], self.modules['nengo_three_oscillators']
        if self.nengo_olfactor or self.nengo_three_oscillators:
            self.nengo_manager = NengoManager(**fly_params['neural_params']['nengo_params'])
            self.flybrain = FlyBrain()
            self.flybrain.build(self.nengo_manager, olfactor=self.nengo_olfactor, num_odor_layers=self.Nlayers,
                                three_oscillators=self.nengo_three_oscillators, **fly_params['neural_params']['nengo_params'], )
            self.flybrain_sim = nengo.Simulator(self.flybrain, dt=0.01)
            self.flybrain_iterations_per_step = int(self.model.dt / self.flybrain_sim.dt)
            # print(self.flybrain_iterations_per_step)
        else:
            self.nengo_manager = None
            self.flybrain = None

        if self.modules['crawler']:
            self.crawler = Crawler(dt=self.model.dt, **fly_params['neural_params']['crawler_params'])
            self.crawler.start_effector()
        else:
            self.crawler = None

        if self.modules['turner']:
            self.turner = Turner(dt=self.model.dt, **fly_params['neural_params']['turner_params'])
            self.turner.start_effector()
        else:
            self.turner = None

        if self.modules['feeder']:
            self.feeder = Feeder(dt=self.model.dt, model=self.model, **fly_params['neural_params']['feeder_params'])
            self.feeder.stop_effector()
            self.reset_feeder()
            self.max_feed_amount = self.compute_max_feed_amount()
        else:
            self.feeder = None

        if self.modules['interference']:
            self.osc_coupling = Oscillator_coupling(**fly_params['neural_params']['interference_params'])
            # self.crawler_interference_free_window = fly_params['neural_params']['interference_params'][
            #     'crawler_interference_free_window']
            # self.crawler_interference_start = fly_params['neural_params']['interference_params'][
            #     'crawler_interference_start']
            # self.feeder_interference_free_window = fly_params['neural_params']['interference_params'][
            #     'feeder_interference_free_window']
            # self.feeder_interference_start = fly_params['neural_params']['interference_params'][
            #     'feeder_interference_start']
            # self.turner_continuous = fly_params['neural_params']['interference_params']['turner_continuous']
            # self.turner_rebound = fly_params['neural_params']['interference_params']['turner_rebound']
            # self.interference_ratio = fly_params['neural_params']['interference_params']['interference_ratio']

        else:
            self.osc_coupling = Oscillator_coupling()

        # Initialize modulators
        if self.modules['intermitter']:
            self.intermitter = Intermitter(dt=self.model.dt,
                                           crawler=self.crawler, turner=self.turner, feeder=self.feeder,
                                           nengo_manager=self.nengo_manager,
                                           **fly_params['neural_params']['intermitter_params'])
            self.intermitter.start_effector()
        else :
            self.intermitter=None

        # Initialize sensors
        if self.modules['olfactor']:
            self.olfactor = Olfactor(dt=self.model.dt, odor_layers=self.model.odor_layers,
                                     **fly_params['neural_params']['olfactor_params'])
        else:
            self.olfactor = None

        # print(self.real_mass, self.real_length)
        # raise



    # def adapt_oscillator_amp_and_threshold(self, length, freq=None, free_window=None, scaled_stride_step=None):
    #     distance = length * scaled_stride_step
    #     amp = distance * np.pi * freq / (np.sin(np.pi * free_window))
    #     thr = np.cos(np.pi * free_window)
    #     return amp, thr

    def compute_next_action(self):
        # print(self.unique_id)
        # Here starts the nervous system step function

        # Sensation
        # print(self.real_length * 1000, self.real_mass * 1000)

        self.sim_time += self.model.dt
        if self.Nlayers > 0:
            pos = self.get_olfactor_position()
            self.odor_concentrations = self.sense_odors(self.model.odor_layers, pos)


        if self.flybrain:
            man = self.nengo_manager
            if self.nengo_olfactor:
                man.set_odor_concentrations(self.odor_concentrations)
            self.flybrain_sim.run_steps(self.flybrain_iterations_per_step, progress_bar=False)
            # TODO Right now the nengo turner is not modulated by olfaction
            # TODO Right now the feeder deoes not work
            if self.nengo_three_oscillators:
                lin=self.flybrain.mean_lin_s(self.flybrain_sim.data, self.flybrain_iterations_per_step)* self.get_sim_length()
                lin_noise = np.random.normal(scale=man.crawler_noise * self.get_sim_length())
                self.set_lin_activity(lin+lin_noise)
                ang=self.flybrain.mean_ang_s(self.flybrain_sim.data, self.flybrain_iterations_per_step)
                ang_noise = np.random.normal(scale=man.turner_noise)
                self.set_ang_activity(ang+ang_noise)
                nengo_feeding_event = self.flybrain.feed_event(self.flybrain_sim.data, self.flybrain_iterations_per_step)
                if nengo_feeding_event & (not self.nengo_feeder_reset_to_be_deleted):
                    self.feeder_motion = True
                    self.nengo_feeder_reset_to_be_deleted = True
                else:
                    self.feeder_motion = False
                    self.nengo_feeder_reset_to_be_deleted = False

        # Use a simple gain on the concentration change since last timestep
        if self.olfactor is not None:
            self.olfactory_activation = self.olfactor.step(self.odor_concentrations)

        elif self.nengo_olfactor:
            self.olfactory_activation = 100 * self.flybrain.mean_odor_change(self.flybrain_sim.data, self.flybrain_iterations_per_step)
        else:
            pass

        # Intermission
        if self.intermitter is not None:
            self.intermitter.step()

        # Step the feeder
        self.feed_success=False
        self.current_amount_eaten = 0
        if self.feeder is not None:
            self.feeder.step()
            if self.feeder.complete_iteration:
                # TODO fix the radius so that it works with any feeder, nengo included
                success, amount_eaten = self.detect_food(mouth_position=self.get_olfactor_position(),
                                                                   radius=self.feeder.feed_radius * self.sim_length,
                                                                   grid=self.model.food_grid,
                                                                   max_amount_eaten=self.max_feed_amount)



                if success :
                    self.feed_success_counter += 1
                    self.feed_success = True
                    self.current_amount_eaten = amount_eaten
                    self.amount_eaten += amount_eaten
                    # if self.modules['energetics']:
                    #     # TODO Connect this to metabolism
                    #     # This adjusts the real_length to mass**(1/3). We need the real_length already, to feed it to the crawler
                    #
                    self.intermitter.feeder_reoccurence_rate = self.intermitter.feeder_reoccurence_rate_on_success
                else:
                    # TODO Maybe intermit here
                    self.intermitter.feeder_reoccurence_rate /= (np.exp(self.intermitter.feeder_reoccurence_decay_coef))

        if self.modules['energetics']:
            if self.deb :
                self.deb.run(f=int(self.feed_success))
                self.real_length = self.deb.get_real_L()
                self.real_mass = self.deb.get_W()
                # if not self.deb.alive :
                #     raise ValueError ('Dead')
            else :
                self.real_mass += self.current_amount_eaten * self.food_to_biomass_ratio
                self.adjust_shape_to_mass()
            self.adjust_body_vertices()
            self.max_feed_amount = self.compute_max_feed_amount()

        if self.crawler is not None:
            self.set_lin_activity(self.crawler.step(self.get_sim_length()))

        # ... and finally step the turner...
        if self.turner is not None:
            self.osc_coupling.step(crawler=self.crawler, feeder=self.feeder)
            self.set_head_contacts_ground(value=self.osc_coupling.turner_inhibition)
            self.set_ang_activity(self.turner.step(inhibited=self.osc_coupling.turner_inhibition,
                                                   interference_ratio=self.osc_coupling.interference_ratio,
                                                   A_olf=self.olfactory_activation))

    def sense_odors(self, odor_layers, pos):
        values = [odor_layers[id].get_value(pos) for id in odor_layers]
        if self.olfactor.noise:
            values = [v + np.random.normal(scale=v * self.olfactor.noise) for v in values]
        return values

        # See computation in MANUAL scientific diary, 20.02.2020

    def detect_food(self, mouth_position, radius=None, grid=None, max_amount_eaten=1.0):
        if grid:
            cell = grid.get_grid_cell(mouth_position)
            if grid.get_value(cell) > 0:
                subtracted_value = grid.subtract_value(cell, max_amount_eaten)
                return True, subtracted_value
            else:
                return False, 0
        else:
            # s = time.time()
            accessible_food = agents_spatial_query(pos=mouth_position, radius=radius,
                                                   agent_list=self.model.get_food())
            # e = time.time()
            # print(e-s)
            # print(len(accessible_food))
            if accessible_food:
                food = random.choice(accessible_food)
                amount_eaten = food.subtract_amount(amount=max_amount_eaten)
                return True, amount_eaten
            else:
                return False, 0

    def reset_feeder(self):
        self.feed_success_counter = 0
        self.amount_eaten = 0
        try:
            self.feeder.reset()
        except:
            pass

    def compute_max_feed_amount(self):
        return self.feeder.max_feed_amount_ratio * self.real_mass
        pass

    def build_modules(self, modules):

        pass
