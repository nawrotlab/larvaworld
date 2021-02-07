import random

import mesa
import nengo
import numpy as np
from copy import deepcopy

from nengo import Simulator

from lib.model.envs._space import agents_spatial_query
from lib.model.larva._effectors import Crawler, Feeder, Oscillator_coupling, Intermitter, Olfactor, Turner, DefaultBrain
from lib.model.larva._sensorimotor import VelocityAgent
from lib.model.larva._bodies import LarvaBody
from lib.aux import functions as fun, naming as nam
from lib.model.larva.deb import DEB
from lib.model.larva.nengo_effectors import NengoBrain, NengoManager, NengoEffector


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
                raise ValueError(f'Defined number of segments {Nsegs} must be either 2 or {self.Npoints-1}')
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

        self.build_energetics(fly_params['energetics_params'])
        VelocityAgent.__init__(self,**fly_params['sensorimotor_params'], **fly_params['body_params'], **kwargs)
        self.brain=self.build_brain(fly_params['neural_params'])
        self.reset_feeder()
    def compute_next_action(self):
        self.sim_time += self.model.dt

        self.odor_concentrations = self.sense_odors(self.model.Nodors, self.model.odor_layers)
        lin, ang, self.feeder_motion, self.olfactory_activation = self.brain.run(self.odor_concentrations, self.get_sim_length())
        self.set_ang_activity(ang)
        self.set_lin_activity(lin)
        self.feed_attempt(self.feeder_motion)

        if self.energetics:
            self.run_energetics(self.feed_success, self.current_amount_eaten)
    def sense_odors(self, Nodors, odor_layers):
        if Nodors==0 :
            return []
        else :
            pos = self.get_olfactor_position()
            values = [odor_layers[id].get_value(pos) for id in odor_layers]
            if self.brain.olfactor.noise :
                values = [v + np.random.normal(scale=v * self.brain.olfactor.noise) for v in values]
            return values


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

    def feed_attempt(self, motion=False):
        if motion :
            r=self.brain.feeder.feed_radius * self.sim_length
            # TODO fix the radius so that it works with any feeder, nengo included
            self.feed_success, self.current_amount_eaten = self.detect_food(mouth_position=self.get_olfactor_position(),
                                                     radius=r,
                                                     grid=self.model.food_grid,
                                                     max_amount_eaten=self.max_feed_amount)

            self.feed_success_counter += int(self.feed_success)
            self.amount_eaten += self.current_amount_eaten
        else :
            self.feed_success = False
            self.current_amount_eaten = 0


    def reset_feeder(self):
        self.feed_success_counter = 0
        self.amount_eaten = 0
        self.feeder_motion=False
        try :
            self.max_feed_amount = self.compute_max_feed_amount()
        except :
            self.max_feed_amount=None
        try:
            self.brain.feeder.reset()
        except:
            pass

    def compute_max_feed_amount(self):
        return self.brain.feeder.max_feed_amount_ratio * self.real_mass


    def build_energetics(self, energetic_pars):
        self.real_length = None
        self.real_mass = None
        if energetic_pars is not None :
            self.energetics=True
            if energetic_pars['deb']:
                self.deb = DEB(species='default', steps_per_day=24 * 60, cv=0, aging=True)
                self.deb.reach_stage('larva')
                self.deb.steps_per_day = int(24 * 60 * 60 / self.model.dt)
                self.real_length = self.deb.get_real_L()
                self.real_mass = self.deb.get_W()
            else:
                self.deb = None
                self.food_to_biomass_ratio = energetic_pars['food_to_biomass_ratio']
        else :
            self.energetics=False

    def build_brain(self, neural_params):
        modules=neural_params['modules']
        if neural_params['nengo'] :
            brain=NengoBrain()
            brain.setup(agent=self, modules=modules, conf=neural_params)
            brain.build(brain.nengo_manager, olfactor=brain.olfactor)
            brain.sim = Simulator(brain, dt=0.01)
            brain.Nsteps = int(self.model.dt / brain.sim.dt)
        else :
            brain= DefaultBrain(agent=self, modules=modules, conf=neural_params)
        return brain

    def run_energetics(self, feed_success, amount_eaten):
        if self.deb:
            self.deb.run(f=int(feed_success))
            self.real_length = self.deb.get_real_L()
            self.real_mass = self.deb.get_W()
            # if not self.deb.alive :
            #     raise ValueError ('Dead')
        else:
            self.real_mass += amount_eaten * self.food_to_biomass_ratio
            self.adjust_shape_to_mass()
        self.adjust_body_vertices()
        self.max_feed_amount = self.compute_max_feed_amount()
