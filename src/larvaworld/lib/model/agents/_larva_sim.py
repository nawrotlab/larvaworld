import numpy as np
from shapely import geometry
from larvaworld.lib import aux
from larvaworld.lib.model.agents._larva import LarvaMotile



class LarvaSim(LarvaMotile):
    def __init__(self, larva_pars, **kwargs):
        super().__init__(**larva_pars, **kwargs)

    @property
    def border_collision(self):
        if len(self.model.borders) == 0:
            return False
        else:
            p0=geometry.Point(self.pos)
            d0 = self.sim_length / 4
            oM = self.head.get_orientation()
            sensor_ray = aux.radar_tuple(p0=p0, angle=oM, distance=d0)
            min_dst, nearest_obstacle = aux.detect_nearest_obstacle(self.model.borders, sensor_ray, p0)

            if min_dst is None:
                return False
            else:
                return True

    def assess_collisions(self, lin_vel, ang_vel):
        if not self.model.larva_collisions:
            ids = self.model.detect_collisions(self.unique_id)
            larva_collision = False if len(ids) == 0 else True
        else:
            larva_collision = False
        if larva_collision:
            lin_vel = 0
            ang_vel += np.sign(ang_vel) * np.pi / 10
            return lin_vel, ang_vel
        res = self.border_collision
        d_ang = np.pi / 20
        if not res:
            return lin_vel, ang_vel
        else:
            lin_vel = 0
            ang_vel += np.sign(ang_vel) * d_ang
            return lin_vel, ang_vel


    def complete_step(self):
        if self.head.get_linearvelocity() < 0:
            self.negative_speed_errors += 1
            self.head.set_lin_vel(0)
        if not self.model.Box2D:
            try:
                self.model.space.move_agent(self, self.pos)
            except:
                self.model.space.move_to(self, np.array(self.pos))
        self.update_larva()


