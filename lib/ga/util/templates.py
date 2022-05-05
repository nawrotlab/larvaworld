import sys
import pygame
import random

from pygame.locals import *
from pygame.locals import Color

from lib.ga.geometry.point import Point
from lib.ga.scene.box import Box
from lib.ga.scene.wall import Wall
from lib.conf.base.dtypes import null_dict
from lib.ga.scene.scene import Scene
from lib.ga.util.color import Color
from lib.ga.util.side_panel import SidePanel


class BaseLauncher:
# class BaseTemplate(LarvaWorldSim):
    SCENE_MAX_SPEED = 3000

    SCENE_MIN_SPEED = 1
    SCENE_SPEED_CHANGE_COEFF = 1.5

    SIDE_PANEL_WIDTH = 480

    SCREEN_MARGIN = 12

    def __init__(self, robot_class, scene_file, scene_type, scene_speed=0, env_params=None, dt=0.1, caption='Template',
                 population_num=30):
        if env_params is None:
            env_params = null_dict('env_conf')
        # super().__init__(env_params=env_params, dt=dt)

        self.dt = dt
        self.arena_width, self.arena_height = env_params.arena.arena_dims
        self.arena_shape = env_params.arena.arena_shape
        self.caption = caption
        self.robot_class = robot_class
        self.scene_file = scene_file
        self.scene_type = scene_type
        self.population_num = population_num
        self.scene_speed = scene_speed

        # self.init_scene()
        # self.scene = Scene.load_from_file(self.scene_file, self.scene_speed, self.SIDE_PANEL_WIDTH)

        # self.screen = self.scene.screen
        # self.side_panel = SidePanel(self.scene, self.Nagents)
        # pygame.init()
        # pygame.display.set_caption(self.caption)
        # self.clock = pygame.time.Clock()

    def init_scene(self):
        self.scene = Scene.load_from_file(self.scene_file, self.scene_speed, self.SIDE_PANEL_WIDTH)
        self.scaling_factor = self.scene.width / self.arena_width

        self.screen = self.scene.screen
        self.side_panel = SidePanel(self.scene)

    def increase_scene_speed(self):
        if self.scene.speed < self.SCENE_MAX_SPEED:
            self.scene.speed *= self.SCENE_SPEED_CHANGE_COEFF
        print('scene.speed:', self.scene.speed)

    def decrease_scene_speed(self):
        if self.scene.speed > self.SCENE_MIN_SPEED:
            self.scene.speed /= self.SCENE_SPEED_CHANGE_COEFF
        print('scene.speed:', self.scene.speed)




class RunTemplate(BaseLauncher):
    # N_ROBOTS = 10
    N_INITIAL_BOXES = 0
    N_INITIAL_WALLS = 0
    # DEFAULT_SCENE_FILE = 'saved_scenes/some_boxes_700.txt'
    # DEFAULT_SCENE_SPEED = 30
    # SCENE_MAX_SPEED = 1000
    # SCENE_MIN_SPEED = 1
    # SCENE_SPEED_CHANGE_COEFF = 1.5
    SAVED_SCENE_FILENAME = 'run_template_scene'

    # SCREEN_MARGIN = 12
    # SIDE_PANEL_WIDTH = 400

    BOX_SIZE = 40
    BOX_SIZE_MIN = 20
    BOX_SIZE_INTERVAL = 60

    N_GENOMES_TO_LOAD_FROM_FILE = 10

    def __init__(self, genome_file=None, load_all_genomes=True, **kwargs):
        super().__init__(scene_type='Run', scene_speed=30, **kwargs)
        # if arena is None:
        #     arena = null_dict('arena')
        # self.dt = dt
        # self.arena_width, self.arena_height = arena.arena_dims
        # self.arena_shape = arena.arena_shape
        # self.scaling_factor = 10000
        # self.scene_type = scene_type
        # self.scene = None
        # self.screen = None
        # self.robots = None
        # self.obstacles = None
        # self.side_panel = None
        # self.scene_speed = None
        # self.scene_file = None
        self.genome_file = genome_file
        self.load_all_genomes = load_all_genomes

        # self.parse_cli_arguments()
        pygame.init()
        pygame.display.set_caption(self.caption)
        clock = pygame.time.Clock()
        self.initialize()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    sys.exit()
                elif event.type == KEYDOWN and event.key == K_r:
                    self.initialize()
                elif event.type == KEYDOWN and event.key == K_j:
                    self.add_robots()
                elif event.type == KEYDOWN and event.key == K_k:
                    self.remove_robot()
                # elif event.type == KEYDOWN and event.key == K_COMMA:
                #     add_boxes()
                # elif event.type == KEYDOWN and event.key == K_PERIOD:
                #     remove_box()
                elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                    self.add_box_at_cursor()
                elif event.type == MOUSEBUTTONDOWN and event.button == 3:
                    self.remove_box_at_cursor()
                elif event.type == KEYDOWN and (event.key == K_PLUS or event.key == 93 or event.key == 270):
                    self.increase_scene_speed()
                elif event.type == KEYDOWN and (event.key == K_MINUS or event.key == 47 or event.key == 269):
                    self.decrease_scene_speed()
                elif event.type == KEYDOWN and event.key == K_s:
                    self.scene.save(self.SAVED_SCENE_FILENAME)

            # teleport at the margins
            for robot in self.robots:
                robot.sense_and_act()

                if robot.x < -self.SCREEN_MARGIN:
                    robot.x = self.scene.width + self.SCREEN_MARGIN
                if robot.x > self.scene.width + self.SCREEN_MARGIN:
                    robot.x = -self.SCREEN_MARGIN
                if robot.y < -self.SCREEN_MARGIN:
                    robot.y = self.scene.height + self.SCREEN_MARGIN
                if robot.y > self.scene.height + self.SCREEN_MARGIN:
                    robot.y = -self.SCREEN_MARGIN

            self.screen.fill(Color.BLACK)

            for obj in self.scene.objects:
                obj.draw(self.screen)

                # Draw object label
                # if self.genome_file is not None and issubclass(type(obj), SensorDrivenRobot) \
                #         or self.scene_file != self.DEFAULT_SCENE_FILE:
                #     obj.draw_label(self.screen)
                # obj.draw_label(self.screen)

            # draw a black background for the side panel
            side_panel_bg_rect = pygame.Rect(self.scene.width, 0, self.SIDE_PANEL_WIDTH, self.scene.height)
            pygame.draw.rect(self.screen, Color.BLACK, side_panel_bg_rect)

            self.side_panel.display_info('an obstacle')

            pygame.display.flip()
            int_scene_speed = int(round(self.scene.speed))
            clock.tick(int_scene_speed)

    def build_box(self, x, y, size, color):
        box = Box(x, y, size, color)
        self.obstacles.append(box)
        return box

    def build_wall(self, point1, point2, color):
        wall = Wall(point1, point2, color)
        self.obstacles.append(wall)
        return wall

    def remove_robot(self):
        if len(self.robots) > 0:
            self.scene.remove(self.robots.pop(0))
        print('Number of robots:', len(self.robots))

    def create_boxes(self, number_to_add=1):
        for i in range(number_to_add):
            x = random.randint(0, self.scene.width)
            y = random.randint(0, self.scene.height)

            size = random.randint(self.BOX_SIZE_MIN, self.BOX_SIZE_INTERVAL)
            box = self.build_box(x, y, size, Color.random_bright())
            self.scene.put(box)

    def add_box_at_cursor(self):
        x, y = pygame.mouse.get_pos()
        box = self.build_box(x, y, self.BOX_SIZE, Color.random_bright())
        self.scene.put(box)

    def remove_box_at_cursor(self):
        x, y = pygame.mouse.get_pos()

        for obstacle in self.obstacles:
            if issubclass(type(obstacle), Box):
                box = obstacle

                if x <= box.x + (box.size / 2) and x >= box.x - (box.size / 2) and y <= box.y + (
                        box.size / 2) and y >= box.y - (box.size / 2):
                    self.scene.remove(box)
                    self.obstacles.remove(box)
                    break

    def add_walls(self, number_to_add=1):
        for i in range(number_to_add):
            x1 = random.randint(0, self.scene.width)
            y1 = random.randint(0, self.scene.height)
            point1 = Point(x1, y1)

            x2 = random.randint(0, self.scene.width)
            y2 = random.randint(0, self.scene.height)
            point2 = Point(x2, y2)

            wall = self.build_wall(point1, point2, Color.random_color(127, 127, 127))
            self.scene.put(wall)

    def initialize(self):
        self.robots = []
        self.obstacles = []
        self.init_scene()
        # self.scaling_factor = self.scene.width / self.arena_width
        for obj in self.scene.objects:
            if issubclass(type(obj), Box):
                self.obstacles.append(obj)

        if self.genome_file is None:
            self.add_robots(self.population_num)
        else:
            self.load_genomes_from_file()

        self.create_boxes(self.N_INITIAL_BOXES)
        self.add_walls(self.N_INITIAL_WALLS)

    #
    # def increase_scene_speed(self):
    #     if self.scene.speed < self.SCENE_MAX_SPEED:
    #         self.scene.speed *= self.SCENE_SPEED_CHANGE_COEFF
    #     print('scene.speed:', self.scene.speed)
    #
    # def decrease_scene_speed(self):
    #     if self.scene.speed > self.SCENE_MIN_SPEED:
    #         self.scene.speed /= self.SCENE_SPEED_CHANGE_COEFF
    #     print('scene.speed:', self.scene.speed)

    def parse_cli_arguments(self):
        pass

    def load_genomes_from_file(self):
        pass

    def add_robots(self, number_to_add=1):
        pass
