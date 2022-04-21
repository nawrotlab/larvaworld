import sys
import pygame
import random
import multiprocessing
import math

from pygame import KEYDOWN, K_ESCAPE, K_r, K_j, K_k, MOUSEBUTTONDOWN, K_PLUS, K_MINUS, K_s
from pygame.locals import *
from pygame.locals import Color

from lib.ga.geometry.point import Point
from lib.ga.obstacle_avoidance.ga_robot import GaRobot, GaLarvaRobot
from lib.ga.obstacle_avoidance.genome import Genome
from lib.ga.robot.actuator import Actuator
from lib.ga.robot.motor_controller import MotorController
from lib.ga.scene.box import Box
from lib.ga.scene.wall import Wall
from lib.ga.sensor.proximity_sensor import ProximitySensor
from lib.ga.util.ga_engine import GA_selector
from lib.ga.util.thread_ga_robot import ThreadGaRobot
from lib.conf.base.dtypes import null_dict
from lib.ga.robot.sensor_driven_robot import SensorDrivenRobot
from lib.ga.scene.scene import Scene
from lib.ga.util.color import Color
from lib.ga.util.side_panel import SidePanel
from lib.ga.util.time_util import TimeUtil
from lib.model.envs._larvaworld import LarvaWorld
from lib.model.envs._larvaworld_sim import LarvaWorldSim


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
        # self.side_panel = SidePanel(self.scene, self.population_num)
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


class GALauncher(BaseLauncher):
    # DEFAULT_SCENE_FILE = 'saved_scenes/obstacle_avoidance_900.txt'
    # DEFAULT_SCENE_SPEED = 0  # 0 = maximum frame rate
    # DEFAULT_VERBOSE_VALUE = 0  # 0, 1, 2
    # SCENE_MAX_SPEED = 3000
    # SIDE_PANEL_WIDTH = 480

    # ROBOT_SIZE = 25
    # DEFAULT_POPULATION_NUM = 10
    # DEFAULT_ELITISM_NUM = 3
    # DEFAULT_OBSTACLE_SENSOR_ERROR = 0
    # DEFAULT_MUTATION_PROBABILITY = 0.3  # 0 < MUTATION_PROBABILITY < 1
    # DEFAULT_MUTATION_COEFFICIENT = 0.1
    # DEFAULT_SELECTION_RATIO = 0.3  # 0 < DEFAULT_SELECTION_RATIO < 1
    # LONG_LASTING_GENERATION_STEP_NUM = 1000
    LONG_LASTING_GENERATION_OBSTACLE_PROB_DELTA = 0.0005  # increasing probability to add a new obstacle in the scene.
    BOX_MIN_SIZE = 20
    BOX_MAX_SIZE = 60
    # MULTICORE = True

    engine_kws = {
        'elitism_num': 3,
        'multicore': True,
        'mutation_probability': 0.3,
        'mutation_coefficient': 0.1,
        'selection_ratio': 0.3,
        'long_lasting_generations': None,
        'long_lasting_generation_step_num': 1000,
        'verbose': 0,
        'robot_random_direction': True,
        'obstacle_sensor_error': 0,
    }

    def __init__(self, GA_engine=None, GA_engine_kws={}, **kwargs):
        super().__init__(scene_type='GA', scene_speed=0, **kwargs)
        # if arena is None:
        #     arena = null_dict('arena')
        # self.dt = dt
        # self.arena_width, self.arena_height = arena.arena_dims
        # self.arena_shape = arena.arena_shape
        self.GA_engine = GA_engine
        self.GA_engine_kws = GA_engine_kws
        # self.scene_type = 'GA'
        # self.scene_file = scene_file
        # self.robot_class = robot_class
        # self.scene = None
        # self.screen = None
        #
        # self.side_panel = None
        # self.population_num = None
        # self.scene_speed = None
        # self.engine = None
        # self.elitism_num = None
        # self.robot_random_direction = None
        # self.multicore = None
        # self.obstacle_sensor_error = None
        # self.mutation_probability = None
        # self.mutation_coefficient = None
        # self.selection_ratio = None
        # self.verbose = None
        # self.long_lasting_generations = None
        # self.long_lasting_generation_step_num = self.LONG_LASTING_GENERATION_STEP_NUM

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
                elif event.type == KEYDOWN and (event.key == K_PLUS or event.key == 93 or event.key == 270):
                    self.increase_scene_speed()
                elif event.type == KEYDOWN and (event.key == K_MINUS or event.key == 47 or event.key == 269):
                    self.decrease_scene_speed()
                elif event.type == KEYDOWN and event.key == K_s:
                    self.engine.save_genomes()

            start_time = TimeUtil.current_time_millis()
            self.engine.step()
            end_time = TimeUtil.current_time_millis()
            step_duration = end_time - start_time
            self.printd(2, 'Step duration: ', step_duration)

            self.screen.fill(Color.BLACK)

            for obj in self.scene.objects:
                obj.draw(self.screen)

                if issubclass(type(obj), SensorDrivenRobot) and obj.label is not None:
                    obj.draw_label(self.screen)

            # draw a black background for the side panel
            side_panel_bg_rect = pygame.Rect(self.scene.width, 0, self.SIDE_PANEL_WIDTH, self.scene.height)
            pygame.draw.rect(self.screen, Color.BLACK, side_panel_bg_rect)

            self.display_info()

            pygame.display.flip()
            int_scene_speed = int(round(self.scene.speed))
            clock.tick(int_scene_speed)

    def display_info(self):
        # self.side_panel.display_ga_info()
        pass

    def initialize(self):
        self.init_scene()
        # self.scene = Scene.load_from_file(self.scene_file, self.scene_speed, self.SIDE_PANEL_WIDTH)
        # self.scaling_factor = self.scene.width / self.arena_width

        # self.screen = self.scene.screen
        # self.side_panel = SidePanel(self.scene, self.population_num)
        self.engine = self.GA_engine(scene=self.scene, side_panel=self.side_panel, population_num=self.population_num,
                                     dt=self.dt, scaling_factor=self.scaling_factor,
                                     robot_class=self.robot_class, **self.engine_kws, **self.GA_engine_kws)

    # def increase_scene_speed(self):
    #     if self.scene.speed < self.SCENE_MAX_SPEED:
    #         self.scene.speed *= 1.5
    #
    #     print('Scene speed:', self.scene.speed)
    #
    # def decrease_scene_speed(self):
    #     if self.scene.speed == 0:
    #         self.scene.speed = self.SCENE_MAX_SPEED
    #
    #     if self.scene.speed > 1:
    #         self.scene.speed /= 1.5
    #
    #     print('Scene speed:', self.scene.speed)

    # def parse_cli_arguments(self):
    #     from lib.ga.util.cli_parser import CliParser
    #     parser = CliParser()
    #     parser.parse_args(self.scene_file, self.DEFAULT_SCENE_SPEED, self.scene_type)
    #
    #     self.elitism_num = parser.elitism_num
    #     self.population_num = parser.population_num
    #     self.mutation_probability = parser.mutation_probability
    #     self.mutation_coefficient = parser.mutation_coefficient
    #     self.robot_random_direction = parser.robot_random_direction
    #     self.scene_speed = parser.scene_speed
    #     # self.scene_file = parser.scene_file
    #     self.obstacle_sensor_error = parser.obstacle_sensor_error
    #     self.selection_ratio = parser.selection_ratio
    #     self.multicore = parser.multicore
    #     self.verbose = parser.verbose
    #     self.long_lasting_generations = parser.long_lasting_generations

    def printd(self, min_debug_level, *args):
        if self.engine.verbose >= min_debug_level:
            msg = ''

            for arg in args:
                msg += str(arg) + ' '

            print(msg)


class GAEngineTemplate(GA_selector):
    ROBOT_SIZE = 25

    def __init__(self, robot_class, genome_class, scene, side_panel,robot_random_direction, multicore,
                 obstacle_sensor_error, dt=0.1, scaling_factor=1,**kwargs):
        # print(kwargs)
        super().__init__(**kwargs)

        self.dt = dt
        self.scaling_factor = scaling_factor
        self.scene = scene
        self.side_panel = side_panel
        self.robot_random_direction = robot_random_direction
        self.multicore = multicore
        self.obstacle_sensor_error = obstacle_sensor_error

        self.robots = []
        self.genomes = []
        self.new_obstacle_probability = 0
        self.obstascles_added = []
        self.genome_class = genome_class
        self.robot_class = robot_class

        for i in range(self.population_num):
            x, y = self.robot_start_position()
            genome = self.genome_class.random(self.generation_num)
            self.genomes.append(genome)
            robot = self.build_robot(x, y, genome, None)
            robot.genome = genome
            scene.put(robot)
            self.robots.append(robot)

        self.side_panel.update_ga_data(self.generation_num, None, None)
        self.side_panel.update_ga_time(0, 0)

        print('\nGeneration', self.generation_num, 'started')

        self.printd(1, 'multicore:', self.multicore, 'num_cpu:', self.num_cpu)

    def step(self):
        start_time = TimeUtil.current_time_millis()

        if self.multicore:
            threads = []
            num_robots = len(self.robots)
            num_robots_per_cpu = math.floor(num_robots / self.num_cpu)

            self.printd(2, 'num_robots_per_cpu:', num_robots_per_cpu)

            for i in range(self.num_cpu - 1):
                start_pos = i * num_robots_per_cpu
                end_pos = (i + 1) * num_robots_per_cpu
                self.printd(2, 'core:', i + 1, 'positions:', start_pos, ':', end_pos)
                robot_list = self.robots[start_pos:end_pos]

                thread = ThreadGaRobot(robot_list)
                thread.start()
                self.printd(2, 'thread', i + 1, 'started')
                threads.append(thread)

            # last sublist of robots
            start_pos = (self.num_cpu - 1) * num_robots_per_cpu
            self.printd(2, 'last core, start_pos', start_pos)
            robot_list = self.robots[start_pos:]

            thread = ThreadGaRobot(robot_list)
            thread.start()
            self.printd(2, 'last thread started')
            threads.append(thread)

            for t in threads:
                t.join()

            if self.verbose >= 2:
                end_time = TimeUtil.current_time_millis()
                partial_duration = end_time - start_time
                print('Step partial duration', partial_duration)

            for robot in self.robots[:]:
                # ensure robot doesn't accidentaly go outside of the scene
                if robot.x < 0 or robot.x > self.scene.width or robot.y < 0 or robot.y > self.scene.height:
                    self.destroy_robot(robot)

                # destroy robot if it collides an obstacle
                if robot.collision_with_object:
                    self.destroy_robot(robot)
        else:
            # multicore = False
            for robot in self.robots[:]:
                robot.sense_and_act()

                # ensure robot doesn't accidentaly go outside of the scene
                if robot.x < 0 or robot.x > self.scene.width or robot.y < 0 or robot.y > self.scene.height:
                    self.destroy_robot(robot)

                # destroy robot if it collides an obstacle
                if robot.collision_with_object:
                    self.destroy_robot(robot)

        # create new obstacles for long lasting generations
        if not self.long_lasting_generations and self.generation_step_num == self.long_lasting_generation_step_num:
            print('Time limit reached (' + str(self.long_lasting_generation_step_num) +
                  ' steps), destroying all remaining robots')

            for robot in self.robots[:]:
                self.destroy_robot(robot)

        # check population extinction
        if not self.robots:
            print('Generation', self.generation_num, 'terminated')
            self.create_new_generation()
            self.build_genomes()
            self.side_panel.update_ga_data(self.generation_num, self.best_genome, self.best_genome.fitness)

        # update statistics time
        current_time = TimeUtil.current_time_millis()
        total_time_seconds = math.floor((current_time - self.start_total_time) / 1000)
        generation_time_seconds = math.floor((current_time - self.start_generation_time) / 1000)
        self.side_panel.update_ga_time(total_time_seconds, generation_time_seconds)

        self.generation_step_num += 1

    def build_robot(self, x, y, genome, label):
        # pass

        return None

    def destroy_robot(self, robot):
        # save fitness value
        fitness_value = self.get_fitness(robot)
        robot.genome.fitness = fitness_value

        self.scene.remove(robot)
        self.robots.remove(robot)
        self.printd(1, 'Destroyed robot with fitness value', fitness_value)

    def build_genomes(self):
        # draw a label for the elite individuals
        elite_label = 1

        for genome in self.genomes:
            if elite_label <= self.elitism_num:
                label = elite_label
                elite_label += 1
            else:
                label = None

            x, y = self.robot_start_position()
            robot = self.build_robot(x, y, genome, label)
            robot.genome = genome
            self.scene.put(robot)
            self.robots.append(robot)

        self.new_obstacle_probability = 0

        # remove all obstacles added to a long lasting generation
        for box in self.obstascles_added:
            self.scene.remove(box)

        self.obstascles_added = []

    def robot_start_position(self):
        x = self.scene.width / 2
        y = self.scene.height / 2
        return x, y


    def create_box(self):
        x = random.randint(0, self.scene.width)
        y = random.randint(0, self.scene.height)

        size = random.randint(self.BOX_MIN_SIZE, self.BOX_MAX_SIZE)
        return Box(x, y, size, Color.random_bright())

    def save_genomes(self):
        if not self.genomes_last_generation:
            # this hapeens at generation 1 only
            genomes_to_save = self.genomes
        else:
            genomes_to_save = sorted(self.genomes_last_generation, key=lambda genome: genome.fitness, reverse=True)

        date_time = TimeUtil.format_date_time()
        file_name = "genomes_" + date_time + ".txt"
        file_path = 'saved_genomes/' + file_name

        with open(file_path, 'w') as f:
            line1 = '# generation_num and fitness are ignored when a genome file is loaded'
            line2 = '# This is the structure of each line:'
            line3 = '# robot_wheel_radius motor_ctrl_coefficient motor_ctrl_min_actuator_value ' + \
                    'sensor_delta_direction sensor_saturation_value sensor_max_distance generation_num fitness'
            f.write(line1 + '\n')
            f.write(line2 + '\n')
            f.write(line3 + '\n')
            f.write('\n')

            for genome in genomes_to_save:
                line = genome.get_saved_genome_repr()
                f.write(line + '\n')
        f.closed
        print('Genomes saved:', file_path)

    def get_fitness(self, robot):
        return None


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
