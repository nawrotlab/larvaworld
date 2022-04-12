import argparse

from lib.ga.obstacle_avoidance_ga import ObstacleAvoidanceGA
from lib.ga.exploration.ga_engine import GaLarvaEngine
from lib.ga.util.scene_type import SceneType


class CliParser:

    def __init__(self):
        self.elitism_num = None
        self.population_num = None
        self.mutation_probability = None
        self.mutation_coefficient = None
        self.robot_random_direction = None
        self.obstacle_sensor_error = None
        self.selection_ratio = None
        self.multicore = None
        self.verbose = None
        self.scene_file = None
        self.scene_speed = None
        self.genome_file = None
        self.load_all_genomes = None
        self.long_lasting_generations = None

    def parse_args(self, default_scene_file, default_scene_speed, scene_type):
        parser = argparse.ArgumentParser()

        if scene_type == SceneType.GA_OBSTACLE_AVOIDANCE:
            parser.add_argument('-v', '--verbose', help='Set verbosity. Default: 0', type=int, choices=range(0, 3))

            parser.add_argument('-p', '--population', help='Number of vehicles in each generation. Default: ' +
                                                           str(GaLarvaEngine.DEFAULT_POPULATION_NUM), type=int, metavar='NUM')

            parser.add_argument('-e', '--elite',
                                help='Number of vehicles carried over unaltered to a new generation. Default: ' + str(
                                    GaLarvaEngine.DEFAULT_ELITISM_NUM), type=int, metavar='NUM')

            parser.add_argument('-m', '--mutation_prob',
                                help='Probability that a mutation occurs on a single gene. Default: ' + str(
                                    GaLarvaEngine.DEFAULT_MUTATION_PROBABILITY), type=float, metavar='NUM')

            parser.add_argument('-M', '--mutation_coeff',
                                help='Coefficient used to alter a gene value during mutation. Default: ' + str(
                                    GaLarvaEngine.DEFAULT_MUTATION_COEFFICIENT), type=float, metavar='NUM')

            parser.add_argument('-S', '--selection_ratio',
                                help='Ratio of parents selected to breed a new generation. Default: ' + str(
                                    GaLarvaEngine.DEFAULT_SELECTION_RATIO), type=float, metavar='NUM')

            parser.add_argument('-r', '--random_direction', help='Set an initial random direction for the vehicles',
                                action='store_true')

            parser.add_argument('-E', '--sensor_error',
                                help='Coefficient used to simulate the obstacle sensor read error. Default: ' + str(
                                    GaLarvaEngine.DEFAULT_OBSTACLE_SENSOR_ERROR) + ', recommended: < 0.2', type=float, metavar='NUM')

            parser.add_argument('-l', '--long_lasting_generations', help='Enable long lasting generations',
                                action='store_true')

            parser.add_argument('-c', '--multicore', help='Enable multicore support (experimental)', action='store_true')

        if scene_type == SceneType.OBSTACLE_AVOIDANCE:
            parser.add_argument('-g', '--genomes', help='Path of the genome file. Default: none', metavar='FILE')

            parser.add_argument('-a', '--load_all_genomes', help='Load all the genomes contained in a genome file. ' +
                                'Applicable with --genomes parameter only', action='store_true')

        parser.add_argument('-s', '--scene', help='Path of the scene file. Default: ' + default_scene_file,
                            metavar='FILE')

        parser.add_argument('-f', '--fps',
                            help='Maximum frame rate (0 = no limit). Default: ' + str(default_scene_speed),
                            type=int, metavar='NUM')

        args = parser.parse_args()

        if scene_type == SceneType.GA_OBSTACLE_AVOIDANCE:
            self.elitism_num = GaLarvaEngine.DEFAULT_ELITISM_NUM if args.elite is None else args.elite
            self.population_num = GaLarvaEngine.DEFAULT_POPULATION_NUM if args.population is None else args.population
            self.mutation_probability = GaLarvaEngine.DEFAULT_MUTATION_PROBABILITY if args.mutation_prob is None else args.mutation_prob
            self.mutation_coefficient = GaLarvaEngine.DEFAULT_MUTATION_COEFFICIENT if args.mutation_coeff is None else args.mutation_coeff
            self.robot_random_direction = args.random_direction
            self.obstacle_sensor_error = GaLarvaEngine.DEFAULT_OBSTACLE_SENSOR_ERROR if args.sensor_error is None else args.sensor_error
            self.selection_ratio = GaLarvaEngine.DEFAULT_SELECTION_RATIO if args.selection_ratio is None else args.selection_ratio
            self.multicore = GaLarvaEngine.MULTICORE if GaLarvaEngine.MULTICORE is not None else args.multicore
            self.verbose = ObstacleAvoidanceGA.DEFAULT_VERBOSE_VALUE if args.verbose is None else args.verbose
            self.long_lasting_generations = args.long_lasting_generations

        if scene_type == SceneType.OBSTACLE_AVOIDANCE:
            self.genome_file = args.genomes
            self.load_all_genomes = args.load_all_genomes

        self.scene_file = default_scene_file if args.scene is None else args.scene
        self.scene_speed = default_scene_speed if args.fps is None else args.fps

        # check parameters value

        if self.scene_speed < 0:
            raise ValueError('FPS argument must be >= 0')

        if scene_type == SceneType.GA_OBSTACLE_AVOIDANCE:
            if self.elitism_num < 0:
                raise ValueError('Elite argument must be >= 0')

            if self.population_num < 2:
                raise ValueError('Population argument must be >= 2')

            if self.population_num <= self.elitism_num:
                raise ValueError('Population argument (' + str(self.population_num) + ') must be > elite argument (' +
                                 str(self.elitism_num) + ')')

            if self.obstacle_sensor_error < 0:
                raise ValueError('Sensor error argument must be >= 0')

            if self.mutation_probability < 0 or self.mutation_probability > 1:
                raise ValueError('Mutation probability must be between 0 and 1')

            if self.mutation_coefficient < 0:
                raise ValueError('Mutation coefficient must be >= 0')

            if self.selection_ratio <= 0 or self.selection_ratio > 1:
                raise ValueError('Selection ratio must be between 0 (exclusive) and 1 (inclusive)')

            if round(self.population_num * self.selection_ratio) < 2:
                raise ValueError('The number of parents selected to breed a new generation is < 2. ' +
                                 'Please increase population (' + str(self.population_num) + ') or selection ratio (' +
                                 str(self.selection_ratio) + ')')
