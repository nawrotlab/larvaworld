import param
from param import Selector,String, ListSelector, Magnitude, Boolean, List

from larvaworld.lib.param import OptionalPositiveNumber, OptionalSelector, PositiveInteger, IntegerRange, \
    OptionalPositiveInteger, ClassAttr, NestedConf, PositiveNumber


class FramerateOps(NestedConf):
    fr = PositiveNumber(10, softmax=100, step=0.1, label='framerate',
                            doc='The tracking/simulation framerate (inverse of timestep) in Hz.')
    dt = PositiveNumber(0.1, softmax=1.0, step=0.01, label='timestep',
                        doc='The tracking/simulation timestep (inverse of tracking framerate) in seconds.')
    constant_framerate = param.Boolean(True, doc='Whether the tracking framerate is constant.')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if 'dt' in kwargs:
            self.update_framerate()
        elif 'fr' in kwargs:
            self.update_timestep()

    @param.depends('dt', watch=True)
    def update_framerate(self):
        self.fr = 1 / self.dt

    @param.depends('fr', watch=True)
    def update_timestep(self):
        self.dt = 1 / self.fr
        # raise

class XYops(NestedConf):
    # dt = PositiveNumber(0.1, softmax=1.0, step=0.01, label='tracker timestep',doc='The tracking timestep (inverse of tracking framerate) in seconds.')
    # constant_framerate = Boolean(True, doc='Whether the tracking framerate is constant.')
    XY_unit = Selector(default='m', objects=['m', 'mm'], doc='The spatial unit of the XY coordinate data')
    Npoints = PositiveInteger(3, softmax=20, label='# midline 2D points',
                              doc='The number of points tracked along the larva midline.')
    Ncontour = PositiveInteger(0, softmax=100, label='# contour 2D points',
                               doc='The number of points tracked around the larva contour.')

class Resolution(FramerateOps,XYops):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

class SimTimeOps(FramerateOps):
    duration = OptionalPositiveNumber(5.0, softmax=100.0, step=0.1,
                                          doc='The duration of the simulation in minutes.')
    Nsteps = OptionalPositiveInteger(label='# simulation timesteps', doc='The number of simulation timesteps.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_Nsteps()

    @param.depends('duration', 'dt', watch=True)
    def update_Nsteps(self):
        self.Nsteps = int(self.duration * 60 / self.dt)

    @param.depends('Nsteps', watch=True)
    def update_duration(self):
        self.duration = self.Nsteps * self.dt / 60


class LabFormatFilesystem(NestedConf):
    read_sequence = List(label='data columns',doc='The sequence of columns in the tracker-exported files.')
    read_metadata = Boolean(False, doc='Whether metadata files are available for the tracker-exported files/folders.')
    folder_pref = String(doc='A prefix for detecting a raw-data folder.')
    folder_suff = String(doc='A suffix for detecting a raw-data folder.')
    file_pref = String(doc='A prefix for detecting a raw-data file.')
    file_suf = String(doc='A suffix for detecting a raw-data file.')
    file_sep = String(doc='A separator for detecting a raw-data file.')



class LabFormat(NestedConf) :
    resolution = ClassAttr(Resolution, doc='The dataset metadata')
    filesystem = ClassAttr(LabFormatFilesystem, doc='The import-relevant lab-format filesystem')

class SimMetricOps(XYops):
    bend = Selector(objects=['from_vectors', 'from_angles'],
                    doc='Whether bending angle is computed as a sum of sequential segmental angles or as the angle between front and rear body vectors.')
    front_vector = IntegerRange((1, 2), softbounds=(-12, 12),
                                doc='The initial & final segment of the front body vector.')
    rear_vector = IntegerRange((-2, -1), softbounds=(-12, 12),
                               doc='The initial & final segment of the rear body vector.')
    front_body_ratio = Magnitude(0.5,
                                 doc='The fraction of the body considered front, relevant for bend computation from angles.')
    point_idx = OptionalPositiveInteger(softmax=20,
                                        doc='Index of midline point to use as the larva spatial position. Default is None meaning use the centroid.')
    use_component_vel = Boolean(False,
                                doc='Whether to use the component velocity ralative to the axis of forward motion.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_vectors()

    @param.depends('Npoints', watch=True)
    def update_vectors(self):
        N=self.Npoints
        # self.param.params('front_vector').softbounds = (-N,N)
        self.param.params('front_vector').bounds = (-N,N)
        # self.param.params('rear_vector').softbounds = (-N,N)
        self.param.params('rear_vector').bounds = (-N,N)
        self.param.params('point_idx').bounds=(hardmin, hardmax) = (0,N)
        self.point_idx=self.param.params('point_idx').crop_to_bounds(self.point_idx)



class SimGeneralOps(NestedConf):
    Box2D = param.Boolean(False,doc='Whether to use the Box2D physics engine or not.')
    # store_data = param.Boolean(True, doc='Whether to store the simulation data')
    larva_collisions = param.Boolean(True, doc='Whether to allow overlap between larva bodies.')
    offline = param.Boolean(False,doc='Whether to launch a full Larvaworld environment')
    multicore = param.Boolean(False,doc='Whether to use multiple cores')
    show_display = param.Boolean(True,doc='Whether to launch the pygame-visualization.')

    def __init__(self,offline=False, show_display=True, **kwargs):
        if offline:
            show_display=False
        super().__init__(show_display=show_display,offline=offline,**kwargs)
        # Define constant parameters
        self.scaling_factor = 1000.0 if self.Box2D else 1.0


    @param.depends('offline','show_display', watch=True)
    def disable_display(self):
        if self.offline :
            self.show_display=False




class PreprocessConf(NestedConf):
    rescale_by = OptionalPositiveNumber(softmax=1000.0, step=0.001, doc='Whether to rescale spatial coordinates by a scalar in meters.')
    filter_f = OptionalPositiveNumber(softmax=5.0, step=0.01, doc='Whether to filter spatial coordinates by a grade-1 low-pass filter of the given cut-off frequency.')
    transposition = OptionalSelector(['origin', 'arena', 'center'],doc='Whether to transpose spatial coordinates.')
    interpolate_nans = Boolean(False, doc='Whether to interpolate missing values.')
    drop_collisions = Boolean(False,doc='Whether to drop timepoints where larva collisions are detected.')

class EnrichConf(NestedConf):
    pre_kws = ClassAttr(PreprocessConf,doc='The preprocessing pipelines')
    proc_keys = ListSelector(default=['angular', 'spatial', 'dispersion', 'tortuosity'],objects=['angular', 'spatial', 'source', 'dispersion', 'tortuosity', 'PI', 'wind'], doc='The processing pipelines')
    anot_keys = ListSelector(default=['bout_detection', 'bout_distribution', 'interference'], objects=['bout_detection', 'bout_distribution', 'interference', 'source_attraction', 'patch_residency'], doc='The annotation pipelines')
    recompute = Boolean(False,doc='Whether to recompute')
    mode = Selector(objects=['minimal', 'full'],doc='The processing mode')



# class DataProcessOps(NestedConf):
#     enrichment = aux.ClassAttr(EnrichConf, doc='The spatiotemporal resolution')
#     metric_definition = ClassAttr(SimMetricOps,doc='The metric_definition')