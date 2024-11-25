"""
Larvaworld parameter class and associated methods
"""

import random
import sys
import typing
from types import FunctionType

import numpy as np
import param

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from .. import reg, util
from ..util import nam

__all__ = [
    "SAMPLING_PARS",
    "sample_ps",
    "get_LarvaworldParam",
    "prepare_LarvaworldParam",
    "build_LarvaworldParam",
]


class LarvaworldParam(param.Parameterized):
    """
    LarvaworldParam is a class that extends param.Parameterized and provides a structured way to define and manage parameters
    for the Larvaworld package. Each parameter has several attributes and methods to facilitate its use and manipulation.

    """

    p = param.String(default="", doc="Name of the parameter")
    d = param.String(default="", doc="Dataset name of the parameter")
    disp = param.String(default="", doc="Displayed name of the parameter")
    k = param.String(default="", doc="Key of the parameter")
    sym = param.String(default="", doc="Symbol of the parameter")
    codename = param.String(default="", doc="Name of the parameter in code")
    flatname = param.String(
        default=None, doc="Name of the parameter in model configuration"
    )
    dtype = param.Parameter(default=float, doc="Data type of the parameter value")
    func = param.Callable(
        default=None,
        doc="Function to get the parameter from a dataset",
        allow_None=True,
    )
    required_ks = param.List(
        default=[], doc="Keys of prerequired parameters for computation in a dataset"
    )

    @property
    def s(self):
        return self.disp

    @property
    def l(self):
        return self.disp + "  " + self.ulabel

    @property
    def symunit(self):
        return self.sym + "  " + self.ulabel

    @property
    def ulabel(self):
        if self.u == reg.units.dimensionless:
            return ""
        else:
            return "(" + self.unit + ")"

    @property
    def unit(self):
        if self.u == reg.units.dimensionless:
            return "-"
        else:
            return rf"${self.u}$"

    @property
    def short(self):
        return self.k

    @property
    def v0(self):
        return self.param.v.default

    @property
    def initial_value(self):
        return self.param.v.default

    @property
    def value(self):
        return self.v

    @property
    def symbol(self):
        return self.sym

    @property
    def label(self):
        return self.param.v.label

    @property
    def parameter(self):
        return self.disp

    @property
    def tooltip(self):
        return self.param.v.doc

    @property
    def description(self):
        return self.param.v.doc

    @property
    def help(self):
        return self.param.v.doc

    @property
    def parclass(self):
        return type(self.param.v)

    @property
    def min(self):
        try:
            vmin, vmax = self.param.v.bounds
            return vmin
        except:
            return None

    @property
    def max(self):
        try:
            vmin, vmax = self.param.v.bounds
            return vmax
        except:
            return None

    @property
    def lim(self):
        try:
            lim = self.param.v.bounds
            return lim
        except:
            return None

    @property
    def step(self):
        if (
            self.parclass in [param.Number, param.Range]
            and self.param.v.step is not None
        ):
            return self.param.v.step
        elif self.parclass == param.Magnitude:
            return 0.01
        elif self.parclass in [param.NumericTuple]:
            return 0.01
        else:
            return None

    @property
    def Ndec(self):
        if self.step is not None:
            return str(self.step)[::-1].find(".")
        else:
            return None

    def exists(self, dataset):
        """
        Check if the parameter exists in the given LarvaDataset.

        Args:
            dataset (LarvaDataset): The dataset to check for the parameter.

        Returns:
            AttrDict: A dictionary-like object with two keys:
                - 'step': A boolean indicating if the parameter exists in the dataset's step_data.
                - 'end': A boolean indicating if the parameter exists in the dataset's endpoint_data.

        """
        return util.AttrDict(
            {"step": self.d in dataset.step_ps, "end": self.d in dataset.end_ps}
        )

    def get(self, dataset, compute=True):
        """
        Retrieve the parameter's value from the dataset if it exists, otherwise compute it.

        Args:
            dataset (LarvaDataset): The dataset object from which to retrieve the parameter.
            compute (bool): Flag indicating whether to compute the parameter if it does not exist. Default is True.

        Returns:
            The parameter value if it exists or is successfully computed, otherwise None.

        Raises:
            None

        Notes:
            - The method first checks if the parameter exists in the dataset.
            - If the parameter exists, it retrieves and returns it.
            - If the parameter does not exist and `compute` is True, it computes the parameter and retries retrieval.
            - If the parameter does not exist and `compute` is False, it prints a message indicating the parameter was not found.

        """
        res = self.exists(dataset)
        for key, exists in res.items():
            if exists:
                return dataset.get_par(key=key, par=self.d)

        if compute:
            self.compute(dataset)
            return self.get(dataset, compute=False)
        else:
            print(f"Parameter {self.disp} not found")

    def compute(self, dataset):
        """
        Compute the parameter using the provided dataset.

        This method applies the parameter's computing function to the dataset if the function is defined.
        If the function is not defined, it prints a message indicating that the
        function is not defined.

        Args:
            dataset (LarvaDataset) : The dataset to be processed by the function.

        """
        if self.func is not None:
            self.func(dataset)
        else:
            print(f"Function to compute parameter {self.disp} is not defined")

    def randomize(self):
        """
        Randomizes the value of the parameter based on its type.

        This method assigns a random value to `self.v` depending on the type of
        the parameter class (`self.parclass`). The behavior varies as follows:

        - If the parameter is a `Number` or its subclass, a random float within
          the parameter's bounds is assigned.
        - If the parameter is an `Integer` or its subclass, a random integer
          within the parameter's bounds is assigned.
        - If the parameter is a `Magnitude` or its subclass, a random float
          between 0.0 and 1.0 is assigned.
        - If the parameter is a `Selector` or its subclass, a random choice
          from the parameter's objects is assigned.
        - If the parameter is a `Boolean`, a random boolean value (True or False)
          is assigned.
        - If the parameter is a `Range` or its subclass, a tuple of two random
          floats within the parameter's bounds is assigned, where the second
          float is greater than or equal to the first.

        """
        p = self.parclass
        if p in [param.Number] + param.Number.__subclasses__():
            vmin, vmax = self.param.v.bounds
            self.v = self.param.v.crop_to_bounds(
                np.round(random.uniform(vmin, vmax), self.Ndec)
            )
        elif p in [param.Integer] + param.Integer.__subclasses__():
            vmin, vmax = self.param.v.bounds
            self.v = random.randint(vmin, vmax)
        elif p in [param.Magnitude] + param.Magnitude.__subclasses__():
            self.v = np.round(random.uniform(0.0, 1.0), self.Ndec)
        elif p in [param.Selector] + param.Selector.__subclasses__():
            self.v = random.choice(self.param.v.objects)
        elif p == param.Boolean:
            self.v = random.choice([True, False])
        elif p in [param.Range] + param.Range.__subclasses__():
            vmin, vmax = self.param.v.bounds
            vv0 = np.round(random.uniform(vmin, vmax), self.Ndec)
            vv1 = np.round(random.uniform(vv0, vmax), self.Ndec)
            self.v = (vv0, vv1)

    def mutate(self, Pmut, Cmut):
        """
        Mutates the value of the parameter based on its class type.

        Parameters
        ----------
        Pmut : float
            Probability of mutation.
        Cmut : float
            Coefficient of mutation.

        Notes
        -----
        - For `param.Magnitude` and its subclasses, the value is mutated using a Gaussian distribution and cropped to bounds.
        - For `param.Integer` and its subclasses, the value is mutated using a Gaussian distribution, converted to an integer, and cropped to bounds.
        - For `param.Number` and its subclasses, the value is mutated using a Gaussian distribution and cropped to bounds.
        - For `param.Selector` and its subclasses, the value is randomly chosen from the available objects.
        - For `param.Boolean`, the value is randomly chosen between True and False.
        - For `param.Range` and its subclasses, the range values are mutated using a Gaussian distribution, clipped to bounds, and rounded.

        """
        if random.random() < Pmut:
            if self.parclass in [param.Magnitude] + param.Magnitude.__subclasses__():
                v0 = self.v if self.v is not None else 0.5
                vv = random.gauss(v0, Cmut)
                self.v = self.param.v.crop_to_bounds(np.round(vv, self.Ndec))
                # self.v = np.round(self.v, self.Ndec)
            elif self.parclass in [param.Integer] + param.Integer.__subclasses__():
                vmin, vmax = self.param.v.bounds
                vr = np.abs(vmax - vmin)
                v0 = self.v if self.v is not None else int(vmin + vr / 2)
                vv = random.gauss(v0, Cmut * vr)
                self.v = self.param.v.crop_to_bounds(int(vv))
            elif self.parclass in [param.Number] + param.Number.__subclasses__():
                vmin, vmax = self.param.v.bounds
                vr = np.abs(vmax - vmin)
                v0 = self.v if self.v is not None else vmin + vr / 2
                vv = random.gauss(v0, Cmut * vr)
                self.v = self.param.v.crop_to_bounds(np.round(vv, self.Ndec))
            elif self.parclass in [param.Selector] + param.Selector.__subclasses__():
                self.v = random.choice(self.param.v.objects)
            elif self.parclass == param.Boolean:
                self.v = random.choice([True, False])
            elif self.parclass in [param.Range] + param.Range.__subclasses__():
                vmin, vmax = self.param.v.bounds
                vr = np.abs(vmax - vmin)
                v0, v1 = self.v if self.v is not None else (vmin, vmax)
                vv0 = random.gauss(v0, Cmut * vr)
                vv1 = random.gauss(v1, Cmut * vr)
                vv0 = np.round(np.clip(vv0, a_min=vmin, a_max=vmax), self.Ndec)
                vv1 = np.round(np.clip(vv1, a_min=vv0, a_max=vmax), self.Ndec)
                self.v = (vv0, vv1)


def get_LarvaworldParam(vparfunc, v0=None, dv=None, **kws):
    class _LarvaworldParam(LarvaworldParam):
        v = vparfunc
        u = param.Parameter(
            default=reg.units.dimensionless, doc="Unit of the parameter values"
        )

    par = _LarvaworldParam(**kws)
    return par


SAMPLING_PARS = util.bidict(
    util.AttrDict(
        {
            "length": "body.length",
            nam.freq(nam.scal(nam.vel(""))): "brain.crawler.freq",
            # nam.freq(nam.scal(nam.vel(''))): 'brain.intermitter.crawl_freq',
            nam.mean(
                nam.chunk_track("stride", nam.scal(nam.dst("")))
            ): "brain.crawler.stride_dst_mean",
            nam.std(
                nam.chunk_track("stride", nam.scal(nam.dst("")))
            ): "brain.crawler.stride_dst_std",
            nam.freq("feed"): "brain.feeder.freq",
            nam.max(
                nam.chunk_track("stride", nam.scal(nam.vel("")))
            ): "brain.crawler.max_scaled_vel",
            nam.phi(nam.max(nam.scal(nam.vel("")))): "brain.crawler.max_vel_phase",
            "attenuation": "brain.interference.attenuation",
            nam.max("attenuation"): "brain.interference.attenuation_max",
            nam.freq(nam.vel(nam.orient("front"))): "brain.turner.freq",
            nam.phi(nam.max("attenuation")): "brain.interference.max_attenuation_phase",
        }
    )
)


def sample_ps(ps, e=None):
    """
    Get the parameters from the given list `ps` that exist on the inverse `SAMPLING_PARS` dictionary.

    Args:
        ps (list): A list of parameters.
        e (optional): The endpoint dataframe of a dataset. If provided parameters are further filtered to exist in `e`.

    Returns:
        list: A list of parameters, filtered to exist in the default `SAMPLING_PARS` dictionary and potentially filtered to exist in `e`.

    """
    Sinv = SAMPLING_PARS.inverse
    ps = util.SuperList([Sinv[k] for k in util.existing_cols(Sinv, ps)]).flatten
    if e:
        ps = ps.existing(e)
    return ps


def get_vfunc(dtype, lim, vs):
    """
    Returns the appropriate Param class based on the provided data type, value limit, and value options.

    Parameters
    ----------
    dtype (type): The data type of the parameter.
    lim (tuple): A tuple representing the limit or range for the parameter.
    vs (any): The value options of the parameter.

    Returns
    -------
    param.Parameter: The corresponding Param class for the given data type, limit, and value set.

    """
    func_dic = {
        float: param.Number,
        int: param.Integer,
        str: param.String,
        bool: param.Boolean,
        dict: param.Dict,
        list: param.List,
        type: param.ClassSelector,
        typing.List[int]: param.List,
        typing.List[str]: param.List,
        typing.List[float]: param.List,
        typing.List[typing.Tuple[float]]: param.List,
        FunctionType: param.Callable,
        typing.Tuple[float]: param.Range,
        typing.Tuple[int]: param.NumericTuple,
        TypedDict: param.Dict,
    }
    if dtype == float and lim == (0.0, 1.0):
        return param.Magnitude
    if type(vs) == list and dtype in [str, int]:
        return param.Selector
    elif dtype in func_dic.keys():
        return func_dic[dtype]
    else:
        return param.Parameter


def vpar(vfunc, v0, doc, lab, lim, dv, vs):
    """
    Create a parameter object with specified attributes.

    Parameters
    ----------
    vfunc (type): The parameter type (e.g., param.List, param.Number, param.Range, param.Selector).
    v0 (any): The default value for the parameter.
    doc (str): Documentation string for the parameter.
    lab (str): Label for the parameter.
    lim (tuple or None): Bounds for the parameter if applicable (e.g., for param.Number or param.Range).
    dv (any or None): Step value for the parameter if applicable (e.g., for param.Number or param.Range).
    vs (list or None): List of objects for the parameter if applicable (e.g., for param.Selector).

    Returns
    -------
    param.Parameter: An instantiated parameter object with the specified attributes.

    """
    f_kws = {"default": v0, "doc": doc, "label": lab, "allow_None": True}
    if vfunc in [param.List, param.Number, param.Range]:
        if lim is not None:
            f_kws["bounds"] = lim
    if vfunc in [param.Range, param.Number]:
        if dv is not None:
            f_kws["step"] = dv
    if vfunc in [param.Selector]:
        f_kws["objects"] = vs
    func = vfunc(**f_kws, instantiate=True)
    return func


def prepare_LarvaworldParam(
    p,
    k=None,
    dtype=float,
    d=None,
    disp=None,
    sym=None,
    codename=None,
    lab=None,
    doc=None,
    flatname=None,
    required_ks=[],
    u=reg.units.dimensionless,
    v0=None,
    v=None,
    lim=None,
    dv=None,
    vs=None,
    vfunc=None,
    vparfunc=None,
    func=None,
    **kwargs,
):
    """
    Method that formats the dictionary of attributes for a parameter in order to create a LarvaworldParam instance.

    Parameters
    ----------
    - p (str): The primary parameter name.
    - k (str, optional): The key for the parameter. Defaults to None.
    - dtype (type, optional): The data type of the parameter. Defaults to float.
    - d (str, optional): The code-related name for the parameter. Defaults to None.
    - disp (str, optional): The display name for the parameter. Defaults to None.
    - sym (str, optional): The symbol for the parameter. Defaults to None.
    - codename (str, optional): The code name for the parameter. Defaults to None.
    - lab (str, optional): The label for the parameter. Defaults to None.
    - doc (str, optional): The documentation string for the parameter. Defaults to None.
    - flatname (str, optional): The flat name for the parameter. Defaults to None.
    - required_ks (list, optional): List of required keys. Defaults to [].
    - u (unit, optional): The unit of the parameter. Defaults to reg.units.dimensionless.
    - v0 (any, optional): The initial value of the parameter. Defaults to None.
    - v (any, optional): The value of the parameter. Defaults to None.
    - lim (tuple, optional): The limits for the parameter. Defaults to None.
    - dv (any, optional): The delta value for the parameter. Defaults to None.
    - vs (any, optional): The value set for the parameter. Defaults to None.
    - vfunc (callable, optional): The value function for the parameter. Defaults to None.
    - vparfunc (callable, optional): The value parameter function for the parameter. Defaults to None.
    - func (callable, optional): The computing function for the parameter. Defaults to None.
    - **kwargs: Additional keyword arguments.

    Returns
    -------
    - util.AttrDict: A dictionary of formatted attributes for creating a LarvaworldParam instance.

    """
    codename = p if codename is None else codename
    d = p if d is None else d
    disp = d if disp is None else disp
    k = k if k is not None else d
    v0 = v if v is not None else v0

    if flatname is None:
        if p in SAMPLING_PARS:
            flatname = SAMPLING_PARS[p]
        else:
            flatname = p

    if sym is None:
        sym = k

    if dv is None:
        if dtype in [
            float,
            typing.List[float],
            typing.List[typing.Tuple[float]],
            typing.Tuple[float],
        ]:
            dv = 0.01
        elif dtype in [int]:
            dv = 1
        else:
            pass

    if vparfunc is None:
        if vfunc is None:
            vfunc = get_vfunc(dtype=dtype, lim=lim, vs=vs)
        if lab is None:
            if u == reg.units.dimensionless:
                lab = f"{disp}"
            else:
                ulab = rf"${u}$"
                lab = rf"{disp} ({ulab})"
        doc = lab if doc is None else doc
        vparfunc = vpar(vfunc, v0, doc, lab, lim, dv, vs)
    else:
        vparfunc = vparfunc()

    return util.AttrDict(
        {
            "name": p,
            "p": p,
            "d": d,
            "k": k,
            "disp": disp,
            "sym": sym,
            "codename": codename,
            "flatname": flatname,
            "dtype": dtype,
            "func": func,
            "u": u,
            "required_ks": required_ks,
            "vparfunc": vparfunc,
            "dv": dv,
            "v0": v0,
        }
    )


def build_LarvaworldParam(p, **kwargs):
    """
    Constructs a Larvaworld parameter object.

    This function prepares the input attributes using the `prepare_LarvaworldParam` function
    and then generates the Larvaworld parameter object using the `get_LarvaworldParam` function.

    Args:
        p: The primary parameter required for building the Larvaworld parameter object.
        **kwargs: Additional keyword arguments that are passed to the `prepare_LarvaworldParam` function.

    Returns:
        The constructed Larvaworld parameter object.

    """
    pre_p = prepare_LarvaworldParam(p=p, **kwargs)
    return get_LarvaworldParam(**pre_p)
