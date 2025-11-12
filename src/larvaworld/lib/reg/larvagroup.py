"""
Configuration and Generator classes for virtual larva groups.
"""

from __future__ import annotations
from typing import Any, Optional

import copy
import numpy as np
import param

from .. import reg, util

from ..param import (
    ClassAttr,
    Larva_Distro,
    Life,
    NestedConf,
    Odor,
    PositiveInteger,
    generate_xyNor_distro,
)

from ..util import AttrDict

__all__: list[str] = [
    "LarvaGroupMutator",
    "LarvaGroup",
    "GTRvsS",
    "update_larva_groups",
]


def update_larva_groups(lgs: AttrDict, **kwargs: Any) -> AttrDict:
    """
    Modifies the experiment's configuration larvagroups.

    Args:
        lgs (dict): The existing larvagroups in the experiment configuration.
        N (int):: Overwrite the number of agents per larva group.
        models (list): Overwrite the larva models used in the experiment. If not None, a larva group per model ID will be simulated.
        groupIDs (list): The displayed IDs of the groups. If None, the model IDs (mIDs) are used.
        sample: The reference dataset.

    Returns:
        The experiment's configuration larvagroups.

    """
    Nold = len(lgs)
    gIDs = list(lgs)
    confs = prepare_larvagroup_args(default_Nlgs=Nold, **kwargs)
    new_lgs = AttrDict()
    for i, conf in enumerate(confs):
        gID = gIDs[i % Nold]
        gConf = lgs[gID]
        gConf.group_id = gID
        lg = LarvaGroup(**gConf)
        new_lg = lg.new_group(**conf)
        new_lgs[new_lg.group_id] = new_lg.entry(as_entry=False, expand=False)

    return new_lgs


class LarvaGroupMutator(NestedConf):
    """
    Configuration for generating multiple larva groups from model IDs.

    Utility class for creating multiple larva groups by specifying
    model IDs, group IDs, and number of agents per group.

    Attributes:
        modelIDs: List of model IDs to instantiate
        groupIDs: Optional custom IDs for the groups
        N: Number of agents per group

    Example:
        >>> mutator = LarvaGroupMutator(modelIDs=['rover', 'sitter'], N=10)
    """

    modelIDs = reg.conf.Model.confID_selector(single=False)
    groupIDs = param.List(
        default=None,
        allow_None=True,
        item_type=str,
        doc="The ids for the generated datasets",
    )
    N = PositiveInteger(5, label="# agents/group", doc="Number of agents per model ID")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


# TODO : Integration of the following function in the LarvaGroupMutator class
def prepare_larvagroup_args(
    Ns: Optional[int | list[int]] = None,
    modelIDs: Optional[str | list[str]] = None,
    groupIDs: Optional[list[str]] = None,
    colors: Optional[list[Any]] = None,
    default_Nlgs: int = 1,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Prepare the arguments for the larva group configuration.

    Args:
    - Ns (int or list): The number of agents per larva group.
    - modelIDs (str or list): The model IDs of the larva groups.
    - groupIDs (str or list): The group IDs of the larva groups.
    - colors (str or list): The colors of the larva groups.
    - default_Nlgs (int): The default number of larva groups.

    Returns:
    - list: A list of dictionaries containing the arguments for the larva group configuration.

    """
    temp = [len(a) for a in [Ns, modelIDs, groupIDs, colors] if isinstance(a, list)]
    if len(temp) > 0:
        Nlgs = int(np.max(temp))
    else:
        Nlgs = default_Nlgs
    if modelIDs is not None:
        if isinstance(modelIDs, str):
            modelIDs = [copy.deepcopy(modelIDs) for i in range(Nlgs)]
        elif isinstance(modelIDs, list):
            assert len(modelIDs) == Nlgs
        else:
            raise
    else:
        modelIDs = [None] * Nlgs
    if groupIDs is not None:
        assert isinstance(groupIDs, list) and len(groupIDs) == Nlgs
    else:
        groupIDs = modelIDs
    assert len(groupIDs) == Nlgs
    if Ns is not None:
        if isinstance(Ns, list):
            assert len(Ns) == Nlgs
        elif isinstance(Ns, int):
            Ns = [Ns for i in range(Nlgs)]
    else:
        Ns = [None] * Nlgs
    if colors is not None:
        assert isinstance(colors, list) and len(colors) == Nlgs
    elif Nlgs == default_Nlgs:
        colors = [None] * Nlgs
    else:
        colors = util.N_colors(Nlgs)
    return [
        {
            "N": Ns[i],
            "model": modelIDs[i],
            "group_id": groupIDs[i],
            "color": colors[i],
            **kwargs,
        }
        for i in range(Nlgs)
    ]


class LarvaGroup(NestedConf):
    """
    Configuration for a group of larvae with shared properties.

    Defines a collection of larvae sharing the same behavioral model,
    spatial distribution, life history, and optional reference dataset.

    Attributes:
        group_id: Unique identifier for the group
        model: Behavioral model ID or configuration dict
        color: Display color for the group
        odor: Odor signature of the agents
        distribution: Spatial distribution configuration
        life_history: Life history and energetics configuration
        sample: Optional reference dataset to sample from
        imitation: Whether to imitate reference dataset trajectories

    Example:
        >>> group = LarvaGroup(model='rover', group_id='rovers', N=20)
        >>> agents = group(parameter_dict={})
    """

    group_id = param.String("LarvaGroup", doc="The distinct ID of the group")
    model = reg.conf.Model.confID_selector()
    color = param.Color("black", doc="The default color of the group")
    odor = ClassAttr(Odor, doc="The odor of the agent")
    distribution = ClassAttr(
        Larva_Distro, doc="The spatial distribution of the group agents"
    )
    life_history = ClassAttr(Life, doc="The life history of the group agents")
    sample = reg.conf.Ref.confID_selector()
    imitation = param.Boolean(
        default=False, doc="Whether to imitate the reference dataset."
    )

    def __init__(
        self,
        model: Optional[str | dict] = None,
        group_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if group_id is None:
            group_id = model if isinstance(model, str) else "LarvaGroup"

        # Handle model parameter: if dict, set after init to bypass selector validation
        if isinstance(model, dict):
            super().__init__(group_id=group_id, **kwargs)
            # Bypass parameter validation by setting directly via param API
            self.param.model.check_on_set = False
            self.model = model
            self.param.model.check_on_set = True
        else:
            super().__init__(model=model, group_id=group_id, **kwargs)

    def entry(self, expand: bool = False, as_entry: bool = True) -> AttrDict:
        C = self.nestedConf
        if expand:
            C.model = self.expanded_model
        if as_entry:
            return AttrDict({self.group_id: C})
        else:
            return C

    @property
    def expanded_model(self) -> dict:
        m = self.model
        assert m is not None
        if isinstance(m, dict):
            return m
        elif isinstance(m, str):
            return reg.conf.Model.getID(m)
        else:
            raise

    def generate_agent_attrs(
        self, parameter_dict: dict[str, Any] = {}
    ) -> tuple[list[str], list, list, list]:
        m = self.expanded_model
        Nids = self.distribution.N
        if self.sample is not None:
            d = reg.conf.Ref.loadRef(self.sample, load=True, step=False)
            m = d.config.get_sample_bout_distros(m.get_copy())
        else:
            d = None

        if not self.imitation:
            ps, ors = generate_xyNor_distro(self.distribution)
            ids = [f"{self.group_id}_{i}" for i in range(Nids)]

            if d is not None:
                sample_dict = d.sample_larvagroup(
                    N=Nids, ps=[k for k in m.flatten() if m.flatten()[k] == "sample"]
                )
            else:
                sample_dict = {}

        else:
            assert d is not None
            ids, ps, ors, sample_dict = d.imitate_larvagroup(N=Nids)
        sample_dict.update(parameter_dict)

        all_pars = [m.get_copy() for i in range(Nids)]
        if len(sample_dict) > 0:
            for i in range(Nids):
                dic = AttrDict({p: vs[i] for p, vs in sample_dict.items()})
                all_pars[i] = all_pars[i].update_nestdict(dic)
        return ids, ps, ors, all_pars

    def __call__(self, parameter_dict: dict[str, Any] = {}) -> list[dict]:
        ids, ps, ors, all_pars = self.generate_agent_attrs(parameter_dict)
        return self.generate_agent_confs(ids, ps, ors, all_pars)

    def generate_agent_confs(
        self, ids: list[str], ps: list, ors: list, all_pars: list[dict[str, Any]]
    ) -> list[dict]:
        confs = []
        for id, p, o, pars in zip(ids, ps, ors, all_pars):
            conf = {
                "pos": p,
                "orientation": o,
                "color": self.color,
                "unique_id": id,
                "group": self.group_id,
                "odor": self.odor,
                "life_history": self.life_history,
                **pars,
            }
            confs.append(conf)
        return confs

    def new_group(
        self,
        N: Optional[int] = None,
        model: Optional[str] = None,
        group_id: Optional[str] = None,
        color: Optional[str] = None,
        **kwargs: Any,
    ) -> "LarvaGroup":
        kws = self.nestedConf
        if N is not None:
            kws.distribution.N = N
        if model is not None:
            kws.model = model
            if group_id is None:
                group_id = model
        if group_id is not None:
            kws.group_id = group_id
        if color is not None:
            kws.color = color
        kws.update(**kwargs)
        return LarvaGroup(**kws)

    def new_groups(
        self, as_dict: bool = False, **kwargs: Any
    ) -> util.ItemList | AttrDict:
        confs = prepare_larvagroup_args(**kwargs)
        lg_list = util.ItemList([self.new_group(**conf) for conf in confs])
        if not as_dict:
            return lg_list
        else:
            return AttrDict(
                {lg.group_id: lg.entry(as_entry=False, expand=False) for lg in lg_list}
            )


def GTRvsS(
    N: int = 1,
    age: float = 72.0,
    q: float = 1.0,
    h_starved: float = 0.0,
    substrate_type: str = "standard",
    expand: bool = False,
) -> AttrDict:
    """
    Create two larva-groups, 'rover' and 'sitter', based on the respective larva-models, with defined life-history to be used in simulations involving energetics.

    Args:
    - N (int): The number of agents in each group.
    - age (float): The age of the larvae in hours.
    - q (float): The rearing quality of the larvae.
    - h_starved (float): The hours the larvae have been starved just before their current age.
    - substrate_type (str): The type of the rearing substrate.
    - expand (bool): Whether to expand the model configuration.

    Returns:
    - dict: A dictionary of the larva-groups.

    """
    kws0 = {
        "distribution": {"N": N, "scale": (0.005, 0.005)},
        "life_history": Life.prestarved(
            age=age,
            h_starved=h_starved,
            rearing_quality=q,
            substrate_type=substrate_type,
        ),
    }
    return AttrDict(
        {
            id: LarvaGroup(color=c, model=id, **kws0).entry(
                expand=expand, as_entry=False
            )
            for id, c in zip(["rover", "sitter"], ["blue", "red"])
        }
    )
