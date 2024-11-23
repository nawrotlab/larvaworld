"""
This module provides classes and functions for managing configurations in the larvaworld package.
It includes classes for configuration types and the unique reference type, as well as functions for resetting configurations.
"""

import os
import param

from ... import vprint, CONF_DIR, CONFTYPES
from .. import reg, util, funcs
from ..param import ClassDict, OptionalSelector

__all__ = [
    "next_idx",
    "ConfType",
    "RefType",
    "conf",
    "resetConfs",
]


def next_idx(id, conftype="Exp"):
    """
    Increment and retrieve the next index for a given configuration type and id.

    Args:
        id (str): The id for which the index is to be incremented.
        conftype (str, optional): The type of configuration. Defaults to 'Exp'.

    Returns:
        int: The next index for the given configuration type and id.

    Raises:
        IOError: If there is an issue reading or writing the index file.

    Notes:
        - The function reads from and writes to a file named 'SimIdx.txt' located in the directory specified by `larvaworld.CONF_DIR`.
        - If the file does not exist, it initializes a new dictionary with default configuration types.
        - If the given configuration type or id does not exist in the dictionary, they are initialized.

    """
    f = f"{CONF_DIR}/SimIdx.txt"
    if not os.path.isfile(f):
        d = util.AttrDict({k: {} for k in ["Exp", "Batch", "Essay", "Eval", "Ga"]})
    else:
        d = util.load_dict(f)
    if conftype not in d:
        d[conftype] = util.AttrDict()
    if id not in d[conftype]:
        d[conftype][id] = 0
    d[conftype][id] += 1
    util.save_dict(d, f)
    return d[conftype][id]


class ConfType(param.Parameterized):
    """
    ConfType is a class that manages different configuration types and their associated dictionaries. It provides methods to load, save, update, and manipulate these configurations.

    Attributes:
        conftype (param.Selector): Selector for available configuration types.
        dict (ClassDict): The configuration dictionary.
        CONFTYPE_SUBKEYS (util.AttrDict): A dictionary of configuration type subkeys built by the `build_ConfTypeSubkeys` method.

    """

    conftype = param.Selector(objects=CONFTYPES, doc="The configuration type")
    dict = ClassDict(
        default=util.AttrDict(), item_type=None, doc="The configuration dictionary"
    )

    def __init__(self, **kwargs):
        """
        Initializes the configuration object.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the superclass initializer.

        Attributes:
            CONFTYPE_SUBKEYS (dict): A dictionary of configuration type subkeys built by the `build_ConfTypeSubkeys` method.

        """
        super().__init__(**kwargs)
        self.CONFTYPE_SUBKEYS = self.build_ConfTypeSubkeys()
        self.update_dict()

    def build_ConfTypeSubkeys(self):
        """
        Constructs a nested dictionary representing configuration subkeys for different configuration types.

        The method initializes a dictionary with keys from `larvaworld.CONFTYPES` and empty dictionaries as values.
        It then updates this dictionary with predefined subkey mappings.

        Returns:
            util.AttrDict: A dictionary-like object with the combined configuration subkeys.

        """
        d0 = {k: {} for k in CONFTYPES}
        d1 = {
            "Batch": {"exp": "Exp"},
            "Ga": {"env_params": "Env"},
            "Exp": {
                "env_params": "Env",
                "trials": "Trial",
                "larva_groups": "Model",
            },
        }
        d0.update(d1)
        return util.AttrDict(d0)

    @property
    def path_to_dict(self):
        """
        Constructs a file path string based on the configuration directory and type.

        Returns:
            str: The constructed file path in the format '{larvaworld.CONF_DIR}/{self.conftype}.txt'.

        """
        return f"{CONF_DIR}/{self.conftype}.txt"

    @param.depends("conftype", watch=True)
    def update_dict(self):
        """
        Updates the dictionary configuration parameter.

        This method is triggered when the 'conftype' parameter changes (actually when it is initialized).
        It updates the item type of the 'dict' parameter to match the 'dict_entry_type' and then
        reloads the configuration.
        """
        self.param.params("dict").item_type = self.dict_entry_type
        self.load()

    def getID(self, id):
        """
        Retrieve the configuration stored under an id from the dictionary.
        If the provided id is a list, the method will recursively retrieve the configuration under each id in the list.

        Args:
            id (int, str, or list): The id(s) to retrieve. Can be a single id or a list of ids.

        Returns:
            The configuration(s) (nested dictionary(ies)) associated with the provided id(s) from the dictionary.

        Raises:
            ValueError: If the provided id does not exist in the dictionary.

        """
        if isinstance(id, list):
            return [self.getID(i) for i in id]
        if id in self.dict:
            return self.dict[id]
        else:
            vprint(f"{self.conftype} Configuration {id} does not exist", 1)
            raise ValueError()

    def get(self, id):
        """
        Retrieve a generator for a configuration entry by id.
        If the provided id is a list, recursively retrieve entries for each id in the list.

        Args:
            id (Union[str, list]): The id of the configuration entry to retrieve, or a list of ids.

        Returns:
            Union[conf_class, list]: A generator for the configuration entry or a list of generators for configuration entries.

        """
        if isinstance(id, list):
            return [self.get(i) for i in id]
        entry = self.getID(id)
        return self.conf_class(**entry, name=id)

    def load(self):
        """
        Loads the dictionary storing the configurations from the specified path and assigns it to the instance variable `dict`.
        This method reads the dictionary from the file located at `self.path_to_dict`.

        Raises:
            FileNotFoundError: If the file at `self.path_to_dict` does not exist.
            IOError: If there is an error reading the file.

        """
        self.dict = util.load_dict(self.path_to_dict)

    def save(self):
        """
        Saves the current state of the dictionary to the specified path.
        This method saves the dictionary (`self.dict`) to the file located at `self.path_to_dict`.

        Returns:
            bool: True if the save operation was successful, False otherwise.

        """
        return util.save_dict(self.dict, self.path_to_dict)

    def set_dict(self, d):
        """
        Sets the dictionary for the configuration type and saves it.

        Args:
            d (dict): The dictionary to be set.

        Returns:
            None

        """
        self.param.params("dict").item_type = self.dict_entry_type
        self.dict = d
        self.save()

    def reset(self, init=False):
        """
        Resets the configuration dictionary from the built-in dictionary of predefined configurations.

        If the configuration dictionary file exists:
            - If `init` is True, prints the current number of entries and returns.
            - If `init` is False, updates the current dictionary with the stored dictionary and prints the updated number of entries.

        If the configuration dictionary file does not exist:
            - Initializes the dictionary with the stored dictionary and prints the number of entries.

        Args:
            init (bool): Flag to indicate if the reset is for initialization purposes. Defaults to False.

        """
        if os.path.isfile(self.path_to_dict):
            if init:
                vprint(
                    f"{self.conftype} configuration dict exists with {len(self.dict)} entries",
                    1,
                )
                return
            else:
                d = self.dict
                Ncur = len(d)
                d.update(self.stored_dict)
                self.set_dict(d)
                vprint(
                    f"{self.conftype} configuration dict of {Ncur} entries enriched to {len(self.dict)}",
                    1,
                )
        else:
            self.set_dict(self.stored_dict)
            vprint(
                f"{self.conftype} configuration dict initialized with {len(self.dict)} entries",
                1,
            )

    def selectIDs(self, dic={}):
        """
        Selects and returns a list of ids from the configuration dictionary that match the criteria specified in the given dictionary.

        Args:
            dic (dict, optional): A dictionary containing key-value pairs to match against the ids' attributes. Defaults to an empty dictionary.

        Returns:
            util.SuperList: A list of ids that match the criteria specified in the dictionary.

        """
        valid = util.SuperList()
        for id in self.confIDs:
            c = self.getID(id).flatten()
            if all([(k in c and c[k] == v) for k, v in dic.items()]):
                valid.append(id)
        return valid

    def setID(self, id, conf, mode="overwrite"):
        """
        Sets the configuration for a given id.

        Parameters
        ----------
        id (str): The identifier for the configuration.
        conf (object): The configuration object to be stored under the id.
        mode (str, optional): The mode of setting the configuration.
                      It can be 'overwrite' (default) or 'update'.

        Behavior:
        - If the id already exists in the dictionary and the mode is 'update',
          the existing configuration is updated with the new configuration.
        - Otherwise, the new configuration is set for the given id.
        - The configuration is saved after setting.
        - If the configuration type is 'Model', updates related parameters in
          genetic algorithm selectors and mutators.

        Side Effects:
        - Calls the save method to update the dictionary.
        - Updates genetic algorithm parameters if the configuration type is 'Model'.
        - Prints a message indicating the configuration has been saved.

        """
        if id in self.dict and mode == "update":
            self.dict[id] = self.dict[id].update_nestdict(conf.flatten())
        else:
            self.dict[id] = conf
        self.save()
        if self.conftype == "Model":
            from ..sim.genetic_algorithm import GAselector

            GAselector.param.objects()["base_model"].objects = self.confIDs
            reg.larvagroup.LarvaGroupMutator.param.objects()[
                "modelIDs"
            ].objects = self.confIDs
            reg.larvagroup.LarvaGroup.param.objects()["model"].objects = self.confIDs
        vprint(f"{self.conftype} Configuration saved under the id : {id}", 1)

    def delete(self, id=None):
        """
        Delete a configuration entry by its id.

        Parameters
        ----------
        id (optional): The id of the configuration entry to delete. If not provided, no action is taken.

        Behavior:
        - If the provided id exists in the configuration dictionary, the entry is removed.
        - The updated configuration dictionary is saved.
        - A message indicating the deletion is printed.

        """
        if id is not None:
            if id in self.dict:
                self.dict.pop(id, None)
                self.save()
                vprint(f"Deleted {self.conftype} configuration under the id : {id}", 1)

    def expand(self, id=None, conf=None):
        """
        Expands the configuration dictionary by resolving subkeys and their ids.

        Args:
            id (str, optional): The identifier for the configuration. Defaults to None.
            conf (dict, optional): The configuration dictionary to expand. If None, it will be fetched from self.dict using the provided id. Defaults to None.

        Returns:
            dict or None: The expanded configuration dictionary if successful, otherwise None.

        """
        if conf is None:
            if id in self.dict:
                conf = self.dict[id]
            else:
                return None
        subks = self.CONFTYPE_SUBKEYS[self.conftype]
        if len(subks) > 0:
            for subID, subk in subks.items():
                ids = reg.conf[subk].confIDs
                if subID == "larva_groups" and subk == "Model":
                    for k, v in conf["larva_groups"].items():
                        if v.model in ids:
                            v.model = reg.conf[subk].getID(v.model)
                else:
                    if conf[subID] in ids:
                        conf[subID] = reg.conf[subk].getID(conf[subID])

        return conf

    # @param.depends('confIDs','dict', watch=True)
    def confID_selector(self, default=None, single=True):
        """
        A configuration selector object.

        Parameters
        ----------
        - default: The default configuration id to select (optional).
        - single (bool): If True, returns an OptionalSelector for a single selection.
                 If False, returns a ListSelector for multiple selections.

        Returns
        -------
        - OptionalSelector or ListSelector: A selector object for configuration IDs.

        """
        kws = {
            "default": default,
            "objects": self.confIDs,
            "label": f"{self.conftype} configuration ID",
            "doc": f"Selection among stored {self.conftype} configurations by ID",
        }
        if single:
            return OptionalSelector(**kws)
        else:
            return param.ListSelector(**kws)

    @property
    def reset_func(self):
        """
        Shortcut for the reset configuration function.

        Returns:
            function: The stored default configuration function that resets the dictionary for the given configuration type.

        """
        return funcs.stored_confs[self.conftype]

    @property
    def stored_dict(self):
        """
        Returns the result of calling the reset_func method.
        This method is an alias for self.reset_func().

        Returns:
            dict: The dictionary returned by self.reset_func().

        """
        return self.reset_func()

    @property
    def confIDs(self):
        """
        Retrieve a sorted list of configuration ids from the configuration dictionary.

        Returns:
            list: A sorted list of keys from the configuration dictionary.

        """
        return sorted(list(self.dict.keys()))

    @property
    def conf_class(self):
        """
        Determines the configuration generator class based on the `conftype` attribute.

        Returns:
            class: The configuration generator class corresponding to `conftype`.
                   - If `conftype` is None, returns None.
                   - If `conftype` is found in `reg.gen`, returns the corresponding class from `reg.gen`.
                   - Otherwise, returns `util.AttrDict`.

        """
        c = self.conftype
        if c is None:
            return None
        elif c in reg.gen:
            return reg.gen[c]
        else:
            return util.AttrDict

    @property
    def dict_entry_type(self):
        """
        Returns the type of dictionary entry used as a configuration.

        This method returns the `AttrDict` type, which is
        a specialized nested dictionary that allows attribute-style access.

        Returns:
            type: The `AttrDict` type.

        """
        return util.AttrDict


class RefType(ConfType):
    """
    RefType class extends ConfType to handle a unique configuration type,
    the one that accesses the reference datasets' configurations associated with specific ids.

    """

    def __init__(self, **kwargs):
        """
        Initialize the RefType instance.

        Parameters
        ----------
        **kwargs: Additional keyword arguments passed to the ConfType initializer.

        """
        super().__init__(conftype="Ref", **kwargs)

    def getRefDir(self, id):
        """
        Get the directory for a given reference id.

        Parameters
        ----------
        id (str): The id associated with a reference dataset.

        Returns
        -------
        str: The directory path for the reference dataset.

        """
        assert id is not None
        return self.getID(id)

    def getRef(self, id=None, dir=None):
        """
        Get the reference dataset's configuration.

        Parameters
        ----------
        id (str, optional): The reference id. Defaults to None.
        dir (str, optional): The directory path. Defaults to None.

        Returns
        -------
        dict: The reference dataset's configuration.

        Raises
        ------
        AssertionError: If the reference file does not exist or the configuration does not contain an 'id'.

        """
        path = self.path_to_Ref(id=id, dir=dir)
        assert os.path.isfile(path)
        c = util.load_dict(path)
        assert "id" in c
        vprint(f"Loaded existing conf {c.id}", 1)
        return c

    def setRef(self, c, id=None, dir=None):
        """
        Save the reference dataset's configuration.

        Parameters
        ----------
        c (dict): The reference dataset's configuration.
        id (str, optional): The reference id. Defaults to None.
        dir (str, optional): The directory path. Defaults to None.

        Raises
        ------
        AssertionError: If the configuration does not contain an 'id'.

        """
        path = self.path_to_Ref(id=id, dir=dir)
        util.save_dict(c, path)
        assert "id" in c
        vprint(f"Saved conf under ID {c.id}", 1)

    def path_to_Ref(self, id=None, dir=None):
        """
        Get the path to the file storing a reference dataset's configuration.

        Parameters
        ----------
        id (str, optional): The reference id. Defaults to None.
        dir (str, optional): The directory path. Defaults to None.

        Returns
        -------
        str: The path to the reference dataset's configuration file.

        """
        if dir is None:
            dir = self.getRefDir(id)
        return f"{dir}/data/conf.txt"

    def loadRef(self, id=None, dir=None, load=False, **kwargs):
        """
        Load a reference dataset.

        Parameters
        ----------
        id (str, optional): The reference id. Defaults to None.
        dir (str, optional): The directory path. Defaults to None.
        load (bool, optional): Whether to load data from the dataset. Defaults to False.
        **kwargs: Additional keyword arguments for loading the dataset.

        Returns
        -------
        LarvaDataset: The loaded reference dataset.

        """
        from ..process.dataset import LarvaDataset

        c = self.getRef(id=id, dir=dir)
        assert c is not None
        d = LarvaDataset(config=c, load_data=False)
        if load:
            d.load(**kwargs)
        vprint(f"Loaded stored reference dataset : {id}", 1)
        return d

    def loadRefs(self, ids=None, dirs=None, **kwargs):
        """
        Load multiple reference datasets.

        Parameters
        ----------
        ids (list, optional): List of reference ids. Defaults to None.
        dirs (list, optional): List of directory paths. Defaults to None.
        **kwargs: Additional keyword arguments for loading the datasets.

        Returns
        -------
        ItemList: A list of loaded reference datasets.

        """
        if ids is None:
            assert dirs is not None
            ids = [None] * len(dirs)
        if dirs is None:
            assert ids is not None
            dirs = [None] * len(ids)
        return util.ItemList(
            [self.loadRef(id=id, dir=dir, **kwargs) for id, dir in zip(ids, dirs)]
        )

    def retrieve_dataset(self, dataset=None, load=True, **kwargs):
        """
        Retrieve a dataset, loading it if necessary.

        Parameters
        ----------
        dataset (LarvaDataset, optional): The dataset to retrieve. Defaults to None.
        load (bool, optional): Whether to load the dataset. Defaults to True.
        **kwargs: Additional keyword arguments for loading the dataset.

        Returns
        -------
        LarvaDataset: The retrieved dataset.

        """
        if dataset is None:
            dataset = self.loadRef(load=load, **kwargs)
        return dataset

    def cleanRefIDs(self):
        """
        Purge reference ids by removing invalid ones.
        """
        ids = self.confIDs
        for id in ids:
            try:
                self.loadRef(id)
            except:
                self.delete(id)

    @property
    def dict_entry_type(self):
        """
        Get the type of dictionary entries.

        Returns:
        type: The type of dictionary entries, which is str.

        """
        return str

    def getRefGroups(self):
        """
        Get reference groups.

        Returns:
        AttrDict: A dictionary of reference groups.

        """
        d = self.Refdict
        gd = util.AttrDict({c.group_id: c for id, c in d.items()})
        gIDs = util.unique_list(list(gd.keys()))
        return util.AttrDict(
            {
                gID: {c.id: c.dir for id, c in d.items() if c.group_id == gID}
                for gID in gIDs
            }
        )

    @property
    def RefGroupIDs(self):
        """
        Get the IDs of reference groups.

        Returns:
        list: A list of reference group IDs.

        """
        d = self.Refdict
        gd = util.AttrDict(
            {c.group_id: c for id, c in d.items() if c.group_id is not None}
        )
        return util.unique_list(list(gd.keys()))

    @property
    def Refdict(self):
        """
        Get a dictionary of all reference datasets' configurations.

        Returns:
        AttrDict: A nested dictionary of all reference datasets' configurations.

        """
        return util.AttrDict({id: self.getRef(id) for id in self.confIDs})

    def getRefGroup(self, group_id):
        """
        Get a reference group by its ID.

        Parameters
        ----------
        group_id (str): The group ID.

        Returns
        -------
        AttrDict: A nested dictionary of all reference datasets' configurations in the group.

        """
        d = self.getRefGroups()[group_id]
        return util.AttrDict({id: self.getRef(dir=dir) for id, dir in d.items()})

    def loadRefGroup(self, group_id, to_return="collection", **kwargs):
        """
        Load a reference group by its ID.

        Parameters
        ----------
        group_id (str): The group ID.
        to_return (str, optional): The format to return the group in. Defaults to 'collection'.
        **kwargs: Additional keyword arguments for loading the references.

        Returns
        -------
        Union[AttrDict, ItemList, LarvaDatasetCollection]: The loaded reference group.

        """
        d = self.getRefGroups()[group_id]
        if to_return == "dict":
            return util.AttrDict(
                {id: self.loadRef(dir=dir, **kwargs) for id, dir in d.items()}
            )
        elif to_return == "list":
            return util.ItemList(
                [self.loadRef(dir=dir, **kwargs) for id, dir in d.items()]
            )
        elif to_return == "collection":
            from ..process.dataset import LarvaDatasetCollection

            return LarvaDatasetCollection(
                datasets=util.ItemList(
                    [self.loadRef(dir=dir, **kwargs) for id, dir in d.items()]
                )
            )


conf = util.AttrDict({k: ConfType(conftype=k) for k in CONFTYPES if k != "Ref"})

conf.Ref = RefType()


def resetConfs(conftypes=None, **kwargs):
    """
    Resets the configurations for the specified configuration types.

    Parameters
    ----------
    conftypes (list, optional): A list of configuration types to reset. If None, defaults to larvaworld.CONFTYPES.
    **kwargs: Additional keyword arguments to pass to the reset method of each configuration type.

    Returns
    -------
    None

    """
    if conftypes is None:
        conftypes = CONFTYPES

    for conftype in conftypes:
        conf[conftype].reset(**kwargs)
