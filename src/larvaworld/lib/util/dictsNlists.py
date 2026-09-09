"""
Classes and methods for managing nested dictionaries and lists
"""

from __future__ import annotations

from typing import Any

import copy
import json
import pickle
import typing

import agentpy.sequences
import pandas as pd

__all__: list[str] = [
    "AttrDict",
    "load_dict",
    "save_dict",
    "bidict",
    "SuperList",
    "ItemList",
    "existing_cols",
    "nonexisting_cols",
    "cols_exist",
    "flatten_list",
    "unique_list",
    "checkEqual",
    "np2Dtotuples",
]


class AttrDict(dict):
    """
    Dictionary subclass with attribute-style access.

    Allows dictionary entries to be accessed using dot notation (as attributes)
    in addition to standard bracket notation. Automatically converts nested
    dictionaries to AttrDict instances.

    Example:
        >>> d = AttrDict({'a': 1, 'b': {'c': 2}})
        >>> d.a
        1
        >>> d.b.c
        2
        >>> d['b']['c']
        2
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the dictionary and expose its entries as attributes.

        Args:
            *args: Positional arguments forwarded to :class:`dict`.
            **kwargs: Keyword arguments forwarded to :class:`dict`.
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self.autonest(d=self)

    def autonest(self, d: dict) -> dict:
        """Convert every nested plain dict into an :class:`AttrDict`, in place.

        Args:
            d: The dictionary to convert.

        Returns:
            The same mapping, with nested dicts replaced.
        """
        for k, data in d.items():
            d[k] = self.from_nested_dicts(data)
        return d

    @classmethod
    def from_nested_dicts(cls, data: Any) -> Any:
        """Construct nested NestDicts from nested dictionaries."""
        if not isinstance(data, dict):
            return data
        else:
            return cls(data)

    def replace_keys(self, pairs: dict = {}) -> "AttrDict":
        """Return a copy with some top-level keys renamed.

        Args:
            pairs: Mapping of old key to new key. Keys absent from it are
                carried over unchanged.

        Returns:
            The renamed dictionary.
        """
        dic = {}
        for k, v in self.items():
            if k in list(pairs.keys()):
                dic[pairs[k]] = v
            else:
                dic[k] = v
        return AttrDict(dic)

    def get_copy(self) -> "AttrDict":
        """Return a deep copy of this dictionary."""
        return AttrDict(copy.deepcopy(self))

    def flatten(self, parent_key: str = "", sep: str = ".") -> "AttrDict":
        """Flatten the nested structure into dotted single-level keys.

        Empty nested dicts are preserved as the sentinel ``"empty_dict"`` so
        that :meth:`unflatten` can restore them.

        Args:
            parent_key: Prefix applied to every key, used when recursing.
            sep: Separator placed between key levels.

        Returns:
            The flattened dictionary.
        """
        items = []
        for k, v in self.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, typing.MutableMapping):
                if len(v) > 0:
                    items.extend(AttrDict(v).flatten(new_key, sep=sep).items())
                else:
                    items.append((new_key, "empty_dict"))

            else:
                items.append((new_key, v))
        return AttrDict(dict(items))

    def unflatten(self, sep: str = ".") -> "AttrDict":
        """Rebuild a nested dictionary from dotted single-level keys.

        Inverse of :meth:`flatten`; the ``"empty_dict"`` sentinel is restored
        to an empty mapping.

        Args:
            sep: Separator between key levels.

        Returns:
            The nested dictionary.
        """
        dic = {}
        for k, v in self.items():
            if v == "empty_dict":
                v = {}
            parts = k.split(sep)
            dd = dic
            for part in parts[:-1]:
                if part not in dd:
                    dd[part] = {}
                dd = dd[part]
            dd[parts[-1]] = v
        return AttrDict(dic)

    def update_existingdict(self, dic: dict) -> None:
        """Update only the keys that already exist, in place.

        Args:
            dic: The new values. Keys absent from this dictionary are ignored.
        """
        for k, v in dic.items():
            if k in list(self.keys()):
                self[k] = v

    def update_existingdict_by_suffix(self, dic: dict) -> None:
        """Update existing keys matched by suffix, in place.

        Each incoming key is matched against the tail of the existing keys.
        Ambiguous suffixes matching more than one key are reported and skipped.

        Args:
            dic: Mapping of key suffix to new value.
        """
        for k, v in dic.items():
            k1s = [k0 for k0 in self.keylist if k0.endswith(k)]
            if len(k1s) == 1:
                k1 = k1s[0]
                self[k1] = v
            elif len(k1s) > 1:
                print(f"Non unique suffix : {k}")

    def update_nestdict(self, dic: dict) -> "AttrDict":
        """Update nested entries addressed by their flattened keys.

        Args:
            dic: Mapping of dotted key to new value. Missing keys are added.

        Returns:
            The updated nested dictionary.
        """
        dic0_f = self.flatten()
        dic0_f.update(dic)
        return dic0_f.unflatten()

    def update_nestdict_copy(self, dic: dict) -> "AttrDict":
        """Return a deep copy updated by :meth:`update_nestdict`.

        Args:
            dic: Mapping of dotted key to new value.

        Returns:
            The updated copy, leaving this dictionary untouched.
        """
        return self.get_copy().update_nestdict(dic)

    def new_dict(self, dic: dict) -> "AttrDict":
        """Return a deep copy updated with the given nested values.

        Args:
            dic: Mapping of dotted key to new value.

        Returns:
            The updated copy.
        """
        return self.get_copy().update_nestdict(dic)

    def update_existingnestdict(self, dic: dict) -> "AttrDict":
        """Update only the nested entries that already exist.

        Args:
            dic: Mapping of dotted key to new value. Unknown keys are ignored.

        Returns:
            The updated nested dictionary.
        """
        dic0_f = self.flatten()
        dic0_f.update_existingdict(dic)
        return dic0_f.unflatten()

    def update_existingnestdict_by_suffix(self, dic: dict) -> "AttrDict":
        """Update existing nested entries matched by key suffix.

        Args:
            dic: Mapping of key suffix to new value.

        Returns:
            The updated nested dictionary.
        """
        dic0_f = self.flatten()
        dic0_f.update_existingdict_by_suffix(dic)
        return dic0_f.unflatten()

    def save(self, file: str) -> None:
        """Write this dictionary to disk.

        Args:
            file: Destination path. Pickle is attempted first, then JSON.
        """
        save_dict(self, file)

    @classmethod
    def load(cls, file: str) -> "AttrDict":
        """Read a dictionary from disk.

        Args:
            file: Source path. Pickle is attempted first, then JSON.

        Returns:
            The loaded dictionary, empty if the file could not be read.
        """
        return load_dict(file)

    def print(self, flat: bool = False) -> None:
        """Print the dictionary for inspection.

        Args:
            flat: When True, print one dotted key per line. When False, print
                the nested structure with indentation.
        """
        if flat:
            for k, v in self.flatten().items():
                print(f"      {k} : {v}")
        else:
            pref0 = "     "

            def print_nested_level(d: dict, pref: str = pref0) -> None:
                """Print one nesting level, recursing into nested dicts."""
                for k, v in d.items():
                    if not isinstance(v, dict):
                        print(f"{pref}{k} : {v}")
                    else:
                        print(f"{pref}{k} : ")
                        print_nested_level(v, pref=f"{pref}{pref0}")

            print_nested_level(self, pref=pref0)

    @property
    def keylist(self) -> "SuperList":
        """The top-level keys, as a :class:`SuperList`."""
        return SuperList(self.keys())

    @classmethod
    def merge_dicts(cls, l: list[dict]) -> "AttrDict":
        """Merge dictionaries at the top level only.

        Args:
            l: The dictionaries to merge. Later entries win on key collisions.

        Returns:
            The merged dictionary.
        """
        D = {}
        for d in l:
            for k, v in d.items():
                D[k] = v
        return cls(D)

    @classmethod
    def merge_nestdicts(cls, l: list[dict]) -> "AttrDict":
        """Merge nested dictionaries entry by entry.

        Merging happens on the flattened representation, so nested branches are
        combined rather than replaced wholesale.

        Args:
            l: The dictionaries to merge. Later entries win on key collisions.

        Returns:
            The merged nested dictionary.
        """
        D = {}
        for d in l:
            for k, v in AttrDict(d).flatten().items():
                D[k] = v
        return cls(D).unflatten()

    # def keys_on_condition(self, conditions={}):
    #     D=[]
    #     ids=self.keylist
    #     d=AttrDict({id:self[id].flatten() for id in ids})
    #     for id in ids :
    #         for k,v in conditions.items():
    #             try:
    #                 if not self[id][k]==v:
    #                     break
    #             except
    #                 pass


def load_dict(file: str) -> AttrDict:
    """
    Load dictionary from pickle or JSON file.

    Attempts to load from pickle first, falls back to JSON if that fails,
    returns empty AttrDict if both fail.

    Args:
        file: Path to file containing pickled or JSON dictionary

    Returns:
        Loaded dictionary as AttrDict, or empty AttrDict on failure

    Example:
        >>> d = load_dict('config.pkl')
        >>> d = load_dict('config.json')
    """
    try:
        with open(file, "rb") as tfp:
            d = pickle.load(tfp)
    except:
        try:
            with open(file) as tfp:
                d = json.load(tfp)
        except:
            d = {}
    return AttrDict(d)


def save_dict(d: dict, file: str) -> None:
    """
    Save dictionary to pickle or JSON file.

    Attempts to save as pickle first, falls back to JSON if that fails.

    Args:
        d: Dictionary to save
        file: Path to output file

    Example:
        >>> save_dict({'a': 1, 'b': 2}, 'config.pkl')
        >>> save_dict({'a': 1, 'b': 2}, 'config.json')
    """
    if file is not None:
        try:
            with open(file, "wb") as fp:
                pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            with open(file, "w") as fp:
                json.dump(d, fp)
        finally:
            pass
            # print('Failed to save dict')


class bidict(dict):
    """
    Bidirectional dictionary with inverse mapping.

    Maintains both forward (key→value) and inverse (value→keys) mappings.
    The inverse attribute maps each value to a list of keys that map to it.

    Attributes:
        inverse: Dictionary mapping values to lists of keys

    Example:
        >>> bd = bidict({'a': 1, 'b': 2, 'c': 1})
        >>> bd.inverse
        {1: ['a', 'c'], 2: ['b']}
        >>> bd['a']
        1
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the mapping and its inverse.

        Args:
            *args: Positional arguments forwarded to :class:`dict`.
            **kwargs: Keyword arguments forwarded to :class:`dict`.
        """
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set an entry, keeping the inverse mapping consistent.

        Args:
            key: The key to set.
            value: The value to associate with it.
        """
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key: Any) -> None:
        """Delete an entry, keeping the inverse mapping consistent.

        Args:
            key: The key to remove.
        """
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


class SuperList(list):
    """
    Enhanced list with utility properties and methods.

    Extends built-in list with convenient properties for sorting, flattening,
    deduplication, grouping, and DataFrame column operations.

    Properties:
        N: Length of list
        sorted: Sorted copy of list
        flatten: Recursively flattened list
        unique: List with duplicates removed (preserves order)
        in_pairs: List grouped into pairs

    Example:
        >>> sl = SuperList([3, 1, 2, 1])
        >>> sl.unique
        [3, 1, 2]
        >>> sl.sorted
        [1, 1, 2, 3]
        >>> SuperList([[1, 2], [3, 4]]).flatten
        [1, 2, 3, 4]
    """

    @property
    def N(self) -> int:
        """The number of elements."""
        return len(self)

    @property
    def sorted(self) -> "SuperList":
        """A sorted copy of the list."""
        return SuperList(sorted(self))

    @property
    def flatten(self) -> "SuperList":
        """A recursively flattened copy, with nested lists spliced in."""
        l = SuperList()
        for a in self:
            if not isinstance(a, list):
                l.append(a)
            else:
                aa = SuperList(a).flatten
                l.extend(aa)
        return l

    @property
    def unique(self) -> "SuperList":
        """A copy with duplicates dropped, keeping first-occurrence order."""
        if len(self) == 0:
            return SuperList()
        elif len(self) == 1:
            return self
        else:
            seen = set()
            seen_add = seen.add
            return SuperList([x for x in self if not (x in seen or seen_add(x))])

    def group_by_n(self, n: int = 2) -> "SuperList":
        """Split the list into consecutive groups of ``n``.

        Args:
            n: The group size. A trailing shorter group is kept as-is.

        Returns:
            The list of groups.
        """
        Nmore = int(len(self) % n)
        N = int((len(self) - Nmore) / n)
        g = [self[i * n : (i + 1) * n] for i in range(N)]
        if Nmore != 0:
            g.append(self[-Nmore:])
        return SuperList(g)

    @property
    def in_pairs(self) -> "SuperList":
        """The list split into consecutive pairs."""
        return self.group_by_n(n=2)

    def existing(self, df: Any) -> "SuperList":
        """Select the entries that name existing columns.

        Args:
            df: A dataframe, or a sequence of column names.

        Returns:
            The subset present in ``df``.
        """
        return SuperList(existing_cols(self, df))

    def nonexisting(self, df: Any) -> "SuperList":
        """Select the entries that do not name existing columns.

        Args:
            df: A dataframe, or a sequence of column names.

        Returns:
            The subset absent from ``df``.
        """
        return SuperList(nonexisting_cols(self, df))

    def exist_in(self, df: Any) -> bool:
        """Report whether every entry names an existing column.

        Args:
            df: A dataframe, or a sequence of column names.

        Returns:
            True if all entries are present in ``df``.
        """
        return cols_exist(self, df)

    def __add__(self, *args: Any, **kwargs: Any) -> "SuperList":
        """Return self+value."""
        return SuperList(super().__add__(*args, **kwargs))

    def suf(self, suf: str = "") -> "SuperList":
        """Select the entries ending with a suffix.

        Args:
            suf: The required suffix.

        Returns:
            The matching entries.
        """
        return SuperList([i for i in self if i.endswith(suf)])

    def pref(self, pref: str = "") -> "SuperList":
        """Select the entries starting with a prefix.

        Args:
            pref: The required prefix.

        Returns:
            The matching entries.
        """
        return SuperList([i for i in self if i.startswith(pref)])

    def contains(self, l: str = "") -> "SuperList":
        """Select the entries containing a substring.

        Args:
            l: The required substring.

        Returns:
            The matching entries.
        """
        return SuperList([i for i in self if l in i])


class ItemList(agentpy.sequences.AgentSequence, list):
    """
    Agent sequence list with mass attribute setting.

    Combines agentpy.AgentSequence with list functionality, allowing
    batch attribute operations on agent collections.

    Example:
        >>> items = ItemList([agent1, agent2, agent3])
        >>> items.speed = 5.0  # Sets speed=5.0 on all agents
    """

    def __init__(
        self, objs: Any = (), cls: type | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """Build the sequence from existing objects, or by constructing them.

        Args:
            objs: The objects to hold, or an integer count of objects to build
                from ``cls``.
            cls: The class instantiated when ``objs`` is a count.
            *args: Positional arguments forwarded to ``cls``.
            **kwargs: Keyword arguments forwarded to ``cls``. List values are
                broadcast element-wise across the constructed objects.
        """
        if isinstance(objs, int):
            objs = self._obj_gen(objs, cls, *args, **kwargs)
        super().__init__(objs)

    @staticmethod
    def _obj_gen(n: int, cls: type, *args: Any, **kwargs: Any) -> Any:
        """Generate objects for the sequence.

        Args:
            n: The number of objects to build.
            cls: The class to instantiate.
            *args: Positional arguments forwarded to ``cls``.
            **kwargs: Keyword arguments forwarded to ``cls``. List values are
                broadcast element-wise.

        Yields:
            The constructed objects.
        """
        for i in range(n):
            # AttrIter values get broadcasted among agents
            i_kwargs = {
                k: arg[i] if isinstance(arg, list) else arg for k, arg in kwargs.items()
            }
            yield cls(**i_kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute on every held object.

        Args:
            name: The attribute name.
            value: The value to assign. A list is distributed element-wise;
                any other value is applied to every object.
        """
        if isinstance(value, list):
            # Apply each value to each agent
            for obj, v in zip(self, value):
                setattr(obj, name, v)
        else:
            # Apply single value to all agents
            for obj in self:
                setattr(obj, name, value)


def existing_cols(cols: list[str], df: pd.DataFrame | list[str]) -> list[str]:
    """
    Filter column names to those that exist in DataFrame.

    Args:
        cols: List of column names to check
        df: DataFrame or list of column names

    Returns:
        List of columns from cols that exist in df

    Example:
        >>> existing_cols(['a', 'b', 'c'], df)
        ['a', 'c']  # if only 'a' and 'c' exist in df
    """
    if isinstance(df, pd.DataFrame):
        df = df.columns.values
    return [col for col in cols if col in df]


def nonexisting_cols(cols: list[str], df: pd.DataFrame | list[str]) -> list[str]:
    """
    Filter column names to those that don't exist in DataFrame.

    Args:
        cols: List of column names to check
        df: DataFrame or list of column names

    Returns:
        List of columns from cols that don't exist in df

    Example:
        >>> nonexisting_cols(['a', 'b', 'c'], df)
        ['b']  # if only 'b' doesn't exist in df
    """
    if isinstance(df, pd.DataFrame):
        df = df.columns.values
    return [col for col in cols if col not in df]


def cols_exist(cols: list[str], df: pd.DataFrame | list[str]) -> bool:
    """
    Check if all columns exist in DataFrame.

    Args:
        cols: List of column names to check
        df: DataFrame or list of column names

    Returns:
        True if all columns in cols exist in df

    Example:
        >>> cols_exist(['a', 'b'], df)
        True  # if both 'a' and 'b' exist
    """
    if isinstance(df, pd.DataFrame):
        df = df.columns.values
    return set(cols).issubset(df)


def flatten_list(l: list[list[Any]]) -> list[Any]:
    """
    Flatten a list of lists into a single list.

    Args:
        l: List of lists to flatten

    Returns:
        Flattened list containing all items from sublists

    Example:
        >>> flatten_list([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
    """
    return [item for sublist in l for item in sublist]


def checkEqual(l1: list[Any], l2: list[Any]) -> bool:
    """
    Check if two lists contain the same elements (order-independent).

    Args:
        l1: First list
        l2: Second list

    Returns:
        True if both lists contain exactly the same elements

    Example:
        >>> checkEqual([1, 2, 3], [3, 2, 1])
        True
        >>> checkEqual([1, 2], [1, 2, 3])
        False
    """
    for a in l1:
        if a not in l2:
            return False
    for a in l2:
        if a not in l1:
            return False
    return True


def unique_list(l: list[Any]) -> SuperList:
    """
    Remove duplicates from list while preserving order.

    Args:
        l: List to deduplicate

    Returns:
        SuperList with duplicates removed, first occurrence preserved

    Example:
        >>> unique_list([1, 2, 1, 3, 2])
        [1, 2, 3]
    """
    if len(l) == 0:
        return SuperList()
    elif len(l) == 1:
        return l
    else:
        seen = set()
        seen_add = seen.add
        return SuperList([x for x in l if not (x in seen or seen_add(x))])


def np2Dtotuples(a: Any) -> list[tuple[Any, Any]]:
    """
    Convert 2D numpy array to list of tuples.

    Args:
        a: 2D numpy array with shape (N, 2) or list of tuples

    Returns:
        List of (x, y) tuples

    Example:
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> np2Dtotuples(arr)
        [(1, 2), (3, 4)]
    """
    if isinstance(a, list) and all([isinstance(aa, tuple) for aa in a]):
        return a
    else:
        return list(zip(a[:, 0], a[:, 1]))
