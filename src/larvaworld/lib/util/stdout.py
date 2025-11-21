"""
Methods for managing context and attributes
"""

from __future__ import annotations

import functools
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout

# from functools import reduce
from operator import and_

import pandas as pd

__all__: list[str] = [
    "TimeUtil",
    "suppress_stdout_stderr",
    "suppress_stdout",
    "remove_prefix",
    "remove_suffix",
    "rsetattr",
    "rgetattr",
    "try_except",
    "storeH5",
    "common_ancestor_class",
]


class TimeUtil:
    """
    Class for managing simulation time
    """

    @staticmethod
    def current_time_millis() -> int:
        return int(round(time.time() * 1000))

    @staticmethod
    def current_time_sec() -> int:
        return int(round(time.time()))

    @staticmethod
    def format_time_seconds(seconds: int) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    @staticmethod
    def format_date_time() -> str:
        return time.strftime("%Y-%m-%d_%H.%M.%S")


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


@contextmanager
def suppress_stdout(show_output: bool):
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if not show_output:
            sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


def remove_suffix(text: str, suffix: str) -> str:
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text  # or whatever


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rsetattr(obj, attr: str, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr: str, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def try_except(success, failure, *exceptions):
    try:
        return success()
    except exceptions or Exception:
        return failure() if callable(failure) else failure


def storeH5(
    df,
    path: str | None = None,
    key: str | None = None,
    mode: str | None = None,
    **kwargs,
):
    if path is not None:
        if mode is None:
            if os.path.isfile(path):
                mode = "a"
            else:
                mode = "w"

        if key is not None:
            try:
                store = pd.HDFStore(path, mode=mode)
                store[key] = df
                store.close()
            except:
                if mode == "a":
                    storeH5(df, path=path, key=key, mode="w", **kwargs)
        elif key is None and isinstance(df, dict):
            store = pd.HDFStore(path, mode=mode)
            for k, v in df.items():
                store[k] = v
            store.close()
        else:
            raise ValueError("H5key not provided.")


def common_ancestor_class(classes):
    return next(iter(functools.reduce(and_, (Counter(cls.mro()) for cls in classes))))
