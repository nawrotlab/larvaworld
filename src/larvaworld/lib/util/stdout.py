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
from typing import Any, Callable, Iterable, Iterator

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
        """Return the current wall-clock time in milliseconds."""
        return int(round(time.time() * 1000))

    @staticmethod
    def current_time_sec() -> int:
        """Return the current wall-clock time in seconds."""
        return int(round(time.time()))

    @staticmethod
    def format_time_seconds(seconds: int) -> str:
        """Format a duration as ``H:MM:SS``.

        Args:
            seconds: The duration in seconds.

        Returns:
            The formatted duration string.
        """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    @staticmethod
    def format_date_time() -> str:
        """Return the current date and time as ``YYYY-MM-DD_HH.MM.SS``."""
        return time.strftime("%Y-%m-%d_%H.%M.%S")


@contextmanager
def suppress_stdout_stderr() -> Iterator[tuple[Any, Any]]:
    """A context manager that redirects stdout and stderr to devnull.

    Yields:
        The redirected stderr and stdout targets.
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


@contextmanager
def suppress_stdout(show_output: bool) -> Iterator[None]:
    """Conditionally suppress stdout within the context.

    Args:
        show_output: When False, stdout is redirected to devnull for the
            duration of the context. When True, output passes through.

    Yields:
        None. The context is entered for its side effect only.
    """
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
    """Strip a leading prefix from a string if present.

    Args:
        text: The string to trim.
        prefix: The prefix to remove.

    Returns:
        The string without the prefix, or unchanged if it did not start with it.
    """
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


def remove_suffix(text: str, suffix: str) -> str:
    """Strip a trailing suffix from a string if present.

    Args:
        text: The string to trim.
        suffix: The suffix to remove.

    Returns:
        The string without the suffix, or unchanged if it did not end with it.
    """
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text  # or whatever


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rsetattr(obj: Any, attr: str, val: Any) -> None:
    """Set a possibly nested attribute using dot notation.

    Args:
        obj: The object to modify.
        attr: The attribute path, e.g. ``"a.b.c"``.
        val: The value to assign.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    """Get a possibly nested attribute using dot notation.

    Args:
        obj: The object to read from.
        attr: The attribute path, e.g. ``"a.b.c"``.
        *args: An optional single default, returned for missing attributes as
            in :func:`getattr`.

    Returns:
        The resolved attribute value.
    """

    def _getattr(obj: Any, attr: str) -> Any:
        """Read one attribute, honouring the optional default."""
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def try_except(
    success: Callable[[], Any], failure: Any, *exceptions: type[BaseException]
) -> Any:
    """Evaluate a callable, falling back to a default on failure.

    Args:
        success: Zero-argument callable to attempt.
        failure: Fallback value, or a zero-argument callable producing it.
        *exceptions: Exception types to catch. Defaults to :class:`Exception`.

    Returns:
        The result of ``success()``, or the resolved fallback on failure.
    """
    try:
        return success()
    except exceptions or Exception:
        return failure() if callable(failure) else failure


def storeH5(
    df: Any,
    path: str | None = None,
    key: str | None = None,
    mode: str | None = None,
    **kwargs: Any,
) -> None:
    """Store a dataframe, or a dict of dataframes, in an HDF5 file.

    Args:
        df: The dataframe to store, or a mapping of key to dataframe when
            ``key`` is None.
        path: Destination file path. Nothing is written when None.
        key: The HDF5 key to store under. When None, ``df`` must be a dict
            whose keys are used instead.
        mode: File mode. Defaults to ``"a"`` for an existing file, else ``"w"``.
        **kwargs: Forwarded to the recursive retry call.

    Raises:
        ValueError: If ``key`` is None and ``df`` is not a dict.
    """
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


def common_ancestor_class(classes: Iterable[type]) -> type:
    """Return the most derived class shared by all given classes.

    Args:
        classes: The classes whose method resolution orders are intersected.

    Returns:
        The first common ancestor in MRO order.
    """
    return next(iter(functools.reduce(and_, (Counter(cls.mro()) for cls in classes))))
