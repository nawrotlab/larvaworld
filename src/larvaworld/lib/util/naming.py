"""
Class managing parameter naming
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__: list[str] = [
    "NamingRegistry",
    # 'tex',
    # 'sub',
    # 'sup',
    # 'subsup',
]

from .dictsNlists import AttrDict, SuperList


def join(s: str, p: str, loc: str, c: str = "_") -> str | None:
    """Join a naming token and a parameter name.

    Args:
        s: The naming token, e.g. ``"velocity"``.
        p: The parameter name being decorated.
        loc: Where ``s`` is placed relative to ``p``: ``"suf"`` puts ``p``
            first, ``"pref"`` and ``"sep"`` put ``s`` first.
        c: The separator character.

    Returns:
        The combined name, or None if ``loc`` is not recognized.
    """
    if loc == "suf":
        return f"{p}{c}{s}"
    elif loc == "pref":
        return f"{s}{c}{p}"
    elif loc == "sep":
        return f"{s}{c}{p}"


def name(
    s: str, ps: str | list[str], loc: str = "suf", c: str = "_"
) -> str | SuperList | None:
    """Apply a naming token to one or many parameter names.

    Args:
        s: The naming token to apply.
        ps: A single parameter name, or a list of them.
        loc: Placement of the token, as in :func:`join`.
        c: The separator character.

    Returns:
        The decorated name for a string input, or a :class:`SuperList` of
        decorated names for a list input. Empty names are passed through
        undecorated.
    """
    if isinstance(ps, str):
        if ps == "":
            return s
        else:
            return join(s, ps, loc, c)
    elif isinstance(ps, list):
        return SuperList([join(s, p, loc, c) if p != "" else s for p in ps])


def _tex(p: str) -> str:
    """Strip math delimiters so a name can be nested inside another expression.

    Args:
        p: The name, possibly already wrapped in ``$``.

    Returns:
        The name without ``$`` characters.
    """
    return p.replace("$", "")


def tex_sym(symbol: str, p: str, sep: str = "") -> str:
    """Wrap a name in a LaTeX command such as ``ar`` or ``	heta``.

    Args:
        symbol: The LaTeX command name, without the leading backslash.
        p: The name to wrap.
        sep: Optional separator inserted between command and argument.

    Returns:
        The LaTeX math expression.
    """
    return rf"$\{symbol}{sep}{{{_tex(p)}}}$"


def tex(p: str, q: str, sep: str = "") -> str:
    """Combine two names into a single LaTeX math expression.

    Args:
        p: The base name.
        q: The decorating name.
        sep: The LaTeX separator, e.g. ``"_"`` for a subscript.

    Returns:
        The LaTeX math expression.
    """
    return rf"${{{_tex(p)}}}{sep}{{{_tex(q)}}}$"


def sub(p: str, q: str) -> str:
    """Render ``p`` with ``q`` as a LaTeX subscript.

    Args:
        p: The base name.
        q: The subscript.

    Returns:
        The LaTeX math expression.
    """
    return tex(p, q, sep="_")


def sup(p: str, q: str) -> str:
    """Render ``p`` with ``q`` as a LaTeX superscript.

    Args:
        p: The base name.
        q: The superscript.

    Returns:
        The LaTeX math expression.
    """
    return tex(p, q, sep="^")


def subsup(p: str, q: str, z: str) -> str:
    """Render ``p`` with both a subscript and a superscript.

    Args:
        p: The base name.
        q: The subscript.
        z: The superscript.

    Returns:
        The LaTeX math expression.
    """
    return rf"${{{_tex(p)}}}_{{{_tex(q)}}}^{{{_tex(z)}}}$"


class TexNaming:
    """
    Accessor for LaTeX-formatted parameter names.

    Exposes the supported LaTeX accents and Greek letters as dynamic attributes:
    ``tex.bar("v")`` returns the LaTeX for a bar-accented ``v``. Explicit
    subscript, superscript, and circledast helpers are also provided.

    Attributes:
        symbols: LaTeX accent commands available as attributes.
        letters: LaTeX Greek/operator commands available as attributes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the accessor and register the supported commands.

        Args:
            *args: Positional arguments forwarded to ``super()``.
            **kwargs: Keyword arguments forwarded to ``super()``.
        """
        super().__init__(*args, **kwargs)

        self.symbols = ["bar", "tilde", "dot", "ddot", "mathring"]
        self.letters = ["theta", "omega", "Delta", "sum", "delta"]

    def get_func(self, symbol: str) -> Callable[..., str]:
        """Build the formatting function for one LaTeX command.

        Args:
            symbol: A command from :attr:`symbols` or :attr:`letters`.

        Returns:
            A callable applying that command to a name.

        Raises:
            AssertionError: If the command is not supported.
        """
        assert symbol in self.symbols + self.letters

        def func(p: str, **kwargs: Any) -> str:
            return tex_sym(symbol=symbol, p=p, **kwargs)

        return func

    def __getattr__(self, item: str) -> Callable[..., str]:
        """Resolve a LaTeX command accessed as an attribute.

        Args:
            item: The command name.

        Returns:
            The corresponding formatting function.
        """
        return self[item]

    def __getitem__(self, k: str) -> Callable[..., str]:
        """Resolve a LaTeX command accessed by key.

        Args:
            k: The command name.

        Returns:
            The corresponding formatting function.
        """
        return self.get_func(k)

    def sub(self, p: str, q: str) -> str:
        """Render ``p`` with ``q`` as a LaTeX subscript.

        Args:
            p: The base name.
            q: The subscript.

        Returns:
            The LaTeX math expression.
        """
        return tex(p, q, sep="_")

    def sup(self, p: str, q: str) -> str:
        """Render ``p`` with ``q`` as a LaTeX superscript.

        Args:
            p: The base name.
            q: The superscript.

        Returns:
            The LaTeX math expression.
        """
        return tex(p, q, sep="^")

    def subsup(self, p: str, q: str, z: str) -> str:
        """Render ``p`` with both a subscript and a superscript.

        Args:
            p: The base name.
            q: The subscript.
            z: The superscript.

        Returns:
            The LaTeX math expression.
        """
        return rf"${{{_tex(p)}}}_{{{_tex(q)}}}^{{{_tex(z)}}}$"

    def circledast(self, p: str) -> str:
        """Render ``p`` with a circled-asterisk superscript.

        Args:
            p: The base name.

        Returns:
            The LaTeX math expression.
        """
        return rf"${_tex(p)}^{{\circledast}}$"


class NamingRegistry(AttrDict):
    """
    Registry for systematic parameter naming and LaTeX formatting.

    Provides dynamic attribute-based naming functions for scientific parameters,
    supporting prefixes, suffixes, separators, and LaTeX math notation. Used
    throughout larvaworld for consistent parameter naming conventions.

    The registry supports:
    - Dynamic naming with prefixes (e.g., 'final_', 'initial_')
    - Parameter name expansions (e.g., 'vel' → 'velocity')
    - XY coordinate generation for body points
    - Midline and contour point naming
    - LaTeX math formatting via TexNaming

    Attributes:
        k_pref: List of supported prefix keywords
        k_pairs: Dictionary mapping short names to full names
        tex: TexNaming instance for LaTeX formatting

    Example:
        >>> nam = NamingRegistry()
        >>> nam.vel('x')  # velocity naming
        'velocity_x'
        >>> nam.final('position')  # prefix naming
        'final_position'
        >>> nam.xy('head')  # XY coordinates
        ['head_x', 'head_y']
        >>> nam.midline(3)  # body points
        ['head', 'point2', 'tail']
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the registry and its naming vocabulary.

        Args:
            *args: Positional arguments forwarded to :class:`AttrDict`.
            **kwargs: Keyword arguments forwarded to :class:`AttrDict`.
        """
        super().__init__(*args, **kwargs)
        self.k_pref = [
            "final",
            "initial",
            "cum",
            "lin",
            "scal",
            "abs",
            "dst_to",
            "bearing_to",
            "non",
        ]

        self.k_pairs = AttrDict(
            {
                "vel": "velocity",
                "acc": "acceleration",
                "scal": "scaled",
                "orient": "orientation",
                "unwrap": "unwrapped",
                # 'scal': 'scaled',
            }
        )

        # self.ks = SuperList(self.k_pref+self.k_pairs.keylist+self.k_ops.keylist+['freq', 'chain', 'dur', 'dst']).unique

        # self.tex_symbols = ['bar', 'tilde', 'wave', 'theta_', 'omega_', 'Delta', 'sum', 'delta', 'dot', 'ddot',
        #                     'mathring']

        self.tex = TexNaming()

    def get_kws(self, k: str) -> dict:
        """Resolve the :func:`name` keyword arguments for a naming token.

        Args:
            k: The naming token, e.g. ``"vel"`` or ``"final"``.

        Returns:
            Keyword arguments for :func:`name`: the expanded token ``s``, plus
            ``loc`` for prefix tokens and ``c`` for tokens joined without a
            separator.
        """
        noseparator = ["chain"]

        kws = {}
        if k in self.k_pref:
            kws["loc"] = "pref"
        if k in self.k_pairs:
            kws["s"] = self.k_pairs[k]
        else:
            kws["s"] = k
        if k in noseparator:
            kws["c"] = ""
        return kws

    def get_func(self, k: str) -> Callable[..., Any]:
        """Build the naming function for one token.

        Args:
            k: The naming token.

        Returns:
            A callable applying that token to one or many parameter names.
        """
        kws = self.get_kws(k)

        def func(ps: str | list[str], **kwargs: Any) -> Any:
            kws.update(kwargs)
            return name(ps=ps, **kws)

        return func

    def __getattr__(self, item: str) -> Callable[..., Any]:
        """Resolve a naming token accessed as an attribute.

        Args:
            item: The naming token.

        Returns:
            The corresponding naming function.
        """
        return self[item]

    def __getitem__(self, k: str) -> Callable[..., Any]:
        """Resolve a naming token accessed by key.

        Args:
            k: The naming token.

        Returns:
            The corresponding naming function.
        """
        return self.get_func(k)

    def num(self, chunk: str) -> str:
        """Name the count of a behavioural chunk, e.g. ``"num_turns"``.

        Args:
            chunk: The chunk name.

        Returns:
            The pluralized count parameter name.
        """
        s = "num"
        temp = name(s, chunk, "pref")
        return name("s", temp, "suf", c="")

    def xy(
        self,
        points: str | list[str],
        flat: bool = False,
        xsNys: bool = False,
    ) -> SuperList:
        """Build the x/y coordinate names for one or many body points.

        Args:
            points: A point name, or a list of them. An empty string yields the
                bare ``["x", "y"]`` trajectory columns.
            flat: When True, flatten a list result into a single sequence.
            xsNys: When True, group the result as ``[all_xs, all_ys]`` instead
                of per-point pairs.

        Returns:
            The coordinate column names.
        """
        if type(points) == str:
            if points == "":
                t = ["x", "y"]
            else:
                t = [f"{points}_x", f"{points}_y"]

        elif type(points) == list:
            t = [self.xy(p) for p in points]
            if xsNys:
                t = [np.array(t)[:, i].tolist() for i in [0, 1]]
            if flat:
                t = [item for sublist in t for item in sublist]
        return SuperList(t)

    def chunk_track(self, chunk_name: str, params: str) -> Any:
        """Name the tracked change of a parameter over a chunk.

        Args:
            chunk_name: The behavioural chunk.
            params: The parameter name, or names, being tracked.

        Returns:
            The chunk-track parameter name(s).
        """
        return self[chunk_name](params, loc="pref")

    def contour(self, Nc: int) -> list[str]:
        """Name the contour points of the body outline.

        Args:
            Nc: The number of contour points.

        Returns:
            The contour point names, in order.
        """
        return [f"contour{i}" for i in range(Nc)]

    def midline(self, N: int, type: str = "point", reverse: bool = False) -> list[str]:
        """Name the midline points along the body axis.

        The first and last points are always ``"head"`` and ``"tail"``; a single
        point is named ``"body"``.

        Args:
            N: The number of midline points.
            type: The stem used for the intermediate points.
            reverse: When True, order the points tail-to-head.

        Returns:
            The midline point names.
        """
        if N >= 2:
            points = ["head"] + [f"{type}{i}" for i in np.arange(2, N, 1)] + ["tail"]
        elif N == 1:
            points = ["body"]
        else:
            points = []
        if reverse:
            points.reverse()
        return points

    def contour_xy(self, Nc: int, flat: bool = False, xsNys: bool = False) -> SuperList:
        """Build the coordinate names for the contour points.

        Args:
            Nc: The number of contour points.
            flat: When True, flatten into a single sequence.
            xsNys: When True, group as ``[all_xs, all_ys]``.

        Returns:
            The contour coordinate column names.
        """
        return self.xy(self.contour(Nc), flat=flat, xsNys=xsNys)

    def midline_xy(
        self, N: int, reverse: bool = False, flat: bool = False, xsNys: bool = False
    ) -> SuperList:
        """Build the coordinate names for the midline points.

        Args:
            N: The number of midline points.
            reverse: When True, order the points tail-to-head.
            flat: When True, flatten into a single sequence.
            xsNys: When True, group as ``[all_xs, all_ys]``.

        Returns:
            The midline coordinate column names.
        """
        return self.xy(self.midline(N, reverse=reverse), flat=flat, xsNys=xsNys)

    @property
    def centroid_xy(self) -> SuperList:
        """The centroid coordinate column names."""
        return self.xy("centroid")

    @property
    def traj_xy(self) -> SuperList:
        """The bare trajectory coordinate column names."""
        return self.xy("")

    def at(self, p: str, t: str) -> str:
        """Name the value of a parameter sampled at a given event.

        Args:
            p: The parameter name.
            t: The event at which it is sampled.

        Returns:
            The sampled parameter name.
        """
        return self[f"{p}_at"](t, loc="pref")

    def atStartStopChunk(self, p: str, chunk: str) -> list[Any]:
        """Name a parameter at a chunk's start, its stop, and across the chunk.

        Args:
            p: The parameter name.
            chunk: The behavioural chunk.

        Returns:
            The names at chunk start, at chunk stop, and the chunk track.
        """
        return [
            self.at(p, self.start(chunk)),
            self.at(p, self.stop(chunk)),
            self.chunk_track(chunk, p),
        ]

    @property
    def on_food(self) -> str:
        """The name of the on-food state."""
        return "on_food"

    @property
    def off_food(self) -> str:
        """The name of the off-food state."""
        return "off_food"
