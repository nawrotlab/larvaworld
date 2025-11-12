"""
Methods for managing colors
"""

from __future__ import annotations

import random

import matplotlib
import numpy as np

__all__: list[str] = [
    "invert_color",
    "random_colors",
    "N_colors",
    "colorname2tuple",
    "colortuple2str",
    "col_range",
    "Color",
    "combine_hex_values",
    "mix2colors",
    "scaled_velocity_to_col",
    "angular_velocity_to_col",
]


def invert_color(col: str | list[int] | tuple[int, int, int]) -> tuple[str, str]:
    """
    Invert a color to its complementary color.

    Accepts color as hex string, RGB name, or RGB tuple and returns both
    the original and inverted colors as hex strings.

    Args:
        col: Color as hex string ('#RRGGBB'), named color ('red'), or RGB tuple (R, G, B)

    Returns:
        Tuple of (original_hex, inverted_hex) color strings

    Example:
        >>> invert_color('red')
        ('#FF0000', '#00FFFF')
        >>> invert_color((255, 0, 0))
        ('#FF0000', '#00FFFF')
    """
    if type(col) in [list, tuple] and len(col) == 3:
        if not all([0 <= i <= 1 for i in col]):
            col = list(np.array(col) / 255)
        col = matplotlib.colors.rgb2hex(col)
    elif col[0] != "#":
        col = matplotlib.colors.cnames[col]
    table = str.maketrans("0123456789abcdef", "fedcba9876543210")
    col2 = "#" + col[1:].lower().translate(table).upper()
    return col, col2


def random_colors(n: int) -> list[np.ndarray]:
    """
    Generate n random distinct RGB colors.

    Creates colors by stepping through RGB space with fixed increments
    to ensure visual distinctness between colors.

    Args:
        n: Number of colors to generate

    Returns:
        List of n RGB arrays, each with shape (3,) and values in range [0, 255]

    Example:
        >>> cols = random_colors(3)
        >>> len(cols)
        3
        >>> cols[0].shape
        (3,)
    """
    ret = []
    r = int(random.random() * 200)
    g = int(random.random() * 200)
    b = int(random.random() * 200)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append(np.array([r, g, b]))
    return ret


def N_colors(N: int, as_rgb: bool = False) -> list:
    """
    Get N predefined distinct colors for plotting.

    Returns a list of color names or RGB tuples optimized for visual
    distinction. Uses predefined palettes for N ≤ 10, colormap for larger N.

    Args:
        N: Number of colors needed
        as_rgb: If True, return RGB tuples instead of color names

    Returns:
        List of N color names (strings) or RGB tuples depending on as_rgb

    Example:
        >>> colors = N_colors(3)
        >>> colors
        ['green', 'red', 'blue']
        >>> rgb_colors = N_colors(2, as_rgb=True)
        >>> len(rgb_colors[0])
        3
    """
    cols = [
        "green",
        "red",
        "blue",
        "purple",
        "orange",
        "magenta",
        "cyan",
        "darkred",
        "lightblue",
    ]
    if N <= len(cols):
        cs = cols[:N]
    elif N == 10:
        cs = [
            "lightgreen",
            "green",
            "red",
            "darkred",
            "lightblue",
            "blue",
            "darkblue",
            "magenta",
            "cyan",
            "orange",
            "purple",
        ]
    else:
        colormap = matplotlib.colormaps["brg"]
        cs = [colormap(i) for i in np.linspace(0, 1, N)]
    if as_rgb:
        cs = [colorname2tuple(c) for c in cs]
    return cs


def colorname2tuple(name: str) -> tuple[float, float, float]:
    """
    Convert color name to RGB tuple with values in [0, 255].

    Args:
        name: Matplotlib color name (e.g., 'red', 'blue') or hex string

    Returns:
        RGB tuple with values scaled to [0, 255]

    Example:
        >>> colorname2tuple('red')
        (255.0, 0.0, 0.0)
    """
    c0 = matplotlib.colors.to_rgb(name)
    c1 = tuple([i * 255 for i in c0])
    return c1


def colortuple2str(t: tuple[float, float, float]) -> str:
    """
    Convert RGB tuple to hex color string.

    Accepts RGB values in [0, 1] or [0, 255] range and returns hex string.

    Args:
        t: RGB tuple with values in [0, 1] or [0, 255]

    Returns:
        Hex color string in format '#RRGGBB'

    Example:
        >>> colortuple2str((255, 0, 0))
        '#ff0000'
        >>> colortuple2str((1.0, 0.0, 0.0))
        '#ff0000'
    """
    if any([tt > 1 for tt in t]):
        t = tuple([tt / 255 for tt in t])
    return matplotlib.colors.rgb2hex(t)


def col_range(
    q: np.ndarray | float,
    low: tuple[int, int, int] | str = (255, 0, 0),
    high: tuple[int, int, int] | str = (255, 255, 255),
    mul255: bool = False,
) -> np.ndarray:
    """
    Map values to colors along a gradient between two colors.

    Linearly interpolates RGB values between low and high colors based on
    input values in range [0, 1].

    Args:
        q: Value(s) in range [0, 1] to map to colors
        low: Low end color as RGB tuple or name (default: red)
        high: High end color as RGB tuple or name (default: white)
        mul255: If True, return values in [0, 255], else [0, 1]

    Returns:
        RGB array(s) with values clipped to valid range

    Example:
        >>> col_range(0.5)  # midpoint between red and white
        array([1. , 0.5, 0.5])
        >>> col_range(np.array([0, 0.5, 1]), mul255=True)
        array([[255., 0., 0.], [255., 127.5, 127.5], [255., 255., 255.]])
    """
    if isinstance(low, str):
        low = colorname2tuple(low)
    if isinstance(high, str):
        high = colorname2tuple(high)
    rr0, gg0, bb0 = q_col1 = np.array(low) / 255
    rr1, gg1, bb1 = q_col2 = np.array(high) / 255
    qrange = np.array([rr1 - rr0, gg1 - gg0, bb1 - bb0])
    res = q_col1 + np.array([q, q, q]).T * qrange
    res = np.clip(res, a_min=0, a_max=1)
    if mul255:
        res *= 255
    return res


class Color:
    """
    Color constants and utility methods for RGB color manipulation.

    Provides commonly used color constants as RGB tuples and static methods
    for generating random colors and mapping timeseries data to color gradients.

    Attributes:
        BLACK: RGB (0, 0, 0)
        WHITE: RGB (255, 255, 255)
        RED: RGB (255, 0, 0)
        GREEN: RGB (0, 255, 0)
        BLUE: RGB (0, 0, 255)
        YELLOW: RGB (255, 255, 0)
        GRAY: RGB (128, 128, 128)
        DARK_GRAY: RGB (64, 64, 64)

    Example:
        >>> Color.RED
        (255, 0, 0)
        >>> col = Color.random_bright()
        >>> all(c >= 200 for c in col)
        True
    """

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    GRAY = (128, 128, 128)
    DARK_GRAY = (64, 64, 64)

    @staticmethod
    def random_color(
        min_r: int = 0,
        min_g: int = 0,
        min_b: int = 0,
        max_r: int = 255,
        max_g: int = 255,
        max_b: int = 255,
    ) -> tuple[int, int, int]:
        r = random.randint(min_r, max_r)
        g = random.randint(min_g, max_g)
        b = random.randint(min_b, max_b)
        return r, g, b

    @staticmethod
    def random_bright(min_value: int = 200) -> tuple[int, int, int]:
        r = random.randint(min_value, 255)
        g = random.randint(min_value, 255)
        b = random.randint(min_value, 255)
        # print(r,g,b)
        return r, g, b
        # return colortuple2str((r, g, b))

    @staticmethod
    def timeseries_to_col(
        a: np.ndarray | float,
        lim: float = 1.0,
        color_range: list[tuple[int, int, int]] = [RED, GREEN],
    ) -> tuple[float, float, float] | list[tuple[float, float, float]]:
        t = np.clip(np.abs(a) / lim, a_min=0, a_max=1)
        (r1, b1, g1), (r2, b2, g2) = color_range
        r, b, g = r2 - r1, b2 - b1, g2 - g1
        if isinstance(a, float):
            return r1 + r * t, b1 + b * t, g1 + g * t
        else:
            return [(r1 + r * tt, b1 + b * tt, g1 + g * tt) for tt in t]

    # @staticmethod


def combine_hex_values(d: dict[str, float] | dict[tuple[int, int, int], float]) -> str:
    """
    Blend multiple colors with weighted average.

    Combines colors specified as hex strings or RGB tuples using their
    weights to produce a blended result color.

    Args:
        d: Dictionary mapping colors (hex strings or RGB tuples) to weights (floats)

    Returns:
        Blended color as hex string '#RRGGBB'

    Example:
        >>> combine_hex_values({'#FF0000': 0.7, '#0000FF': 0.3})
        '#b2004d'
        >>> combine_hex_values({(255, 0, 0): 1.0, (0, 0, 255): 1.0})
        '#800080'
    """
    dd = {}
    for k, v in d.items():
        if type(k) == str:
            k = matplotlib.colors.to_hex(k)
        k = k.lstrip("#")
        dd[k] = v
    d_items = sorted(dd.items())
    tot_weight = sum(dd.values())
    red = int(sum([int(k[:2], 16) * v for k, v in d_items]) / tot_weight)
    green = int(sum([int(k[2:4], 16) * v for k, v in d_items]) / tot_weight)
    blue = int(sum([int(k[4:6], 16) * v for k, v in d_items]) / tot_weight)
    zpad = lambda x: x if len(x) == 2 else "0" + x
    return "#" + zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])

    # @staticmethod


def mix2colors(c0: str, c1: str) -> str:
    """
    Mix two colors with 70-30 weight ratio.

    Convenience function for blending two colors with primary color
    weighted at 70% and secondary at 30%.

    Args:
        c0: Primary color (hex string or name)
        c1: Secondary color (hex string or name)

    Returns:
        Mixed color as hex string

    Example:
        >>> mix2colors('#FF0000', '#0000FF')  # red + blue
        '#b2004d'
    """
    cc = combine_hex_values(d={c0: 0.7, c1: 0.3})
    return cc


def scaled_velocity_to_col(
    a: np.ndarray | float,
) -> np.ndarray | tuple[float, float, float]:
    """
    Map scaled velocity values to color gradient.

    Converts velocity values to colors using red-to-green gradient
    with limit of 0.8 for saturation.

    Args:
        a: Velocity value(s) to map to colors

    Returns:
        RGB tuple for single value or list of tuples for array

    Example:
        >>> col = scaled_velocity_to_col(0.4)
        >>> len(col)
        3
    """
    return Color.timeseries_to_col(a, lim=0.8)


def angular_velocity_to_col(
    a: np.ndarray | float,
) -> np.ndarray | tuple[float, float, float]:
    """
    Map angular velocity values to color gradient.

    Converts angular velocity (in radians) to colors using red-to-green
    gradient with limit of 100 degrees for saturation.

    Args:
        a: Angular velocity value(s) in radians to map to colors

    Returns:
        RGB tuple for single value or list of tuples for array

    Example:
        >>> col = angular_velocity_to_col(np.deg2rad(50))
        >>> len(col)
        3
    """
    return Color.timeseries_to_col(a, lim=np.deg2rad(100))
