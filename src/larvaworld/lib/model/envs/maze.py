from __future__ import annotations
from typing import Any, List
import random

from shapely.geometry import LineString

__all__: list[str] = [
    "Cell",
    "Maze",
]


class Cell:
    """
    Single cell in maze grid.

    Represents a grid point that may be surrounded by walls in cardinal
    directions (North, East, South, West). Walls can be knocked down
    to create maze passages.

    Attributes:
        x: Cell x-coordinate in grid
        y: Cell y-coordinate in grid
        walls: Dict of wall states {"N": bool, "S": bool, "E": bool, "W": bool}
        wall_pairs: Class-level mapping of opposing wall directions

    Example:
        >>> cell = Cell(x=5, y=3)
        >>> cell.knock_down_wall(neighbor_cell, "N")
        >>> has_walls = cell.has_all_walls()  # False
    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {"N": "S", "S": "N", "E": "W", "W": "E"}

    def __init__(self, x: int, y: int) -> None:
        """Initialize the cell at (x,y). At first it is surrounded by walls."""
        self.x, self.y = x, y
        self.walls = {"N": True, "S": True, "E": True, "W": True}

    def has_all_walls(self) -> bool:
        """Does this cell still have all its walls?"""
        return all(self.walls.values())

    def knock_down_wall(self, other: "Cell", wall: str) -> None:
        """Knock down the wall between cells self and other."""
        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False


class Maze:
    """
    Maze generator and representation as grid of cells.

    Creates procedurally generated mazes using depth-first search algorithm.
    Produces maze as grid of Cell objects with wall connectivity, and can
    export to SVG or shapely LineString format for simulation obstacles.

    Attributes:
        nx: Number of cells in x-direction
        ny: Number of cells in y-direction
        ix: Starting cell x-coordinate for generation (default: 0)
        iy: Starting cell y-coordinate for generation (default: 0)
        height: Physical height of maze in simulation units (default: 1.0)
        maze_map: 2D array of Cell objects forming the maze

    Example:
        >>> maze = Maze(nx=10, ny=10, height=0.5)
        >>> maze.make_maze()  # Generate maze structure
        >>> lines = maze.maze_lines()  # Get wall LineStrings
        >>> maze.write_svg("output.svg")  # Export to SVG
    """

    def __init__(
        self, nx: int, ny: int, ix: int = 0, iy: int = 0, height: float = 1.0
    ) -> None:
        """
        Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).

        """
        self.height = height
        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x: int, y: int) -> Cell:
        """Return the Cell object_class at (x,y)."""
        return self.maze_map[x][y]

    def __str__(self) -> str:
        """Return a (crude) string representation of the maze."""
        maze_rows = ["-" * self.nx * 2]
        for y in range(self.ny):
            maze_row = ["|"]
            for x in range(self.nx):
                if self.maze_map[x][y].walls["E"]:
                    maze_row.append(" |")
                else:
                    maze_row.append("  ")
            maze_rows.append("".join(maze_row))
            maze_row = ["|"]
            for x in range(self.nx):
                if self.maze_map[x][y].walls["S"]:
                    maze_row.append("-+")
                else:
                    maze_row.append(" +")
            maze_rows.append("".join(maze_row))
        return "\n".join(maze_rows)

    def write_svg(self, filename: str) -> None:
        """Write an SVG image of the maze to filename."""
        aspect_ratio = self.nx / self.ny
        # Pad the maze all around by this amount.
        padding = 10
        # Height and width of the maze image (excluding padding), in pixels
        height = 500
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx

        def write_wall(ww_f, ww_x1, ww_y1, ww_x2, ww_y2):
            """Write a single wall to the SVG image file handle f."""
            print(
                f'<line x1="{ww_x1}" y1="{ww_y1}" x2="{ww_x2}" y2="{ww_y2}"/>',
                file=ww_f,
            )

        # Write the SVG image file for maze
        with open(filename, "w") as f:
            # SVG preamble and styles.
            print('<?xml version="1.0" encoding="utf-8"?>', file=f)
            print('<svg xmlns="http://www.w3.org/2000/svg"', file=f)
            print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=f)
            print(
                f'    width="{width + 2 * padding:d}" height="{height + 2 * padding:d}" viewBox="{-padding} {-padding} {width + 2 * padding} {height + 2 * padding}">',
                file=f,
            )
            print('<defs>\n<style mode="text/css"><![CDATA[', file=f)
            print("line {", file=f)
            print("    stroke: #000000;\n    stroke-linecap: square;", file=f)
            print("    stroke-width: 5;\n}", file=f)
            print("]]></style>\n</defs>", file=f)
            # Draw the "South" and "East" walls of each cell, if present (these
            # are the "North" and "West" walls of a neighbouring cell in
            # general, of course).
            for x in range(self.nx):
                for y in range(self.ny):
                    if self.cell_at(x, y).walls["S"]:
                        x1, y1, x2, y2 = (
                            x * scx,
                            (y + 1) * scy,
                            (x + 1) * scx,
                            (y + 1) * scy,
                        )
                        write_wall(f, x1, y1, x2, y2)
                    if self.cell_at(x, y).walls["E"]:
                        x1, y1, x2, y2 = (
                            (x + 1) * scx,
                            y * scy,
                            (x + 1) * scx,
                            (y + 1) * scy,
                        )
                        write_wall(f, x1, y1, x2, y2)
            # Draw the North and West maze border, which won't have been drawn
            # by the procedure above.
            print(f'<line x1="0" y1="0" x2="{width}" y2="0"/>', file=f)
            print(f'<line x1="0" y1="0" x2="0" y2="{height}"/>', file=f)
            print("</svg>", file=f)

    def find_valid_neighbours(self, cell: Cell) -> List[tuple[str, Cell]]:
        """Return a list of unvisited neighbours to cell."""
        delta = [("W", (-1, 0)), ("E", (1, 0)), ("S", (0, 1)), ("N", (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self) -> None:
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1

    def maze_lines(self) -> list[LineString]:
        lines = []
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = self.height / self.ny, self.height / self.ny

        for x in range(self.nx):
            for y in range(self.ny):
                if self.cell_at(x, y).walls["S"]:
                    lines.append(
                        LineString(
                            [(x * scx, (y + 1) * scy), ((x + 1) * scx, (y + 1) * scy)]
                        )
                    )
                if self.cell_at(x, y).walls["E"]:
                    lines.append(
                        LineString(
                            [((x + 1) * scx, y * scy), ((x + 1) * scx, (y + 1) * scy)]
                        )
                    )
        return lines
