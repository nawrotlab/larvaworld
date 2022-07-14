import random
from matplotlib.patches import Circle
from shapely.geometry import LineString, Point

from lib.aux.dictsNlists import group_list_by_n
import lib.aux.colsNstr as fun
from lib.aux import dictsNlists as dNl, ang_aux, sim_aux, shapely_aux



class Cell:
    """A cell in the maze.

    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.

    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False


class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny, ix=0, iy=0, height=1.0):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).

        """
        self.height=height
        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x, y):
        """Return the Cell object_class at (x,y)."""

        return self.maze_map[x][y]

    def __str__(self):
        """Return a (crude) string representation of the maze."""

        maze_rows = ['-' * self.nx * 2]
        for y in range(self.ny):
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['E']:
                    maze_row.append(' |')
                else:
                    maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)

    def write_svg(self, filename):
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

            print('<line x1="{}" y1="{}" x2="{}" y2="{}"/>'
                  .format(ww_x1, ww_y1, ww_x2, ww_y2), file=ww_f)

        # Write the SVG image file for maze
        with open(filename, 'W') as f:
            # SVG preamble and styles.
            print('<?xml version="1.0" encoding="utf-8"?>', file=f)
            print('<svg xmlns="http://www.w3.org/2000/svg"', file=f)
            print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=f)
            print('    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'
                  .format(width + 2 * padding, height + 2 * padding,
                          -padding, -padding, width + 2 * padding, height + 2 * padding),
                  file=f)
            print('<defs>\n<style mode="text/css"><![CDATA[', file=f)
            print('line {', file=f)
            print('    stroke: #000000;\n    stroke-linecap: square;', file=f)
            print('    stroke-width: 5;\n}', file=f)
            print(']]></style>\n</defs>', file=f)
            # Draw the "South" and "East" walls of each cell, if present (these
            # are the "North" and "West" walls of a neighbouring cell in
            # general, of course).
            for x in range(self.nx):
                for y in range(self.ny):
                    if self.cell_at(x, y).walls['S']:
                        x1, y1, x2, y2 = x * scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
                    if self.cell_at(x, y).walls['E']:
                        x1, y1, x2, y2 = (x + 1) * scx, y * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
            # Draw the North and West maze border, which won't have been drawn
            # by the procedure above.
            print('<line x1="0" y1="0" x2="{}" y2="0"/>'.format(width), file=f)
            print('<line x1="0" y1="0" x2="0" y2="{}"/>'.format(height), file=f)
            print('</svg>', file=f)

    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self):
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

    def maze_lines(self):
        lines=[]
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = self.height / self.ny, self.height / self.ny

        for x in range(self.nx):
            for y in range(self.ny):
                if self.cell_at(x, y).walls['S']:
                    lines.append(LineString([(x * scx, (y + 1) * scy), ((x + 1) * scx, (y + 1) * scy)]))
                if self.cell_at(x, y).walls['E']:
                    lines.append(LineString([((x + 1) * scx, y * scy), ((x + 1) * scx, (y + 1) * scy)]))
        return lines

class Border:
    def __init__(self, model, points=None, unique_id=None, width=0.001, default_color='black'):
        from lib.model.space.obstacle import Wall
        self.model=model
        if type(default_color)==str :
            default_color=fun.colorname2tuple(default_color)
        elif default_color is None :
            default_color=self.model.screen_color
        self.default_color=default_color
        if unique_id is None :
            unique_id= f'Border_{len(self.model.border_xy)}'
        self.unique_id = unique_id
        self.width=width * self.model.scaling_factor
        self.points=points
        lines = [LineString([tuple(p1), tuple(p2)]) for p1, p2 in group_list_by_n(points, 2)]
        self.border_xy, self.border_lines = self.model.create_borders(lines)
        self.border_bodies = self.model.create_border_bodies(self.border_xy)
        self.border_walls=[]
        for l in self.border_lines :
            # print(list(l.coords))
            (x1, y1),(x2,y2)=list(l.coords)
            point1 = shapely_aux.Point(x1, y1)
            point2 = shapely_aux.Point(x2, y2)
            wall=Wall(point1, point2, color=self.default_color)
            # edges = [[point1, point2]]
            self.border_walls.append(wall)

        self.selected=False

    def delete(self):
        for xy in self.border_xy :
            self.model.border_xy.remove(xy)
        for l in self.border_lines:
            self.model.border_lines.remove(l)
        if len(self.border_bodies)>0 :
            for b in self.border_bodies :
                self.model.border_bodies.remove(b)
                self.model.space.delete(b)
        del self

    def draw(self, screen):
        for b in self.border_xy :
            screen.draw_polyline(b, color=self.default_color, width=self.width, closed=False)
            if self.selected :
                screen.draw_polyline(b, color=self.model.selection_color, width=self.width*0.5, closed=False)

    def contained(self,p):
        return any([l.distance(Point(p))<self.width for l in self.border_lines])

    def set_id(self,id):
        self.unique_id=id