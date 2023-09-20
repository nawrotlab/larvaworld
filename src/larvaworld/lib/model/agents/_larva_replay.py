import numpy as np

from larvaworld.lib import aux
from larvaworld.lib.aux import nam
from larvaworld.lib.model.agents._larva import Larva, LarvaContoured, LarvaSegmented

__all__ = [
    'LarvaReplay',
    'LarvaReplayContoured',
    'LarvaReplaySegmented',
]

__displayname__ = 'Experimental replay larva'

class LarvaReplay(Larva):
    """
    Class representing a larva used to replay recorded data.

    Parameters
    ----------
    data : ReplayData
        The recorded data for the replay larva.
    **kwargs
        Additional keyword arguments to pass to the base `Larva` class constructor.

    Notes
    -----
    This class extends the base `Larva` class to create a replay larva using recorded data. It initializes the replay
    larva's position and orientation based on the provided data.

    """

    __displayname__ = 'Replay larva'

    def __init__(self, data, **kwargs):
        self.data=data
        fo0=self.data.front_orientation[0]
        if np.isnan(fo0):
            fo0=0

        super().__init__(pos=self.data.pos[0],orientation=fo0,**kwargs)





    def step(self):
        """
        Update the replay larva's position and orientation based on recorded data.

        Notes
        -----
        This method updates the replay larva's position and orientation for each time step based on recorded data.
        It also updates the trajectory and orientation_trajectory attributes.

        """

        self.pos = self.data.pos[self.model.t]
        self.trajectory.append(self.pos)
        self.orientation_trajectory.append(self.front_orientation)
        if not np.isnan(self.pos).any():
            self.model.space.move_to(self, np.array(self.pos))


    @property
    def midline_xy(self):
        """
        Get the xy coordinates of the midline points based on recorded data.

        Returns
        -------
        list of tuple
            A list of tuples representing the xy coordinates of the midline points.

        """

        return aux.np2Dtotuples(self.data.midline[self.model.t])

    @property
    def front_orientation(self):
        """
        Get the front orientation of the replay larva based on recorded data.

        Returns
        -------
        float
            The front orientation in radians.

        """

        return self.data.front_orientation[self.model.t]

    @property
    def rear_orientation(self):
        """
        Get the rear orientation of the replay larva based on recorded data.

        Returns
        -------
        float
            The rear orientation in radians.

        """

        return self.data.rear_orientation[self.model.t]




class LarvaReplayContoured(LarvaReplay, LarvaContoured):
    """
    Class representing a replay larva with contour data based on recorded data.

    This class extends the `LarvaReplay` class and adds contour data to the replay larva based on recorded data.

    """

    __displayname__ = 'Contoured replay larva'

    def step(self):
        """
        Update the replay larva's position, orientation, and contour based on recorded data.

        Notes
        -----
        This method updates the replay larva's position and orientation using the base class `LarvaReplay` and adds
        contour data to the replay larva.

        """

        super().step()
        self.vertices=self.contour_xy

    @property
    def contour_xy(self):
        """
        Get the xy coordinates of the contour points based on recorded data.

        Returns
        -------
        list of tuple
            A list of tuples representing the xy coordinates of the contour points.

        """

        a = self.data.contour[self.model.t]
        a = a[~np.isnan(a)].reshape(-1, 2)
        return aux.np2Dtotuples(a)



class LarvaReplaySegmented(LarvaReplay, LarvaSegmented):

    """
    Class representing a segmented replay larva based on recorded data.

    This class extends the `LarvaReplay` class and creates a segmented replay larva with multiple body segments based on
    recorded data.

    """

    __displayname__ = 'Segmented replay larva'

    def step(self):
        """
        Update the replay larva's position, orientation, and body segments based on recorded data.

        Notes
        -----
        This method updates the replay larva's position and orientation using the base class `LarvaReplay` and creates
        and positions multiple body segments based on recorded data.

        """

        super().step()
        mid = self.midline_xy
        ors = self.data.seg_orientations[self.model.t]
        for i, seg in enumerate(self.segs):
            seg.set_position(mid[i])
            try:
                seg.set_orientation(ors[i])
            except:
                pass