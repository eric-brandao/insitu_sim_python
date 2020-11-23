import numpy as np
from controlsair import sph2cart, cart2sph
from rayinidir import RayInitialDirections


class Source():
    """ A sound source class to initialize some types of sources

    With this class you can instantiate single sources or arrays of sources and so on.

    Attributes
    ----------
    coord : 1x3 nparray (not fail proof)

    Methods
    ----------
    add_sources(coord)
        Adds a source to the list of sources.
    set_arc_sources(radius = 1.0, ns = 100, angle_span = (-90, 90), random = False)
        Generate an array of sound sources in an 2D arc
    set_ssph_sources(radius = 1.0, ns = 100, angle_max=90, random = False, plot=False)
        Generate an array of sound sources over a surface of a sphere.
    """

    def __init__(self, coord = [0.0, 0.0, 1.0], q = 1.0):
        """ constructor

        Instantiate a single source with a given 3D coordinates
        User must be sure that the source lies in a feasable space.

        Parameters
        ----------
        coord : 1x3 list (not fail proof)
        """
        self.coord = np.reshape(np.array(coord), (1,3))
        self.q = np.array(q, dtype = np.float32)

    def add_sources(self, coord):
        """ Adds a source to the list of sources.

        The method uses the coord passed to append a source to a list of sources.

        Parameters
        ----------
        coord : 1x3 list (not fail proof)
        """
        coord = np.array(coord)
        ns = int(len(coord.flatten())/3)
        coord = np.reshape(coord, (ns, 3))
        self.coord = np.append(self.coord, coord, axis = 0)

    def set_arc_sources(self, radius = 1.0, ns = 100, angle_span = (-90, 90), random = False):
        """ Generate an array of sound sources in an 2D arc

        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            radius : float
                radii of the source arc (how far from the sample they are)
            ns : int
                the number of sound sources in the arc
            angle_span : tuple
                the angle range in [deg] for which the sources span
            random : bool
                if True, then the complex amplitudes are randomized. It is not
                in use to this point. Maybe in the future.
        """
        theta = np.linspace(start = -np.pi/2, stop = np.pi/2, num = ns)
        x_coord = radius * np.sin(theta)
        y_coord = np.zeros(len(theta))
        z_coord = radius * np.cos(theta)
        self.coord = np.zeros((len(theta), 3), dtype=np.float32)
        self.coord[:,0] = x_coord.flatten()
        self.coord[:,1] = y_coord.flatten()
        self.coord[:,2] = z_coord.flatten()

    def set_ssph_sources(self, radius = 1.0, ns = 100, angle_max=90, random = False, plot=False):
        """ Generate an array of sound sources over a surface of a sphere.

        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver. The method will use icosahedron
        decomposition to generate evenly distributed points on the surface of the
        sphere. The actual number of sources generated may differ from the intended
        number of sources originally intended.

        Parameters
        ----------
            radius float
                radii of the hemisphere (how far from the sample they are)
            ns : int
                Intended number of sound sources in the sphere
            angle_max : float
                the maximum elevation angle in [deg] for which the sources span
            random : bool
                if True, then the complex amplitudes are randomized. It is not
                in use to this point. Maybe in the future.
        """
        # Define theta and phi discretization
        directions = RayInitialDirections()
        directions, _ = directions.isotropic_rays(Nrays = ns)
        #print('The number of sources is: {}'.format(n_waves))
        if plot:
            directions.plot_points()
        _, theta,_ = cart2sph(directions[:,0], directions[:,1], directions[:,2])
        theta_id = np.where(np.logical_and(theta > np.deg2rad(0), theta < np.deg2rad(angle_max)))
        self.coord = radius * directions[theta_id[0]]


    # def set_vsph_sources(self, radii_span = (1.0, 10.0), ns = 100, random = False):
    #     '''
    #     This method is used to generate an array of sound sources over the volume of a sphere
    #     Inputs:
    #         radii_span - tuple with the range for which the sources span
    #         ns - the number of sound sources in the sphere
    #         random (bool) - if True, then the complex amplitudes are randomized
    #     '''
    #     pass

    # def set_sphdir_sources(self, ns = 100, random = False):
    #     '''
    #     This method is used to generate an array of sound sources directions for plane wave simulation
    #     Inputs:
    #         radius - radii of the source arc (how far from the sample they are)
    #         ns - the number of sound sources in the sphere
    #         random (bool) - if True, then the complex amplitudes are randomized
    #     '''
    #     pass
