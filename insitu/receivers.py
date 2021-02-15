import numpy as np
#import toml
from controlsair import load_cfg, sph2cart

class Receiver():
    """ A receiver class to initialize many types of receivers

    With this class you can instantiate single receivers, arrays and so on.

    Attributes
    ----------
    coord : 1x3 nparray (not fail proof)

    Methods
    ----------
    double_rec(z_dist = 0.01)
        Initializes a double receiver separated by z_dist.
    line_array(startat = 0.001, line_len = 1.0, n_rec = 10, direction = 'z')
        Initializes a line array of receivers.
    arc(radii = 20.0, n_recs = 36)
        Initializes a receiver arc.
    planar_array(x_len = 1.0, n_x = 10, y_len = 1.0, n_y = 10, zr = 0.1)
        Initializes a planar array of receivers (z/xy plane).
    double_planar_array(x_len = 1.0, n_x = 8, y_len = 1.0, n_y = 8, zr = 0.01, dz = 0.01)
        Initializes a double layer planar array of receivers (z/xy plane).
    brick_array(x_len = 1.0, n_x = 8, y_len = 1.0, n_y = 8, z_len = 1.0, n_z = 8, zr = 0.1)
        Initializes a regular 3D array of receivers on a parallelepipid.
    random_3d_array(x_len = 1.0, y_len = 1.0, z_len = 1.0, zr = 0.1, n_total = 192, seed = 0)
        Initializes a randomized 3D array of receivers
    """

    def __init__(self, coord = [0.0, 0.0, 0.01]):
        """ constructor

        Instantiate a single receiver with a given 3D coordinates
        The default is a height of 1 [cm]. User must be sure that the receiver
        lies in a feasable space.

        Parameters
        ----------
        coord : 1x3 list (not fail proof)
        """
        self.coord = np.reshape(np.array(coord, dtype = np.float32), (1,3))

    def double_rec(self, z_dist = 0.01):
        """ Initializes a double receiver separated by z_dist.

        The method uses the coord passed to constructor and will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver.
        In impedance measurements, it is commoen to use the double microphone technique.

        Parameters
        ----------
        z_dist : float
            vertical distance between the pair of receivers.
        """
        self.coord = np.append(self.coord, [self.coord[0,0], self.coord[0,1], self.coord[0,2]+z_dist])
        self.coord = np.reshape(self.coord, (2,3))

    def line_array(self, startat = 0.001, line_len = 1.0, n_rec = 10, direction = 'z'):
        """ Initializes a line array of receivers. 

        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            startat : float
                location of the start the array
            line_len : float
                the length of the line
            n_rec : int
                the number of receivers in the line array
            direction : sty
                direction of the line ('x', 'y' or 'z' - for simplicity)
        """
        self.coord = np.zeros((n_rec,3))
        line = np.linspace(start = startat, stop = line_len+startat, num = n_rec)
        if direction == 'x':
            self.coord[:,0] = line
        elif direction == 'y':
            self.coord[:,1] = line
        else:
            self.coord[:,2] = line

    def arc(self, radii = 20.0, n_recs = 36):
        """" Initializes a receiver arc.

        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Inputs:
            radii - radius of the receiver
            n_recs - number of receivers in the arc

        Parameters
        ----------
            radii : float
                radius of the arc of receivers
            n_recs : int
                number of receivers in the arc
        """
        # angles
        thetas = np.linspace(0, np.pi, n_recs)
        # initialize receiver list in memory
        self.coord = np.zeros((n_recs, 3), dtype = np.float32)
        self.coord[:, 0], self.coord[:, 1], self.coord[:, 2] =\
            sph2cart(radii, thetas, 0)

    def planar_array(self, x_len = 1.0, n_x = 10, y_len = 1.0, n_y = 10, zr = 0.1):
        """ Initializes a planar array of receivers (z/xy plane).

        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            x_len : float
                the length of the x direction (array goes from -x_len/2 to +x_len/2).
            n_x : int
                the number of receivers in the x direction
            y_len : float
                the length of the y direction (array goes from -y_len/2 to +y_len/2).
            n_y : int
                the number of receivers in the y direction
            zr : float
                distance from the microphones to the sample
        """
        # size of the array in x and y directions
        self.x_len = x_len
        self.y_len = y_len
        # spacing between the microphones in x and y directions
        self.ax = self.x_len/n_x
        self.ay = self.y_len/n_y
        # x and y coordinates of the grid
        xc = np.linspace(-x_len/2, x_len/2, n_x)
        yc = np.linspace(-y_len/2, y_len/2, n_y)
        # meshgrid
        xv, yv = np.meshgrid(xc, yc)
        # initialize receiver list in memory
        self.coord = np.zeros((n_x * n_y, 3), dtype = np.float32)
        self.coord[:, 0] = xv.flatten()
        self.coord[:, 1] = yv.flatten()
        self.coord[:, 2] = zr

    def double_planar_array(self, x_len = 1.0, n_x = 8, y_len = 1.0, n_y = 8, zr = 0.01, dz = 0.01):
        """ Initializes a double layer planar array of receivers (z/xy plane).

        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            x_len : float
                the length of the x direction (array goes from -x_len/2 to +x_len/2).
            n_x : int
                the number of receivers in the x direction
            y_len : float
                the length of the y direction (array goes from -y_len/2 to +y_len/2).
            n_y : int
                the number of receivers in the y direction
            zr : float
                distance from the closest layer to the sample
            dz : float
                separation distance between the two layers
        """
        # size of the array in x and y directions
        self.x_len = x_len
        self.y_len = y_len
        # spacing between the microphones in x and y directions
        self.ax = self.x_len/n_x
        self.ay = self.y_len/n_y
        # x and y coordinates of the grid
        xc = np.linspace(-x_len/2, x_len/2, n_x)
        yc = np.linspace(-y_len/2, y_len/2, n_y)
        # meshgrid
        xv, yv = np.meshgrid(xc, yc)
        # initialize receiver list in memory
        self.coord = np.zeros((2 * n_x * n_y, 3), dtype = np.float32)
        self.coord[0:n_x*n_y, 0] = xv.flatten()
        self.coord[0:n_x*n_y, 1] = yv.flatten()
        self.coord[0:n_x*n_y, 2] = zr
        self.coord[n_x*n_y:, 0] = xv.flatten()
        self.coord[n_x*n_y:, 1] = yv.flatten()
        self.coord[n_x*n_y:, 2] = zr + dz

    def brick_array(self, x_len = 1.0, n_x = 8, y_len = 1.0, n_y = 8, z_len = 1.0, n_z = 8, zr = 0.1):
        """ Initializes a regular 3D array of receivers on a parallelepipid.

        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            x_len : float
                the length of the x direction (array goes from -x_len/2 to +x_len/2).
            n_x : int
                the number of receivers in the x direction
            y_len : float
                the length of the y direction (array goes from -y_len/2 to +y_len/2).
            n_y : int
                the number of receivers in the y direction
            z_len : float
                the length of the z direction (array goes from -z_len/2 to +z_len/2).
            n_z : int
                the number of receivers in the z direction
            zr : float
                distance from the closest layer to the sample
        """
        # size of the array in x and y directions
        self.x_len = x_len
        self.y_len = y_len
        # spacing between the microphones in x and y directions
        self.ax = self.x_len/n_x
        self.ay = self.y_len/n_y
        # x and y coordinates of the grid
        xc = np.linspace(-x_len/2, x_len/2, n_x)
        yc = np.linspace(-y_len/2, y_len/2, n_y)
        zc = np.linspace(zr, zr+z_len, n_z)
        # print('sizes: xc {}, yc {}, zc {}'.format(xc.size, yc.size, zc.size))
        # meshgrid
        xv, yv, zv = np.meshgrid(xc, yc, zc)
        # print('sizes: xv {}, yv {}, zv {}'.format(xv.shape, yv.shape, zv.shape))
        # initialize receiver list in memory
        self.coord = np.zeros((n_x * n_y * n_z, 3), dtype = np.float32)
        self.coord[0:n_x*n_y*n_z, 0] = xv.flatten()
        self.coord[0:n_x*n_y*n_z, 1] = yv.flatten()
        self.coord[0:n_x*n_y*n_z, 2] = zv.flatten()

    def random_3d_array(self, x_len = 1.0, y_len = 1.0, z_len = 1.0, zr = 0.1, n_total = 192, seed = 0):
        """ Initializes a randomized 3D array of receivers

        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            x_len : float
                the length of the x direction (array goes from -x_len/2 to +x_len/2).
            y_len : float
                the length of the y direction (array goes from -y_len/2 to +y_len/2).
            z_len : float
                the length of the z direction (array goes from -z_len/2 to +z_len/2).
            zr : float
                distance from the closest microphone to the sample
            n_total : int
                the number of receivers in the z direction
            seed : int
                seed of random number generator
        """
        # size of the array in x and y directions
        self.x_len = x_len
        self.y_len = y_len
        # spacing between the microphones in x and y directions
        self.ax = self.x_len/(n_total**(1/3))
        self.ay = self.y_len/(n_total**(1/3))
        # x and y coordinates of the grid
        np.random.seed(seed)
        xc = -x_len/2 + x_len * np.random.rand(n_total)#np.linspace(-x_len/2, x_len/2, n_x)
        yc = -y_len/2 + y_len * np.random.rand(n_total)
        zc = zr + z_len * np.random.rand(n_total)
        # initialize receiver list in memory
        self.coord = np.zeros((n_total, 3), dtype = np.float32)
        self.coord[0:n_total, 0] = xc.flatten()
        self.coord[0:n_total, 1] = yc.flatten()
        self.coord[0:n_total, 2] = zc.flatten()

    def planar_xz(self, x_len = 1.0, n_x = 10, z0 = 0 ,z_len = 1.0, n_z = 10, yr = 0.0):
        """ Initializes a planar array of receivers (xz plane).

        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            x_len : float
                the length of the x direction (array goes from -x_len/2 to +x_len/2).
            n_x : int
                the number of receivers in the x direction
            z0 : float
                initial z coordinate
            z_len : float
                the length of the z direction (array goes from z0 to z0+z_len).
            n_z : int
                the number of receivers in the z direction
            yr : float
                y coordinate of the plane
        """
        # x and y coordinates of the grid
        xc = np.linspace(-x_len/2, x_len/2, n_x)
        zc = np.linspace(z0, z_len, n_z)
        # meshgrid
        self.x_grid, self.z_grid = np.meshgrid(xc, zc)
        # initialize receiver list in memory
        self.coord = np.zeros((n_x * n_z, 3), dtype = np.float32)
        self.coord[:, 0] = self.x_grid.flatten()
        self.coord[:, 1] = yr
        self.coord[:, 2] = self.z_grid.flatten()

    def spherical_array(self, radius = 0.1, n_rec = 32, center_dist = 0.5):
        """ Initializes a spherical array of receivers. To be done.

        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
        Inputs:
            radius : float
                the radius of the sphere
            n-rec : int
                the number of receivers in the array
            center_dist : float
                distance from the origin to the center of the array
        """
        pass



# class Receivers():
#     def __init__(self, config_file):
#         '''
#         Set up the receivers
#         '''
#         config = load_cfg(config_file)
#         coord = []
#         orientation = []
#         for r in config['receivers']:
#             coord.append(r['position'])
#             orientation.append(r['orientation'])
#         self.coord = np.array(coord)
#         self.orientation = np.array(orientation)

    

# def setup_receivers(config_file):
#     '''
#     Set up the sound sources
#     '''
#     receivers = [] # An array of empty receiver objects
#     config = load_cfg(config_file) # toml file
#     for r in config['receivers']:
#         coord = np.array(r['position'], dtype=np.float32)
#         orientation = np.array(r['orientation'], dtype=np.float32)
#         ################### cpp receiver class #################
#         receivers.append(insitu_cpp.Receivercpp(coord, orientation)) # Append the source object
#         ################### py source class ################
#         # receivers.append(Receiver(coord, orientation))
#     return receivers

# # class Receiver from python side


# def planar_xz(self, x_len = 1.0, n_x = 10, z0 = 0 ,z_len = 1.0, n_z = 10, yr = 0.0):
#         '''
#         This method initializes a planar array of receivers (z/xy plane). It will overwrite
#         self.coord to be a matrix where each line gives a 3D coordinate for each receiver
#         Inputs:
#             x_len - the length of the x direction (array goes from -x_len/2 to +x_len/2).
#             n_x - the number of receivers in the x direction
#             y_len - the length of the y direction (array goes from -x_len/2 to +x_len/2).
#             n_y - the number of receivers in the y direction
#             zr - distance from the closest microphone layer to the sample
#         '''
#         # x and y coordinates of the grid
#         xc = np.linspace(-x_len/2, x_len/2, n_x)
#         zc = np.linspace(z0, z_len, n_z)
#         # meshgrid
#         self.x_grid, self.z_grid = np.meshgrid(xc, zc)
#         # initialize receiver list in memory
#         self.coord = np.zeros((n_x * n_z, 3), dtype = np.float32)
#         self.coord[:, 0] = self.x_grid.flatten()
#         self.coord[:, 1] = yr
#         self.coord[:, 2] = self.z_grid.flatten()