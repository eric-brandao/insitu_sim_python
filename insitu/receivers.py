import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from mpl_toolkits.mplot3d import Axes3D
#import toml
from controlsair import load_cfg, sph2cart, cart2sph
from rayinidir import RayInitialDirections
import utils_insitu as ut_is


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
        """ Constructor

        Instantiate a single receiver with a given 3D coordinates
        The default is a height of 1 [cm]. User must be sure that the receiver
        lies in a feasable space.

        Parameters
        ----------
        coord : list or 1dArray
            The coordinates of a point in space (x,y,z) (not fail proof)
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

    def line_array(self, line_len = 1.0, step = 0.1, axis = 'z', 
                   start_at = 0, zr = 0.1):
        """ Initializes a line array of receivers. 

        Line array of receiver along x, y or z axis. It includes the line_len coord.
        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            line_len : float
                the length of the line array in [m]
            step : float
                the spacing between the points of the line in [m]
            axis : str
                axis along which the line is created. It can be either 'x', 'y' or 'z'
            start_at : float
                location of the start the array
            zr : float
                distance from the z = 0 plane in [m]
        """
        line = np.arange(start = 0, stop = line_len+step, step = step)
        self.ax = step
        self.coord = np.zeros((len(line),3))
        if axis == 'x':
            self.coord[:,0] = line
        elif axis == 'y':
            self.coord[:,1] = line
        elif axis == 'z':
            self.coord[:,2] = line
        
        self.translate_array(axis = axis, delta = start_at)
        self.translate_array(axis = 'z', delta = zr)
            
    def double_line_array(self, line_len = 1.0, step = 0.1, axis = 'z', 
                   start_at = 0, zr = 0.1, dz = 0.1):
        """ Initializes a double line array of receivers. 

        Double line array of receiver along x, y or z axis. It includes the line_len coord.
        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            line_len : float
                the length of the line array in [m]
            step : float
                the spacing between the points of the line in [m]
            axis : str
                axis along which the line is created. It can be either 'x', 'y' or 'z'
            start_at : float
                location of the start the array
            zr : float
                distance from the z = 0 plane in [m]
            dz : float
                distance between the lines in [m]
        """
        # start with 1 line array
        rec1 = Receiver()
        rec1.line_array(line_len = line_len, step = step, axis = axis, 
                       start_at = start_at, zr = zr)
        # initialize coordinate vector of the double line array
        coord_2line = np.zeros((int(2*rec1.coord.shape[0]), rec1.coord.shape[1]))
        
        # Create the 2nd line array
        rec2 = Receiver()
        rec2.line_array(line_len = line_len, step = step, axis = axis, 
                       start_at = start_at, zr = zr + dz)
        # fill coordinate vector
        coord_2line[:int(rec1.coord.shape[0]),:] = rec1.coord
        coord_2line[int(rec1.coord.shape[0]):,:] = rec2.coord
        self.ax = step
        self.coord = coord_2line

    def planar_array(self, x_len = 1.0, n_x = 10, y_len = 1.0, n_y = 10, zr = 0.1):
        """ Initializes a planar array of receivers (xy plane).

        Creates a planar array with regular spacing between the receivers.
        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            x_len : float
                the length of the x direction in [m] (array goes from -x_len/2 to +x_len/2).
            n_x : int
                the number of receivers in the x direction
            y_len : float
                the length of the y direction in [m] (array goes from -y_len/2 to +y_len/2).
            n_y : int
                the number of receivers in the y direction
            zr : float
                distance from the microphones to the z=0 plane
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
        self.coord = np.zeros((n_x * n_y, 3))
        self.coord[:, 0] = xv.flatten()
        self.coord[:, 1] = yv.flatten()
        self.coord[:, 2] = zr
        
    def planar_array_xz(self, x_len = 1.0, n_x = 10, z_len = 1.0, n_z = 10,
                        zr = 0, yr = 0.0):
        """ Initializes a planar array of receivers (xz plane).

        Creates a planar array with regular spacing between the receivers along
        xz axis - useful for validation.
        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            x_len : float
                the length of the x direction in [m] (array goes from -x_len/2 to +x_len/2).
            n_x : int
                the number of receivers in the x direction
            y_len : float
                the length of the y direction in [m] (array goes from -y_len/2 to +y_len/2).
            n_y : int
                the number of receivers in the y direction
            zr : float
                distance from the microphones to the z=0 plane
            yr : float
                distance from the microphones to the y=0 plane
        """            
        # x and y coordinates of the grid
        xc = np.linspace(-x_len/2, x_len/2, n_x)
        zc = np.linspace(zr, z_len, n_z)
        # meshgrid
        self.x_grid, self.z_grid = np.meshgrid(xc, zc)
        # initialize receiver list in memory
        self.coord = np.zeros((n_x * n_z, 3), dtype = np.float32)
        self.coord[:, 0] = self.x_grid.flatten()
        self.coord[:, 1] = yr
        self.coord[:, 2] = self.z_grid.flatten()
    
    def random_planar_array(self, x_len = 0.8, y_len = 0.6,  
                         zr = 0.1, nx = 10, ny = 10,
                         delta_xy = None, seed = 0, plot = False):
        """ Initializes a randomized 3D array of receivers with minimal sampling distance

        Creates a planar array with irregular spacing between the receivers.
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
                distance from the microphones to the z=0 plane
            nx : int
                number of microphones along the x axis
            ny : int
                number of microphones along the y axis
            delta_xy : float
                You can specify the smallest spacing. The default is None, in which 
                case nx and ny are used. If you do specify delta_xy, then nx and ny 
                will be computed for you according to the specified spacing.
            seed : int
                the seed of the random sampling
            plot : bool
                Say True to see all corners, center points and array coordinates
        """
        # size of the array in x and y directions
        self.x_len = x_len
        self.y_len = y_len
                
        if delta_xy is None:
            number_pts_line = np.array([nx+1, ny+1])
            # spacing between the microphones in x and y directions
            self.ax = x_len/nx
            self.ay = y_len/ny
        else:
            delta_diagonal = delta_xy / np.sqrt(2) # the diagonal of each cube
            number_pts_line = np.ceil(np.array([self.x_len/delta_diagonal, 
                                        self.y_len/delta_diagonal]))
            # spacing between the microphones in x and y directions
            self.ax = delta_xy
            self.ay = delta_xy
        
        id_num_lesseq_than_1 = np.where(number_pts_line<=1)[0]
        number_pts_line[id_num_lesseq_than_1] = 2
        # lines x, y, z - vertices
        line_x = np.linspace(-self.x_len/2, self.x_len/2, int(number_pts_line[0]))
        line_y = np.linspace(-self.y_len/2, self.y_len/2, int(number_pts_line[1]))
        
        # line spacing
        dx = line_x[1]-line_x[0]
        dy = line_y[1]-line_y[0]
                
        # lines x, y, z - centers
        line_x_c = 0.5*(line_x[1:]-line_x[0:-1]) + line_x[0:-1]
        line_y_c = 0.5*(line_y[1:]-line_y[0:-1]) + line_y[0:-1]
        
        # center coordinates
        num_of_sq = int(len(line_x_c)*len(line_y_c))
        center_coord = np.zeros((num_of_sq, 3))
        counter = 0
        for xc in line_x_c:
            for yc in line_y_c:
                center_coord[counter, :] = np.array([xc, yc, zr])
                counter += 1
        
        # Cube vertices - not in order, but it does not matter
        x_mesh, y_mesh = np.meshgrid(line_x, line_y, indexing='xy')
        sq_vertices = np.array([x_mesh.flatten(), y_mesh.flatten()]).T
         
        # define a random point for each cube
        np.random.seed(seed)
        self.coord = np.zeros((num_of_sq, 3))
        for sq_id in range(num_of_sq):
            
            xc = center_coord[sq_id, 0] - dx/2 + dx * np.random.rand(1)
            yc = center_coord[sq_id, 1] - dy/2 + dy * np.random.rand(1)
            self.coord[sq_id, :] = np.array([xc, yc, zr]).T
        
        if plot:
            ## Figure
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d') #fig.gca()#projection='3d'
            ax.scatter(sq_vertices[:, 0], sq_vertices[:, 1], zr, 
                        marker='o', s=12, color='green', alpha = 0.4)
            
            ax.scatter(center_coord[:, 0], center_coord[:, 1], center_coord[:, 2], 
                        marker='o', s=10, color='red', alpha = 0.3)
            
            ax.scatter(self.coord[:, 0], self.coord[:, 1], self.coord[:, 2], 
                        marker='o', s=10, color='blue', alpha = 0.9)
            
            ax.set_title("Cube")
            ax.set_xlabel(r'$x$ [m]')
            #ax.set_xticks([-0.5, -0.5, L_x/2, baffle_size/2])
            ax.set_ylabel(r'$y$ [m]')
            #ax.set_yticks([-L_y/2, L_y/2])
            ax.set_zlabel(r'$z$ [m]')
            #ax.set_zticks([-sample_thickness, 1.0, 2, 3])
            ax.grid(False)
            ax.set_xlim((-2*np.amax(sq_vertices[:,0]), 2*np.amax(sq_vertices[:,0])))
            ax.set_ylim((-2*np.amax(sq_vertices[:,1]), 2*np.amax(sq_vertices[:,1])))
            ax.set_zlim((-0.05, 2*zr))
            #ax.view_init(elev=elev, azim=azim)
            plt.tight_layout()
            plt.show()

    def double_planar_array(self, x_len = 1.0, n_x = 8, y_len = 1.0, n_y = 8, 
                            zr = 0.01, dz = 0.01):
        """ Initializes a double layer planar array of receivers (z/xy plane).

        Creates a double layer planar array with regular spacing between the receivers.
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
                distance from the microphones to the z=0 plane
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

        Creates a 3D array with regular spacing between the receivers.
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
                distance from the microphones to the z=0 plane
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

        Creates a 3D array with irregular spacing between the receivers. Here,
        the number of desired microphones is the priority. The consequence is that
        random points are sampled within the desired volume, and there is no way to
        control the spacing between the microphones.        
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
                distance from the microphones to the z=0 plane
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
        
    def random_3d_array2(self, x_len = 0.8, y_len = 0.6, z_len = 0.25, 
                         zr = 0.1, nx = 10, ny = 10, nz = 10,
                         delta_xyz = None, seed = 0, plot = False):
        """ Initializes a randomized 3D array of receivers with minimal sampling distance
        
        Creates a 3D array with irregular spacing between the receivers. Here,
        the spacing between the microphones is the priority. You can either specify
        the desired number of microphones along each axis (default) or specify the minimum
        desired spacing (then, the number of microphones along each axis will be computed for you)
        The consequence is that random points are sampled within small voxels, and there is no way to
        control the number of microphones in the array.
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
                distance from the microphones to the z=0 plane
            nx : int
                number of microphones along x axis
            ny : int
                number of microphones along y axis
            nz : int
                number of microphones along z axis
            delta_xy : float
                You can specify the smallest spacing. The default is None, in which 
                case nx and ny are used. If you do specify delta_xy, then nx and ny 
                will be computed for you according to the specified spacing.
            seed : int
                the seed of the random sampling
            plot : bool
                Say True to see all corners, center points and array coordinates
        """
        # size of the array in x and y directions
        self.x_len = x_len
        self.y_len = y_len
        self.z_len = z_len
                
        if delta_xyz is None:
            number_pts_line = np.array([nx+1, ny+1, nz+1])
            # spacing between the microphones in x and y directions
            self.ax = x_len/nx
            self.ay = y_len/ny
        else:
            delta_diagonal = delta_xyz / np.sqrt(3) # the diagonal of each cube
            number_pts_line = np.ceil(np.array([self.x_len/delta_diagonal, 
                                        self.y_len/delta_diagonal, self.z_len/delta_diagonal]))
            # spacing between the microphones in x and y directions
            self.ax = delta_xyz
            self.ay = delta_xyz
        
        id_num_lesseq_than_1 = np.where(number_pts_line<=1)[0]
        number_pts_line[id_num_lesseq_than_1] = 2
        # lines x, y, z - vertices
        line_x = np.linspace(-self.x_len/2, self.x_len/2, int(number_pts_line[0]))
        line_y = np.linspace(-self.y_len/2, self.y_len/2, int(number_pts_line[1]))
        line_z = np.linspace(zr, self.z_len + zr, int(number_pts_line[2]))
        
        # line spacing
        dx = line_x[1]-line_x[0]
        dy = line_y[1]-line_y[0]
        dz = line_z[1]-line_z[0]
                
        # lines x, y, z - centers
        line_x_c = 0.5*(line_x[1:]-line_x[0:-1]) + line_x[0:-1]
        line_y_c = 0.5*(line_y[1:]-line_y[0:-1]) + line_y[0:-1]
        line_z_c = 0.5*(line_z[1:]-line_z[0:-1]) + line_z[0:-1]
        
        # center coordinates
        num_of_cubes = int(len(line_x_c)*len(line_y_c)*len(line_z_c))
        center_coord = np.zeros((num_of_cubes, 3))
        counter = 0
        for xc in line_x_c:
            for yc in line_y_c:
                for zc in line_z_c:
                    center_coord[counter, :] = np.array([xc, yc, zc])
                    counter += 1
        
        # Cube vertices - not in order, but it does not matter
        x_mesh, y_mesh, z_mesh = np.meshgrid(line_x, line_y, line_z, indexing='xy')
        cube_vertices = np.array([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()]).T
         
        # define a random point for each cube
        np.random.seed(seed)
        self.coord = np.zeros((num_of_cubes, 3))
        for cube_id in range(num_of_cubes):
            
            xc = center_coord[cube_id, 0] - dx/2 + dx * np.random.rand(1)
            yc = center_coord[cube_id, 1] - dy/2 + dy * np.random.rand(1)
            zc = center_coord[cube_id, 2] - dz/2 + dz * np.random.rand(1)
            self.coord[cube_id, :] = np.array([xc, yc, zc]).T
        
        if plot:
            ## Figure
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d') #fig.gca()#projection='3d'
            ax.scatter(cube_vertices[:, 0], cube_vertices[:, 1], cube_vertices[:, 2], 
                        marker='o', s=12, color='green', alpha = 0.4)
            
            ax.scatter(center_coord[:, 0], center_coord[:, 1], center_coord[:, 2], 
                        marker='o', s=10, color='red', alpha = 0.3)
            
            ax.scatter(self.coord[:, 0], self.coord[:, 1], self.coord[:, 2], 
                        marker='o', s=10, color='blue', alpha = 0.9)
            
            ax.set_title("Cube")
            ax.set_xlabel(r'$x$ [m]')
            #ax.set_xticks([-0.5, -0.5, L_x/2, baffle_size/2])
            ax.set_ylabel(r'$y$ [m]')
            #ax.set_yticks([-L_y/2, L_y/2])
            ax.set_zlabel(r'$z$ [m]')
            #ax.set_zticks([-sample_thickness, 1.0, 2, 3])
            ax.grid(False)
            ax.set_xlim((-2*np.amax(cube_vertices[:,0]), 2*np.amax(cube_vertices[:,0])))
            ax.set_ylim((-2*np.amax(cube_vertices[:,1]), 2*np.amax(cube_vertices[:,1])))
            ax.set_zlim((-0.05, 2*np.amax(cube_vertices[:,2])))
            #ax.view_init(elev=elev, azim=azim)
            plt.tight_layout()
            plt.show()
        
    def arc(self, radius = 20.0, n_recs = 36):
        """" Initializes a receiver arc.

        Receiver arc with a given radius and number of receivers.
        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

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
            sph2cart(radius, thetas, 0)

    def hemispherical_array(self, radius = 1, n_rec_target = 32):
        """ Initializes a hemispherical array of receivers (icosahedron).

        Receiver hemisphere with a given radius and target number of receivers.
        you might get more. 
        The method will overwrite self.coord to be a matrix where each line
        gives a 3D coordinate for each receiver

        Parameters
        ----------
            radius : float
                the radius of the sphere
            n-rec : int
                the number of receivers in the array
            center_dist : float
                distance from the origin to the center of the array
        """
        directions = RayInitialDirections()
        directions, n_sph, elements = directions.isotropic_rays(Nrays = int(n_rec_target))
        # elements = directions.indices
        id_dir = np.where(directions[:,2]>=0)
        self.id_dir = id_dir
        self.coord = directions[id_dir[0],:]
        self.pdir_all = directions
        self.n_prop = len(self.coord[:,0])
        self.conectivities_all = elements
        self.conectivity_correction()
        r, theta, phi = cart2sph(self.coord[:,0], self.coord[:,1], self.coord[:,2])
        r = radius*r
        self.coord[:,0], self.coord[:,1], self.coord[:,2] = sph2cart(r,theta,phi)

    def conectivity_correction(self,):
        """ Connectivity correction for mesh plotting
        """
        self.sign_vec = np.array([self.pdir_all[self.conectivities_all[:,0], 2],
                             self.pdir_all[self.conectivities_all[:,1], 2],
                             self.pdir_all[self.conectivities_all[:,2], 2]])
        self.sign_z = np.sign(self.sign_vec)
        n_rows = np.sum(self.sign_z.T < 0 , 1)
        self.p_rows = np.where(n_rows == 0)[0]
        self.conectivities = self.conectivities_all[self.p_rows,:]
        self.delta = self.id_dir[0]-np.arange(self.coord.shape[0])

    # def hemispherical_array2(self, radius = 1, n_rec_target = 32):
    #     from decomposition_ev_ig import DecompositionEv2
    #     pk = DecompositionEv2()
    #     pk.prop_dir(n_waves = n_rec_target, plot = False)
    #     self.coord = pk.pdir
    #     self.conectivities = pk.conectivities
    
    def round_array(self, num_of_dec_cases = 3):
        """ Round the coordinates of the array
        
        You can round the self.coord (coordinates) to a desired number of decimal places.
        This is useful for measurements, since the precision of the positioning system is not
        perfect
        
        Parameters
        ----------
        num_of_dec_cases : int
            number of decimal cases to round to.
        """
        self.coord = np.round(self.coord, decimals = num_of_dec_cases)
        
    def translate_array(self, axis = 'x', delta = 0.1):
        """Translate array over x, y or z axis
        
        Translate the array by a distance along an axis
        
        Parameters
        ----------
        axis : str
            axis along which to translate
        delta : float
            distance to translate in [m]
        """     
        if axis == 'x':
            self.coord[:, 0] += delta
        elif axis == 'y':
            self.coord[:, 1] += delta
        elif axis == 'z':
            self.coord[:, 2] += delta
        else:
            self.coord[:, 2] += 0

    def rotate_array(self, axis = 'x', theta_deg = 45):
        """ Rotate array over x, y or z axis by an angle
        
        Parameters
        ----------
        axis : str
            axis along which to rotate
        theta_deg : float
            angle in [deg] to rotate.
        """
        if axis == 'x':
            rot_mxt = self.get_rotmtx_x(theta_deg = theta_deg)
        elif axis == 'y':
            rot_mxt = self.get_rotmtx_y(theta_deg = theta_deg)
        elif axis == 'z':
            rot_mxt = self.get_rotmtx_z(theta_deg = theta_deg)
        else:
            rot_mxt = self.get_rotmtx_z(theta_deg = 0)
            print("no valid rotation")
            
        self.coord = self.coord @ rot_mxt
    
    def get_rotmtx_x(self, theta_deg = 45):
        """ Get a rotation matrix over x axis
        """
        theta_rad = np.deg2rad(theta_deg)
        rot_mtx = np.array([[1, 0, 0], 
                            [0, np.cos(theta_rad), -np.sin(theta_rad)],
                            [0, np.sin(theta_rad), np.cos(theta_rad)]])
        return rot_mtx
    
    def get_rotmtx_y(self, theta_deg = 45):
        """ Get a rotation matrix over y axis
        """
        theta_rad = np.deg2rad(theta_deg)
        rot_mtx = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)], 
                            [0, 1, 0],
                            [-np.sin(theta_rad), 0, np.cos(theta_rad)]])
        return rot_mtx
        
    def get_rotmtx_z(self, theta_deg = 45):
        """ Get a rotation matrix over z axis
        """
        theta_rad = np.deg2rad(theta_deg)
        rot_mtx = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0], 
                            [np.sin(theta_rad), np.cos(theta_rad), 0],
                            [0, 0, 1]])
        return rot_mtx
    
    def plot_array(self,):
        """ plot the array coordinates as dots in space
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d') #fig.gca()#projection='3d'
        ax.scatter(self.coord[:, 0], self.coord[:, 1], self.coord[:, 2], 
                    marker='o', s=12, color='blue', alpha = 0.7)
        
        ax.set_title("Receiver array (num of receivers: {})".format(self.coord.shape[0]))
        ax.set_xlabel(r'$x$ [m]')
        #ax.set_xticks([-0.5, -0.5, L_x/2, baffle_size/2])
        ax.set_ylabel(r'$y$ [m]')
        #ax.set_yticks([-L_y/2, L_y/2])
        ax.set_zlabel(r'$z$ [m]')
        #ax.set_zticks([-sample_thickness, 1.0, 2, 3])
        ax.grid(False)
        ax.set_xlim((-2*np.amax(self.coord[:,0]), 2*np.amax(self.coord[:,0])))
        ax.set_ylim((-2*np.amax(self.coord[:,1]), 2*np.amax(self.coord[:,1])))
        ax.set_zlim((-0.05, 2*np.amax(self.coord[:,2])))
        #ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()

    def save(self, filename = 'qdt', path = ''):
        """ To save the decomposition object as pickle
        """
        ut_is.save(self, filename = filename, path = path)

    def load(self, filename = 'qdt', path = ''):
        """ To load the decomposition object as pickle

        You can instantiate an empty object of the class and load a saved one.
        It will overwrite the empty object.
        """
        ut_is.load(self, filename = filename, path = path)   