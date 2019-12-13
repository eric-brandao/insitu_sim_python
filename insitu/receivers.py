import numpy as np
import toml
from insitu.controlsair import load_cfg
# import insitu_cpp


class Receiver():
    '''
    A receiver class to initialize the following receiver properties:
    cood - 3D coordinates of a receiver (p, u or pu)
    There are several types of receivers to be implemented.
    - single_rec: is a single receiver (this is the class __init__)
    - double_rec: is a pair of receivers (tipically used in impedance measurements - separated by a z distance)
    - line_array: an line trough z containing receivers
    - planar_array: a regular grid of microphones
    - double_planar_array: a double regular grid of microphones separated by a z distance
    - spherical_array: a sphere of receivers
    - arc: an arc of receivers
    '''
    def __init__(self, coord = [0.0, 0.0, 0.01]):
        '''
        The class constructor initializes a single receiver with a given 3D coordinates
        The default is a height of 1 [cm]. User must be sure that the receiver lies out of
        the sample being emulated. This can go wrong if we allow the sample to have a thickness
        going on z>0
        '''
        self.coord = np.reshape(np.array(coord, dtype = np.float32), (1,3))

    def double_rec(self, z_dist = 0.01):
        '''
        This method initializes a double receiver separated by z_dist. It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        '''
        self.coord = np.append(self.coord, [self.coord[0,0], self.coord[0,1], self.coord[0,2]+z_dist])
        self.coord = np.reshape(self.coord, (2,3))

    def line_array(self, line_len = 1.0, n_rec = 10):
        '''
        This method initializes a line array of receivers. It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        Inputs:
            line_len - the length of the line. The first sensor will be at coordinates given by
            the class constructor. Receivers will span in z-direction
            n_rec - the number of receivers in the line array
        '''
        pass

    def planar_array(self, x_len = 1.0, n_x = 10, y_len = 1.0, n_y = 10):
        '''
        This method initializes a planar array of receivers (z/xy plane). It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        Inputs:
            x_len - the length of the x direction (array goes from -x_len/2 to +x_len/2).
            n_x - the number of receivers in the x direction
            y_len - the length of the y direction (array goes from -x_len/2 to +x_len/2).
            n_y - the number of receivers in the y direction
        '''
        pass

    def double_planar_array(self, x_len = 1.0, n_x = 10, y_len = 1.0, n_y = 10, z_dist = 0.01):
        '''
        This method initializes a double planar array of receivers (z/xy plane)
        separated by z_dist. It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        Inputs:
            x_len - the length of the x direction (array goes from -x_len/2 to +x_len/2).
            n_x - the number of receivers in the x direction
            y_len - the length of the y direction (array goes from -x_len/2 to +x_len/2).
            n_y - the number of receivers in the y direction
        '''
        pass

    def spherical_array(self, radius = 0.1, n_rec = 32, center_dist = 0.5):
        '''
        This method initializes a spherical array of receivers. The array coordinates are
        separated by center_dist from the origin. It will overwrite
        self.coord to be a matrix where each line gives a 3D coordinate for each receiver
        Inputs:
            radius - the radius of the sphere.
            n_rec - the number of receivers in the spherical array
            center_dist - center distance from the origin
        '''
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


