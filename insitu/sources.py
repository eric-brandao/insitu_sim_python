import numpy as np

class Source():
    '''
    A sound source class to initialize the following sound source properties:
    Inputs:
        cood - 3D coordinates of a single sound source
        q - volume velocity [m^3/s]
    '''
    def __init__(self, coord = [0.0, 0.0, 1.0], q = 1.0):
        self.coord = np.reshape(np.array(coord, dtype = np.float32), (1,3))
        self.q = np.array(q, dtype = np.float32)

    def set_arc_sources(self, radius = 1.0, ns = 10, angle_span = (-90, 90), random = False):
        '''
        This method is used to generate an array of sound sources in an 2D arc
        Inputs:
            radius - radii of the source arc (how far from the sample they are)
            ns - the number of sound sources in the arc
            angle_span - tuple with the range for which the sources span
            random (bool) - if True, then the complex amplitudes are randomized
        '''
        pass

    def set_ssph_sources(self, radius = 1.0, ns = 100, random = False):
        '''
        This method is used to generate an array of sound sources over a surface of a sphere
        Inputs:
            radius - radii of the source arc (how far from the sample they are)
            ns - the number of sound sources in the sphere
            random (bool) - if True, then the complex amplitudes are randomized
        '''
        pass

    def set_vsph_sources(self, radii_span = (1.0, 10.0), ns = 100, random = False):
        '''
        This method is used to generate an array of sound sources over the volume of a sphere
        Inputs:
            radii_span - tuple with the range for which the sources span
            ns - the number of sound sources in the sphere
            random (bool) - if True, then the complex amplitudes are randomized
        '''
        pass

# def setup_sources(config_file, rays):
#     '''
#     Set up the sound sources
#     '''
#     sources = [] # An array of empty souce objects
#     config = load_cfg(config_file) # toml file
#     for s in config['sources']:
#         coord = np.array(s['position'], dtype=np.float32)
#         orientation = np.array(s['orientation'], dtype=np.float32)
#         power_dB = np.array(s['power_dB'], dtype=np.float32)
#         eq_dB = np.array(s['eq_dB'], dtype=np.float32)
#         power_lin = 10.0e-12 * 10**((power_dB + eq_dB) / 10.0)
#         delay = s['delay'] / 1000
#         ################### cpp source class #################
#         sources.append(insitu_cpp.Sourcecpp(coord, orientation,
#             power_dB, eq_dB, power_lin, delay)) # Append the source object
#         # sources.append([insitu_cpp.Sourcecpp(coord, orientation,
#         #     power_dB, eq_dB, power_lin, delay),
#         #     rays]) # Append the source object
#         ################### py source class ################
#         # sources.append(Source(coord, orientation,
#         #     power_dB, eq_dB, power_lin, delay)) # Append the source object
#     return sources
# # class Souces from python side