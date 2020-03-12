import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from insitu.controlsair import load_cfg
# import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
import pickle
import time
from controlsair import sph2cart, cart2sph


# import impedance-py/C++ module and other stuff
import insitu_cpp
from controlsair import plot_spk

class FreeField(object):
    '''
    A class to calculate the sound pressure and particle velocity
    using free field. This can be of use with array methods
    The inputs are the objects: air, controls, material, sources, receivers
    '''
    def __init__(self, air, controls, sources, receivers):
        self.air = air
        self.controls = controls
        self.sources = sources
        self.material = []
        self.receivers = receivers
        self.pres_s = []
        self.uz_s = []

    def monopole_ff(self,):
        '''
        Method used to calculate the free field response due to a monopole with a given source coordinate
        '''
        # Loop the receivers
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            # hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = np.linalg.norm(r_coord - s_coord) # distance source-receiver
                pres_rec[jrec, :] = (np.exp(-1j * self.controls.k0 * r)) / r
            self.pres_s.append(pres_rec)

    def mirrorsource(self,):
        '''
        Method used to calculate the half-space response due to a monopole with a given source coordinate.
        The original monopole is mirroed against the plane z=0 and its strength is the same as the original
        monopole
        '''
        # Loop the receivers
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            # hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r1 = np.linalg.norm(r_coord - s_coord) # distance source-receiver
                r2 = np.linalg.norm(r_coord + s_coord) # distance source-receiver
                pres_rec[jrec, :] = (np.exp(-1j * self.controls.k0 * r1)) / r1 +\
                    (np.exp(-1j * self.controls.k0 * r2)) / r2
            self.pres_s.append(pres_rec)

    def planewave_ff(self, theta = 0, phi = 0):
        '''
        Method used to calculate the free field response due to a plane wave
        Inputs:
            theta - the elevation angle
            phi - the azimuth angle
        '''
        # Loop the receivers
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            # hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                # r = np.linalg.norm(r_coord) # distance source-receiver
                for jf, k0 in enumerate(self.controls.k0):
                    kx = k0 * np.cos(phi) * np.sin(theta)
                    ky = k0 * np.sin(phi) * np.sin(theta)
                    kz = k0 * np.cos(theta)
                    k_vec = np.array([kx, ky, kz])
                    pres_rec[jrec, jf] = np.exp(1j * np.dot(k_vec, r_coord))
            self.pres_s.append(pres_rec)

    def planewave_diffuse(self, randomize = True, seed = 0):
        '''
        Method used to calculate the free field response due to multiple plane wave incidence.
        The directions are supposed to come from a list of unity vectors contained in sources.coord
        Inputs:
            theta - the elevation angle
            phi - the azimuth angle
        '''
        # r, theta, phi = sph2cart(self.sources.coord[:,0], self.sources.coord[:,1], self.sources.coord[:,2])
        pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
        bar = ChargingBar('Calculating sound pressure at each receiver', max=len(self.receivers.coord), suffix='%(percent)d%%')
        for jrec, r_coord in enumerate(self.receivers.coord):
            # r = np.linalg.norm(r_coord) # distance source-receiver
            for jf, k0 in enumerate(self.controls.k0):
                k_vec = k0 * self.sources.coord
                if randomize:
                    np.random.seed(seed)
                    q = np.random.randn(1) + 1j*np.random.randn(1)
                else:
                    q = 1
                pres_rec[jrec, jf] = np.sum(q * np.exp(1j * np.dot(k_vec, r_coord)))
            bar.next()
        bar.finish()
        self.pres_s = [pres_rec]

    def add_noise(self, snr = np.Inf):
        '''
        Method used to artificially add gaussian noise to the measued data
        '''
        for js, s_coord in enumerate(self.sources.coord):
            # hs = s_coord[2] # source height
            pres_rec = self.pres_s[js]
            # N = (pres_rec).shape
            # print(N)
            for jrec, r_coord in enumerate(self.receivers.coord):
                for jf, k0 in enumerate(self.controls.k0):
                    # Here, the signal power is 2 (1 for real and imaginary component)
                    signal = pres_rec[jrec, jf]
                    signalPower_lin = np.abs(signal)**2
                    signalPower_dB = 10 * np.log10(signalPower_lin)
                    noisePower_dB = signalPower_dB - snr
                    noisePower_lin = 10 ** (noisePower_dB/10)
                    noise = np.sqrt(noisePower_lin/2)*(np.random.randn(1) + 1j*np.random.randn(1))
                    self.pres_s[js][jrec][jf] = signal + noise

    def plot_pres(self):
        '''
        Method to plot the spectrum of the sound pressure
        '''
        plot_spk(self.controls.freq, self.pres_s, ref = 20e-6)

    def plot_scene(self, vsam_size = 2):
        '''
        a simple plot of the scene using matplotlib - not redered
        '''
        fig = plt.figure()
        fig.canvas.set_window_title("Measurement scene")
        ax = fig.gca(projection='3d')
        vertices = np.array([[-vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, vsam_size/2, 0.0],
            [-vsam_size/2, vsam_size/2, 0.0]])
        verts = [list(zip(vertices[:,0],
                vertices[:,1], vertices[:,2]))]
        # patch plot
        collection = Poly3DCollection(verts,
            linewidths=2, alpha=0.3, edgecolor = 'black', zorder=2)
        collection.set_facecolor('silver')
        ax.add_collection3d(collection)
        # plot source
        for s_coord in self.sources.coord:
            ax.scatter(s_coord[0], s_coord[1], s_coord[2],
                color='red',  marker = "o", s=20)
        # plot receiver
        for r_coord in self.receivers.coord:
            ax.scatter(r_coord[0], r_coord[1], r_coord[2],
                color='blue',  marker = "o")
        ax.set_xlabel('X axis')
        # plt.xticks([], [])
        ax.set_ylabel('Y axis')
        # plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='both')
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        # ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        # ax.set_zlim((0, 1.2))

        # ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        # ax.set_zticks((0, 1.2))
        ax.view_init(elev=30, azim=-50)
        # ax.invert_zaxis()
        plt.show() # show plot
