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
        # Loop the receivers
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            # hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = np.linalg.norm(r_coord - s_coord) # distance source-receiver
                pres_rec[jrec, :] = (np.exp(-1j * self.controls.k0 * r)) / r
            self.pres_s.append(pres_rec)

    def planewave_ff(self, theta = 0, phi = 0):
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
                color='red',  marker = "o", s=200)
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
        ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.view_init(elev=30, azim=-50)
        # ax.invert_zaxis()
        plt.show() # show plot
