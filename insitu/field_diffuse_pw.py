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
# import insitu_cpp
from controlsair import plot_spk, update_progress
from material import PorousAbsorber

class PWDifField(object):
    '''
    A class to calculate the sound pressure and particle velocity
    using plane wave incidence in a diffuse field setting. This can be of use with array methods
    The inputs are the objects: air, controls, sources, receivers. Material properties will have to be 
    calculated along the way.
    '''
    def __init__(self,  air, controls, sources, receivers):
        self.air = air
        self.controls = controls
        self.material = []
        self.sources = sources
        self.receivers = receivers
        self.pres_s = []
        self.uz_s = []

    def p_fps(self, resistivity = 25000, thickness = 0.1, thick2=0, locally = True, randomize = True, seed = 0):
        '''
        Method to calculate the diffuse field incidence at every receiver.
        '''
        # We loop over each source and create the correct boundary conditions
        Vp = np.zeros((len(self.sources.coord), len(self.controls.freq)), dtype = np.csingle)
        # np.random.seed(seed)
        for js, s_coord in enumerate(self.sources.coord):
            r, theta, phi = cart2sph(s_coord[0], s_coord[1], s_coord[2])
            material = PorousAbsorber(self.air, self.controls)
            material.miki(resistivity=resistivity)
            if locally:
                if thick2 == 0:
                    material.layer_over_rigid(thickness = thickness, theta = 0)
                else:
                    material.layer_over_airgap(thick1 = thickness, thick2 = thick2, theta = 0)
                # Vp[js,:] = np.divide(material.Zs * np.cos(theta) - self.air.c0*self.air.rho0,
                #     material.Zs * np.cos(theta) + self.air.c0*self.air.rho0)
                Vp[js,:] = np.divide(material.Zs - self.air.c0*self.air.rho0,
                    material.Zs + self.air.c0*self.air.rho0)
            else:
                if thick2 == 0:
                    material.layer_over_rigid(thickness = thickness, theta = theta)
                else:
                    material.layer_over_airgap(thick1 = thickness, thick2 = thick2, theta = theta)
                Vp[js,:] = material.Vp
            self.material.append(material)
        ns = len(self.sources.coord)
        # if randomize:
        #     # amp = np.sqrt(np.random.randn(ns)**2 + np.random.randn(ns)**2)
        #     amp = np.sqrt(np.random.normal(0,1,ns)**2 + np.random.normal(0,1,ns)**2)
        #     # amp = amp/np.amax(amp)
        #     phase = np.random.rand(ns)
        #     q = amp * np.exp(1j*phase)#np.random.randn(ns) + 1j*np.random.randn(ns)
        # else:
        #     q = np.ones(ns)

        pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
        # bar = ChargingBar('Calculating sound pressure at each receiver', max=len(self.receivers.coord), suffix='%(percent)d%%')
        for jrec, r_coord in enumerate(self.receivers.coord):
            update_progress(jrec/len(self.receivers.coord))
            np.random.seed(seed)
            # r = np.linalg.norm(r_coord) # distance source-receiver
            for jf, k0 in enumerate(self.controls.k0):
                if randomize:
                    # amp = np.sqrt(np.random.randn(ns)**2 + np.random.randn(ns)**2)
                    amp = np.sqrt(np.random.normal(0,2,ns)**2 + np.random.normal(0,2,ns)**2)
                    # amp = amp/np.amax(amp)
                    phase = np.random.rand(ns)
                    q = amp * np.exp(1j*phase)#np.random.randn(ns) + 1j*np.random.randn(ns)
                else:
                    q = np.ones(ns)
                k_vec = k0 * self.sources.coord
                pres_rec[jrec, jf] = np.sum(q * (np.exp(1j * np.dot(k_vec, r_coord))+\
                Vp[:,jf] * np.exp(-1j * np.dot(k_vec, r_coord))))
        #     bar.next()
        # bar.finish()
        self.pres_s = [pres_rec]

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
        # vertexes plot
        vertices = np.array([[-vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, vsam_size/2, 0.0],
            [-vsam_size/2, vsam_size/2, 0.0]])
        # ax.scatter(vertices[:,0], vertices[:,1],
        #     vertices[:,2], color='blue')
        verts = [list(zip(vertices[:,0],
                vertices[:,1], vertices[:,2]))]
        # patch plot
        collection = Poly3DCollection(verts,
            linewidths=1, alpha=0.9, edgecolor = 'gray')
        collection.set_facecolor('grey')
        ax.add_collection3d(collection)
        # plot source
        for s_coord in self.sources.coord:
            ax.scatter(s_coord[0], s_coord[1], s_coord[2],
                color='red',  marker = "o", s=20)
        for r_coord in self.receivers.coord:
            ax.scatter(r_coord[0], r_coord[1], r_coord[2],
                color='blue',  marker = "o")
        ax.set_xlabel('X axis')
        plt.xticks([], [])
        ax.set_ylabel('Y axis')
        plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='none')
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        ax.set_zlim((0, 1))
        ax.set_zticks((0, 0.3))
        ax.view_init(elev=5, azim=-55)
        # ax.invert_zaxis()
        plt.show() # show plot

    def save(self, filename = 'my_pw', path = '/home/eric/dev/insitu/data/'):
        '''
        This method is used to save the simulation object
        '''
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'my_pw', path = '/home/eric/dev/insitu/data/'):
        '''
        This method is used to load a simulation object. You build a empty object
        of the class and load a saved one. It will overwrite the empty one.
        '''
        lpath_filename = path + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)