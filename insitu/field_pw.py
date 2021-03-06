import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from controlsair import load_cfg, cart2sph, sph2cart
from material import PorousAbsorber

# import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
import pickle
import time

# import impedance-py/C++ module and other stuff
#import insitu_cpp
from controlsair import plot_spk

class PWField(object):
    '''
    A class to calculate the sound pressure and particle velocity
    using plane wave incidence. This can be of use with array methods
    The inputs are the objects: air, controls, material, sources, receivers
    '''
    def __init__(self,  air = [], controls = [], material = [], receivers = [], theta = 0, phi = 0):
        self.air = air
        self.controls = controls
        self.material = material
        self.sources = []
        self.receivers = receivers
        self.pres_s = []
        self.uz_s = []
        self.theta = theta
        self.phi = phi

    def p_fps(self,):
        # Loop the receivers
        self.pres_s = []
        pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
        for jrec, r_coord in enumerate(self.receivers.coord):
            # r = np.linalg.norm(r_coord) # distance source-receiver
            for jf, k0 in enumerate(self.controls.k0):
                # kx = k0 * np.cos(self.phi) * np.sin(self.theta)
                # ky = k0 * np.sin(self.phi) * np.sin(self.theta)
                # kz = k0 * np.cos(self.theta)
                kx, ky, kz = sph2cart(k0, np.pi/2-self.theta, self.phi)
                k_veci = np.array([-kx, -ky, -kz])
                k_vecr = np.array([-kx, -ky, kz])
                # print('Incident wave: ({})'.format(k_veci/k0))
                # print('Reflected wave: ({})'.format(k_vecr/k0))
                pres_rec[jrec, jf] = np.exp(-1j * np.dot(k_veci, r_coord)) +\
                    self.material.Vp[jf] * np.exp(-1j * np.dot(k_vecr, r_coord))
        self.pres_s.append(pres_rec)

    def uz_fps(self,):
        # Loop the receivers
        self.uz_s = []
        uz_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
        for jrec, r_coord in enumerate(self.receivers.coord):
            # r = np.linalg.norm(r_coord) # distance source-receiver
            for jf, k0 in enumerate(self.controls.k0):
                # kx = k0 * np.cos(self.phi) * np.sin(self.theta)
                # ky = k0 * np.sin(self.phi) * np.sin(self.theta)
                # kz = k0 * np.cos(self.theta)
                kx, ky, kz = sph2cart(k0, np.pi/2-self.theta, self.phi)
                k_veci = np.array([kx, ky, -kz])
                k_vecr = np.array([kx, ky, kz])
                uz_rec[jrec, jf] = (-kz/k0) * (np.exp(-1j * np.dot(k_veci, r_coord)) -\
                    self.material.Vp[jf] * np.exp(-1j * np.dot(k_vecr, r_coord)))
                # uz_rec[jrec, jf] = (kz/k0) *(np.exp(1j * k_vec[2] * r_coord[2]) -\
                #     self.material.Vp[jf] * np.exp(-1j * k_vec[2] * r_coord[2]))*\
                #     (np.exp(1j * k_vec[0] * r_coord[0]))*\
                #     (np.exp(1j * k_vec[1] * r_coord[1]))
        self.uz_s.append(uz_rec)

    def p_mult(self,):
        # Loop the receivers
        self.pres_s = []
        pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
        for jrec, r_coord in enumerate(self.receivers.coord):
            for jel, el in enumerate(self.theta):
                material_m = PorousAbsorber(self.air, self.controls)
                material_m.miki(resistivity=self.material.resistivity)
                material_m.layer_over_rigid(thickness = self.material.thickness, theta = el)
                for jf, k0 in enumerate(self.controls.k0):
                    kx, ky, kz = sph2cart(k0, np.pi/2-el, self.phi[jel])
                    k_veci = np.array([kx, ky, kz])
                    k_vecr = np.array([kx, ky, -kz])
                    pres_rec[jrec, jf] += np.exp(1j * np.dot(k_veci, r_coord)) +\
                        material_m.Vp[jf] * np.exp(1j * np.dot(k_vecr, r_coord))
        self.pres_s.append(pres_rec)

    def uz_mult(self,):
        # Loop the receivers
        self.uz_s = []
        uz_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
        for jrec, r_coord in enumerate(self.receivers.coord):
            for jel, el in enumerate(self.theta):
                material_m = PorousAbsorber(self.air, self.controls)
                material_m.miki(resistivity=self.material.resistivity)
                material_m.layer_over_rigid(thickness = self.material.thickness, theta = el)
                for jf, k0 in enumerate(self.controls.k0):
                    kx, ky, kz = sph2cart(k0, np.pi/2-el, self.phi[jel])
                    k_veci = np.array([kx, ky, kz])
                    k_vecr = np.array([kx, ky, -kz])
                    uz_rec[jrec, jf] += (kz/k0) * (np.exp(1j * np.dot(k_veci, r_coord)) -\
                    material_m.Vp[jf] * np.exp(1j * np.dot(k_vecr, r_coord)))
        self.uz_s.append(uz_rec)

    def pev_fps(self, kx_e = [0], ky_e = [0], Ae = [0]):
        '''
        Method calculates the pressure field due to a propagating incident, a progating reflected
        and a set of evanescent plane waves specified by the vectors kx_e, ky_e and Ae
        '''
        # Loop the receivers
        self.pres_s = []
        pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
        for jrec, r_coord in enumerate(self.receivers.coord):
            # r = np.linalg.norm(r_coord) # distance source-receiver
            for jf, k0 in enumerate(self.controls.k0):
                # Calculate propagating wave-numbers
                kx, ky, kz = sph2cart(k0, np.pi/2-self.theta, self.phi)
                k_veci_p = np.array([kx, ky, kz])
                k_vecr_p = np.array([kx, ky, -kz])
                # Evanescent wave-numbers
                kz_e = (kx_e**2 + ky_e**2 - k0**2)**0.5
                p_ev = np.sum(Ae * (np.exp(-kz_e * r_coord[2])) * (np.exp(1j * (kx_e * r_coord[0] + ky_e * r_coord[1]))))
                pres_rec[jrec, jf] = np.exp(1j * np.dot(k_veci_p, r_coord)) +\
                    self.material.Vp[jf] * np.exp(1j * np.dot(k_vecr_p, r_coord)) + p_ev
        self.pres_s.append(pres_rec)

    def add_noise(self, snr = 30, uncorr = False):
        """ Add gaussian noise to the simulated data.

        The function is used to add noise to the pressure data.
        it reads the clean signal and estimate its power. Then, it estimates the power
        of the noise that would lead to the target SNR. Then it draws random numbers
        from a Normal distribution with standard deviation =  noise power

        Parameters
        ----------
        snr : float
            The signal to noise ratio you want to emulate
        uncorr : bool
            If added noise to each receiver is uncorrelated or not.
            If uncorr is True the the noise power is different for each receiver
            and frequency. If uncorr is False the noise power is calculated from
            the average signal magnitude of all receivers (for each frequency).
            The default value is False
        """
        signal = self.pres_s[0]
        if uncorr:
            signalPower_lin = (np.abs(signal)/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
        else:
            # signalPower_lin = (np.abs(np.mean(signal, axis=0))/np.sqrt(2))**2
            signalPower_lin = ((np.mean(np.abs(signal), axis=0))/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
        np.random.seed(0)
        noise = np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape)
        self.pres_s[0] = signal + noise

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
        # plot the plane wave
        # for jel, el in enumerate(self.theta):
        #     nx = np.cos(self.phi[jel]) * np.sin(self.theta[jel])
        #     ny = np.sin(self.phi[jel]) * np.sin(self.theta[jel])
        #     nz = np.cos(self.theta[jel])
        #     normal = np.array([nx, ny, nz])
        #     center = 0.7 * normal
        #     basex = np.array([nx, ny, nz+0.3])
        #     basey = np.array([nx, ny+0.3, nz])
        #     y = np.cross(normal, basex)
        #     u = y/np.linalg.norm(y)
        #     z = np.cross(normal, basey)
        #     v = z/np.linalg.norm(z)
        #     area = 0.05
        #     radius = np.sqrt(area/2)
        #     v1 = center + radius * u
        #     v2 = center + radius * v
        #     v3 = center - radius * u
        #     v4 = center - radius * v
        #     vertices = np.array([v1, v2, v3, v4])
        #     verts = [list(zip(vertices[:,0],
        #         vertices[:,1], vertices[:,2]))]
        #     # patch plot
        #     collection = Poly3DCollection(verts,
        #         linewidths=1, alpha=0.3, edgecolor = 'red', zorder=1, linewidth=3)
        #     collection.set_facecolor('silver')
        #     ax.add_collection3d(collection)
        #     ax.quiver(center[0], center[1], center[2],
        #         -normal[0], -normal[1], -normal[2], length=0.1, normalize=True, color='red')
        # for s_coord in self.sources:
        #     ax.scatter(s_coord[0], s_coord[1], s_coord[2],
                # color='red',  marker = "o", s=50)
        # ax.scatter(center[0], center[1], center[2],
        #         color='red',  marker = "o", s=500)
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

    def plot_pres(self):
        '''
        Method to plot the spectrum of the sound pressure
        '''
        plot_spk(self.controls.freq, self.pres_s, ref = 20e-6)

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

