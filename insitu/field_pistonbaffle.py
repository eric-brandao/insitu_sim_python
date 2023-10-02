import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from controlsair import load_cfg, cart2sph, sph2cart, plot_spk

# import scipy.integrate as integrate
import scipy as spy
from scipy import integrate
import time
import sys
from tqdm import tqdm
#from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
import pickle


class PistonOnBaffle(object):
    '''
    A class to calculate the radiated sound pressure for a piston on a baffle
    '''
    def __init__(self,  air = [], controls = [], receivers = []):
        self.air = air
        self.controls = controls
        self.receivers = receivers
        # self.pres_s = []
        # self.uz_s = []

    def p_rigid_squared(self, w = 1, Lx = 0.2, Ly = 0.2):
        '''
        This method calculates the sound pressure for a squared rigid piston on a
        baffle.
        Inputs: velocity (w), length (Lx) and width (Ly)
        '''
        # Loop the receivers
        # self.pres_s = []
        self.Lx = Lx
        self.Ly = Ly
        self.pres_s = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
        bar = tqdm(total = len(self.controls.k0)*len(self.receivers.coord),
            desc = 'Calculating Field...')
        for jrec, r_coord in enumerate(self.receivers.coord):
            # r = np.linalg.norm(r_coord) # distance source-receiver
            for jf, k0 in enumerate(self.controls.k0):
                # integrand - real part
                fxy_r = lambda yl, xl: np.real(np.exp(-1j * k0 * np.sqrt((r_coord[0]-xl)**2 +\
                (r_coord[1]-yl)**2 + (r_coord[2])**2))/\
                np.sqrt((r_coord[0]-xl)**2 + (r_coord[1]-yl)**2 + (r_coord[2])**2))
                # integrand - imag part
                fxy_i = lambda yl, xl: np.imag(np.exp(-1j * k0 * np.sqrt((r_coord[0]-xl)**2 +\
                (r_coord[1]-yl)**2 + (r_coord[2])**2))/\
                np.sqrt((r_coord[0]-xl)**2 + (r_coord[1]-yl)**2 + (r_coord[2])**2))
                # integrals
                Ir = integrate.dblquad(fxy_r, -Lx/2, Lx/2, lambda x: -Ly/2, lambda x: Lx/2)
                Ii = integrate.dblquad(fxy_i, -Lx/2, Lx/2, lambda x: -Ly/2, lambda x: Lx/2)
                self.pres_s[jrec, jf] = ((1j*self.air.rho0 *self.air.c0*k0)/(2*np.pi))*\
                    (Ir[0] + 1j*Ii[0])
                bar.update(1)
        bar.close()
        # self.pres_s.append(pres_rec)

    def uz_rigid_squared(self, ):
        '''
        This method calculates the z-dir particle velocity for a squared rigid piston on a
        baffle.
        Inputs: velocity (w), length (Lx) and width (Ly)
        '''
        # Loop the receivers
        self.uz_s = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
        bar = tqdm(total = len(self.controls.k0)*len(self.receivers.coord),
            desc = 'Calculating Particle velocity Field...')
        for jrec, r_coord in enumerate(self.receivers.coord):
            # r = np.linalg.norm(r_coord) # distance source-receiver
            for jf, k0 in enumerate(self.controls.k0):
                # integrand - real part
                fxy_r = lambda yl, xl: np.real(np.exp(1j * k0 * np.sqrt((r_coord[0]-xl)**2 +\
                (r_coord[1]-yl)**2 + (r_coord[2])**2))/\
                np.sqrt((r_coord[0]-xl)**2 + (r_coord[1]-yl)**2 + (r_coord[2])**2) *\
                (1/(1j*k0*np.sqrt((r_coord[0]-xl)**2+(r_coord[1]-yl)**2+(r_coord[2])**2))-1) *\
                r_coord[2]/np.sqrt((r_coord[0]-xl)**2+(r_coord[1]-yl)**2+(r_coord[2])**2))
                # integrand - imag part
                fxy_i = lambda yl, xl: np.imag(np.exp(1j * k0 * np.sqrt((r_coord[0]-xl)**2 +\
                (r_coord[1]-yl)**2 + (r_coord[2])**2))/\
                np.sqrt((r_coord[0]-xl)**2 + (r_coord[1]-yl)**2 + (r_coord[2])**2) *\
                (1/(1j*k0*np.sqrt((r_coord[0]-xl)**2+(r_coord[1]-yl)**2+(r_coord[2])**2))-1) *\
                r_coord[2]/np.sqrt((r_coord[0]-xl)**2+(r_coord[1]-yl)**2+(r_coord[2])**2))
                # integrals
                Ir = integrate.dblquad(fxy_r, -self.Lx/2, self.Lx/2, lambda x: -self.Ly/2, lambda x: self.Lx/2)
                Ii = integrate.dblquad(fxy_i, -self.Lx/2, self.Lx/2, lambda x: -self.Ly/2, lambda x: self.Lx/2)
                self.uz_s[jrec, jf] = ((1j*self.air.rho0 *self.air.c0*k0)/(2*np.pi))*\
                    (Ir[0] + 1j*Ii[0])
                bar.update(1)
        bar.close()

    def p_trav(self, w = 1, m=1, Lx = 0.2, Ly = 0.2):
        '''
        This method calculates the sound pressure for a squared rigid piston on a
        baffle.
        Inputs: velocity (w), length (Lx) and width (Ly)
        '''
        # Loop the receivers
        # self.pres_s = []
        self.Lx = Lx
        self.Ly = Ly
        self.pres_s = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
        bar = tqdm(total = len(self.controls.k0)*len(self.receivers.coord),
            desc = 'Calculating Field...')
        for jrec, r_coord in enumerate(self.receivers.coord):
            # r = np.linalg.norm(r_coord) # distance source-receiver
            for jf, k0 in enumerate(self.controls.k0):
                # integrand - real part
                fxy_r = lambda yl, xl: np.real(np.cos(m*np.pi*xl/Lx)*\
                    np.exp(1j * k0 * np.sqrt((r_coord[0]-xl)**2 +\
                    (r_coord[1]-yl)**2 + (r_coord[2])**2))/\
                    np.sqrt((r_coord[0]-xl)**2 + (r_coord[1]-yl)**2 + (r_coord[2])**2))
                # integrand - imag part
                fxy_i = lambda yl, xl: np.imag(np.cos(m*np.pi*xl/Lx)*\
                    np.exp(1j * k0 * np.sqrt((r_coord[0]-xl)**2 +\
                    (r_coord[1]-yl)**2 + (r_coord[2])**2))/\
                    np.sqrt((r_coord[0]-xl)**2 + (r_coord[1]-yl)**2 + (r_coord[2])**2))
                # integrals
                Ir = integrate.dblquad(fxy_r, -Lx/2, Lx/2, lambda x: -Ly/2, lambda x: Lx/2)
                Ii = integrate.dblquad(fxy_i, -Lx/2, Lx/2, lambda x: -Ly/2, lambda x: Lx/2)
                self.pres_s[jrec, jf] = ((1j*self.air.rho0 *self.air.c0*k0)/(2*np.pi))*\
                    (Ir[0] + 1j*Ii[0])
                bar.update(1)
        bar.close()

    def add_noise(self, snr = 30, uncorr = True):
        '''
        Method used to artificially add gaussian noise to the measued data
        '''
        signal = self.pres_s
        # try:
        #     signal_u = self.uz_s
        # except:
        #     signal_u = np.zeros(1)
        # Estimate signal power
        signalPower_lin = (np.abs(np.mean(signal, axis=0))/np.sqrt(2))**2
        signalPower_dB = 10 * np.log10(signalPower_lin)
        # Estimate noise power
        noisePower_dB = signalPower_dB - snr
        noisePower_lin = 10 ** (noisePower_dB/10)
        # if signal_u.any() != 0:
        #     signalPower_lin_u = (np.abs(np.mean(signal_u, axis=0))/np.sqrt(2))**2
        #     signalPower_dB_u = 10 * np.log10(signalPower_lin_u)
        #     noisePower_dB_u = signalPower_dB_u - snr
        #     noisePower_lin_u = 10 ** (noisePower_dB_u/10)
        # Add noise
        np.random.seed(0)
        noise = np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape) +\
            1j*np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape)
        self.pres_s = signal + noise
        # if signal_u.any() != 0:
        #     print('Adding noise to particle velocity')
        #     noise_u = np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape) +\
        #         1j*np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape)
        #     self.uz_s[0] = signal_u + noise_u

    def plot_scene(self, vsam_size = 1):
        '''
        a simple plot of the scene using matplotlib - not redered
        '''
        fig = plt.figure()
        ax = plt.axes(projection ="3d")
        vertices = np.array([[-self.Lx/2, -self.Ly/2, 0.0],
            [self.Lx/2, -self.Ly/2, 0.0],
            [self.Lx/2, self.Ly/2, 0.0],
            [-self.Lx/2, self.Ly/2, 0.0]])
        verts = [list(zip(vertices[:,0],
                vertices[:,1], vertices[:,2]))]
        # patch plot
        collection = Poly3DCollection(verts,
            linewidths=2, alpha=0.5, edgecolor = 'black', zorder=2)
        collection.set_facecolor('gold')
        ax.add_collection3d(collection)
        
        # Baffle
        vertices = np.array([[-vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, vsam_size/2, 0.0],
            [-vsam_size/2, vsam_size/2, 0.0]])
        verts = [list(zip(vertices[:,0],
                vertices[:,1], vertices[:,2]))]
        # patch plot
        collection = Poly3DCollection(verts,
            linewidths=2, alpha=0.5, edgecolor = 'silver', zorder=2)
        collection.set_facecolor('silver')
        ax.add_collection3d(collection)
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
        ax.set_zlim((0, np.amax([vsam_size, np.amax(self.receivers.coord[:,2])])))
        # ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.view_init(elev=30, azim=-50)
        # ax.invert_zaxis()
        plt.show() # show plot

    def save(self, filename = 'piston', path = ''):
        '''
        This method is used to save the simulation object
        '''
        filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'piston', path = ''):
        '''
        This method is used to load a simulation object. You build a empty object
        of the class and load a saved one. It will overwrite the empty one.
        '''
        lpath_filename = path + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)