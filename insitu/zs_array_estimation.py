import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
import toml
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
#from tqdm._tqdm_notebook import tqdm
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import cvxpy as cp
from scipy import linalg # for svd
from scipy.sparse.linalg import lsqr, lsmr
from lcurve_functions import csvd, l_cuve
import pickle
from receivers import Receiver
from material import PorousAbsorber
from controlsair import cart2sph, sph2cart, update_progress, compare_alpha, compare_zs
from rayinidir import RayInitialDirections
from parray_estimation import octave_freq, octave_avg
from decompositionclass import Decomposition, filter_evan

class ZsArray(Decomposition):
    '''
    This class inherits the Decomposition class and adds the methods for impedance estimation.
    '''
    def __init__(self, p_mtx = None, controls = None, material = None, receivers = None):
        '''
        Init - we first retrive general data, then we process some receiver data
        '''
        Decomposition.__init__(self, p_mtx, controls, material, receivers)
        super().__init__(p_mtx, controls, material, receivers)

    def zs(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [0], avgZs = True):
        '''
        Method to calculate the absorption coefficient straight from 3D array data.
        Inputs:
            Lx - The length of calculation aperture
            Ly - The width of calculation aperture
            n_x - The number of calculation points in x dir
            n_y - The number of calculation points in y dir
            theta [list] - list of target angles to calculate the absorption from reconstructed impedance
            avgZs (bool) - whether to average over <Zs> (if True - default) or over <p>/<uz> (if False)
        '''
        # Set the grid used to reconstruct the surface impedance
        grid = Receiver()
        grid.planar_array(x_len=Lx, y_len=Ly, zr=0.0, n_x = n_x, n_y = n_x)
        if n_x > 1 or n_y > 1:
            self.grid = grid.coord
        else:
            self.grid = np.array([0,0,0])
        # Alocate some memory prior to loop
        self.Zs = np.zeros(len(self.controls.k0), dtype=complex)
        self.p_s = np.zeros((len(self.grid), len(self.controls.k0)), dtype=complex)
        self.uz_s = np.zeros((len(self.grid), len(self.controls.k0)), dtype=complex)
        bar = ChargingBar('Calculating zs (backpropagation whole sphere) for angle: ',\
            max=len(self.controls.k0), suffix='%(percent)d%%')
        for jf, k0 in enumerate(self.controls.k0):
            # Wave number vector
            k_vec = k0 * self.dir
            # Form H matrix
            h_mtx = np.exp(1j*self.grid @ k_vec.T)
            # complex amplitudes of all waves
            x = self.pk[:,jf]
            # reconstruct pressure and particle velocity at surface
            p_surf_mtx = h_mtx @ x
            uz_surf_mtx = ((np.divide(k_vec[:,2], k0)) * h_mtx) @ x
            self.p_s[:,jf] =  p_surf_mtx
            self.uz_s[:,jf] =  uz_surf_mtx
            # Average impedance at grid
            if avgZs:
                Zs_pt = np.divide(p_surf_mtx, uz_surf_mtx)
                self.Zs[jf] = np.mean(Zs_pt)
            else:
                self.Zs[jf] = np.mean(p_surf_mtx) / (np.mean(uz_surf_mtx))
            bar.next()
        bar.finish()
        # Calculate the sound absorption coefficient for targeted angles
        self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        for jtheta, dtheta in enumerate(theta):
            self.alpha[jtheta,:] = 1 - (np.abs(np.divide((self.Zs  * np.cos(dtheta) - 1),\
                (self.Zs * np.cos(dtheta) + 1))))**2
        return self.alpha

    def zs_ev(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [0], avgZs = True):
        '''
        Method to calculate the absorption coefficient straight from 3D array data, including evanescent waves.
        Inputs:
            Lx - The length of calculation aperture
            Ly - The width of calculation aperture
            n_x - The number of calculation points in x dir
            n_y - The number of calculation points in y dir
            theta [list] - list of target angles to calculate the absorption from reconstructed impedance
            avgZs (bool) - whether to average over <Zs> (if True - default) or over <p>/<uz> (if False)
        '''
        # Set the grid used to reconstruct the surface impedance
        grid = Receiver()
        grid.planar_array(x_len=Lx, y_len=Ly, zr=0.0, n_x = n_x, n_y = n_x)
        if n_x > 1 or n_y > 1:
            self.grid = grid.coord
        else:
            self.grid = np.array([0,0,0])
        # Alocate some memory prior to loop
        self.Zs = np.zeros(len(self.controls.k0), dtype=complex)
        self.p_s = np.zeros((len(self.grid), len(self.controls.k0)), dtype=complex)
        self.uz_s = np.zeros((len(self.grid), len(self.controls.k0)), dtype=complex)
        bar = ChargingBar('Calculating zs (backpropagation whole sphere) for angle: ',\
            max=len(self.controls.k0), suffix='%(percent)d%%')
        for jf, k0 in enumerate(self.controls.k0):
            # Wave number vector (propagating)
            k_p = k0 * self.dir
            # Wave number vector (evanescent)
            kx_e, ky_e, self.n_evan = filter_evan(k0, self.kx_e, self.ky_e, plot=False)
            # print('Number of evanescent is {}'.format(self.n_evan))
            kz_e = np.sqrt(k0**2 - kx_e**2 - ky_e**2+0j)
            k_ev = np.array([kx_e, ky_e, kz_e]).T
            # Total wave number
            k_vec = np.vstack((k_p, k_ev))
            # Form H matrix
            # h_p = np.exp(1j*self.grid @ k_vec.T)
            # h_e = np.exp(1j*self.grid @ k_ev.T)
            h_mtx = np.exp(1j*self.grid @ k_vec.T) #np.hstack((h_p, h_e))
            # complex amplitudes of all waves
            # x = np.concatenate((self.pk[:,jf], self.pk_ev[jf]))
            # reconstruct pressure and particle velocity at surface
            p_surf_mtx = h_mtx @ np.concatenate((self.pk[:,jf], self.pk_ev[jf]))
            uz_surf_mtx = ((np.divide(k_vec[:,2], k0)) * h_mtx) @ np.concatenate((self.pk[:,jf], self.pk_ev[jf]))
            self.p_s[:,jf] =  p_surf_mtx
            self.uz_s[:,jf] =  uz_surf_mtx
            # Average impedance at grid
            if avgZs:
                Zs_pt = np.divide(p_surf_mtx, uz_surf_mtx)
                self.Zs[jf] = np.mean(Zs_pt)
            else:
                self.Zs[jf] = np.mean(p_surf_mtx) / (np.mean(uz_surf_mtx))
            bar.next()
        bar.finish()
        # Calculate the sound absorption coefficient for targeted angles
        self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        for jtheta, dtheta in enumerate(theta):
            self.alpha[jtheta,:] = 1 - (np.abs(np.divide((self.Zs  * np.cos(dtheta) - 1),\
                (self.Zs * np.cos(dtheta) + 1))))**2
        return self.alpha