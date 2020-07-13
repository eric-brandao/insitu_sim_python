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
from parray_estimation import octave_freq, octave_avg, get_hemispheres
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
        # bar = ChargingBar('Calculating zs (backpropagation whole sphere) for angle: ',\
        #     max=len(self.controls.k0), suffix='%(percent)d%%')
        bar = tqdm(total = len(self.controls.k0), desc = 'Calculating zs (backpropagation whole sphere)')
        for jf, k0 in enumerate(self.controls.k0):
            # Wave number vector
            k_vec = -k0 * self.dir
            # Form H matrix
            h_mtx = np.exp(1j*self.grid @ k_vec.T)
            # complex amplitudes of all waves
            x = self.pk[:,jf]
            # reconstruct pressure and particle velocity at surface
            p_surf_mtx = h_mtx @ x
            uz_surf_mtx = -((np.divide(k_vec[:,2], k0)) * h_mtx) @ x
            self.p_s[:,jf] =  p_surf_mtx
            self.uz_s[:,jf] =  uz_surf_mtx
            # Average impedance at grid
            if avgZs:
                Zs_pt = np.divide(p_surf_mtx, uz_surf_mtx)
                self.Zs[jf] = np.mean(Zs_pt)
            else:
                self.Zs[jf] = np.mean(p_surf_mtx) / (np.mean(uz_surf_mtx))
            bar.update(1)
        #     bar.next()
        # bar.finish()
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
        # bar = ChargingBar('Calculating zs (backpropagation whole sphere) for angle: ',\
        #     max=len(self.controls.k0), suffix='%(percent)d%%')
        bar = tqdm(total = len(self.controls.k0), desc = 'Calculating zs (backpropagation whole sphere)')

        for jf, k0 in enumerate(self.controls.k0):
            # Wave number vector (propagating)
            k_p = -k0 * self.dir
            # Wave number vector (evanescent)
            kx_e, ky_e, n_e = filter_evan(k0, self.kx_e, self.ky_e, plot=False)
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
            uz_surf_mtx = -((np.divide(k_vec[:,2], k0)) * h_mtx) @ np.concatenate((self.pk[:,jf], self.pk_ev[jf]))
            self.p_s[:,jf] =  p_surf_mtx
            self.uz_s[:,jf] =  uz_surf_mtx
            # Average impedance at grid
            if avgZs:
                Zs_pt = np.divide(p_surf_mtx, uz_surf_mtx)
                self.Zs[jf] = np.mean(Zs_pt)
            else:
                self.Zs[jf] = np.mean(p_surf_mtx) / (np.mean(uz_surf_mtx))
            bar.update(1)
        #     bar.next()
        # bar.finish()
        # Calculate the sound absorption coefficient for targeted angles
        self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        for jtheta, dtheta in enumerate(theta):
            self.alpha[jtheta,:] = 1 - (np.abs(np.divide((self.Zs  * np.cos(dtheta) - 1),\
                (self.Zs * np.cos(dtheta) + 1))))**2
        return self.alpha

    def alpha_from_pk(self, ):
        '''
        Method to calculate the absorption coefficient straight from decomposition data.
        There is no target angle in this method. Simply, the total reflected energy is
        divided by the total incident energy
        '''
        theta = self.grid_theta.flatten()
        phi = self.grid_phi.flatten()
        theta_inc_id = np.where(np.logical_and(theta >= 0, theta < np.pi/2))
        theta_ref_id = np.where(np.logical_and(theta > -np.pi/2, theta < 0))
        # Get the directions of interpolated data.
        # xx, yy, zz = sph2cart(1, theta, phi)
        # dirs = np.transpose(np.array([xx, yy, zz]))
        # Get the incident and reflected hemispheres
        # theta_inc_id, theta_ref_id = get_hemispheres(theta)
        # We will loop through the list of desired_thetas
        # Initialize
        if self.flag_oct_interp:
            self.alpha_pk = np.zeros(len(self.freq_oct)) # Nfreq
            # # Get the list of indexes for angles you want
            # thetainc_des_list, thetaref_des_list = desired_theta_list(theta_inc_id, theta_ref_id,
            #     theta, desired_theta = dtheta, target_range = target_range)
            # Loop over frequency
            bar = tqdm(total = len(self.controls.k0), desc = 'Calculating absorption from P(k)')
            # bar = ChargingBar('Calculating absorption (avg...) for angle: ' +\
            #     str(np.rad2deg(dtheta)) + ' deg.',\
            #     max=len(self.freq_oct), suffix='%(percent)d%%')
            for jf, fc in enumerate(self.freq_oct):
                pk = self.grid_pk[jf].flatten()
                pk_inc = np.abs(pk[theta_inc_id[0]])**2 # incident energy
                pk_ref = np.abs(pk[theta_ref_id[0]])**2 # reflected energy
                self.alpha_pk[jf] = 1 - pk_ref/pk_inc
                bar.update(1)
            bar.close()
        else:
            self.alpha_pk = np.zeros(len(self.controls.k0)) # Nfreq
            # Get the list of indexes for angles you want
            # thetainc_des_list, thetaref_des_list = desired_theta_list(theta_inc_id, theta_ref_id,
            #     theta, desired_theta = dtheta, target_range = target_range)
            # Loop over frequency
            bar = tqdm(total = len(self.controls.k0), desc = 'Calculating absorption from P(k)')
            for jf, k0 in enumerate(self.controls.k0):
                pk = self.grid_pk[jf].flatten()
                pk_inc = np.mean(np.abs(pk[theta_inc_id[0]])**2) # incident energy
                pk_ref = np.mean(np.abs(pk[theta_ref_id[0]])**2) # reflected energy
                # pk_inc = np.mean(np.abs(pk[10201:])**2) # incident energy
                # pk_ref = np.mean(np.abs(pk[0:10201])**2) # reflected energy
                self.alpha_pk[jf] = 1 - pk_ref/pk_inc
                bar.update(1)
            bar.close()
        return self.alpha_pk