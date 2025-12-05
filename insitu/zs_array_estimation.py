import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
import toml
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
#from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
#from tqdm._tqdm_notebook import tqdm
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#import cvxpy as cp
from scipy import linalg # for svd
from scipy.sparse.linalg import lsqr, lsmr
# from lcurve_functions import csvd, l_cuve, tikhonov, ridge_solver, direct_solver
import pickle
from receivers import Receiver
from material import PorousAbsorber
from controlsair import cart2sph, sph2cart, update_progress, compare_alpha, compare_zs
from rayinidir import RayInitialDirections
#from parray_estimation import octave_freq, octave_avg, get_hemispheres
from decompositionclass import Decomposition, filter_evan

class ZsArray(Decomposition):
    """ Decomposition and impedance estimation using propagating plane waves.

    The class inherit the attributes and methods of the decomposition class
    Decomposition, which has several methods to perform sound field decomposition
    into a set of incident and reflected plane waves. These sets of plane waves are composed of
    propagating waves. The grid for the propagating plane waves is
    created from the uniform angular subdivision of the surface of a sphere of
    radius k [rad/m].

    ZsArray then adds methods for the estimation of the surface impedance and absorption
    coefficient.

    Attributes
    ----------
    p_mtx : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of all the receivers
        Each column is a set of sound pressure at all receivers for a frequency.
    controls : object (AlgControls)
        Controls of the decomposition (frequency spam)
    material : object (PorousAbsorber)
        Contains the material properties (surface impedance). This can be used as reference
        when simulations is what you want to do.
    receivers : object (Receiver)
        The receivers in the field - this contains the information of the coordinates of
        the microphones in your array
    decomp_type : str
        Decomposition description
    cond_num : (1 x N_freq) numpy 1darray
        condition number of sensing matrix
    pk : list
        List of estimated amplitudes of all plane waves.
        Each element in the list is relative to a frequency of the measurement spectrum.
    fpts : object (Receiver)
        The field points in the field where pressure and velocity are reconstructed
    p_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed sound pressure
        at all the field points
    ux_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed particle vel (z)
        at all the field points
    uy_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed particle vel (z)
        at all the field points
    uz_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed particle vel (z)
        at all the field points

    Methods
    ----------
    wavenum_dir(n_waves = 642, plot = False, halfsphere = False)
        Create the propagating wave number directions

    pk_tikhonov(method = 'direct', f_ref = 1.0, f_inc = 1.0, factor = 1, z0 = 1.5, plot_l = False)
        Wave number spectrum estimation using Tikhonov inversion

    pk_constrained(snr=30, headroom = 0)
        Wave number spectrum estimation using constrained optimization

    pk_cs(snr=30, headroom = 0)
        Wave number spectrum estimation using constrained optimization

    pk_oct_interpolate(nband = 3):
        Interpolate wavenumber spectrum over an fractional octave bands

    reconstruct_pu(receivers)
        Reconstruct the sound pressure and particle velocity at a receiver object

    zs(Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [0], avgZs = True)
        Reconstruct the surface impedance and estimate the absorption

    pk_interpolate(npts=100):
        Interpolate the wave number spectrum on a finer regular grid

    plot_pk_sphere(freq = 1000, db = False, dinrange = 40, save = False, name='name', travel=True)
        plot the magnitude of P(k) as a scatter plot of evanescent and propagating waves

    plot_colormap(self, freq = 1000, total_pres = True)
        Plots a color map of the pressure field.

    plot_pk_map(freq = 1000, db = False, dinrange = 40, phase = False,
        save = False, name='', path = '', fname='', color_code = 'viridis')
        Plot wave number spectrum  - propagating only (vs. phi and theta)

    save(filename = 'my_bemflush', path = '/home/eric/dev/insitu/data/bem_simulations/')
        To save the simulation object as pickle

    load(filename = 'my_qterm', path = '/home/eric/dev/insitu/data/bem_simulations/')
        Load a simulation object.
    """

    def __init__(self, p_mtx = None, controls = None, material = None, receivers = None):
        """

        Parameters
        ----------
        p_mtx : (N_rec x N_freq) numpy array
            A matrix containing the complex amplitudes of all the receivers
            Each column is a set of sound pressure at all receivers for a frequency.
        controls : object (AlgControls)
            Controls of the decomposition (frequency spam)
        material : object (PorousAbsorber)
            Contains the material properties (surface impedance).
        receivers : object (Receiver)
            The receivers in the field

        The objects are stored as attributes in the class (easier to retrieve).
        """
        Decomposition.__init__(self, p_mtx, controls, material, receivers)
        super().__init__(p_mtx, controls, material, receivers)

    def zs(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [0], avgZs = True):
        """ Reconstruct the surface impedance and estimate the absorption

        Reconstruct pressure and particle velocity at a grid of points
        on ther surface of the absorber (z = 0.0). The absorption coefficient
        is also calculated.

        Parameters
        ----------
        Lx : float
            The length of calculation aperture
        Ly : float
            The width of calculation aperture
        n_x : int
            The number of calculation points in x
        n_y : int
            The number of calculation points in y dir
        theta : list
            Target angles to calculate the absorption from reconstructed impedance
        avgZs : bool
            Whether to average over <Zs> (default - True) or over <p>/<uz> (if False)

        Returns
        -------
        alpha : (N_theta x Nfreq) numpy ndarray
            The absorption coefficients for each target incident angle.
        """
        # Set the grid used to reconstruct the surface impedance
        grid = Receiver()
        grid.planar_array(x_len=Lx, y_len=Ly, zr=0.0, n_x = n_x, n_y = n_x)
        if n_x > 1 or n_y > 1:
            self.grid = grid.coord
        else:
            self.grid = np.array([0,0,0])
        # # Alocate some memory prior to loop
        # self.reconstruct_pu(receivers=grid)
        # Zs_pt = np.divide(self.p_recon, self.uz_recon)
        # self.Zs = np.mean(Zs_pt, axis=0)#np.zeros(len(self.controls.k0), dtype=complex)
        # self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        # for jtheta, dtheta in enumerate(theta):
        #     self.alpha[jtheta,:] = 1 - (np.abs(np.divide((self.Zs  * np.cos(dtheta) - 1),\
        #         (self.Zs * np.cos(dtheta) + 1))))**2
        # return self.alpha
        # Alocate some memory prior to loop
        self.Zs = np.zeros(len(self.controls.k0), dtype=complex)
        self.p_s = np.zeros((len(self.grid), len(self.controls.k0)), dtype=complex)
        self.uz_s = np.zeros((len(self.grid), len(self.controls.k0)), dtype=complex)
        # bar = ChargingBar('Calculating zs (backpropagation whole sphere) for angle: ',\
        #     max=len(self.controls.k0), suffix='%(percent)d%%')
        bar = tqdm(total = len(self.controls.k0), desc = 'Calculating zs (backpropagation whole sphere)')
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
        """ Calculate the absorption coefficient from wave-number spectra.

        There is no target angle in this method. Simply, the total reflected energy is
        divided by the total incident energy
        """
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




 # def zs_ev(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [0], avgZs = True):
    #     '''
    #     Method to calculate the absorption coefficient straight from 3D array data, including evanescent waves.
    #     Inputs:
    #         Lx - The length of calculation aperture
    #         Ly - The width of calculation aperture
    #         n_x - The number of calculation points in x dir
    #         n_y - The number of calculation points in y dir
    #         theta [list] - list of target angles to calculate the absorption from reconstructed impedance
    #         avgZs (bool) - whether to average over <Zs> (if True - default) or over <p>/<uz> (if False)
    #     '''
    #     # Set the grid used to reconstruct the surface impedance
    #     grid = Receiver()
    #     grid.planar_array(x_len=Lx, y_len=Ly, zr=0.0, n_x = n_x, n_y = n_x)
    #     if n_x > 1 or n_y > 1:
    #         self.grid = grid.coord
    #     else:
    #         self.grid = np.array([0,0,0])
    #     # Alocate some memory prior to loop
    #     self.Zs = np.zeros(len(self.controls.k0), dtype=complex)
    #     self.p_s = np.zeros((len(self.grid), len(self.controls.k0)), dtype=complex)
    #     self.uz_s = np.zeros((len(self.grid), len(self.controls.k0)), dtype=complex)
    #     # bar = ChargingBar('Calculating zs (backpropagation whole sphere) for angle: ',\
    #     #     max=len(self.controls.k0), suffix='%(percent)d%%')
    #     bar = tqdm(total = len(self.controls.k0), desc = 'Calculating zs (backpropagation whole sphere)')

    #     for jf, k0 in enumerate(self.controls.k0):
    #         # Wave number vector (propagating)
    #         k_p = -k0 * self.dir
    #         # Wave number vector (evanescent)
    #         kx_e, ky_e, n_e = filter_evan(k0, self.kx_e, self.ky_e, plot=False)
    #         # print('Number of evanescent is {}'.format(self.n_evan))
    #         kz_e = np.sqrt(k0**2 - kx_e**2 - ky_e**2+0j)
    #         k_ev = np.array([kx_e, ky_e, kz_e]).T
    #         # Total wave number
    #         k_vec = np.vstack((k_p, k_ev))
    #         # Form H matrix
    #         # h_p = np.exp(1j*self.grid @ k_vec.T)
    #         # h_e = np.exp(1j*self.grid @ k_ev.T)
    #         h_mtx = np.exp(1j*self.grid @ k_vec.T) #np.hstack((h_p, h_e))
    #         # complex amplitudes of all waves
    #         # x = np.concatenate((self.pk[:,jf], self.pk_ev[jf]))
    #         # reconstruct pressure and particle velocity at surface
    #         p_surf_mtx = h_mtx @ np.concatenate((self.pk[:,jf], self.pk_ev[jf]))
    #         uz_surf_mtx = -((np.divide(k_vec[:,2], k0)) * h_mtx) @ np.concatenate((self.pk[:,jf], self.pk_ev[jf]))
    #         self.p_s[:,jf] =  p_surf_mtx
    #         self.uz_s[:,jf] =  uz_surf_mtx
    #         # Average impedance at grid
    #         if avgZs:
    #             Zs_pt = np.divide(p_surf_mtx, uz_surf_mtx)
    #             self.Zs[jf] = np.mean(Zs_pt)
    #         else:
    #             self.Zs[jf] = np.mean(p_surf_mtx) / (np.mean(uz_surf_mtx))
    #         bar.update(1)
    #     #     bar.next()
    #     # bar.finish()
    #     # Calculate the sound absorption coefficient for targeted angles
    #     self.alpha = np.zeros((len(theta), len(self.controls.k0)))
    #     for jtheta, dtheta in enumerate(theta):
    #         self.alpha[jtheta,:] = 1 - (np.abs(np.divide((self.Zs  * np.cos(dtheta) - 1),\
    #             (self.Zs * np.cos(dtheta) + 1))))**2
    #     return self.alpha