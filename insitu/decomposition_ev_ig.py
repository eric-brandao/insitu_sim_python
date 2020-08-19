import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
# from matplotlib import cm
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
from scipy.interpolate import griddata
from sklearn.linear_model import Ridge
import time
from tqdm import tqdm
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
#from tqdm._tqdm_notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import cvxpy as cp
from scipy import linalg # for svd
from scipy import signal
from scipy.sparse.linalg import lsqr, lsmr
from lcurve_functions import csvd, l_cuve, tikhonov
import pickle
from receivers import Receiver
from material import PorousAbsorber
from controlsair import cart2sph, sph2cart, cart2sph, update_progress, compare_alpha, compare_zs
from rayinidir import RayInitialDirections
from parray_estimation import octave_freq, octave_avg, get_hemispheres, get_inc_ref_dirs
from decompositionclass import filter_evan

SMALL_SIZE = 11
BIGGER_SIZE = 13
#plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.family': 'serif'})
plt.rc('legend', fontsize=SMALL_SIZE)
#plt.rc('title', fontsize=SMALL_SIZE)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('figure', titlesize=BIGGER_SIZE)


class DecompositionEv2(object):
    '''
    This class inherits the DecompositionEv class and adds the methods for decomposition based on
    a separate grid approach for propagating and evanescent waves.
    '''
    def __init__(self, p_mtx = None, controls = None, material = None, receivers = None):
        '''
        Init - we first retrive general data, then we process some receiver data
        '''
        # self.pres_s = sim_field.pres_s[source_num] #FixMe
        # self.air = sim_field.air
        self.controls = controls
        self.material = material
        # self.sources = sim_field.sources
        self.receivers = receivers
        self.pres_s = p_mtx
        self.flag_oct_interp = False

    def prop_dir(self, n_waves = 642, plot = False):
        '''
        This method is used to create wave number directions uniformily distributed over the surface of
        a hemisphere. The directions of propagation that later will become propagating wave-number vectors.
        The directions of propagation are calculated with the triangulation of an icosahedron used previously
        in the generation of omnidirectional rays (originally implemented in a ray tracing algorithm).
        Inputs:
            n_waves - The number of directions (wave-directions) to generate (integer)
            plot - whether you plot or not the wave points in space (bool)
        '''
        directions = RayInitialDirections()
        directions, n_sph = directions.isotropic_rays(Nrays = int(n_waves))
        id_dir = np.where(directions[:,2]>=0)
        self.pdir = directions[id_dir[0],:]
        self.n_prop = len(self.pdir[:,0])
        if plot:
            fig = plt.figure()
            fig.canvas.set_window_title('Dir test')
            ax = fig.gca(projection='3d')
            p=ax.scatter(self.pdir[:,0], self.pdir[:,1], self.pdir[:,2])

    def pk_tikhonov_ev_ig(self, f_ref = 1.0, f_inc = 1.0, factor = 1, z0 = 1.5, plot_l = False):
        '''
        Method to estimate wave number spectrum based on the Tikhonov matrix inversion technique.
        This version includes the evanescent waves
        Inputs:
        '''
        self.decomp_type = 'Tikhonov (transparent array) w/ evanescent waves - uses irregular grid'
        self.f_ref = f_ref
        self.f_inc = f_inc
        # self.factor = factor
        # self.z0 = z0
        self.zp = -factor * np.amax([self.receivers.ax, self.receivers.ay])
        print('freq: {}'.format(self.controls.freq))
        print('zp: {}, rel to lam {}'.format(self.zp, self.zp/(2*np.pi/self.controls.k0)))
        self.zm = factor * (z0 + np.amax([self.receivers.ax, self.receivers.ay]))
        # self.zm = z0 + factor * np.amax([self.receivers.ax, self.receivers.ay]) # Try
        # Generate kx and ky for evanescent wave grid
        self.kx = np.linspace(start = -np.pi/self.receivers.ax,
            stop = np.pi/self.receivers.ax, num = 3*int(self.n_prop**0.5))
        self.ky = np.linspace(start = -np.pi/self.receivers.ay,
            stop = np.pi/self.receivers.ay, num = 3*int(self.n_prop**0.5))
        kx_grid, ky_grid = np.meshgrid(self.kx,self.ky)
        kx_e = kx_grid.flatten()
        ky_e = ky_grid.flatten()
        # Initialize variables
        self.cond_num = np.zeros(len(self.controls.k0))
        self.pk = []
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0),
            desc = 'Calculating Tikhonov inversion (with evanescent waves and irregular grid)...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # print(self.controls.freq[jf])
            # 1 - for smooth transition from continous to discrete k domain
            # kappa = np.sqrt(self.delta_kx*self.delta_ky/(2*np.pi*k0**2))
            # 2 - form kz
            kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            # 3 Wavenumbers
            k_vec_inc = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T,
                np.array([kx_eig, ky_eig, -kz_eig]).T))
            k_vec_ref = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
                np.array([kx_eig, ky_eig, kz_eig]).T))
            # k_vec_inc = np.vstack((k0 * self.pdir, np.array([kx_eig, ky_eig, -kz_eig]).T))
            # k_vec_ref = np.vstack((k0 * self.pdir, np.array([kx_eig, ky_eig, kz_eig]).T))
            # fz_ref = f_ref * np.sqrt(k0/np.abs(kz_f))
            # 4 - receivers 
            recs_inc = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
                self.receivers.coord[:,2]-self.zm]).T
            recs_ref = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
                self.receivers.coord[:,2]-self.zp]).T
            # 5 - psi and sensing matrix
            psi_inc = np.exp(-1j * recs_inc @ k_vec_inc.T)
            psi_ref = np.exp(-1j * recs_ref @ k_vec_ref.T)
            h_mtx = np.hstack((psi_inc, psi_ref))
            self.cond_num[jf] = np.linalg.cond(h_mtx)
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            # 6 - finding the optimal regularization parameter.
            u, sig, v = csvd(h_mtx)
            lambd_value = l_cuve(u, sig, pm, plotit=plot_l)
            Hm = np.matrix(h_mtx)
            x = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
            self.pk.append(x)#[0:self.n_waves]
            bar.update(1)
        bar.close()

    def reconstruct_pu(self, receivers):
        '''
        reconstruct sound pressure and particle velocity at a receiver object
        '''
        self.p_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        self.uz_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        # Generate kx and ky for evanescent wave grid
        # kx = np.linspace(start = -np.pi/self.receivers.ax,
        #     stop = np.pi/self.receivers.ax, num = int(self.n_prop**0.5))
        # ky = np.linspace(start = -np.pi/self.receivers.ay,
        #     stop = np.pi/self.receivers.ay, num = int(self.n_prop**0.5))
        kx_grid, ky_grid = np.meshgrid(self.kx, self.ky)
        kx_e = kx_grid.flatten()
        ky_e = ky_grid.flatten()

        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing sound field...')
        for jf, k0 in enumerate(self.controls.k0):
            # 1 - for smooth transition from continous to discrete k domain
            # kappa = np.sqrt(self.delta_kx*self.delta_ky/(2*np.pi*k0**2))
            # 3 Wavenumbers
            kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
            kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
            k_vec_inc = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T,
                np.array([kx_eig, ky_eig, -kz_eig]).T))
            k_vec_ref = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
                np.array([kx_eig, ky_eig, kz_eig]).T))
            # 4 - receivers 
            recs_inc = np.array([receivers.coord[:,0], receivers.coord[:,1],
                receivers.coord[:,2]-self.zm]).T
            recs_ref = np.array([receivers.coord[:,0], receivers.coord[:,1],
                receivers.coord[:,2]-self.zp]).T
            # 5 - psi and sensing matrix
            psi_inc = np.exp(-1j * recs_inc @ k_vec_inc.T)
            psi_ref = np.exp(-1j * recs_ref @ k_vec_ref.T)
            h_mtx = np.hstack((psi_inc, psi_ref))
            # p = h_mtx @ self.pk[jf].T
            self.p_recon[:,jf] = np.squeeze(np.asarray(h_mtx @ self.pk[jf].T))
            self.uz_recon[:,jf] = np.squeeze(np.asarray(-((np.divide(np.concatenate((k_vec_inc[:,2], k_vec_ref[:,2])), k0)) *\
                h_mtx) @ self.pk[jf].T))
            bar.update(1)
        bar.close()

    def plot_pkmap_prop(self, freq = 1000, db = False, dinrange = 20,
        save = False, name='name', color_code = 'viridis'):
        '''
        Method to plot the magnitude of the spatial fourier transform as a map of
        propagating waves in terms of theta and phi. The map is first interpolated into a regular grid
        It is a normalized version of the magnitude, either between 0 and 1 or between -dinrange and 0.
        inputs:
            freq - Which frequency you want to see. If the calculated spectrum does not contain it
                we plot the closest frequency before the asked one.
            dB (bool) - Whether to plot in linear scale (default) or decibel scale.
            dinrange - You can specify a dinamic range for the decibel scale. It will not affect the
            linear scale.
            save (bool) - Whether to save or not the figure. PDF file with simple standard name
            color_code - can be anything that matplotlib supports. Some recomendations given below:
                'viridis' (default) - Perceptually Uniform Sequential
                'Greys' - White (cold) to black (hot)
                'seismic' - Blue (cold) to red (hot) with a white in the middle
        '''
        id_f = np.where(self.controls.freq <= freq)
        # id_f = np.where(self.freq_oct <= freq)
        id_f = id_f[0][-1]
        # Quatinties from simulation
        k0 = self.controls.k0[id_f]
        pk = np.squeeze(np.asarray(self.pk[id_f]))
        # Incident and reflected with evanescent
        pk_i = pk[:self.n_prop] # incident
        pk_r = pk[int(len(pk)/2):int(len(pk)/2)+self.n_prop] # reflected
        pk_p = np.hstack((pk_i, pk_r))
        # We must select only the propagating wave components
        # kx_grid, ky_grid = np.meshgrid(self.kx, self.ky)
        # kx_e = kx_grid.flatten()
        # ky_e = ky_grid.flatten()
        # ke_norm = (kx_e**2 + ky_e**2)**0.5
        # kx_p = kx_e[ke_norm <= k0]
        # ky_p = ky_e[ke_norm <= k0]
        # kz_p = np.sqrt(k0**2 - (kx_p**2+ky_p**2))
        # The directions
        directions = np.vstack((-k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
            -k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T))
        # Pk - propgating
        # pk_ip = pk_i[ke_norm <= k0]
        # pk_rp = pk_r[ke_norm <= k0]
        # pk_p = np.hstack((pk_ip, pk_rp))
        # Transform uninterpolated data to spherical coords
        r, theta, phi = cart2sph(directions[:,0], directions[:,1], directions[:,2])
        thetaphi = np.transpose(np.array([phi, theta]))
        # Create the new grid to iterpolate
        new_phi = np.linspace(-np.pi, np.pi, 100)
        new_theta = np.linspace(-np.pi/2, np.pi/2,  100)#(0, np.pi, nn)
        grid_phi, grid_theta = np.meshgrid(new_phi, new_theta)
        # Interpolate
        pk_grid = griddata(thetaphi, np.abs(pk_p), (grid_phi, grid_theta),
            method='cubic', fill_value=np.finfo(float).eps, rescale=False)
        # Calculate colors
        if db:
            color_par = 20*np.log10(np.abs(pk_grid)/np.amax(np.abs(pk_grid)))
            color_range = np.linspace(-dinrange, 0, dinrange+1)
        else:
            color_par = np.abs(pk_grid)/np.amax(np.abs(pk_grid))
            color_range = np.linspace(0, 1, 21)
        # Figure
        fig = plt.figure()
        fig.canvas.set_window_title('Interpolated map of |P(k)| for freq {} Hz'.format(self.controls.freq[id_f]))
        plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - PE decomp. '+ name)
        p = plt.contourf(np.rad2deg(grid_phi), np.rad2deg(grid_theta)+90, color_par,
            color_range, extend='both', cmap = color_code)
        fig.colorbar(p)
        plt.xlabel(r'$\phi$ (azimuth) [deg]')
        plt.ylabel(r'$\theta$ (elevation) [deg]')
        plt.tight_layout()
        if save:
            filename = 'data/colormaps/cmat_' + str(int(freq)) + 'Hz_' + name
            plt.savefig(fname = filename, format='pdf')

class ZsArrayEvIg(DecompositionEv2):
    '''
    This class inherits the DecompositionEv2 class and adds the methods for impedance estimation.
    '''
    def __init__(self, p_mtx = None, controls = None, material = None, receivers = None):
        '''
        Init - we first retrive general data, then we process some receiver data
        '''
        DecompositionEv2.__init__(self, p_mtx, controls, material, receivers)
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
        self.reconstruct_pu(receivers=grid)
        Zs_pt = np.divide(self.p_recon, self.uz_recon)
        self.Zs = np.mean(Zs_pt, axis=0)#np.zeros(len(self.controls.k0), dtype=complex)
        self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        for jtheta, dtheta in enumerate(theta):
            self.alpha[jtheta,:] = 1 - (np.abs(np.divide((self.Zs  * np.cos(dtheta) - 1),\
                (self.Zs * np.cos(dtheta) + 1))))**2
        return self.alpha


