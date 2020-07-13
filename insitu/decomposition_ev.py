import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
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

class DecompositionEv(object):
    '''
    Decomposition class for array processing. Decomposes the measurement into a set of
    plane propagating waves and plane evanescent waves. 
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

    def create_kx_ky(self, n_kx = 20, n_ky = 20, plot=False, freq = 1000):
        '''
        This method is used to create a regular grid of kx and ky.
        This will be used to create all propagating and evanescent waves at the decompostition time.
        '''
        #### With linspace and n_waves
        self.kx = np.linspace(start = -np.pi/self.receivers.ax,
            stop = np.pi/self.receivers.ax, num = n_kx)
        self.ky = np.linspace(start = -np.pi/self.receivers.ay,
            stop = np.pi/self.receivers.ay, num = n_ky)
        self.delta_kx = self.kx[1] - self.kx[0]
        self.delta_ky = self.ky[1] - self.ky[0]

        self.kx_grid, self.ky_grid = np.meshgrid(self.kx,self.ky)
        self.kx_f = self.kx_grid.flatten()
        self.ky_f = self.ky_grid.flatten()
        if plot:
            k0 = 2*np.pi*freq / self.controls.c0
            fig = plt.figure()
            fig.canvas.set_window_title('Non filtered evanescent waves')
            plt.plot(self.kx_f, self.ky_f, 'o')
            plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
                k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
            plt.xlabel('kx')
            plt.ylabel('ky')
            plt.show()

    def pk_tikhonov_ev(self, method = 'Rigde', f_ref = 1.0, f_inc = 1.0, factor = 1, z0 = 1.5, plot_l = False):
        '''
        Method to estimate wave number spectrum based on the Tikhonov matrix inversion technique.
        This version includes the evanescent waves
        Inputs:
            method: string defining the method to be used on finding the correct P(k).
                It can be:
                    (1) - 'scipy': using scipy.linalg.lsqr
                    (2) - 'direct': via x= (Hm^H) * ((Hm * Hm^H + lambd_value * I)^-1) * pm
                    (3) - else: via cvxpy
        '''
        self.decomp_type = 'Tikhonov (transparent array) w/ evanescent waves'
        self.f_ref = f_ref
        self.f_inc = f_inc
        # self.factor = factor
        # self.z0 = z0
        self.zp = -factor * np.amax([self.receivers.ax, self.receivers.ay])
        self.zm = factor * (z0 + np.amax([self.receivers.ax, self.receivers.ay]))

        bar = tqdm(total = len(self.controls.k0),
            desc = 'Calculating Tikhonov inversion (with evanescent waves)...')
        self.cond_num = np.zeros(len(self.controls.k0))
        self.pk = np.zeros((len(self.kx_f), len(self.controls.k0)), dtype=complex)
        self.kx_ef = [] # Filtered version
        self.ky_ef = [] # Filtered version
        self.pk_ev = []
        for jf, k0 in enumerate(self.controls.k0):
            # 1 - for smooth transition from continous to discrete k domain
            kappa = np.sqrt(self.delta_kx*self.delta_ky/(2*np.pi*k0**2))
            # 2 - form kz
            kz_f = form_kz(k0, self.kx_f, self.ky_f, plot=True)
            k_vec_ref = np.array([self.kx_f, self.ky_f, kz_f])
            # 3 - Reflected or radiating part
            fz_ref = f_ref * np.sqrt(k0/np.abs(kz_f))
            recs = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
                self.receivers.coord[:,2]-self.zp]).T
            psi_ref = fz_ref * kappa * np.exp(-1j * recs @ k_vec_ref)
            # 4 - Incident part
            if f_inc != 0:
                k_vec_inc = np.array([self.kx_f, self.ky_f, -kz_f])
                fz_inc = f_inc * np.sqrt(k0/np.abs(kz_f))
                recs = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
                    self.receivers.coord[:,2]-self.zm]).T
                psi_inc = fz_inc * kappa * np.exp(-1j * recs @ k_vec_inc)
            # 5 - Form sensing matrix
            if f_inc == 0:
                h_mtx = psi_ref
            else:
                h_mtx = np.hstack((psi_inc, psi_ref))
            self.cond_num[jf] = np.linalg.cond(h_mtx)
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            # 6 - finding the optimal regularization parameter.
            u, sig, v = csvd(h_mtx)
            lambd_value = l_cuve(u, sig, pm, plotit=plot_l)
            # ## Choosing the method to find the P(k)
            if method == 'scipy':
                from scipy.sparse.linalg import lsqr, lsmr
                x = lsqr(h_mtx, self.pres_s[:,jf], damp=lambd_value)[0]
            elif method == 'direct':
                Hm = np.matrix(h_mtx)
                x = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
            elif method == 'Ridge':
                # Form a real H2 matrix and p2 measurement
                H2 = np.vstack((np.hstack((h_mtx.real, -h_mtx.imag)),
                    np.hstack((h_mtx.imag, h_mtx.real))))
                p2 = np.vstack((pm.real,pm.imag)).flatten()
                regressor = Ridge(alpha=lambd_value, fit_intercept = False, solver = 'svd')
                x2 = regressor.fit(H2, p2).coef_
                x = x2[:h_mtx.shape[1]]+1j*x2[h_mtx.shape[1]:]
            elif method == 'tikhonov':
                u, sig, v = csvd(h_mtx)
                x = tikhonov(u, sig, v, pm, lambd_value)
            # #### Performing the Tikhonov inversion with cvxpy #########################
            else:
                H = h_mtx.astype(complex)
                x_cvx = cp.Variable(h_mtx.shape[1], complex = True)
                lambd = cp.Parameter(nonneg=True)
                lambd.value = lambd_value[0]
                # Create the problem and solve
                problem = cp.Problem(cp.Minimize(objective_fn(H, pm, x_cvx, lambd)))
                # problem.solve()
                problem.solve(solver=cp.SCS, verbose=False) # Fast but gives some warnings
                # problem.solve(solver=cp.ECOS, abstol=1e-3) # slow
                # problem.solve(solver=cp.ECOS_BB) # slow
                # problem.solve(solver=cp.NAG) # not installed
                # problem.solve(solver=cp.CPLEX) # not installed
                # problem.solve(solver=cp.CBC)  # not installed
                # problem.solve(solver=cp.CVXOPT) # not installed
                # problem.solve(solver=cp.MOSEK) # not installed
                # problem.solve(solver=cp.OSQP) # did not work
                # self.pk[:,jf] = x.value[0:self.n_waves]
                # self.pk_ev.append(x.value[self.n_waves:])
                x = x_cvx.value
            self.pk[:,jf] = x#[0:self.n_waves]
            # if include_evan:
            #     self.pk_ev.append(x[self.n_waves:])
            bar.update(1)
        bar.close()

    def reconstruct_pu(self, receivers):
        '''
        reconstruct sound pressure and particle velocity at a receiver object
        '''
        self.p_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        self.uz_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing sound field...')
        for jf, k0 in enumerate(self.controls.k0):
            # 1 - for smooth transition from continous to discrete k domain
            kappa = np.sqrt(self.delta_kx*self.delta_ky/(2*np.pi*k0**2))
            # 2 - form kz
            kz_f = form_kz(k0, self.kx_f, self.ky_f)
            k_vec_ref = np.array([self.kx_f, self.ky_f, kz_f])
            # 3 - Reflected or radiating part
            fz_ref = self.f_ref * np.sqrt(k0/np.abs(kz_f))
            recs = np.array([receivers.coord[:,0], receivers.coord[:,1],
                receivers.coord[:,2]-self.zp]).T
            psi_ref = fz_ref * kappa * np.exp(-1j * recs @ k_vec_ref)
            # 4 - Incident part
            if self.f_inc != 0:
                k_vec_inc = np.array([self.kx_f, self.ky_f, -kz_f])
                fz_inc = self.f_inc * np.sqrt(k0/np.abs(kz_f))
                recs = np.array([receivers.coord[:,0], receivers.coord[:,1],
                    receivers.coord[:,2]-self.zm]).T
                psi_inc = fz_inc * kappa * np.exp(-1j * recs @ k_vec_inc)
            # 5 - Form sensing matrix
            if self.f_inc == 0:
                h_mtx = psi_ref
            else:
                h_mtx = np.hstack((psi_inc, psi_ref))
            self.p_recon[:,jf] = h_mtx @ self.pk[:,jf]
            self.uz_recon[:,jf] = -((np.divide(kz_f, k0)) * h_mtx) @ self.pk[:,jf]
            bar.update(1)
        bar.close()

    def plot_pk_evmap(self, freq = 1000, db = False, dinrange = 12, save = False, name='', path = '', fname='', contourplot = True, plot_kxky = False):
        '''
        Method to plot the magnitude of the spatial fourier transform of the evanescent components
        The map of interpolated to a kx and ky wave numbers.
        It is a normalized version of the magnitude, either between 0 and 1 or between -dinrange and 0.
        inputs:
            freq (float) - Which frequency you want to see. If the calculated spectrum does not contain it
                we plot the closest frequency before the asked one.
            dB (bool) - Whether to plot in linear scale (default) or decibel scale.
            dinrange (float) - You can specify a dinamic range for the decibel scale. It will not affect the
            linear scale.
            save (bool) - Whether to save or not the figure (png file)
            path (str) - path to save fig
            fname (str) - name file of the figure
            plot_kxky (bool) - whether to plot or not the kx and ky points that are part of the evanescent map.
        '''
        import matplotlib.tri as tri
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        k0 = self.controls.k0[id_f]
        if db:
            color_par = 20*np.log10(np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f])))
            id_outofrange = np.where(color_par < -dinrange)
            color_par[id_outofrange] = -dinrange
        else:
            color_par = np.abs(self.pk[:,id_f])#/np.amax(np.abs(pk_ev_grid))
        ############### Countourf ##########################
        # Create the Triangulation; no triangles so Delaunay triangulation created.
        x = self.kx_f
        y = self.ky_f
        triang = tri.Triangulation(x, y)
        # Mask off unwanted triangles.
        # triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
        #     y[triang.triangles].mean(axis=1)) < k0)
        fig = plt.figure()
        fig.canvas.set_window_title('Filtered evanescent waves')
        plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
            k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
        if contourplot:
            p = plt.tricontourf(triang, color_par,
                levels=dinrange)
        else:
            p=plt.scatter(self.kx_f, self.ky_f, c = color_par)
        fig.colorbar(p)
        # if plot_kxky:
        #     plt.scatter(self.kx_f, self.ky_f, c = 'grey', alpha = 0.4)
        plt.xlabel(r'$k_x$ rad/m')
        plt.ylabel(r'$k_y$ rad/m')
        plt.title("|P(k)| (evanescent) at {0:.1f} Hz (k = {1:.2f} rad/m) {2}".format(self.controls.freq[id_f],k0, name))
        plt.tight_layout()
        if save:
            filename = path + fname + '_' + str(int(freq)) + 'Hz'
            plt.savefig(fname = filename, format='png')

def form_kz(k0, kx_f, ky_f, plot=False):
    '''
    This auxiliary function will exclude all propagating wave numbers from the evanescent wave numbers.
    This is necessary because we are creating an arbitrary number of wave numbers (to be used in the decomposition).
    '''
    ke_norm = (kx_f**2 + ky_f**2)**0.5
    kz_f = np.zeros(len(kx_f), dtype = complex)
    # propagating part
    idp = np.where(ke_norm <= k0)[0]
    # kx_p = kx_f[idp]
    # ky_p = ky_f[idp]
    kz_f[idp] = np.sqrt(k0**2 - (kx_f[idp]**2+ky_f[idp]**2))
    # kx_p = kx_f[ke_norm <= k0]
    # ky_p = ky_f[ke_norm <= k0]
    # kz_p = np.sqrt(k0**2 - (kx_p**2+ky_p**2))
    # evanescent part
    ide = np.where(ke_norm > k0)[0]
    kz_f[ide] = -1j*np.sqrt(kx_f[ide]**2+ky_f[ide]**2-k0**2)

    # kx_e = kx_f[ke_norm > k0]
    # ky_e = ky_f[ke_norm > k0]
    # kz_e = -1j*np.sqrt(kx_e**2+ky_e**2-k0**2)
    # # whole thing
    # kz_f = np.concatenate((kz_p, kz_e))
    # # n_evan = len(kx_e_filtered)
    if plot:
        fig = plt.figure()
        fig.canvas.set_window_title('Scatter plot of wave-number directions (half)')
        ax = fig.gca(projection='3d')
        ax.scatter(kx_f[idp], ky_f[idp], np.real(kz_f[idp]))
        ax.scatter(kx_f[ide], ky_f[ide])
        ax.set_xlabel(r'$k_x$ [rad/m]')
        ax.set_ylabel(r'$k_y$ [rad/m]')
        ax.set_zlabel(r'$k_z$ [rad/m]')
        plt.title('Wave number directions (half)')
        plt.tight_layout()
        # plt.scatter(kx_f, ky_f, kz_f, 'ob')
        # plt.xlabel(r'$k_x$ [rad/m]')
        # plt.ylabel(r'$k_y$ [rad/m]')
        # plt.zlabel(r'$k_z$ [rad/m]')
        plt.show()
    return kz_f

def loss_fn(H, pm, x):
    return cp.pnorm(cp.matmul(H, x) - pm, p=2)**2

def regularizer(x):
    return cp.pnorm(x, p=2)**2

def objective_fn(H, pm, x, lambd):
    return loss_fn(H, pm, x) + lambd * regularizer(x)