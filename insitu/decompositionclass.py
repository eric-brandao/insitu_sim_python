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

class Decomposition(object):
    '''
    Decomposition class for array processing
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

    def wavenum_dir(self, n_waves = 642, plot = False, halfsphere = False):
        '''
        This method is used to create wave number directions uniformily distributed over the surface of a sphere.
        The directions of propagation that later will bevome wave-number vectors.
        The directions of propagation are calculated with the triangulation of an icosahedron used previously
        in the generation of omnidirectional rays (originally implemented in a ray tracing algorithm).
        Inputs:
            n_waves - The number of directions (wave-directions) to generate (integer)
            plot - whether you plot or not the wave points in space (bool)
        '''
        directions = RayInitialDirections()
        self.dir, self.n_waves = directions.isotropic_rays(Nrays = int(n_waves))
        if halfsphere:
            r, theta, phi = cart2sph(self.dir[:,0],self.dir[:,1],self.dir[:,2])
            theta_inc_id, theta_ref_id = get_hemispheres(theta)
            incident_dir, reflected_dir = get_inc_ref_dirs(self.dir, theta_inc_id, theta_ref_id)
            self.dir = reflected_dir
            self.n_waves = len(self.dir)
        print('The number of created waves is: {}'.format(self.n_waves))
        if plot:
            directions.plot_points()

    def wavenum_direv(self, n_waves = 20, plot = False, freq=1000):
        '''
        This method is used to create wave number directions that will be used to decompose the evanescent part 
        of the wave field. This part will be the kx and ky componentes. They only depend on the array size and 
        on the microphone spacing. When performing the decomposition, kz will depend on the calculated kx and ky.
        Furthermore, the evanescent part will be separated from the propagating part, so that they can be easily
        filtered out.
        Inputs:
            n_waves - The number of directions (wave-directions) to generate (integer)
            plot - whether you plot or not the wave points in space (bool)
            freq - to have a notion of the radiation circle when plotting kx and ky
        '''
        # Figure out the size of the array in x and y directions
        # Figure out the spacing between the microphones in x and y directions
        # Create kx ad ky (this includes prpagating waves - we'll deal with them later)
        # kx = np.arange(start = -np.pi/self.receivers.ax,
        #     stop = np.pi/self.receivers.ax+2*np.pi/self.receivers.x_len, step = 2*np.pi/self.receivers.x_len)
        # ky = np.arange(start = -np.pi/self.receivers.ay,
        #     stop = np.pi/self.receivers.ay+2*np.pi/self.receivers.y_len, step = 2*np.pi/self.receivers.y_len)
        #### With linspace and n_waves
        kx = np.linspace(start = -np.pi/self.receivers.ax,
            stop = np.pi/self.receivers.ax, num = n_waves)
        ky = np.linspace(start = -np.pi/self.receivers.ay,
            stop = np.pi/self.receivers.ay, num = n_waves)

        self.kx_grid, self.ky_grid = np.meshgrid(kx,ky)
        self.kx_e = self.kx_grid.flatten()
        self.ky_e = self.ky_grid.flatten()
        if plot:
            k0 = 2*np.pi*freq / self.controls.c0
            fig = plt.figure()
            fig.canvas.set_window_title('Non filtered evanescent waves')
            plt.plot(self.kx_e, self.ky_e, 'o')
            plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
                k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
            plt.xlabel('kx')
            plt.ylabel('ky')
            plt.show()

    def pk_tikhonov(self, lambd_value = [], method = 'scipy'):
        '''
        Method to estimate wave number spectrum based on the Tikhonov matrix inversion technique.
        Inputs:
            lambd_value: Value of the regularization parameter. The user can specify that.
                If it comes empty, then we use L-curve to determine the optmal value.
            method: string defining the method to be used on finding the correct P(k).
                It can be:
                    (1) - 'scipy': using scipy.linalg.lsqr
                    (2) - 'direct': via x= (Hm^H) * ((Hm * Hm^H + lambd_value * I)^-1) * pm
                    (3) - else: via cvxpy
        '''
        # Bars
        self.decomp_type = 'Tikhonov (transparent array)'
        bar = ChargingBar('Calculating Tikhonov inversion...', max=len(self.controls.k0), suffix='%(percent)d%%')
        # bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Tikhonov inversion...')
        # Initialize p(k) as a matrix of n_waves x n_freq
        self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=complex)
        self.cond_num = np.zeros(len(self.controls.k0))
        # loop over frequencies
        for jf, k0 in enumerate(self.controls.k0):
            # update_progress(jf/len(self.controls.k0))
            k_vec = k0 * self.dir
            # Form H matrix
            h_mtx = np.exp(1j*self.receivers.coord @ k_vec.T)
            self.cond_num[jf] = np.linalg.cond(h_mtx)
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            # finding the optimal lambda value if the parameter comes empty.
            # if not we use the user supplied value.
            if not lambd_value:
                u, sig, v = csvd(h_mtx)
                lambd_value = l_cuve(u, sig, pm, plotit=False)
            ## Choosing the method to find the P(k)
            if method == 'scipy':
                x = lsqr(h_mtx, self.pres_s[:,jf], damp=lambd_value)
                self.pk[:,jf] = x[0]
            elif method == 'direct':
                Hm = np.matrix(h_mtx)
                self.pk[:,jf] = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
            #### Performing the Tikhonov inversion with cvxpy #########################
            else:
                H = h_mtx.astype(complex)
                x = cp.Variable(h_mtx.shape[1], complex = True)
                lambd = cp.Parameter(nonneg=True)
                lambd.value = lambd_value[0]
                # Create the problem and solve
                problem = cp.Problem(cp.Minimize(objective_fn(H, pm, x, lambd)))
                # problem.is_dcp(dpp = True)
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
                self.pk[:,jf] = x.value
            bar.next()
            # bar.update(1)
        bar.finish()
        # bar.close()
        return self.pk

    def pk_tikhonov_ev(self, lambd_value = [], method = 'scipy'):
        '''
        Method to estimate wave number spectrum based on the Tikhonov matrix inversion technique.
        This version includes the evanescent waves
        Inputs:
            lambd_value: Value of the regularization parameter. The user can specify that.
                If it comes empty, then we use L-curve to determine the optmal value.
            method: string defining the method to be used on finding the correct P(k).
                It can be:
                    (1) - 'scipy': using scipy.linalg.lsqr
                    (2) - 'direct': via x= (Hm^H) * ((Hm * Hm^H + lambd_value * I)^-1) * pm
                    (3) - else: via cvxpy
        '''
        self.decomp_type = 'Tikhonov (transparent array) w/ evanescent waves'
        # loop over frequencies
        bar = ChargingBar('Calculating Tikhonov inversion (with evanescent waves)...', max=len(self.controls.k0), suffix='%(percent)d%%')
        # bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Tikhonov inversion...')
        self.cond_num = np.zeros(len(self.controls.k0))
        self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=complex)
        self.kx_ef = [] # Filtered version
        self.ky_ef = [] # Filtered version
        self.pk_ev = []
        for jf, k0 in enumerate(self.controls.k0):
            # update_progress(jf/len(self.controls.k0))
            # First, we form the propagating wave-numbers ans sensing matrix
            k_vec = k0 * self.dir
            h_p = np.exp(-1j*self.receivers.coord @ k_vec.T)
            # Then, we have to form the remaining evanescent wave-numbers and evanescent sensing matrix
            kx_e, ky_e, self.n_evan = filter_evan(k0, self.kx_e, self.ky_e, plot=False)
            # print('Number of evanescent is {}'.format(self.n_evan))
            kz_e = np.sqrt(k0**2 - kx_e**2 - ky_e**2+0j)
            k_ev = np.array([kx_e, ky_e, kz_e]).T
            h_ev = np.exp(1j*self.receivers.coord @ k_ev.T)
            self.kx_ef.append(kx_e)
            self.ky_ef.append(ky_e)
            # Form H matrix
            h_mtx = np.hstack((h_p, h_ev))
            self.cond_num[jf] = np.linalg.cond(h_mtx)
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            # finding the optimal lambda value if the parameter comes empty.
            # if not we use the user supplied value.
            if not lambd_value:
                u, sig, v = csvd(h_mtx)
                lambd_value = l_cuve(u, sig, pm, plotit=False)
            # ## Choosing the method to find the P(k)
            # # print('reg par: {}'.format(lambd_value))
            if method == 'scipy':
                from scipy.sparse.linalg import lsqr, lsmr
                x = lsqr(h_mtx, self.pres_s[:,jf], damp=lambd_value)
                self.pk[:,jf] = x[0][0:self.n_waves]
                self.pk_ev.append(x[0][self.n_waves:])
                print(x[0].shape)
            elif method == 'direct':
                Hm = np.matrix(h_mtx)
                x = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
                # print(x.shape)
                self.pk[:,jf] = x[0:self.n_waves]
                self.pk_ev.append(x[self.n_waves:])
            elif method == 'Ridge':
                # Form a real H2 matrix and p2 measurement
                H2 = np.vstack((np.hstack((h_mtx.real, -h_mtx.imag)),
                    np.hstack((h_mtx.imag, h_mtx.real))))
                p2 = np.vstack((pm.real,pm.imag)).flatten()
                # form Ridge regressor using the regularization from L-curve
                regressor = Ridge(alpha=lambd_value, fit_intercept = False, solver = 'svd')
                x2 = regressor.fit(H2, p2).coef_
                x = x2[:h_mtx.shape[1]]+1j*x2[h_mtx.shape[1]:]
                # print(x.shape)
                # separate propagating from evanescent
                self.pk[:,jf] = x[0:self.n_waves]
                self.pk_ev.append(x[self.n_waves:])
            elif method == 'tikhonov':
                u, sig, v = csvd(h_mtx)
                x = tikhonov(u, sig, v, pm, lambd_value)
                self.pk[:,jf] = x[0:self.n_waves]
                self.pk_ev.append(x[self.n_waves:])
                # print(x.shape)
            # #### Performing the Tikhonov inversion with cvxpy #########################
            else:
                H = h_mtx.astype(complex)
                x = cp.Variable(h_mtx.shape[1], complex = True)
                lambd = cp.Parameter(nonneg=True)
                lambd.value = lambd_value[0]
                # Create the problem and solve
                problem = cp.Problem(cp.Minimize(objective_fn(H, pm, x, lambd)))
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
                self.pk[:,jf] = x.value[0:self.n_waves]
                self.pk_ev.append(x.value[self.n_waves:])
            bar.next()
            # bar.update(1)
        bar.finish()
        # bar.close()
        # sys.stdout.write("]\n")
        # return self.pk

    def pk_constrained(self, epsilon = 0.1):
        '''
        Method to estimate wave number spectrum based on constrained optimization matrix inversion technique.
        Inputs:
            epsilon - upper bound of noise floor vector
        '''
        # loop over frequencies
        bar = ChargingBar('Calculating bounded optmin...', max=len(self.controls.k0), suffix='%(percent)d%%')
        self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=np.csingle)
        # print(self.pk.shape)
        for jf, k0 in enumerate(self.controls.k0):
            k_vec = k0 * self.dir
            # Form H matrix
            h_mtx = np.exp(1j*self.receivers.coord @ k_vec.T)
            H = h_mtx.astype(complex) # cvxpy does not accept floats, apparently
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            #### Performing the Tikhonov inversion with cvxpy #########################
            x = cp.Variable(h_mtx.shape[1], complex = True) # create x variable
            # Create the problem
            problem = cp.Problem(cp.Minimize(cp.norm2(x)**2),
                [cp.pnorm(cp.matmul(H, x) - pm, p=2) <= epsilon])
            problem.solve(solver=cp.SCS, verbose=False)
            self.pk[:,jf] = x.value
            bar.next()
        bar.finish()
        return self.pk

    def pk_cs(self, lambd_value = [], method = 'scipy'):
        '''
        Method to estimate wave number spectrum based on the l1 inversion technique.
        This is supposed to give us a sparse solution for the sound field decomposition.
        Inputs:
            method: string defining the method to be used on finding the correct P(k).
            It can be:
                (1) - 'scipy': using scipy.linalg.lsqr
                (2) - 'direct': via x= (Hm^H) * ((Hm * Hm^H + lambd_value * I)^-1) * pm
                (3) - else: via cvxpy
        '''
        # loop over frequencies
        bar = ChargingBar('Calculating CS inversion...', max=len(self.controls.k0), suffix='%(percent)d%%')
        self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=np.csingle)
        # print(self.pk.shape)
        for jf, k0 in enumerate(self.controls.k0):
            # wave numbers
            k_vec = k0 * self.dir
            # Form H matrix
            h_mtx = np.exp(1j*self.receivers.coord @ k_vec.T)
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            ## Choosing the method to find the P(k)
            if method == 'scipy':
                # from scipy.sparse.linalg import lsqr, lsmr
                # x = lsqr(h_mtx, self.pres_s[:,jf], damp=np.sqrt(lambd_value))
                # self.pk[:,jf] = x[0]
                pass
            elif method == 'direct':
                # Hm = np.matrix(h_mtx)
                # self.pk[:,jf] = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + lambd_value*np.identity(len(pm))) @ pm
                pass
            # print('x values: {}'.format(x[0]))
            #### Performing the Tikhonov inversion with cvxpy #########################
            else:
                H = h_mtx.astype(complex)
                x = cp.Variable(h_mtx.shape[1], complex = True)
                objective = cp.Minimize(cp.pnorm(x, p=1))
                constraints = [H*x == pm]
                # Create the problem and solve
                problem = cp.Problem(objective, constraints)
                # problem.solve()
                # problem.solve(verbose=False) # Fast but gives some warnings
                problem.solve(solver=cp.SCS, verbose=True) # Fast but gives some warnings
                # problem.solve(solver=cp.ECOS, abstol=1e-3) # slow
                # problem.solve(solver=cp.ECOS_BB) # slow
                # problem.solve(solver=cp.NAG) # not installed
                # problem.solve(solver=cp.CPLEX) # not installed
                # problem.solve(solver=cp.CBC)  # not installed
                # problem.solve(solver=cp.CVXOPT) # not installed
                # problem.solve(solver=cp.MOSEK) # not installed
                # problem.solve(solver=cp.OSQP) # did not work
                self.pk[:,jf] = x.value
            bar.next()
        bar.finish()
        return self.pk

    def pk_oct_interpolate(self, nband = 3):
        '''
        method to interpolate pk over an octave or 1/3 ocatave band
        '''
        # Set flag to true
        self.flag_oct_interp = True
        self.freq_oct, flower, fupper = octave_freq(self.controls.freq, nband = nband)
        self.pk_oct = np.zeros((self.n_waves, len(self.freq_oct)), dtype=complex)
        # octave avg each direction
        for jdir in np.arange(0, self.n_waves):
            self.pk_oct[jdir,:] = octave_avg(self.controls.freq, self.pk[jdir, :], self.freq_oct, flower, fupper)

    def reconstruct_pu(self, receivers):
        '''
        reconstruct sound pressure and particle velocity at a receiver object
        '''
        self.p_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        self.uz_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing sound field...')
        for jf, k0 in enumerate(self.controls.k0):
            # First, we form the sensing matrix
            k_p = -k0 * self.dir
            # h_p = np.exp(-1j*receivers.coord @ k_p.T)
            kz_e = np.sqrt(k0**2 - self.kx_ef[jf]**2 - self.ky_ef[jf]**2+0j)
            k_ev = np.array([self.kx_ef[jf], self.ky_ef[jf], kz_e]).T
            k_vec = np.vstack((k_p, k_ev))
            # h_ev = np.exp(1j*receivers.coord @ k_ev.T)
            # h_mtx = np.hstack((h_p, h_ev))
            h_mtx = np.exp(1j*receivers.coord @ k_vec.T)
            # pressure and particle velocity at surface
            self.p_recon[:,jf] = h_mtx @ np.concatenate((self.pk[:,jf],self.pk_ev[jf]))
            self.uz_recon[:,jf] = -((np.divide(k_vec[:,2], k0)) * h_mtx) @ np.concatenate((self.pk[:,jf], self.pk_ev[jf]))
            # self.p_s[:,jf] =  p_surf_mtx
            #  =  uz_surf_mtx
            bar.update(1)

    def plot_condnum(self, save = False, path = '', fname = ''):
        '''
        Method to plot the condition number
        '''
        fig = plt.figure()
        fig.canvas.set_window_title('Condition number')
        plt.title('Condition number - {}'.format(self.decomp_type))
        plt.loglog(self.controls.freq, self.cond_num, color = 'black', label = self.decomp_type, linewidth = 2)
        plt.grid(linestyle = '--', which='both')
        plt.legend(loc = 'best')
        plt.xticks([50, 100, 500, 1000, 2000, 4000, 8000, 10000],
            ['50', '100', '500', '1k', '2k', '4k', '8k', '10k'])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'$\kappa$ [-]')
        plt.ylim((0.0001, 2*np.amax(self.cond_num)))
        plt.xlim((0.8*self.controls.freq[0], 1.2*self.controls.freq[-1]))
        if save:
            filename = path + fname
            plt.savefig(fname = filename, format='pdf')

    def pk_interpolate(self, npts=100):
        '''
        Method to interpolate the wave number spectrum.
        '''
        # Recover the actual measured points
        r, theta, phi = cart2sph(self.dir[:,0], self.dir[:,1], self.dir[:,2])
        # r, theta, phi = cart2sph(self.dir[:,2], self.dir[:,1], self.dir[:,0])
        thetaphi_pts = np.transpose(np.array([phi, theta]))
        # Create a grid to interpolate on
        nphi = int(2*(npts+1))
        ntheta = int(npts+1)
        sorted_phi = np.sort(phi)
        new_phi = np.linspace(sorted_phi[0], sorted_phi[-1], nphi)
        sorted_theta = np.sort(theta)
        new_theta = np.linspace(sorted_theta[0], sorted_theta[-1], ntheta)#(0, np.pi, nn)
        self.grid_phi, self.grid_theta = np.meshgrid(new_phi, new_theta)
        # interpolate
        from scipy.interpolate import griddata
        self.grid_pk = []
        bar = ChargingBar('Interpolating the grid for P(k)',\
            max=len(self.controls.k0), suffix='%(percent)d%%')
        if self.flag_oct_interp:
            for jf, f_oct in enumerate(self.freq_oct):
                # update_progress(jf/len(self.freq_oct))
                ###### Cubic with griddata #################################
                self.grid_pk.append(griddata(thetaphi_pts, self.pk_oct[:,jf],
                    (self.grid_phi, self.grid_theta), method='cubic', fill_value=np.finfo(float).eps, rescale=False))
        else:
            for jf, k0 in enumerate(self.controls.k0):
                # update_progress(jf/len(self.controls.k0))
                ###### Cubic with griddata #################################
                self.grid_pk.append(griddata(thetaphi_pts, self.pk[:,jf],
                    (self.grid_phi, self.grid_theta), method='cubic', fill_value=np.finfo(float).eps, rescale=False))
                bar.next()
            bar.finish()

    def plot_pk_sphere(self, freq = 1000, db = False, dinrange = 40, save = False, name=''):
        '''
        Method to plot the magnitude of the spatial fourier transform on the surface of a sphere.
        It is a normalized version of the magnitude, either between 0 and 1 or between -dinrange and 0.
        inputs:
            freq - Which frequency you want to see. If the calculated spectrum does not contain it
                we plot the closest frequency before the asked one.
            dB (bool) - Whether to plot in linear scale (default) or decibel scale.
            dinrange - You can specify a dinamic range for the decibel scale. It will not affect the
            linear scale.
            save (bool) - Whether to save or not the figure. PDF file with simple standard name
        '''
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        fig = plt.figure()
        fig.canvas.set_window_title('Scatter plot of |P(k)| for freq {} Hz'.format(self.controls.freq[id_f]))
        ax = fig.gca(projection='3d')
        if db:
            color_par = 20*np.log10(np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f])))
            # color_par = 20*np.log10(np.abs(self.pk_oct[:,id_f])/np.amax(np.abs(self.pk_oct[:,id_f])))
            id_outofrange = np.where(color_par < -dinrange)
            color_par[id_outofrange] = -dinrange
        else:
            color_par = np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f]))
        p=ax.scatter(self.dir[:,0], self.dir[:,1], self.dir[:,2],
            c = color_par)
        fig.colorbar(p)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - ' + name)
        plt.tight_layout()
        if save:
            filename = 'data/colormaps/cmat_' + str(int(freq)) + 'Hz_' + name
            plt.savefig(fname = filename, format='pdf')

    def plot_pk_map(self, freq = 1000, db = False, dinrange = 40, phase = False, save = False, name='', path = '', fname=''):
        '''
        Method to plot the magnitude of the spatial fourier transform on a map of interpolated theta and phi.
        It is a normalized version of the magnitude, either between 0 and 1 or between -dinrange and 0.
        inputs:
            freq - Which frequency you want to see. If the calculated spectrum does not contain it
                we plot the closest frequency before the asked one.
            dB (bool) - Whether to plot in linear scale (default) or decibel scale.
            dinrange - You can specify a dinamic range for the decibel scale. It will not affect the
            linear scale.
            save (bool) - Whether to save or not the figure. PDF file with simple standard name
        '''
        if self.flag_oct_interp:
            id_f = np.where(self.freq_oct <= freq)
        else:
            id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        fig = plt.figure()
        fig.canvas.set_window_title('Interpolated map of |P(k)| for freq {} Hz'.format(self.controls.freq[id_f]))
        if db:
            color_par = 20*np.log10(np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f])))
            id_outofrange = np.where(color_par < -dinrange)
            color_par[id_outofrange] = -dinrange
        else:
            if phase:
                color_par = np.rad2deg(np.angle(self.grid_pk[id_f]))
            else:
                # color_par = np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f]))
                color_par = np.abs(self.grid_pk[id_f])
        p=plt.contourf(np.rad2deg(self.grid_phi),
            90-np.rad2deg(self.grid_theta), color_par)
        fig.colorbar(p)
        plt.xlabel(r'$\phi$ (azimuth) [deg]')
        plt.ylabel(r'$\theta$ (elevation) [deg]')
        if self.flag_oct_interp:
            plt.title('|P(k)| at ' + str(self.freq_oct[id_f]) + 'Hz - '+ name)
        else:
            plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - '+ name)
        plt.tight_layout()
        if save:
            filename = path + fname + '_' + str(int(freq)) + 'Hz'
            plt.savefig(fname = filename, format='png')

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
            color_par = 20*np.log10(np.abs(self.pk_ev[id_f])/np.amax(np.abs(self.pk_ev[id_f])))
            id_outofrange = np.where(color_par < -dinrange)
            color_par[id_outofrange] = -dinrange
        else:
            color_par = np.abs(self.pk_ev[id_f])#/np.amax(np.abs(pk_ev_grid))
        ############### Countourf ##########################
        # Create the Triangulation; no triangles so Delaunay triangulation created.
        x = self.kx_ef[id_f]
        y = self.ky_ef[id_f]
        triang = tri.Triangulation(x, y)
        # Mask off unwanted triangles.
        triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
            y[triang.triangles].mean(axis=1)) < k0)
        fig = plt.figure()
        fig.canvas.set_window_title('Filtered evanescent waves')
        plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
            k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
        if contourplot:
            p = plt.tricontourf(triang, color_par,
                levels=dinrange)
        else:
            p=plt.scatter(self.kx_ef[id_f], self.ky_ef[id_f], c = color_par)
        fig.colorbar(p)
        if plot_kxky:
            plt.scatter(self.kx_ef[id_f], self.ky_ef[id_f], c = 'grey', alpha = 0.4)
        plt.xlabel('kx rad/m')
        plt.ylabel('ky rad/m')
        plt.title("|P(k)| (evanescent) at {0:.1f} Hz (k = {1:.2f} rad/m)".format(self.controls.freq[id_f],k0))
        plt.tight_layout()
        if save:
            filename = path + fname + '_' + str(int(freq)) + 'Hz'
            plt.savefig(fname = filename, format='png')

    def save(self, filename = 'array_zest', path = '/home/eric/dev/insitu/data/zs_recovery/'):
        '''
        This method is used to save the simulation object
        '''
        filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'array_zest', path = '/home/eric/dev/insitu/data/zs_recovery/'):
        '''
        This method is used to load a simulation object. You build a empty object
        of the class and load a saved one. It will overwrite the empty one.
        '''
        lpath_filename = path + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
#### Auxiliary functions
def filter_evan(k0, kx_e, ky_e, plot=False):
    '''
    This auxiliary function will exclude all propagating wave numbers from the evanescent wave numbers.
    This is necessary because we are creating an arbitrary number of wave numbers (to be used in the decomposition).
    '''
    ke_norm = (kx_e**2 + ky_e**2)**0.5
    kx_e_filtered = kx_e[ke_norm > k0]
    ky_e_filtered = ky_e[ke_norm > k0]
    n_evan = len(kx_e_filtered)
    if plot:
        fig = plt.figure()
        fig.canvas.set_window_title('Filtered evanescent waves')
        plt.plot(kx_e_filtered, ky_e_filtered, 'o')
        plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
            k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.show()
    return kx_e_filtered, ky_e_filtered, n_evan


def loss_fn(H, pm, x):
    return cp.pnorm(cp.matmul(H, x) - pm, p=2)**2

def regularizer(x):
    return cp.pnorm(x, p=2)**2

def objective_fn(H, pm, x, lambd):
    return loss_fn(H, pm, x) + lambd * regularizer(x)