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

class Decomposition(object):
    '''
    Decomposition class for array processing
    '''
    def __init__(self, p_mtx = None, controls = None, receivers = None):
        '''
        Init - we first retrive general data, then we process some receiver data
        '''
        # self.pres_s = sim_field.pres_s[source_num] #FixMe
        # self.air = sim_field.air
        self.controls = controls
        # self.material = sim_field.material
        # self.sources = sim_field.sources
        self.receivers = receivers
        self.pres_s = p_mtx
        self.flag_oct_interp = False

    def wavenum_dir(self, n_waves = 642, plot = False):
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
        print('The number of created waves is: {}'.format(self.n_waves))
        if plot:
            directions.plot_points()

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
        bar = ChargingBar('Calculating Tikhonov inversion...', max=len(self.controls.k0), suffix='%(percent)d%%')
        # bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Tikhonov inversion...')
        # Initialize p(k) as a matrix of n_waves x n_freq
        self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=complex)
        # loop over frequencies
        for jf, k0 in enumerate(self.controls.k0):
            # update_progress(jf/len(self.controls.k0))
            k_vec = k0 * self.dir
            # Form H matrix
            h_mtx = np.exp(1j*self.receivers.coord @ k_vec.T)
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            # finding the optimal lambda value if the parameter comes empty.
            # if not we use the user supplied value.
            if not lambd_value:
                u, sig, v = csvd(h_mtx)
                lambd_value = l_cuve(u, sig, pm, plotit=False)
            ## Choosing the method to find the P(k)
            if method == 'scipy':
                x = lsqr(h_mtx, self.pres_s[:,jf], damp=np.sqrt(lambd_value))
                self.pk[:,jf] = x[0]
            elif method == 'direct':
                Hm = np.matrix(h_mtx)
                self.pk[:,jf] = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + lambd_value*np.identity(len(pm))) @ pm
            #### Performing the Tikhonov inversion with cvxpy #########################
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
                self.pk[:,jf] = x.value
            bar.next()
            # bar.update(1)
        bar.finish()
        # bar.close()
        return self.pk

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
        # plt.show()
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
                color_par = np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f]))
        p=plt.contourf(np.rad2deg(self.grid_phi),
            90-np.rad2deg(self.grid_theta), color_par)
        fig.colorbar(p)
        plt.xlabel('phi (azimuth) [deg]')
        plt.ylabel('theta (elevation) [deg]')
        if self.flag_oct_interp:
            plt.title('|P(k)| at ' + str(self.freq_oct[id_f]) + 'Hz - '+ name)
        else:
            plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - '+ name)
        # plt.show()
        if save:
            filename = path + fname + '_' + str(int(freq)) + 'Hz'
            plt.savefig(fname = filename, format='png')

#### Auxiliary functions
def loss_fn(H, pm, x):
    return cp.pnorm(cp.matmul(H, x) - pm, p=2)**2

def regularizer(x):
    return cp.pnorm(x, p=2)**2

def objective_fn(H, pm, x, lambd):
    return loss_fn(H, pm, x) + lambd * regularizer(x)