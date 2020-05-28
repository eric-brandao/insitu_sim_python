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
from lcurve_functions import csvd, l_cuve
import pickle
from receivers import Receiver
from material import PorousAbsorber
from controlsair import cart2sph, sph2cart, update_progress, compare_alpha, compare_zs

# from insitu.field_calc import LocallyReactive

from rayinidir import RayInitialDirections

class PArrayDeduction(object):
    '''
    Impedance deduction class for array processing
    '''
    def __init__(self, sim_field = [], source_num = 0):
        '''
        Init - we first retrive general data, then we process some receiver data
        '''
        # self.pres_s = sim_field.pres_s[source_num] #FixMe
        try:
            self.air = sim_field.air
            self.controls = sim_field.controls
            self.material = sim_field.material
            self.sources = sim_field.sources
            self.receivers = sim_field.receivers
            self.pres_s = sim_field.pres_s[source_num] #FixMe
        except:
            self.air = sim_field
            self.controls = sim_field
            self.material = sim_field
            self.sources = sim_field
            self.receivers = sim_field
            self.pres_s = []
        try:
            self.uz_s = sim_field.uz_s[source_num] #FixMe
        except:
            self.uz_s = []
        self.flag_oct_interp = False

    def wavenum_dir(self, n_waves = 50, plot = False, icosphere = True):
        '''
        This method is used to create wave number directions uniformily distributed over the surface of a sphere.
        It is mainly used when the sensing matrix is made of plane waves. In that case one creates directions of
        propagation that later will bevome wave-number vectors. The directions of propagation are calculated
        with the triangulation of an icosahedron used previously in the generation of omnidirectional rays
        (originally implemented in a ray tracing algorithm).
        Inputs:
            n_waves - The number of directions (wave-directions) to generate (integer)
            plot - whether you plot or not the wave points in space (bool)
            icosphere - method used in the calculation of directions (bool, default= icosahedron)
        '''
        directions = RayInitialDirections()
        if icosphere:
            self.dir, self.n_waves = directions.isotropic_rays(Nrays = int(n_waves))
        else: # FixME with random rays it does not work so well. Try other methods
            self.dir, self.n_waves = directions.random_rays(Nrays = n_waves)
        print('The number of created waves is: {}'.format(self.n_waves))
        if plot:
            directions.plot_points()

    def wavenum_direv(self, n_waves = 642, plot = False, freq=1000):
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
        # Test some filtering
        # kx_e_filtered, ky_e_filtered, self.n_evan = filter_evan(2*np.pi*freq/self.air.c0, self.kx_e, self.ky_e, plot=True)
        # print('The number of created waves is: {}'.format(self.n_waves))
        if plot:
            k0 = 2*np.pi*freq / self.air.c0
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
        # loop over frequencies
        bar = ChargingBar('Calculating Tikhonov inversion...', max=len(self.controls.k0), suffix='%(percent)d%%')
        # bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Tikhonov inversion...')
        self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=np.csingle)
        # print(self.pk.shape)
        # toolbar_width = 20
        # sys.stdout.write("[%s]" % (" " * toolbar_width))
        # sys.stdout.flush()
        # sys.stdout.write(" " * (toolbar_width+1)) # return to start of line, after '['
        for jf, k0 in enumerate(self.controls.k0):
            # sys.stdout.write('-')
            # sys.stdout.flush()
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
            # print('reg par: {}'.format(lambd_value))
            if method == 'scipy':
                from scipy.sparse.linalg import lsqr, lsmr
                x = lsqr(h_mtx, self.pres_s[:,jf], damp=np.sqrt(lambd_value))
                self.pk[:,jf] = x[0]
            elif method == 'direct':
                Hm = np.matrix(h_mtx)
                self.pk[:,jf] = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + lambd_value*np.identity(len(pm))) @ pm
            # print('x values: {}'.format(x[0]))
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
        # sys.stdout.write("]\n")
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
        # loop over frequencies
        bar = ChargingBar('Calculating Tikhonov inversion (with evanescent waves)...', max=len(self.controls.k0), suffix='%(percent)d%%')
        # bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Tikhonov inversion...')
        self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=complex)
        self.pk_ev = []
        for jf, k0 in enumerate(self.controls.k0):
            # update_progress(jf/len(self.controls.k0))
            # First, we form the propagating wave-numbers ans sensing matrix
            k_vec = k0 * self.dir
            h_p = np.exp(1j*self.receivers.coord @ k_vec.T)
            # Then, we have to form the remaining evanescent wave-numbers and evanescent sensing matrix
            kx_e, ky_e, self.n_evan = filter_evan(k0, self.kx_e, self.ky_e, plot=False)
            # print('Number of evanescent is {}'.format(self.n_evan))
            kz_e = np.sqrt(k0**2 - kx_e**2 - ky_e**2+0j)
            k_ev = np.array([kx_e, ky_e, kz_e]).T
            h_ev = np.exp(1j*self.receivers.coord @ k_ev.T)
            self.kx_e = kx_e
            self.ky_e = ky_e
            # Form H matrix
            h_mtx = np.hstack((h_p, h_ev))
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
                x = lsqr(h_mtx, self.pres_s[:,jf], damp=np.sqrt(lambd_value))
                self.pk[:,jf] = x[0][0:self.n_waves]
                self.pk_ev.append(x[0][self.n_waves:])
            # elif method == 'direct':
            #     Hm = np.matrix(h_mtx)
            #     self.pk[:,jf] = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + lambd_value*np.identity(len(pm))) @ pm
            # # print('x values: {}'.format(x[0]))
            # #### Performing the Tikhonov inversion with cvxpy #########################
            # else:
            #     H = h_mtx.astype(complex)
            #     x = cp.Variable(h_mtx.shape[1], complex = True)
            #     lambd = cp.Parameter(nonneg=True)
            #     lambd.value = lambd_value[0]
            #     # Create the problem and solve
            #     problem = cp.Problem(cp.Minimize(objective_fn(H, pm, x, lambd)))
            #     # problem.solve()
            #     problem.solve(solver=cp.SCS, verbose=False) # Fast but gives some warnings
            #     # problem.solve(solver=cp.ECOS, abstol=1e-3) # slow
            #     # problem.solve(solver=cp.ECOS_BB) # slow
            #     # problem.solve(solver=cp.NAG) # not installed
            #     # problem.solve(solver=cp.CPLEX) # not installed
            #     # problem.solve(solver=cp.CBC)  # not installed
            #     # problem.solve(solver=cp.CVXOPT) # not installed
            #     # problem.solve(solver=cp.MOSEK) # not installed
            #     # problem.solve(solver=cp.OSQP) # did not work
            #     self.pk[:,jf] = x.value
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
            # wave numbers
            # kx = k0 * np.cos(self.phi) * np.sin(self.theta)
            # ky = k0 * np.sin(self.phi) * np.sin(self.theta)
            # kz = k0 * np.cos(self.theta)
            # k_vec = (np.array([kx.flatten(), ky.flatten(), kz.flatten()])).T
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
        self.pk_oct = np.zeros((self.n_waves, len(self.freq_oct)), dtype=np.csingle)
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
        # bar = ChargingBar('Interpolating the grid for P(k)',\
        #     max=len(self.controls.k0), suffix='%(percent)d%%')
        if self.flag_oct_interp:
            for jf, f_oct in enumerate(self.freq_oct):
                update_progress(jf/len(self.freq_oct))
                ###### Cubic with griddata #################################
                self.grid_pk.append(griddata(thetaphi_pts, self.pk_oct[:,jf],
                    (self.grid_phi, self.grid_theta), method='cubic', fill_value=np.finfo(float).eps, rescale=False))
        else:
            for jf, k0 in enumerate(self.controls.k0):
                update_progress(jf/len(self.controls.k0))
                ###### Nearest with griddata #################################
                # self.grid_pk.append(griddata(thetaphi_pts, self.pk[:,jf],
                #     (self.grid_phi, self.grid_theta), method='nearest'))
                ###### Cubic with griddata #################################
                self.grid_pk.append(griddata(thetaphi_pts, self.pk[:,jf],
                    (self.grid_phi, self.grid_theta), method='cubic', fill_value=np.finfo(float).eps, rescale=False))
            #     bar.next()
            # bar.finish()

    def alpha_from_array(self, desired_theta = [0], target_range = 3, plot = False):
        '''
        Method to calculate the absorption coefficient straight from 3D array data.
        Inputs:
            desired_theta: a target angle of incidence for which you desire information
                (has to be between 0deg and 90deg)
        '''
        # Transform to spherical coordinates
        self.desired_theta = desired_theta
        r, theta, phi = cart2sph(self.dir[:,0], self.dir[:,1], self.dir[:,2])
        # Get the incident and reflected hemispheres
        theta_inc_id, theta_ref_id = get_hemispheres(theta)
        # We will loop through the list of desired_thetas
        # Initialize
        self.alpha_avg = np.zeros((len(desired_theta), len(self.controls.k0)))
        for jtheta, dtheta in enumerate(desired_theta):
            # Get the list of indexes for angles you want
            thetainc_des_list, thetaref_des_list = desired_theta_list(theta_inc_id, theta_ref_id,
                theta, desired_theta = dtheta, target_range = target_range)
            # Loop over frequency
            bar = ChargingBar('Calculating absorption (avg w/o interp...) for angle: ' +\
                str(np.rad2deg(dtheta)) + ' deg.',\
                max=len(self.controls.k0), suffix='%(percent)d%%')
            for jf, k0 in enumerate(self.controls.k0):
                pk_inc = self.pk[theta_inc_id[0], jf] # hemisphere
                pk_ref = self.pk[theta_ref_id[0], jf] # hemisphere
                pk_inc_target = pk_inc[thetainc_des_list] # at target angles along phi
                pk_ref_target = pk_ref[thetaref_des_list] # at target angles along phi
                inc_energy = np.mean(np.abs(pk_inc_target)**2) 
                ref_energy = np.mean(np.abs(pk_ref_target)**2)
                self.alpha_avg[jtheta, jf] = 1 - ref_energy/inc_energy
                bar.next()
            bar.finish()
            if plot:
                # Get the incident and reflected directions (coordinates to plot)
                incident_dir, reflected_dir = get_inc_ref_dirs(self.dir, theta_inc_id, theta_ref_id)
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.scatter(incident_dir[thetainc_des_list,0], incident_dir[thetainc_des_list,1], incident_dir[thetainc_des_list,2],
                    color='blue')
                ax.scatter(reflected_dir[thetaref_des_list,0], reflected_dir[thetaref_des_list,1], reflected_dir[thetaref_des_list,2],
                    color='red')
                ax.scatter(self.dir[:,0], self.dir[:,1], self.dir[:,2], 
                    color='silver', alpha=0.2)
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                ax.set_zlim((-1, 1))
                plt.show()

    def alpha_from_array2(self, desired_theta = [0], target_range = 3, plot = False):
        '''
        Method to calculate the absorption coefficient straight from 3D array data.
        Inputs:
            desired_theta: a target angle of incidence for which you desire information
                (has to be between 0deg and 90deg)
        '''
        # Get theta and phi in a flat list
        self.desired_theta = desired_theta
        theta = self.grid_theta.flatten()
        phi = self.grid_phi.flatten()
        # Get the directions of interpolated data.
        xx, yy, zz = sph2cart(1, theta, phi)
        dirs = np.transpose(np.array([xx, yy, zz]))
        # Get the incident and reflected hemispheres
        theta_inc_id, theta_ref_id = get_hemispheres(theta)
        # We will loop through the list of desired_thetas
        # Initialize
        if self.flag_oct_interp:
            self.alpha_avg2 = np.zeros((len(desired_theta), len(self.freq_oct))) # Nangles x Nfreq
            for jtheta, dtheta in enumerate(desired_theta):
                # Get the list of indexes for angles you want
                thetainc_des_list, thetaref_des_list = desired_theta_list(theta_inc_id, theta_ref_id,
                    theta, desired_theta = dtheta, target_range = target_range)
                # Loop over frequency
                bar = ChargingBar('Calculating absorption (avg...) for angle: ' +\
                    str(np.rad2deg(dtheta)) + ' deg.',\
                    max=len(self.freq_oct), suffix='%(percent)d%%')
                for jf, fc in enumerate(self.freq_oct):
                    pk = self.grid_pk[jf].flatten()
                    pk_inc = pk[theta_inc_id[0]] # hemisphere
                    pk_ref = pk[theta_ref_id[0]] # hemisphere
                    pk_inc_target = pk_inc[thetainc_des_list] # at target angles along phi
                    pk_ref_target = pk_ref[thetaref_des_list] # at target angles along phi
                    inc_energy = np.mean(np.abs(pk_inc_target)**2)
                    ref_energy = np.mean(np.abs(pk_ref_target)**2)
                    self.alpha_avg2[jtheta, jf] = 1 - ref_energy/inc_energy
                    bar.next()
                bar.finish()
        else:
            self.alpha_avg2 = np.zeros((len(desired_theta), len(self.controls.k0))) # Nangles x Nfreq
            for jtheta, dtheta in enumerate(desired_theta):
                # Get the list of indexes for angles you want
                thetainc_des_list, thetaref_des_list = desired_theta_list(theta_inc_id, theta_ref_id,
                    theta, desired_theta = dtheta, target_range = target_range)
                # Loop over frequency
                bar = ChargingBar('Calculating absorption (avg...) for angle: ' +\
                    str(np.rad2deg(dtheta)) + ' deg.',\
                    max=len(self.controls.k0), suffix='%(percent)d%%')
                for jf, k0 in enumerate(self.controls.k0):
                    pk = self.grid_pk[jf].flatten()
                    pk_inc = pk[theta_inc_id[0]] # hemisphere
                    pk_ref = pk[theta_ref_id[0]] # hemisphere
                    pk_inc_target = pk_inc[thetainc_des_list] # at target angles along phi
                    pk_ref_target = pk_ref[thetaref_des_list] # at target angles along phi
                    inc_energy = np.mean(np.abs(pk_inc_target)**2)
                    ref_energy = np.mean(np.abs(pk_ref_target)**2)
                    self.alpha_avg2[jtheta, jf] = 1 - ref_energy/inc_energy
                    bar.next()
                bar.finish()
                if plot:
                    # Get the incident and reflected directions (coordinates to plot)
                    incident_dir, reflected_dir = get_inc_ref_dirs(dirs, theta_inc_id, theta_ref_id)
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    ax.scatter(incident_dir[thetainc_des_list,0], incident_dir[thetainc_des_list,1], incident_dir[thetainc_des_list,2],
                        color='blue', alpha=1)
                    ax.scatter(reflected_dir[thetaref_des_list,0], reflected_dir[thetaref_des_list,1], reflected_dir[thetaref_des_list,2],
                        color='red', alpha=1)
                    ax.scatter(dirs[:,0], dirs[:,1], dirs[:,2], 
                        color='silver', alpha=0.2)
                    ax.set_xlabel('X axis')
                    ax.set_ylabel('Y axis')
                    ax.set_zlabel('Z axis')
                    ax.set_zlim((-1, 1))
                    plt.show()

    def zs(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [0], avgZs = True):
        '''
        Method to calculate the absorption coefficient straight from 3D array data.
        Inputs:
            Lx - The length of calculation aperture
            Ly - The width of calculation aperture
            n_x - The number of calculation points in x dir
            n_y - The number of calculation points in y dir
        '''
        grid = Receiver()
        grid.planar_array(x_len=Lx, y_len=Ly, zr=0.0, n_x = n_x, n_y = n_x)
        if n_x > 1 or n_y > 1:
            self.grid = grid.coord
        else:
            self.grid = np.array([0,0,0])
        # print('the grid: {}'.format(self.grid))
        # loop over frequency dommain
        # self.Zs = np.zeros((len(theta), len(self.controls.k0)), dtype=complex)
        self.Zs = np.zeros(len(self.controls.k0), dtype=complex)
        # self.alpha = np.zeros(len(self.controls.k0))
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
            # pressure and particle velocity at surface
            p_surf_mtx = h_mtx @ x
            uz_surf_mtx = ((np.divide(k_vec[:,2], k0)) * h_mtx) @ x
            self.p_s[:,jf] =  p_surf_mtx
            self.uz_s[:,jf] =  uz_surf_mtx
            if avgZs:
                Zs_pt = np.divide(p_surf_mtx, uz_surf_mtx)
                self.Zs[jf] = np.mean(Zs_pt)
            else:
                self.Zs[jf] = np.mean(p_surf_mtx) / (np.mean(uz_surf_mtx)) 
            bar.next()
        bar.finish()
        # try:
        #     theta = self.material.theta
        # except:
        #     theta = 0
        self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        for jtheta, dtheta in enumerate(theta):
            self.alpha[jtheta,:] = 1 - (np.abs(np.divide((self.Zs  * np.cos(dtheta) - 1),\
                (self.Zs * np.cos(dtheta) + 1))))**2
        # self.alpha = 1 - (np.abs(np.divide((self.Zs - 1),\
        #     (self.Zs + 1))))**2
        return self.alpha

    def zs_selective(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, desired_theta = 0, target_range = 3, avgZs = True):
        '''
        Method to calculate the absorption coefficient straight from 3D array data.
        Inputs:
            Lx - The length of calculation aperture
            Ly - The width of calculation aperture
            n_x - The number of calculation points in x dir
            n_y - The number of calculation points in y dir
        '''
        self.desired_theta = desired_theta
        ######### Find target angles for calculations ###########################
        # Get theta and phi in a flat list
        theta = self.grid_theta.flatten()
        phi = self.grid_phi.flatten()
        # Get the directions of interpolated data.
        xx, yy, zz = sph2cart(1, theta, phi)
        dirs = np.transpose(np.array([xx, yy, zz]))
        # Get the incident and reflected hemispheres
        theta_inc_id, theta_ref_id = get_hemispheres(theta)
        # Initialize Zs_sel
        self.Zs_sel = np.zeros((len(desired_theta), len(self.controls.k0)), dtype=complex)
        self.alpha_sel = np.zeros((len(desired_theta), len(self.controls.k0)))
        for jtheta, dtheta in enumerate(desired_theta):
            # Get the list of indexes for angles you want
            thetainc_des_list, thetaref_des_list = desired_theta_list(theta_inc_id, theta_ref_id,
                theta, desired_theta = dtheta, target_range = target_range)
            # Get the incident and reflected directions
            dirs_inc = dirs[theta_inc_id[0]] # hemisphere
            dirs_ref = dirs[theta_ref_id[0]] # hemisphere
            dirs_inc_target = dirs_inc[thetainc_des_list] # at target angles along phi
            dirs_ref_target = dirs_ref[thetaref_des_list] # at target angles along phi
            dirs_target = np.concatenate((dirs_inc_target, dirs_ref_target))
            ################# Reconstruction pts ############################
            grid = Receiver()
            grid.planar_array(x_len=Lx, y_len=Ly, zr=0.0, n_x = n_x, n_y = n_x)
            if n_x > 1 or n_y > 1:
                self.grid_sel = grid.coord
            else:
                self.grid_sel = np.array([0,0,0])
        ################ Initialize variables ####################################
        # self.Zs_sel = np.zeros(len(self.controls.k0), dtype=complex)
            self.p_s_sel = np.zeros((len(self.grid_sel), len(self.controls.k0)), dtype=complex)
            self.uz_s_sel = np.zeros((len(self.grid_sel), len(self.controls.k0)), dtype=complex)
            # loop over frequency dommain
            bar = ChargingBar('Calculating Zs and absorption (sel. backprop.) for angle {}'.format(np.rad2deg(dtheta)),\
                max=len(self.controls.k0), suffix='%(percent)d%%')
            for jf, k0 in enumerate(self.controls.k0):
                # Wave number vector
                k_vec = k0 * dirs_target
                # Form H matrix
                h_mtx = np.exp(1j*self.grid_sel @ k_vec.T)
                # complex amplitudes of selected waves
                pk = self.grid_pk[jf].flatten()
                pk_inc = pk[theta_inc_id[0]] # hemisphere
                pk_ref = pk[theta_ref_id[0]] # hemisphere
                pk_inc_target = pk_inc[thetainc_des_list] # at target angles along phi
                pk_ref_target = pk_ref[thetaref_des_list] # at target angles along phi
                x = np.concatenate((pk_inc_target, pk_ref_target))
                # pressure and particle velocity at surface
                p_surf_mtx = h_mtx @ x
                uz_surf_mtx = ((np.divide(k_vec[:,2], k0)) * h_mtx) @ x
                self.p_s_sel[:,jf] =  p_surf_mtx
                self.uz_s_sel[:,jf] =  uz_surf_mtx
                if avgZs:
                    Zs_pt = np.divide(p_surf_mtx, uz_surf_mtx)
                    self.Zs_sel[jtheta,jf] = np.mean(Zs_pt)
                else:
                    self.Zs_sel[jtheta,jf] = np.mean(p_surf_mtx) / (np.mean(uz_surf_mtx)) 
                bar.next()
            bar.finish()
            self.alpha_sel[jtheta,:] = 1 - (np.abs(np.divide((self.Zs_sel[jtheta,:]  * np.cos(dtheta) - 1),\
                (self.Zs_sel[jtheta,:] * np.cos(dtheta) + 1))))**2
        # return self.alpha_sel

    def plot_pk_sphere(self, freq = 1000, db = False, dinrange = 40, save = False, name='name'):
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
        # id_f = np.where(self.freq_oct <= freq)
        id_f = id_f[0][-1]
        fig = plt.figure()
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
        # p=ax.plot_surface(self.dir[:,0], self.dir[:,1], self.dir[:,2],
        #     color = color_par)
        fig.colorbar(p)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - ' + name)
        # plt.show()
        if save:
            filename = 'data/colormaps/cmat_' + str(int(freq)) + 'Hz_' + name
            plt.savefig(fname = filename, format='pdf')

    def plot_pk_sphere_interp(self, freq = 1000, db = False, dinrange = 40, save = False, name='name'):
        '''
        Method to plot the magnitude of the spatial fourier transform on the surface of a sphere.
        The data should be interpolated first. Then, you can see a smooth representation of the colors
        on the surface of a sphere.
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
        ax = fig.gca(projection='3d')
        if db:
            color_par = 20*np.log10(np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f])))
            id_outofrange = np.where(color_par < -dinrange)
            color_par[id_outofrange] = -dinrange
        else:
            color_par = np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f]))
        norm = plt.Normalize()
        facecolors = plt.cm.jet(norm(color_par))
        zz, yy, xx = sph2cart(1, self.grid_theta, self.grid_phi)
        p=ax.plot_surface(xx, yy, zz,
            facecolors=facecolors, linewidth=0, antialiased=False, shade=False, cmap=plt.cm.coolwarm)
        # p=ax.plot_trisurf(xx.flatten(), yy.flatten(), zz.flatten(),
        #     color=facecolors, antialiased=False, shade=False)
        fig.colorbar(p, shrink=0.5, aspect=5)
        # fig.colorbar(p)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - ' + name)
        # plt.show()
        if save:
            filename = 'data/colormaps/cmatinterp_' + str(int(freq)) + 'Hz_' + name
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
        # plt.subplot(111, projection="lambert")
        # ax = fig.gca(projection='3d')
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

    def plot_pk_evmap(self, freq = 1000, n_evani = 50, db = False, dinrange = 12, save = False, name='', path = '', fname=''):
        '''
        Method to plot the magnitude of the spatial fourier transform of the evanescent components
        The map of interpolated to a kx and ky wave numbers.
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
        # First, let us interpolate to have a nice colormap
        new_kx = np.linspace(self.kx_e[0], self.kx_e[-1], n_evani)
        new_ky = np.linspace(self.ky_e[0], self.ky_e[-1], n_evani)
        kx, ky = np.meshgrid(new_kx, new_ky)
        # interpolate
        from scipy.interpolate import griddata
        pk_ev_grid = griddata(np.transpose(np.array([self.kx_e, self.ky_e])), self.pk_ev[id_f],
            (kx, ky), method='cubic', fill_value=np.finfo(float).eps, rescale=False)
        # plot
        fig = plt.figure()
        fig.canvas.set_window_title('Wave number spk of evanescent waves')
        if db:
            color_par = 20*np.log10(np.abs(pk_ev_grid))#20*np.log10(np.abs(pk_ev_grid)/np.amax(np.abs(pk_ev_grid)))
            # id_outofrange = np.where(color_par < -dinrange)
            # color_par[id_outofrange] = -dinrange
        else:
            color_par = np.abs(pk_ev_grid)#/np.amax(np.abs(pk_ev_grid))
        p=plt.contourf(kx, ky, color_par)
        fig.colorbar(p)
        plt.xlabel(r'kx [m$^{-1}$]')
        plt.ylabel(r'ky [m$^{-1}$]')
        plt.title('Evanescent |P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - '+ name)
        # plt.show()
        if save:
            filename = path + fname + '_' + str(int(freq)) + 'Hz'
            plt.savefig(fname = filename, format='png')

    def plot_pk_mapscat(self, freq = 1000, db = False, dinrange = 40, save = False, name='name'):
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
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        if db:
            color_par = 20*np.log10(np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f])))
            id_outofrange = np.where(color_par < -dinrange)
            color_par[id_outofrange] = -dinrange
        else:
            color_par = np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f]))
        r, theta, phi = cart2sph(self.dir[:,0], self.dir[:,1], self.dir[:,2])
        p=plt.scatter(np.rad2deg(theta), np.rad2deg(phi), c = color_par)
        fig.colorbar(p)
        # ax.set_xlabel('phi (azimuth) [deg]')
        # ax.set_ylabel('theta (elevation) [deg]')
        plt.xlabel('phi (azimuth) [deg]')
        plt.ylabel('theta (elevation) [deg]')
        plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - ' + name)
        # plt.show()
        if save:
            filename = 'data/colormaps/map_' + str(int(freq)) + 'Hz_' + name
            plt.savefig(fname = filename, format='pdf')

    def plot_flat_pk(self, freq = 1000):
        '''
        Auxiliary method to plot the wave number spectrum in a xy plot
        '''
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        pk = self.pk[:,id_f]
        xk = np.linspace(0, 1, len(pk))
        pk_int = (self.grid_pk[id_f]).flatten()
        # pk_int = np.roll(pk_int, 500)
        # pk_int = np.flip(pk_int)
        # print('pk {}'.format(pk_int))
        xk_int = np.linspace(0, 1, len(pk_int))
        plt.figure()
        plt.title('Flat P(k)')
        plt.plot(xk, np.abs(pk)/np.amax(np.abs(pk)), label = 'not interpolated')
        plt.plot(xk_int,np.abs(pk_int)/np.amax(np.abs(pk_int)), '--r', label = 'interpolated')
        plt.grid(linestyle = '--', which='both')
        plt.legend(loc = 'upper left')
        plt.xlabel('k/len(k) - index [-]')
        plt.ylabel('|P(k)| [-]')
        plt.ylim((-0.2, 1.2))
        # plt.show()

    def plot_alpha(self, theta = [0], save = False, path = 'single_pw/', filename = 'absorption'):
        '''
        A method to plot the absorption coefficient. If your acoustic field is composed of a single
        sound source, then the "material" object passed to this class is used as a reference basis of
        comparison (the "except" case). In that case "self.desired_theta" is composed of a single angle of incidence targeting
        the only existing sound source in the acoustic field. The incidence angle comes with the passed "material".
        It is the same as self.desired_theta.

        Now, if the acoustic field is composed of a more difuse sound incidence (the "try" case)
        you would need to recreate a reference material. In that case, as the flow resistivity and thickness
        of the sample do not change you can just take this data out of the list of materials in "field"
        (use material[0]). The incidence angle is the theta passed as an argument to the function.
        '''
        id_t = np.where(self.desired_theta <= theta)
        id_t = id_t[0][-1]
        # try:
        #     material = PorousAbsorber(self.air, self.controls)
        #     material.miki(resistivity=self.material[0].resistivity)
        #     material.layer_over_rigid(thickness = self.material[0].thickness, theta = theta)
        # except:
        #     material = self.material
        material = PorousAbsorber(self.air, self.controls)
        material.miki(resistivity=self.material.resistivity)
        material.layer_over_rigid(thickness = self.material.thickness, theta = self.desired_theta[id_t])
        leg_ref = str(int(1000*material.thickness)) + ' mm, ' +\
            str(int(material.resistivity)) + ' resist. (' +\
            str(int(np.ceil(np.rad2deg(material.theta)))) + ' deg.)'
        if self.flag_oct_interp:
            freq = self.freq_oct
        else:
            freq = self.controls.freq

        compare_alpha(
            {'freq': material.freq, leg_ref: material.alpha, 'color': 'black', 'linewidth': 4},
            {'freq': self.controls.freq, 'backpropagation': self.alpha[id_t,:], 'color': 'blue', 'linewidth': 3},
            # {'freq': self.controls.freq, 'backpropagation sel': self.alpha_sel, 'color': 'orange', 'linewidth': 2},
            # {'freq': self.controls.freq, "Melanie's way": self.alpha_avg[id_t,:], 'color': 'red', 'linewidth': 1},
            {'freq': freq, "Melanie's way with interp": self.alpha_avg2[id_t,:], 'color': 'green', 'linewidth': 2})

        if save:
            filename = '/home/eric/research/insitu_arrays/results/figures/' + path + '/absorption/' + filename
            plt.savefig(fname = filename, format='pdf')

    def plot_zs(self, theta = [0], save = False, path = 'single_pw/', filename = 'zs'):
        try:
            material = PorousAbsorber(self.air, self.controls)
            material.miki(resistivity=self.material[0].resistivity)
            material.layer_over_rigid(thickness = self.material[0].thickness, theta = theta)
        except:
            material = self.material
        leg_ref = str(int(1000*material.thickness)) + ' mm, ' +\
            str(int(material.resistivity)) + ' resist. (' +\
            str(int(np.ceil(np.rad2deg(material.theta)))) + ' deg.)'

        compare_zs(
            {'freq': material.freq, leg_ref: material.Zs/(self.air.c0*self.air.rho0), 'color': 'black', 'linewidth': 4},
            {'freq': self.controls.freq, 'backpropagation': self.Zs, 'color': 'blue', 'linewidth': 2},
            {'freq': self.controls.freq, 'backpropagation sel': self.Zs_sel, 'color': 'orange', 'linewidth': 2})
        if save:
            filename = '/home/eric/research/insitu_arrays/results/figures/' + path + '/surfimpedance/' + filename
            plt.savefig(fname = filename, format='pdf')

    def plot_alphavstheta(self, save = False, path = 'single_pw/', filename = 'absorption'):
        '''
        Function used to plot alpha vs. theta per frequency band

        If your acoustic field is composed of a single sound source, it does not make sense to use this method,
        as outputs away from the angle where the source is will be very imprecise.

        Now, if the acoustic field is composed of a more difuse sound incidence (the "try" case)
        you would need to recreate a reference material. In that case, as the flow resistivity and thickness
        of the sample do not change you can just take this data out of the list of materials in "field"
        (use material[0]). The target incidence angles are the ones in self.desired_theta.

        Frequency bands from 315 to 2000 are ploted at this stage.
        '''
        # Let's test if the acoustic field is diffuse or composed of a single sound source.
        if (not self.sources or len(self.sources.coord) == 1):
            print('You only have a single sound source incidence. It does not make sense to print angle variation of absorption coefficient')
        # if diffuse field is the case, then:
        else:
            # Create a reference:
            alpha_ref = alpha_vs_angle(self.desired_theta, self.air, self.controls,
                self.material[0].resistivity, self.material[0].thick1, thick2 = self.material[0].thick2)
            freq_oct, flower, fupper = octave_freq(self.controls.freq, nband = 3)
            # octave avg each direction
            alpha_ref_oct = np.zeros((len(self.desired_theta), len(freq_oct)), dtype=np.csingle)
            for jang in np.arange(0, len(self.desired_theta)):
                alpha_ref_oct[jang,:] = octave_avg(self.controls.freq, alpha_ref[jang, :], freq_oct, flower, fupper)
            # Find frequency bands to plot from 315 - 2000 (Melanie's papper)
            id_fplt = np.where(np.logical_and(self.freq_oct > 250,
                self.freq_oct < 2500))
            freq_oct = self.freq_oct[id_fplt[0]]
            alpha_ref_oct = alpha_ref_oct[:,id_fplt[0]]
            #############################################################################
            # porous = PorousAbsorber(self.air, self.controls)
            # porous.freq = freq_oct
            # porous.miki(resistivity=self.material[0].resistivity)

            # alpha_ref = np.zeros((len(self.desired_theta), len(freq_oct)))
            # for jel, el in enumerate(self.desired_theta):
            #     Zs, Vp, alpha_ref[jel, :] = porous.layer_over_rigid(thickness=self.material[0].thickness,
            #         theta = el)
            #############################################################################
            # The figure
            fig, axs = plt.subplots(3,3)
            jfc = 0
            jfc_data = id_fplt[0][0]
            for jl in np.arange(0, 3):
                for jc in np.arange(0,3):
                    axs[jl, jc].plot(np.rad2deg(self.desired_theta),
                        alpha_ref_oct[:, jfc], 'black', label = 'data', linewidth = 2)
                    axs[jl, jc].plot(np.rad2deg(self.desired_theta),
                        self.alpha_avg2[:, jfc_data], 'green', label = 'data', linewidth = 2)
                    axs[jl, jc].grid(linestyle = '--', which='both')
                    # axs[0].legend(loc = 'best')
                    axs[jl, jc].set(title = 'fc = ' + str("{:.1f}".format(freq_oct[jfc])) + 'Hz')
                    # axs[jl, jc].set(title = 'fc = ' + str(freq_oct[jfc]) + 'Hz')
                    axs[jl, jc].set(ylabel = 'absorption coefficient')
                    axs[jl, jc].set(xlabel = 'angle [deg]')
                    plt.setp(axs[jl, jc], ylim=(0.0, 1.0))
                    jfc += 1
                    jfc_data +=1
            print('done')

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

#%% Auxiliary functions
def get_hemispheres(theta):
    # Get incident and reflected ids
    theta_inc_id = np.where(np.logical_and(theta >= 0, theta <= np.pi/2))
    theta_ref_id = np.where(np.logical_and(theta >= -np.pi/2, theta <= 0))
    return theta_inc_id, theta_ref_id

def get_inc_ref_dirs(directions, theta_inc_id, theta_ref_id):
    incident_dir = directions[theta_inc_id[0]]
    reflected_dir = directions[theta_ref_id[0]]
    return incident_dir, reflected_dir

def find_desiredangle(desired_angle, angles, target_range = 3):
    ang_sorted_list = np.unique(np.sort(angles))
    ang_range = np.mean(ang_sorted_list[1:]-ang_sorted_list[0:-1])
    if ang_range < np.deg2rad(target_range):
        ang_range = np.deg2rad(target_range)
    theta_des_list = np.where(np.logical_and(angles <= desired_angle+ang_range/2,
        angles >= desired_angle-ang_range/2))
    angles_in_range = angles[theta_des_list[0]]
    return angles_in_range, theta_des_list[0]

def desired_theta_list(theta_inc_id, theta_ref_id, theta, desired_theta = 0, target_range = 3):
    # Rotate desired angle to be correct and transform to radians
    desired_theta = np.pi/2-desired_theta
    # Get the incident directions
    # theta_inc_id = np.where(np.logical_and(theta >= 0, theta <= np.pi/2))
    # incident_dir = directions[theta_inc_id[0]]
    incident_theta = theta[theta_inc_id[0]]
    # Get the reflected directions
    # theta_ref_id = np.where(np.logical_and(theta >= -np.pi/2, theta <= 0))
    # reflected_dir = directions[theta_ref_id[0]]
    reflected_theta = theta[theta_ref_id[0]]
    # Get the indexes for and the desired angle
    thetainc_des, thetainc_des_list = find_desiredangle(desired_theta, incident_theta, target_range=target_range)
    thetaref_des, thetaref_des_list = find_desiredangle(-desired_theta, reflected_theta, target_range=target_range)
    return thetainc_des_list, thetaref_des_list

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

#%% Functions to use with cvxpy
def loss_fn(H, pm, x):
    return cp.pnorm(cp.matmul(H, x) - pm, p=2)**2

def regularizer(x):
    return cp.pnorm(x, p=2)**2

def objective_fn(H, pm, x, lambd):
    return loss_fn(H, pm, x) + lambd * regularizer(x)

def lcurve_der(lambd_values, solution_norm, residual_norm, plot_print = False):
    '''
    Function to determine the best regularization parameter
    '''
    dxi = (np.array(solution_norm[1:])**2 - np.array(solution_norm[0:-1])**2)/\
        (np.array(lambd_values[1:]) - np.array(lambd_values[0:-1]))
    dpho = (np.array(residual_norm[1:])**2 - np.array(residual_norm[0:-1])**2)/\
        (np.array(lambd_values[1:]) - np.array(lambd_values[0:-1]))
    clam = (2**np.array(lambd_values[1:])*(dxi**2))/\
        ((dpho**2 + dxi**2)**3/2)
    id_maxcurve = np.where(clam == np.amax(clam))
    lambd_ideal = lambd_values[id_maxcurve[0]+1]
    if plot_print:
        print('The ideal value of lambda is: {}'.format(lambd_ideal))
        plt.plot(lambd_values[1:], clam)
        plt.show()
    print(id_maxcurve[0] + 1)
    return int(id_maxcurve[0] + 1)

def octave_freq(freq, nband = 3):
    '''
    Function calculates center frequencies and mean value of complex signal for the band 
    '''
    # Center frequencies
    fcentre  = (10**nband) * (2**(np.arange(-18, 13) / nband))
    # Find the center frequencies in the calculated band
    id_foct = np.where(np.logical_and(fcentre > freq[0],
        fcentre < freq[-1]))
    fcentre = fcentre[id_foct[0]]
    # Lower and upper frequencies
    fd = 2**(1/(2 * nband))
    fupper = fcentre * fd
    flower = fcentre / fd
    return fcentre, flower, fupper
    #print('bla')

def octave_avg(freq, signal, fcentre, flower, fupper):
    # Loop over each freq band and average
    signal_oct = np.zeros(len(fcentre), dtype=np.csingle)
    for jfc, fc in enumerate(fcentre):
        # find indexes belonging to the frequency band in question
        id_f = np.where(np.logical_and(freq >= flower[jfc],
            freq <= fupper[jfc]))
        # print('for fc of {}, frequencies are: {}'.format(fc, freq[id_f[0]]))
        signal_oct[jfc] = np.mean(signal[id_f[0]])
    return signal_oct

def alpha_vs_angle(desired_theta, air, controls, resistivity, thick1, thick2 = 0):
    porous = PorousAbsorber(air, controls)
    # porous.freq = controls.freq
    porous.miki(resistivity=resistivity)
    alpha_ref = np.zeros((len(desired_theta), len(porous.freq)))
    for jel, el in enumerate(desired_theta):
        Zs, Vp, alpha_ref[jel, :] = porous.layer_over_rigid(thickness = thick1,
            theta = el)
    return alpha_ref

# ######### Find target angles for calculations ###########################
#         # Get theta and phi in a flat list
#         theta = self.grid_theta.flatten()
#         phi = self.grid_phi.flatten()
#         # Get the directions of interpolated data.
#         xx, yy, zz = sph2cart(1, theta, phi)
#         dirs = np.transpose(np.array([xx, yy, zz]))
#         # Get the incident and reflected hemispheres
#         theta_inc_id, theta_ref_id = get_hemispheres(theta)
#         # Get the list of indexes for angles you want
#         thetainc_des_list, thetaref_des_list = desired_theta_list(theta_inc_id, theta_ref_id,
#             theta, desired_theta = desired_theta, target_range = target_range)
#         # Get the incident and reflected directions
#         dirs_inc = dirs[theta_inc_id[0]] # hemisphere
#         dirs_ref = dirs[theta_ref_id[0]] # hemisphere
#         dirs_inc_target = dirs_inc[thetainc_des_list] # at target angles along phi
#         dirs_ref_target = dirs_ref[thetaref_des_list] # at target angles along phi
#         dirs_target = np.concatenate((dirs_inc_target, dirs_ref_target))
#         ################# Reconstruction pts ############################
#         grid = Receiver()
#         grid.planar_array(x_len=Lx, y_len=Ly, zr=0.0, n_x = n_x, n_y = n_x)
#         if n_x > 1 or n_y > 1:
#             self.grid_sel = grid.coord
#         else:
#             self.grid_sel = np.array([0,0,0])
#         ################ Initialize variables ####################################
#         self.Zs_sel = np.zeros(len(self.controls.k0), dtype=complex)
#         self.p_s_sel = np.zeros((len(self.grid_sel), len(self.controls.k0)), dtype=complex)
#         self.uz_s_sel = np.zeros((len(self.grid_sel), len(self.controls.k0)), dtype=complex)
#         # loop over frequency dommain
#         bar = ChargingBar('Calculating impedance and absorption (selective backpropagation)',\
#             max=len(self.controls.k0), suffix='%(percent)d%%')
#         for jf, k0 in enumerate(self.controls.k0):
#             # Wave number vector
#             k_vec = k0 * dirs_target
#             # Form H matrix
#             h_mtx = np.exp(1j*self.grid_sel @ k_vec.T)
#             # complex amplitudes of selected waves
#             pk = self.grid_pk[jf].flatten()
#             pk_inc = pk[theta_inc_id[0]] # hemisphere
#             pk_ref = pk[theta_ref_id[0]] # hemisphere
#             pk_inc_target = pk_inc[thetainc_des_list] # at target angles along phi
#             pk_ref_target = pk_ref[thetaref_des_list] # at target angles along phi
#             x = np.concatenate((pk_inc_target, pk_ref_target))
#             # pressure and particle velocity at surface
#             p_surf_mtx = h_mtx @ x
#             uz_surf_mtx = ((np.divide(k_vec[:,2], k0)) * h_mtx) @ x
#             self.p_s_sel[:,jf] =  p_surf_mtx
#             self.uz_s_sel[:,jf] =  uz_surf_mtx
#             if avgZs:
#                 Zs_pt = np.divide(p_surf_mtx, uz_surf_mtx)
#                 self.Zs_sel[jf] = np.mean(Zs_pt)
#             else:
#                 self.Zs_sel[jf] = np.mean(p_surf_mtx) / (np.mean(uz_surf_mtx)) 
#             bar.next()
#         bar.finish()
#         # try:
#         #     theta = self.material.theta
#         # except:
#         #     theta = 0
#         self.alpha_sel = 1 - (np.abs(np.divide((self.Zs_sel  * np.cos(desired_theta) - 1),\
#             (self.Zs_sel * np.cos(desired_theta) + 1))))**2
#         # self.alpha = 1 - (np.abs(np.divide((self.Zs - 1),\
#         #     (self.Zs + 1))))**2
#         return self.alpha_sel