import numpy as np
import matplotlib.pyplot as plt
import toml
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import cvxpy as cp
from scipy import linalg # for svd
from lcurve_functions import csvd, l_cuve
import pickle
from receivers import Receiver

# from insitu.field_calc import LocallyReactive

from rayinidir import RayInitialDirections

class PArrayDeduction(object):
    '''
    Impedance deduction class for array processing
    '''
    def __init__(self, sim_field, source_num = 0):
        '''
        Init - we first retrive general data, then we process some receiver data
        '''
        self.air = sim_field.air
        self.controls = sim_field.controls
        self.material = sim_field.material
        self.sources = sim_field.sources
        self.receivers = sim_field.receivers
        # self.pres_s = sim_field.pres_s[source_num] #FixMe
        try:
            self.pres_s = sim_field.pres_s[source_num] #FixMe
        except:
            self.pres_s = []
        try:
            self.uz_s = sim_field.uz_s[source_num] #FixMe
        except:
            self.uz_s = []

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
        # rho = np.sqrt(self.dir[:,0]**2+self.dir[:,1]**2+self.dir[:,2]**2)
        # self.theta = np.arccos(self.dir[:,2]/rho)
        # np.seterr(divide = 'ignore')
        # self.phi = np.arccos(self.dir[:,1] / (np.sqrt(self.dir[:,0]**2+self.dir[:,1]**2)))
        # id_phi0 = np.where(np.isnan(self.phi) == True)
        # self.phi[id_phi0] = 0
        print('The number of created waves is: {}'.format(self.n_waves))
        if plot:
            directions.plot_points()
        # self.thetau = np.unique(self.theta)
        # self.phiu = np.unique(self.phi)
        # print('size of theta u: {}'.format(len(self.thetau)))
        # print('size of phi u: {}'.format(len(self.phiu)))



    def pk_a3d_tikhonov(self, lambd_value = [], method = 'scipy'):
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
        bar.finish()
        return self.pk

    def pk_a3d_constrained(self, epsilon = 0.1):
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

    def alpha_from_array3d(self,):
        '''
        Method to calculate the absorption coefficient straight from 3D array data.
        '''
        pass

    def zs(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, avgZs = True):
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
        self.Zs = np.zeros(len(self.controls.k0), dtype=complex)
        # self.alpha = np.zeros(len(self.controls.k0))
        self.p_s = np.zeros((len(self.grid), len(self.controls.k0)), dtype=complex)
        self.uz_s = np.zeros((len(self.grid), len(self.controls.k0)), dtype=complex)
        bar = ChargingBar('Calculating impedance and absorption (backpropagation)',\
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
        theta = self.material.theta
        self.alpha = 1 - (np.abs(np.divide((self.Zs  * np.cos(theta) - 1),\
            (self.Zs * np.cos(theta) + 1))))**2
        # self.alpha = 1 - (np.abs(np.divide((self.Zs - 1),\
        #     (self.Zs + 1))))**2
        return self.alpha

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
        id_f = id_f[0][-1]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if db:
            color_par = 10*np.log10(np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f])))
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

            # print('shape of k_vec: {}'.format(k_vec.shape))
            # print('shape of r: {}'.format(self.receivers.coord.shape))
            # print('shape of H: {}'.format(h_mtx.shape))
    # def tamura_pp(self,):
    #     '''
    #     Tamura's method to measure surface impedance based on measurements taked with a double planar array
    #     '''
    #     #print('shape of press {}'.format(self.pres_s[:,0].shape))
    #     Nmics = self.receivers.coord.shape[0]
    #     N = Nmics/2
    #     z1 = self.receivers.coord[0,2]
    #     z2 = self.receivers.coord[Nmics-1,2]
    #     Dx = self.receivers.coord[1,0]-self.receivers.coord[0,0]
    #     Dy = self.receivers.coord[int(N**0.5),1]-self.receivers.coord[0,1]
    #     for jf, k0 in enumerate(self.controls.k0):
    #         # wave numbers
    #         kx = np.reshape(np.linspace(-((N-1)/(2*N))*k0, ((N-1)/(2*N))*k0, N),
    #             (int(N**0.5), int(N**0.5)))
    #         ky = np.reshape(np.linspace(-((N-1)/(2*N))*k0, ((N-1)/(2*N))*k0, N),
    #             (int(N**0.5), int(N**0.5)))
    #         kz = (k0**2 - kx**2 - ky**2)**0.5
    #         # angles
    #         theta = np.arcsin(((kx**2 + ky**2)**0.5)/k0)
    #         phi = np.arctan(ky/kx)
    #         print('theta angles: {}'.format(theta))
    #         print('phi angles: {}'.format(phi))
    #         # kx = np.linspace(0, ((N-1)/N)*k0, N)
    #         # ky = np.linspace(0, ((N-1)/N)*k0, N)#(-((N-1)/(2*N))*(1/Dy), ((N-1)/(2*N))*(1/Dy), N)
    #         # kz = (k0**2 - kx**2 - ky**2)**0.5
    #         # print(kz)
    #         # # list of pressure signals
    #         # pxy_list = self.pres_s[:,jf]
    #         # # reshape and perform fft2
    #         # p_z1 = np.reshape(pxy_list[0:int(Nmics/2)], (int(N**0.5), int(N**0.5)))
    #         # # p_kxy_z1 = np.fft.fft2(p_z1)
    #         # p_kxy_z1 = np.reshape(np.fft.fft2(p_z1), int(N))
    #         # p_z2 = np.reshape(pxy_list[int(Nmics/2):int(Nmics)], (int(N**0.5), int(N**0.5)))
    #         # # p_kxy_z2 = np.fft.fft2(p_z2)
    #         # p_kxy_z2 = np.reshape(np.fft.fft2(p_z2), int(N))
    #         # # incident and reflected field
    #         # p_i = (p_kxy_z1 * np.exp(1j*kz*z2) - p_kxy_z2 * np.exp(1j*kz*z1)) /\
    #         #     (2*1j*np.sin(kz*(z2-z1)))
    #         # p_r = (p_kxy_z2 * np.exp(-1j*kz*z1) - p_kxy_z1 * np.exp(1j*kz*z2)) /\
    #         #     (2*1j*np.sin(kz*(z2-z1)))
    #         # # reflection coefficient
    #         # vp_kxy = p_r / p_i
    #         # # print(vp_kxy.shape)

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


# def plot_pk_sphere2(self, freq = 1000, db = False):
#         '''
#         Just try plotting the spatial fourier transformation on the surface of a sphere
#         '''
#         id_f = np.where(self.controls.freq <= freq)
#         id_f = id_f[0][-1]
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         if db:
#             color_par = 10*np.log10(np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f])))
#         else:
#             color_par = np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f]))
        
#         p=ax.plot_trisurf(self.theta, self.phi, color_par)
#         fig.colorbar(p)
#         ax.set_xlabel('X axis')
#         ax.set_ylabel('Y axis')
#         ax.set_zlabel('Z axis')
#         plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz')

    # def pk_a3d_tikhonov_lbrute(self,lam_init = -2, lam_end = 1, n_lam = 20):
    #     '''
    #     Method to estimate the wave number spk using Tikhonov regularization and L-curve (brute force). Applied to a 3D array
    #     Inputs:
    #         lam_init: initial value for lambd (reg. par.) - in log scale, so that -2 is equivalent to lambd=0.01 (10^⁻2)
    #         lam_end: final value for lambd (reg. par.) - in log scale, so that 1 is equivalent to lambd=10 (10^1)
    #         n_lam: number of lambda parameters tryed
    #     '''
    #     # loop over frequencies
    #     bar = ChargingBar('Calculating...', max=len(self.controls.k0), suffix='%(percent)d%%')
    #     self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=np.csingle)
    #     # print(self.pk.shape)
    #     for jf, k0 in enumerate(self.controls.k0):
    #         # wave numbers
    #         # kx = k0 * np.cos(self.phi) * np.sin(self.theta)
    #         # ky = k0 * np.sin(self.phi) * np.sin(self.theta)
    #         # kz = k0 * np.cos(self.theta)
    #         # k_vec = (np.array([kx.flatten(), ky.flatten(), kz.flatten()])).T
    #         k_vec = k0 * self.dir
    #         # Form H matrix
    #         h_mtx = np.exp(1j*self.receivers.coord @ k_vec.T)
    #         H = h_mtx.astype(complex) # cvxpy does not accept floats, apparently
    #         # measured data
    #         pm = self.pres_s[:,jf].astype(complex)
    #         #### Performing the Tikhonov inversion with cvxpy #########################
    #         x = cp.Variable(h_mtx.shape[1], complex = True) # create x variable
    #         # Regularization parameters
    #         lambd = cp.Parameter(nonneg=True) # create a regularization parameter
    #         lambd_values = np.logspace(lam_init, lam_end, n_lam)
    #         # Create the problem
    #         problem = cp.Problem(cp.Minimize(objective_fn(H, pm, x, lambd)))
    #         # now, we loop through several attempts of lambda and save values to compute L-curve
    #         x_values = []
    #         solution_norm = []
    #         residual_norm = []
    #         for jv, v in enumerate(lambd_values):
    #             lambd.value = v
    #             print('Solving {} of {}'.format(jv+1, n_lam))
    #             problem.solve()
    #             x_values.append(x.value)
    #             solution_norm.append(np.linalg.norm(x.value, ord=2))
    #             residual_norm.append(np.linalg.norm(H @ x.value - pm, ord=2))
    #         # determine best attempted lambda
    #         id_sol = lcurve_der(lambd_values, solution_norm, residual_norm)
    #         self.pk[:,jf] = x_values[id_sol]

    #         ## Using lsqr (scipy)
    #         # x = lsqr(h_mtx, self.pres_s[:,jf], damp=0.1)
    #         # self.pk[:,jf] = x[0]
    #         # print('x values: {}'.format(x[0]))

    #         bar.next()
    #     bar.finish()