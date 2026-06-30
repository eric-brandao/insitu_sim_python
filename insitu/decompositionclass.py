import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
# from insitu.controlsair import load_cfg
# import scipy.integrate as integrate
# import scipy as spy
# from sklearn.linear_model import Ridge
# import time
from tqdm import tqdm
# import sys
# from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
#from tqdm._tqdm_notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#import cvxpy as cp
# from scipy import linalg # for svd
# from scipy import signal
# from scipy.sparse.linalg import lsqr, lsmr
# from lcurve_functions import csvd, l_cuve, tikhonov, ridge_solver, direct_solver
import lcurve_functions as lc
import pickle
from receivers import Receiver
# from material import PorousAbsorber
from controlsair import cart2sph, sph2cart
from rayinidir import RayInitialDirections
#from parray_estimation import octave_freq, octave_avg, get_hemispheres, get_inc_ref_dirs

# SMALL_SIZE = 11
# BIGGER_SIZE = 13
# #plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'font.family': 'serif'})
# plt.rc('legend', fontsize=SMALL_SIZE)
# #plt.rc('title', fontsize=SMALL_SIZE)
# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('figure', titlesize=BIGGER_SIZE)

class PPWE(object):
    """ Decomposition of the sound field using ony propagating waves.

    The class has several methods to perform sound field decomposition into a set of
    incident and reflected plane propagating waves. These sets of plane waves are composed of
    propagating waves only. The propagating waves are created by segmentation of the
    surface of a sphere into equal solid angles.

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

    def __init__(self, p_mtx = None, controls = None, material = None, receivers = None,
                 regu_par = 'L-curve'):
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
        self.controls = controls
        self.material = material
        self.receivers = receivers
        self.pres_s = p_mtx
        self.flag_oct_interp = False
        if regu_par == 'L-curve' or regu_par == 'l-curve':
            self.regu_par_fun = lc.l_curve
            print("You choose L-curve to find optimal regularization parameter")
        elif regu_par == 'gcv' or regu_par == 'GCV':
            self.regu_par_fun = lc.gcv_lambda
            print("You choose GCV to find optimal regularization parameter")
        else:
            self.regu_par_fun = lc.l_curve
            print("Returning to default L-curve to find optimal regularization parameter")
            
    def iso_wavenum_dir(self, n_waves = 642, plot = False, halfsphere = False):
        """ Create the propagating wave number directions

        The propagating wave number directions are uniformily distributed
        over the surface of a sphere (which will have radius k [rad/m] during the
        decomposition). The directions of propagating waves are calculated with the
        triangulation of an icosahedron used previously (originally implemented in a
        ray tracing algorithm).

        Parameters
        ----------
            n_waves : int
                The number of intended wave-directions to generate (Default is 642).
                Usually the subdivision of the sphere will return an equal or higher
                number of directions. Then, we take the reflected part only (half of it).
            plot : bool
                whether you plot or not the directions in space (bool)
            halfsphere : bool
                whether to use only half a sphere - used in radiation problems only
        """
        dir_obj = Receiver()
        dir_obj.isospherical_array(radius = 1, n_rec_target = n_waves)
        self.dir = np.copy(dir_obj.coord)
        self.n_waves = self.dir.shape[0]
        self.connectivities = np.copy(dir_obj.connectivities)
        # directions = RayInitialDirections()
        # self.dir, self.n_waves,_ = directions.isotropic_rays(Nrays = int(n_waves))
        # if halfsphere:
        #     _, theta, _ = cart2sph(self.dir[:,0],self.dir[:,1],self.dir[:,2])
        #     theta_inc_id, theta_ref_id = get_hemispheres(theta)
        #     _, reflected_dir = get_inc_ref_dirs(self.dir, theta_inc_id, theta_ref_id)
        #     self.dir = reflected_dir
        #     self.n_waves = len(self.dir)
        # print('The number of created waves is: {}'.format(self.n_waves))
        if plot:
            self.plot_wn_dir()
            
    def sg_wavenum_dir(self, delta_theta_deg = 4, plot = False):
        """ Create the propagating wave number directions

        The propagating wave number directions are uniformily distributed
        over the surface of a sphere (which will have radius k [rad/m] during the
        decomposition). The directions of propagating waves are calculated with the
        semi-gaussian sphere (allowing uniform sampling over theta).

        Parameters
        ----------
            n_waves : int
                The number of intended wave-directions to generate (Default is 642).
                Usually the subdivision of the sphere will return an equal or higher
                number of directions. Then, we take the reflected part only (half of it).
            plot : bool
                whether you plot or not the directions in space (bool)
            halfsphere : bool
                whether to use only half a sphere - used in radiation problems only
        """
        dir_obj = Receiver()
        dir_obj.semigaussian_sphere(radius = 1, delta_theta_deg = delta_theta_deg, 
                                    hemispherical = False)
        self.dir = np.copy(dir_obj.coord)
        self.n_waves = self.dir.shape[0]
        self.connectivities = np.copy(dir_obj.connectivities)
        if plot:
            self.plot_wn_dir()
            
    def plot_wn_dir(self, idx = None):
        """ Plot wave number directions
        """
        # directions.plot_points()
        plt.figure()
        ax = plt.axes(projection ="3d")
        if idx is not None:
            ax.scatter(self.dir[idx,0], self.dir[idx,1], self.dir[idx,2],
                color='red', s = 20, marker = 'o') 
        ax.scatter(self.dir[:,0], self.dir[:,1], self.dir[:,2],
            color='blue', s=5, alpha = 0.4)
        ax.set_title("{} directions".format(self.dir.shape[0]), loc = 'center')
        ax.set_xlabel(r'$k_x$  [rad/m]')
        ax.set_ylabel(r'$k_y$ [rad/m]')
        ax.set_zlabel(r'$k_z$ [rad/m]')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.tight_layout()
        
    def hmtx_p(self, k0, recs, uni_sphere):  # pressure matrix
        """ Form p-matrix
        
        Parameters
        ----------
        k0 : float
            wave-number magnitude
        recs : numpy1dArray
            receiver coordinates of shape Kx3
        uni_sphere : numpy1dArray
            coordinates on the surface of the unit sphere (of shape Lx3)            
        """
        k_vec = k0 * uni_sphere
        h_mtx = np.exp(-1j * recs @ k_vec.T)
        return h_mtx
    
    def hmtx_u(self, k0, recs, uni_sphere,  direction = 2):  # x,y,z-particle velocity matrix
        """ Form p-matrix
        
        Parameters
        ----------
        k0 : float
            wave-number magnitude
        recs : numpy1dArray
            receiver coordinates of shape Kx3
        uni_sphere : numpy1dArray
            coordinates on the surface of the unit sphere (of shape Lx3)  
        direction : int
            0 (for x direction), 1 (for y direction), 2 (for z direction)
        """
        
        k_vec = k0 * uni_sphere
        h_mtx = (np.divide(k_vec[:,direction], k0)) * np.exp(-1j * recs @ k_vec.T)
        return h_mtx

    def pk_tikhonov(self, method = 'Tikhonov', plot_l = False):
        """ Wave number spectrum estimation using Tikhonov inversion

        Estimate the wave number spectrum using regularized Tikhonov inversion.
        The choice of the regularization parameter is baded on the L-curve criterion.
        This sound field is modelled by a set of propagating waves. This
        method is an adaptation of DTU methods, implemented in:
            Mélanie Nolan. Estimation of angle-dependent absorption coefficients 
            from spatially distributed in situ measurements , J Acoust Soc Am (EL).
            2019 147(2):EL119-EL124. doi: 10.1121/10.0000716 

        The inversion steps are: (i) - Get the scaled version of the propagating directions;
        (ii) - form the sensing matrix; (iii) - compute SVD of the sensing matix;
        (iv) - compute the regularization parameter (L-curve); (vii) - matrix inversion.

        Parameters
        ----------
        method : str
            Determines which method to use to compute the pseudo-inverse.
                'direct' (default) - analytical solution - fastest, but maybe
                inacurate on noiseless situations. The following uses optimization 
                algorithms
                'scipy' - uses scipy.linalg.lsqr (sparse matrix) -fast but less acurate
                'Ridge - uses sklearn Ridge regression - slower, but accurate.
                'cvx' - uses cvxpy - slower, but accurate.
        plot_l : bool
            Whether to plot the L-curve or not. Default is false.
        """
        self.decomp_type = 'Tikhonov (transparent array)'
        bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Tikhonov inversion...')
        # Initialize variables
        self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=complex)
        self.lambd_value_vec = np.zeros(len(self.controls.k0))
        self.cond_num = np.zeros(len(self.controls.k0))
        # loop over frequencies
        for jf, k0 in enumerate(self.controls.k0):
            # get sensing matrix
            h_mtx = self.hmtx_p(k0, self.receivers.coord, self.dir)
            # k_vec = k0 * self.dir
            # # Form the sensing matrix
            # h_mtx = np.exp(-1j*self.receivers.coord @ k_vec.T)
            self.cond_num[jf] = np.linalg.cond(h_mtx)
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            # compute SVD of the sensing matix
            u, sig, v = lc.csvd(h_mtx)
            # compute the regularization parameter (L-curve)
            lambd_value = self.regu_par_fun(u, sig, pm, plot_l)
            self.lambd_value_vec[jf] = lambd_value
            if method == 'direct':
                Hm = np.matrix(h_mtx)
                self.pk[:,jf] = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
            elif method == 'Ridge':
                x = lc.ridge_solver(h_mtx,pm,lambd_value)
                self.pk[:,jf] = x
            elif method == 'Tikhonov':
                x = lc.tikhonov(u,sig,v,pm,lambd_value)
                self.pk[:,jf] = x
            elif method == 'cvx':
                x = lc.cvx_tikhonov(h_mtx.astype(complex), pm, lambd_value, l_norm = 2)
                self.pk[:,jf] = x
            bar.update(1)
        bar.close()
        # return self.pk

    def pk_constrained(self, snr=30, headroom = 0):
        """ Wave number spectrum estimation using constrained optimization

        Estimate the wave number spectrum using constrained optimization. The problem
        solved is min(||x||_2), subjected to ||Ax - b||_2 < e.

        This method is an adaptation of DTU methods, implemented in:
        Efren Fernandez-Grande. Sound field reconstruction using a spherical microphone
        array, J Acoust Soc Am. 2016 139(3):1168-1178. doi: 10.1121/1.4943545

        The inversion steps are: (i) - Get the scaled version of the propagating directions;
        (ii) - form the sensing matrix; (iii) - compute the inverse problem.

        Parameters
        ----------
        snr : float
            Signal to noise ratio in the simulation
        headroom : float
            Apply some headroom to the noise level to compute "e". 
        """
        # Initialize
        self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=np.csingle)
        # loop over frequencies
        bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Constrained Optim.')
        for jf, k0 in enumerate(self.controls.k0):
            # get the scaled version of the propagating directions
            k_vec = k0 * self.dir
            # Form the sensing matrix
            h_mtx = np.exp(-1j*self.receivers.coord @ k_vec.T)
            H = h_mtx.astype(complex) # cvxpy does not accept floats, apparently
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            # Performing constrained optmization cvxpy
            x_cvx = cp.Variable(h_mtx.shape[1], complex = True) # create x variable
            # Create the problem
            epsilon = 10**(-(snr-headroom)/10)
            problem = cp.Problem(cp.Minimize(cp.norm2(x_cvx)**2),
                [cp.pnorm(pm - cp.matmul(H, x_cvx), p=2) <= epsilon])
            problem.solve(solver=cp.SCS, verbose=False)
            self.pk[:,jf] = x_cvx.value
            bar.update(1)
        bar.close()

    def pk_cs(self, snr=30, headroom = 0):
        """ Wave number spectrum estimation using constrained optimization

        Estimate the wave number spectrum using constrained optimization. The problem
        solved is min(||x||_1), subjected to ||Ax - b||_2 < e.

        This method is an adaptation of DTU methods, implemented in:
        Efren Fernandez-Grande. Compressive sensing with a spherical microphone array,
        J Acoust Soc Am (EL). 2016 139(2):EL45-EL49. doi: 10.1121/1.4942546 

        The inversion steps are: (i) - Get the scaled version of the propagating directions;
        (ii) - form the sensing matrix; (iii) - compute the inverse problem.

        Parameters
        ----------
        snr : float
            Signal to noise ratio in the simulation
        headroom : float
            Apply some headroom to the noise level to compute "e". 
        """
        # Initialize
        import cvxpy as cvx
        self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=np.csingle)
        # loop over frequencies
        bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Constrained Optim.')
        # print(self.pk.shape)
        for jf, k0 in enumerate(self.controls.k0):
            # get the scaled version of the propagating directions
            k_vec = k0 * self.dir
            # Form the sensing matrix
            h_mtx = np.exp(-1j*self.receivers.coord @ k_vec.T)
            H = h_mtx.astype(complex)
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            epsilon = 10**(-(snr-headroom)/10)
            # x_cvx = lc.cvx_solver_c(H, pm, epsilon, l_norm = 2)
            # Performing constrained optmization cvxpy
            x_cvx = cvx.Variable(h_mtx.shape[1], complex = True)
            # Create the problem
            epsilon = 10**(-(snr-headroom)/10)
            objective = cvx.Minimize(cvx.pnorm(x_cvx, p=1))
            constraints = [cvx.pnorm(pm - cvx.matmul(H, x_cvx), p=2) <= epsilon]#[H*x == pm]
            # Create the problem and solve
            problem = cvx.Problem(objective, constraints)
            problem.solve(solver=cvx.SCS, verbose=False) 
            #problem.solve() 
            self.pk[:,jf] = x_cvx.value
            bar.update(1)
        bar.close()
        return self.pk

    def pk_oct_interpolate(self, nband = 3):
        """ Interpolate wavenumber spectrum over an fractional octave bands

        Interpolates the wavenumber spectrum on 1/3 octave bands. Useful
        when doing diffuse field measurements. Based on:

            Mélanie Nolan. Estimation of angle-dependent absorption coefficients 
            from spatially distributed in situ measurements , J Acoust Soc Am (EL).
            2019 147(2):EL119-EL124. doi: 10.1121/10.0000716

        Parameters
        ----------
        nbands : int
            Fractional octave bands. Default is 3 for 1/3 octave bands
        """
        # Set flag to true
        self.flag_oct_interp = True
        # Find the fractional octave bands
        self.freq_oct, flower, fupper = octave_freq(self.controls.freq, nband = nband)
        # initialize
        self.pk_oct = np.zeros((self.n_waves, len(self.freq_oct)), dtype=complex)
        # octave avg each direction
        for jdir in np.arange(0, self.n_waves):
            self.pk_oct[jdir,:] = octave_avg(self.controls.freq, self.pk[jdir, :],
                self.freq_oct, flower, fupper)

    def reconstruct_p(self, receivers):
        """ Reconstruct sound pressure at an array of receivers
        
        Parameters
        ----------
        receivers : receiver object
            receiver object - where to reconstruct
        """
        # Initialize
        p_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        # Loop over frequency
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing pressure field...')
        for jf, k0 in enumerate(self.controls.k0):
            # get sensing matrix
            h_mtx = self.hmtx_p(k0, receivers.coord, self.dir)
            p_recon[:,jf] = h_mtx @ self.pk[:,jf]
            bar.update(1)
        bar.close()
        return p_recon
        
    def reconstruct_u(self, receivers, direction = 2):
        """ Reconstruct particle velocity at an array of receivers
        
        Parameters
        ----------
        receivers : receiver object
            receiver object - where to reconstruct
        direction : int
            0 (for x direction), 1 (for y direction), 2 (for z direction)
        """
        # Initialize
        u_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        # Loop over frequency
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing velocity field...')
        for jf, k0 in enumerate(self.controls.k0):
            # get sensing matrix
            grad_h_mtx = self.hmtx_u(k0, receivers.coord, self.dir, direction = direction)
            u_recon[:,jf] = grad_h_mtx @ self.pk[:,jf]
            bar.update(1)
        bar.close()
        return u_recon
    
    def reconstruct_zs(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, theta = [0], avgZs = True):
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
        # grid at surface
        grid = Receiver()
        grid.planar_array(x_len = Lx, y_len = Ly, zr = 0.0, n_x = n_x, n_y = n_x)
        p_recon = self.reconstruct_p(grid)
        uz_recon = self.reconstruct_u(grid, direction = 2)
        Zs_pt = -np.divide(p_recon, uz_recon)
        self.Zs = np.mean(Zs_pt, axis = 0)
        self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        for jthe, the in enumerate(theta):
            Vp = np.divide((self.Zs  * np.cos(the) - 1),\
                (self.Zs * np.cos(the) + 1))
            self.alpha[jthe,:] = 1 - (np.abs(Vp))**2
        return self.alpha
    
    def reconstruct_zs_theta(self, Lx = 0.1, Ly = 0.1, n_x = 21, n_y = 21, avgZs = True):
        """ Reconstruct the angle-dependent surface impedance and estimate the absorption

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
        # grid at surface
        grid = Receiver()
        grid.planar_array(x_len = Lx, y_len = Ly, zr = 0.0, n_x = n_x, n_y = n_x)
        # Sphere angles
        theta_deg, theta_deg_unique = self.sphere_elev_angles()
        # init variables
        self.Zs = np.zeros((len(theta_deg_unique), len(self.controls.k0)), dtype = complex)
        self.alpha = np.zeros((len(theta_deg_unique), len(self.controls.k0)))
        # loog angles and freq
        bar = tqdm(total = len(self.controls.k0) * len(theta_deg_unique), 
                   desc = 'Reconstructing angle dependent Zs...')
        for jf, k0 in enumerate(self.controls.k0):
            for jthe, the in enumerate(theta_deg_unique):
                # Get sphere indexes of the angle
                idthe = self.select_sphere_idx(theta_deg, theta_deg_unique[jthe])
                # Get reconstruction matrices
                h_mtx = self.hmtx_p(k0, grid.coord, self.dir[idthe,:])
                grad_h_mtx = self.hmtx_u(k0, grid.coord, self.dir[idthe,:], direction = 2)
                # Pressure and z-velocity
                p_recon = h_mtx @ self.pk[idthe,jf]
                uz_recon = grad_h_mtx @ self.pk[idthe,jf]
                Zs_pt = -np.divide(p_recon, uz_recon)
                self.Zs[jthe, jf] = np.mean(Zs_pt, axis = 0)
                Vp = np.divide((self.Zs[jthe, jf]  * np.cos(np.deg2rad(the)) - 1),\
                    (self.Zs[jthe, jf] * np.cos(np.deg2rad(the)) + 1))
                self.alpha[jthe, jf] = 1 - (np.abs(Vp))**2
                bar.update(1)
            
        bar.close()
        return self.alpha
    
    def sphere_elev_angles(self):
        """ Get sphere angles
        
        seems like the bottom of the sphere is the reflected part and 0 deg.
        """
        r, theta, phi = cart2sph(self.dir[:,0], self.dir[:,1], self.dir[:,2])
        theta_deg = np.round(np.rad2deg(theta), decimals = 1) + 90.0
        theta_deg_unique = np.unique(theta_deg)
        return theta_deg, theta_deg_unique[theta_deg_unique <= 90]

    def select_sphere_idx(self, theta_deg, theta_deg_val = 0):
        """ Get sphere indices for angle dependent reconstruction
        """
        idthe = np.where((theta_deg == theta_deg_val) | (theta_deg == 180 - theta_deg_val))
        return idthe[0]
        
        
        # p_recon = self.reconstruct_p(grid)
        # uz_recon = self.reconstruct_u(grid, direction = 2)
        # Zs_pt = -np.divide(p_recon, uz_recon)
        # self.Zs = np.mean(Zs_pt, axis = 0)
        # self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        # for jthe, the in enumerate(theta):
        #     Vp = np.divide((self.Zs  * np.cos(the) - 1),\
        #         (self.Zs * np.cos(the) + 1))
        #     self.alpha[jthe,:] = 1 - (np.abs(Vp))**2
        # return self.alpha
    

    def reconstruct_pu(self, receivers, compute_uxy = True):
        """ Reconstruct the sound pressure and particle velocity at a receiver object

        Reconstruct the pressure and particle velocity at a set of desired field points.
        This can be used on impedance estimation or to plot spatial maps of pressure,
        velocity, intensity.

        The steps are: (i) - Get the scaled version of the propagating directions;
        (ii) - form the new sensing matrix; (iii) - compute p and u.

        Parameters
        ----------
        receivers : object (Receiver)
            contains a set of field points at which to reconstruct
        compute_uxy : bool
            Whether to compute x and y components of particle velocity or not (Default is False)
        """
        # Initialize
        self.p_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        self.uz_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        if compute_uxy:
            self.ux_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
            self.uy_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
        # Loop over frequency
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing sound field...')
        for jf, k0 in enumerate(self.controls.k0):
            # get the scaled version of the propagating directions
            k_p = k0 * self.dir
            # Form the new sensing matrix
            h_mtx = np.exp(-1j*receivers.coord @ k_p.T)
            # compute P and U
            self.p_recon[:,jf] = h_mtx @ self.pk[:,jf]
            self.uz_recon[:,jf] = -((np.divide(k_p[:,2], k0)) * h_mtx) @ self.pk[:,jf]
            if compute_uxy:
                self.ux_recon[:,jf] = -((np.divide(k_p[:,0], k0)) * h_mtx) @ self.pk[:,jf]
                self.uy_recon[:,jf] = -((np.divide(k_p[:,1], k0)) * h_mtx) @ self.pk[:,jf]
            bar.update(1)
        bar.close()
        
    

    def pk_interpolate(self, npts=100):
        """ Interpolate the wave number spectrum on a finer regular grid.

        Also based on:
            Mélanie Nolan. Estimation of angle-dependent absorption coefficients
            from spatially distributed in situ measurements , J Acoust Soc Am (EL).
            2019 147(2):EL119-EL124. doi: 10.1121/10.0000716

        Parameters
        ----------
        npts : int
            Number of points on thehta and phi axis. The resulting interpolation grid
            will be of size 2*npts+1 x npts+1
        """
        # Recover the actual measured points
        _, theta, phi = cart2sph(self.dir[:,0], self.dir[:,1], self.dir[:,2])
        thetaphi_pts = np.transpose(np.array([phi, theta]))
        # Create a grid to interpolate on
        nphi = int(2*(npts+1))
        ntheta = int(npts+1)
        new_phi = np.linspace(-np.pi, np.pi, nphi)
        new_theta = np.linspace(-np.pi/2, np.pi/2, ntheta)#(0, np.pi, nn)
        self.grid_phi, self.grid_theta = np.meshgrid(new_phi, new_theta)
        # interpolate
        from scipy.interpolate import griddata
        self.grid_pk = []
        bar = tqdm(total = len(self.controls.k0), desc = 'Interpolating the grid for P(k)')

        if self.flag_oct_interp:
            for jf, f_oct in enumerate(self.freq_oct):
                # Cubic with griddata
                self.grid_pk.append(griddata(thetaphi_pts, self.pk_oct[:,jf],
                    (self.grid_phi, self.grid_theta), method='cubic',
                    fill_value=np.finfo(float).eps, rescale=False))
        else:
            for jf, k0 in enumerate(self.controls.k0):
                # Cubic with griddata 
                self.grid_pk.append(griddata(thetaphi_pts, np.abs(self.pk[:,jf]),
                    (self.grid_phi, self.grid_theta), method='cubic',
                    fill_value=np.finfo(float).eps, rescale=False))
                bar.update(1)
        bar.close()

    def plot_pk_sphere(self, freq = 1000, db = False, dinrange = 12,
        save = False, name='', travel = True):
        """ plot the magnitude of P(k) as a scatter plot of propagating waves

        Plot the magnitude of the wave number spectrum as a scatter plot of
        propagating  waves. It is a normalized version of the magnitude, either between
        0 and 1 or between -dinrange and 0. The maps are ploted as color as function
        of phi and theta.

        Parameters
        ----------
            freq : float
                Which frequency you want to see. If the calculated spectrum does not contain it
                we plot the closest frequency before the target.
            db : bool
                Whether to plot in linear scale (default) or decibel scale.
            dinrange : float
                You can specify a dinamic range for the decibel scale. It will not affect the
                linear scale.
            save : bool
                Whether to save or not the figure. PDF file with simple standard name
            name : str
                Name of the figure file #FixMe
            travel : bool
                Whether to plot travel direction or arrival direction. Default is True
        """
        id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        fig = plt.figure()
        ax = plt.axes(projection ="3d")
        vmin = 0
        vmax = 1
        if db:
            color_par = 20*np.log10(np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f])))
            id_outofrange = np.where(color_par < -dinrange)
            color_par[id_outofrange] = -dinrange
            vmin = -dinrange
            vmax = 0
        else:
            color_par = np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f]))
        if travel:
            p=ax.scatter(self.dir[:,0], self.dir[:,1], -self.dir[:,2], c = color_par,
                         vmin = vmin, vmax = vmax)
        else:
            p=ax.scatter(self.dir[:,0], self.dir[:,1], self.dir[:,2], c = color_par,
                         vmin = vmin, vmax = vmax)
        fig.colorbar(p)
        ax.set_xlabel(r'$k_x$ axis')
        ax.set_ylabel(r'$k_y$ axis')
        ax.set_zlabel(r'$k_z$ axis')
        plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - ' + name)
        plt.tight_layout()
        if save:
            filename = 'data/colormaps/cmat_' + str(int(freq)) + 'Hz_' + name
            plt.savefig(fname = filename, format='pdf')

    def plot_pk_map(self, freq = 1000, db = False, dinrange = 40, phase = False,
        save = False, name='', path = '', fname='', color_code = 'viridis'):
        """ Plot wave number spectrum  - propagating only (vs. phi and theta)

        Plot the magnitude of the wave number spectrum as a map of
        propagating waves. Assumes the map has been interpolated into
        a regular grid of azimuth (phi) and elevation (theta) angle.
        It is a normalized version of the magnitude, either between
        0 and 1 or between -dinrange and 0. The maps are ploted as color as function
        of phi and theta.

        Parameters
        ----------
            freq : float
                Which frequency you want to see. If the calculated spectrum does not contain it
                we plot the closest frequency before the target.
            db : bool
                Whether to plot in linear scale (default) or decibel scale.
            dinrange : float
                You can specify a dinamic range for the decibel scale. It will not affect the
                linear scale.
            save : bool
                Whether to save or not the figure. PDF file with simple standard name
            name : str
                Name of the figure file #FixMe
            path : str
                Path to save the figure file
            fname : str
                File name to save the figure file
            color_code : str
                Can be anything that matplotlib supports. Some recomendations given below:
                'viridis' (default) - Perceptually Uniform Sequential
                'Greys' - White (cold) to black (hot)
                'seismic' - Blue (cold) to red (hot) with a white in the middle
        """
        if self.flag_oct_interp:
            id_f = np.where(self.freq_oct <= freq)
        else:
            id_f = np.where(self.controls.freq <= freq)
        id_f = id_f[0][-1]
        fig = plt.figure()
        if db:
            color_par = 20*np.log10(np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f])))
            color_range = np.linspace(-dinrange, 0, dinrange+1)
        else:
            color_par = np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f]))
            color_range = np.linspace(0, 1, 21)
        p=plt.contourf(np.rad2deg(self.grid_phi), np.rad2deg(self.grid_theta), color_par,
            color_range, extend='both', cmap = color_code)
        fig.colorbar(p)
        plt.xlabel(r'$\phi$ (azimuth) [deg]')
        plt.ylabel(r'$\theta$ (elevation) [deg]')
        if self.flag_oct_interp:
            plt.title('|P(k)| at ' + str(self.freq_oct[id_f]) + 'Hz - '+ name)
        else:
            plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - P decomp. '+ name)
        plt.tight_layout()
        if save:
            filename = path + fname + '_' + str(int(freq)) + 'Hz'
            plt.savefig(fname = filename, format='png')

    def save(self, filename = 'array_zest', path = '/home/eric/dev/insitu/data/zs_recovery/'):
        """ To save the decomposition object as pickle

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'array_zest', path = '/home/eric/dev/insitu/data/zs_recovery/'):
        """ To load the decomposition object as pickle

        You can instantiate an empty object of the class and load a saved one.
        It will overwrite the empty object.

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        lpath_filename = path + filename + '.pkl'
        f = open(lpath_filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
#### Auxiliary functions
def filter_evan(k0, kx_e, ky_e, plot=False):
    """ Filter the propagating waves

    This auxiliary function will exclude all propagating wave numbers from
    the evanescent wave numbers. This is necessary because we are creating
    an arbitrary number of wave numbers (to be used in the decomposition).
    """
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


