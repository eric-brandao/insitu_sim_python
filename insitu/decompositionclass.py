import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
# from insitu.controlsair import load_cfg
# import scipy.integrate as integrate
# import scipy as spy
from sklearn.linear_model import Ridge
import time
from tqdm import tqdm
import sys
# from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
#from tqdm._tqdm_notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import cvxpy as cp
# from scipy import linalg # for svd
# from scipy import signal
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
        self.controls = controls
        self.material = material
        self.receivers = receivers
        self.pres_s = p_mtx
        self.flag_oct_interp = False

    def wavenum_dir(self, n_waves = 642, plot = False, halfsphere = False):
        """ Create the propagating wave number directions

        The propagating wave number directions are uniformily distributed
        over the surface of an hemisphere (which will have radius k [rad/m] during the
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
        directions = RayInitialDirections()
        self.dir, self.n_waves = directions.isotropic_rays(Nrays = int(n_waves))
        if halfsphere:
            _, theta, _ = cart2sph(self.dir[:,0],self.dir[:,1],self.dir[:,2])
            theta_inc_id, theta_ref_id = get_hemispheres(theta)
            _, reflected_dir = get_inc_ref_dirs(self.dir, theta_inc_id, theta_ref_id)
            self.dir = reflected_dir
            self.n_waves = len(self.dir)
        print('The number of created waves is: {}'.format(self.n_waves))
        if plot:
            directions.plot_points()

    def pk_tikhonov(self, method = 'direct', plot_l = False):
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
        self.cond_num = np.zeros(len(self.controls.k0))
        # loop over frequencies
        for jf, k0 in enumerate(self.controls.k0):
            # get the scaled version of the propagating directions
            k_vec = k0 * self.dir
            # Form the sensing matrix
            h_mtx = np.exp(-1j*self.receivers.coord @ k_vec.T)
            self.cond_num[jf] = np.linalg.cond(h_mtx)
            # measured data
            pm = self.pres_s[:,jf].astype(complex)
            # compute SVD of the sensing matix
            u, sig, v = csvd(h_mtx)
            # compute the regularization parameter (L-curve)
            lambd_value = l_cuve(u, sig, pm, plotit=False)
            ## Choosing the method to find the P(k)
            if method == 'scipy':
                x = lsqr(h_mtx, self.pres_s[:,jf], damp=lambd_value)
                self.pk[:,jf] = x[0]
            elif method == 'direct':
                Hm = np.matrix(h_mtx)
                self.pk[:,jf] = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
            elif method == 'Ridge':
                # Form a real H2 matrix and p2 measurement
                H2 = np.vstack((np.hstack((h_mtx.real, -h_mtx.imag)),
                    np.hstack((h_mtx.imag, h_mtx.real))))
                p2 = np.vstack((pm.real,pm.imag)).flatten()
                # form Ridge regressor using the regularization from L-curve
                regressor = Ridge(alpha=lambd_value, fit_intercept = False, solver = 'svd')
                x2 = regressor.fit(H2, p2).coef_
                self.pk[:,jf] = x2[:h_mtx.shape[1]]+1j*x2[h_mtx.shape[1]:]
                # # separate propagating from evanescent
                # self.pk[:,jf] = x[0:self.n_waves]
            #### Performing the Tikhonov inversion with cvxpy #########################
            else:
                H = h_mtx.astype(complex)
                x = cp.Variable(h_mtx.shape[1], complex = True)
                lambd = cp.Parameter(nonneg=True)
                lambd.value = lambd_value[0]
                # Create the problem and solve
                problem = cp.Problem(cp.Minimize(objective_fn(H, pm, x, lambd)))
                problem.solve(solver=cp.SCS, verbose=False) # Fast but gives some warnings
                self.pk[:,jf] = x.value
            bar.update(1)
        bar.close()
        return self.pk

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
            # Performing constrained optmization cvxpy
            x_cvx = cp.Variable(h_mtx.shape[1], complex = True)
            # Create the problem
            epsilon = 10**(-(snr-headroom)/10)
            objective = cp.Minimize(cp.pnorm(x_cvx, p=1))
            constraints = [cp.pnorm(pm - cp.matmul(H, x_cvx), p=2) <= epsilon]#[H*x == pm]
            # Create the problem and solve
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=True) 
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
        fig.canvas.set_window_title('Scatter plot of |P(k)| for freq {} Hz'.format(self.controls.freq[id_f]))
        ax = fig.gca(projection='3d')
        if db:
            color_par = 20*np.log10(np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f])))
            id_outofrange = np.where(color_par < -dinrange)
            color_par[id_outofrange] = -dinrange
        else:
            color_par = np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f]))
        if travel:
            p=ax.scatter(self.dir[:,0], self.dir[:,1], -self.dir[:,2], c = color_par)
        else:
            p=ax.scatter(self.dir[:,0], self.dir[:,1], self.dir[:,2], c = color_par)
        fig.colorbar(p)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
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
        fig.canvas.set_window_title('Interpolated map of |P(k)| for freq {} Hz'.format(self.controls.freq[id_f]))
        if db:
            color_par = 20*np.log10(np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f])))
            color_range = np.linspace(-dinrange, 0, dinrange+1)
        else:
            color_par = np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f]))
            color_range = np.linspace(0, 1, 21)
        p=plt.contourf(np.rad2deg(self.grid_phi), 90-np.rad2deg(self.grid_theta), color_par,
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


# ###################################### Copy of original implementation with evenescent waves ########
# class Decomposition(object):
#     '''
#     Decomposition class for array processing
#     '''
#     def __init__(self, p_mtx = None, controls = None, material = None, receivers = None):
#         '''
#         Init - we first retrive general data, then we process some receiver data
#         '''
#         # self.pres_s = sim_field.pres_s[source_num] #FixMe
#         # self.air = sim_field.air
#         self.controls = controls
#         self.material = material
#         # self.sources = sim_field.sources
#         self.receivers = receivers
#         self.pres_s = p_mtx
#         self.flag_oct_interp = False

#     def wavenum_dir(self, n_waves = 642, plot = False, halfsphere = False):
#         '''
#         This method is used to create wave number directions uniformily distributed over the surface of a sphere.
#         The directions of propagation that later will bevome wave-number vectors.
#         The directions of propagation are calculated with the triangulation of an icosahedron used previously
#         in the generation of omnidirectional rays (originally implemented in a ray tracing algorithm).
#         Inputs:
#             n_waves - The number of directions (wave-directions) to generate (integer)
#             plot - whether you plot or not the wave points in space (bool)
#         '''
#         directions = RayInitialDirections()
#         self.dir, self.n_waves = directions.isotropic_rays(Nrays = int(n_waves))
#         if halfsphere:
#             r, theta, phi = cart2sph(self.dir[:,0],self.dir[:,1],self.dir[:,2])
#             theta_inc_id, theta_ref_id = get_hemispheres(theta)
#             incident_dir, reflected_dir = get_inc_ref_dirs(self.dir, theta_inc_id, theta_ref_id)
#             self.dir = reflected_dir
#             self.n_waves = len(self.dir)
#         print('The number of created waves is: {}'.format(self.n_waves))
#         if plot:
#             directions.plot_points()

#     def wavenum_direv(self, n_waves = 20, plot = False, freq=1000):
#         '''
#         This method is used to create wave number directions that will be used to decompose the evanescent part 
#         of the wave field. This part will be the kx and ky componentes. They only depend on the array size and 
#         on the microphone spacing. When performing the decomposition, kz will depend on the calculated kx and ky.
#         Furthermore, the evanescent part will be separated from the propagating part, so that they can be easily
#         filtered out.
#         Inputs:
#             n_waves - The number of directions (wave-directions) to generate (integer)
#             plot - whether you plot or not the wave points in space (bool)
#             freq - to have a notion of the radiation circle when plotting kx and ky
#         '''
#         # Figure out the size of the array in x and y directions
#         # Figure out the spacing between the microphones in x and y directions
#         # Create kx ad ky (this includes prpagating waves - we'll deal with them later)
#         # kx = np.arange(start = -np.pi/self.receivers.ax,
#         #     stop = np.pi/self.receivers.ax+2*np.pi/self.receivers.x_len, step = 2*np.pi/self.receivers.x_len)
#         # ky = np.arange(start = -np.pi/self.receivers.ay,
#         #     stop = np.pi/self.receivers.ay+2*np.pi/self.receivers.y_len, step = 2*np.pi/self.receivers.y_len)
#         self.n_evan = n_waves
#         #### With linspace and n_waves
#         kx = np.linspace(start = -np.pi/self.receivers.ax,
#             stop = np.pi/self.receivers.ax, num = n_waves)
#         ky = np.linspace(start = -np.pi/self.receivers.ay,
#             stop = np.pi/self.receivers.ay, num = n_waves)

#         self.kx_grid, self.ky_grid = np.meshgrid(kx,ky)
#         self.kx_e = self.kx_grid.flatten()
#         self.ky_e = self.ky_grid.flatten()
#         if plot:
#             k0 = 2*np.pi*freq / self.controls.c0
#             fig = plt.figure()
#             fig.canvas.set_window_title('Non filtered evanescent waves')
#             plt.plot(self.kx_e, self.ky_e, 'o')
#             plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
#                 k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
#             plt.xlabel('kx')
#             plt.ylabel('ky')
#             plt.show()

#     def pk_tikhonov(self, lambd_value = [], method = 'scipy'):
#         '''
#         Method to estimate wave number spectrum based on the Tikhonov matrix inversion technique.
#         Inputs:
#             lambd_value: Value of the regularization parameter. The user can specify that.
#                 If it comes empty, then we use L-curve to determine the optmal value.
#             method: string defining the method to be used on finding the correct P(k).
#                 It can be:
#                     (1) - 'scipy': using scipy.linalg.lsqr
#                     (2) - 'direct': via x= (Hm^H) * ((Hm * Hm^H + lambd_value * I)^-1) * pm
#                     (3) - else: via cvxpy
#         '''
#         # Bars
#         self.decomp_type = 'Tikhonov (transparent array)'
#         # bar = ChargingBar('Calculating Tikhonov inversion...', max=len(self.controls.k0), suffix='%(percent)d%%')
#         bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Tikhonov inversion...')
#         # Initialize p(k) as a matrix of n_waves x n_freq
#         self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=complex)
#         self.cond_num = np.zeros(len(self.controls.k0))
#         # loop over frequencies
#         for jf, k0 in enumerate(self.controls.k0):
#             # update_progress(jf/len(self.controls.k0))
#             k_vec = k0 * self.dir
#             # Form H matrix
#             h_mtx = np.exp(-1j*self.receivers.coord @ k_vec.T)
#             self.cond_num[jf] = np.linalg.cond(h_mtx)
#             # measured data
#             pm = self.pres_s[:,jf].astype(complex)
#             # finding the optimal lambda value if the parameter comes empty.
#             # if not we use the user supplied value.
#             # if not lambd_value:
#             u, sig, v = csvd(h_mtx)
#             lambd_value = l_cuve(u, sig, pm, plotit=False)
#             ## Choosing the method to find the P(k)
#             if method == 'scipy':
#                 x = lsqr(h_mtx, self.pres_s[:,jf], damp=lambd_value)
#                 self.pk[:,jf] = x[0]
#             elif method == 'direct':
#                 Hm = np.matrix(h_mtx)
#                 self.pk[:,jf] = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
#             elif method == 'Ridge':
#                 # Form a real H2 matrix and p2 measurement
#                 H2 = np.vstack((np.hstack((h_mtx.real, -h_mtx.imag)),
#                     np.hstack((h_mtx.imag, h_mtx.real))))
#                 p2 = np.vstack((pm.real,pm.imag)).flatten()
#                 # form Ridge regressor using the regularization from L-curve
#                 regressor = Ridge(alpha=lambd_value, fit_intercept = False, solver = 'svd')
#                 x2 = regressor.fit(H2, p2).coef_
#                 self.pk[:,jf] = x2[:h_mtx.shape[1]]+1j*x2[h_mtx.shape[1]:]
#                 # # separate propagating from evanescent
#                 # self.pk[:,jf] = x[0:self.n_waves]
#             #### Performing the Tikhonov inversion with cvxpy #########################
#             else:
#                 H = h_mtx.astype(complex)
#                 x = cp.Variable(h_mtx.shape[1], complex = True)
#                 lambd = cp.Parameter(nonneg=True)
#                 lambd.value = lambd_value[0]
#                 # Create the problem and solve
#                 problem = cp.Problem(cp.Minimize(objective_fn(H, pm, x, lambd)))
#                 # problem.is_dcp(dpp = True)
#                 # problem.solve()
#                 problem.solve(solver=cp.SCS, verbose=False) # Fast but gives some warnings
#                 # problem.solve(solver=cp.ECOS, abstol=1e-3) # slow
#                 # problem.solve(solver=cp.ECOS_BB) # slow
#                 # problem.solve(solver=cp.NAG) # not installed
#                 # problem.solve(solver=cp.CPLEX) # not installed
#                 # problem.solve(solver=cp.CBC)  # not installed
#                 # problem.solve(solver=cp.CVXOPT) # not installed
#                 # problem.solve(solver=cp.MOSEK) # not installed
#                 # problem.solve(solver=cp.OSQP) # did not work
#                 self.pk[:,jf] = x.value
#             # bar.next()
#             bar.update(1)
#         # bar.finish()
#         bar.close()
#         return self.pk

#     def pk_tikhonov_ev(self, method = 'scipy', include_evan = False):
#         '''
#         Method to estimate wave number spectrum based on the Tikhonov matrix inversion technique.
#         This version includes the evanescent waves
#         Inputs:
#             lambd_value: Value of the regularization parameter. The user can specify that.
#                 If it comes empty, then we use L-curve to determine the optmal value.
#             method: string defining the method to be used on finding the correct P(k).
#                 It can be:
#                     (1) - 'scipy': using scipy.linalg.lsqr
#                     (2) - 'direct': via x= (Hm^H) * ((Hm * Hm^H + lambd_value * I)^-1) * pm
#                     (3) - else: via cvxpy
#         '''
#         self.decomp_type = 'Tikhonov (transparent array) w/ evanescent waves'
#         # loop over frequencies
#         # bar = ChargingBar('Calculating Tikhonov inversion (with evanescent waves)...', max=len(self.controls.k0), suffix='%(percent)d%%')
#         if include_evan:
#             bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Tikhonov inversion (with evanescent waves)...')
#         else:
#             bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Tikhonov inversion (without evanescent waves)...')
#         self.cond_num = np.zeros(len(self.controls.k0))
#         self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=complex)
#         self.kx_ef = [] # Filtered version
#         self.ky_ef = [] # Filtered version
#         self.pk_ev = []
#         for jf, k0 in enumerate(self.controls.k0):
#             # print('freq {} Hz'.format(self.controls.freq[jf]))
#             # update_progress(jf/len(self.controls.k0))
#             # First, we form the propagating wave-numbers ans sensing matrix
#             k_vec = k0 * self.dir
#             h_p = np.exp(-1j*self.receivers.coord @ k_vec.T)
#             if include_evan:
#                 # Then, we have to form the remaining evanescent wave-numbers and evanescent sensing matrix
#                 kx_e, ky_e, n_e = filter_evan(k0, self.kx_e, self.ky_e, plot=False)
#                 # print('Number of evanescent is {}'.format(self.n_evan))
#                 kz_e = np.sqrt(k0**2 - kx_e**2 - ky_e**2+0j)
#                 k_ev = np.array([kx_e, ky_e, kz_e]).T
#                 # Fkz_ev = np.sqrt(k0/np.abs(k_ev[:,2]))
#                 # h_ev = Fkz_ev * np.exp(1j*self.receivers.coord @ k_ev.T)
#                 h_ev = np.exp(-1j*self.receivers.coord @ k_ev.T)
#                 self.kx_ef.append(kx_e)
#                 self.ky_ef.append(ky_e)
#                 # Form H matrix
#                 h_mtx = np.hstack((h_p, h_ev))
#             else:
#                 h_mtx = h_p
#             self.cond_num[jf] = np.linalg.cond(h_mtx)
#             # measured data
#             pm = self.pres_s[:,jf].astype(complex)
#             # finding the optimal lambda value if the parameter comes empty.
#             # if not we use the user supplied value.
#             # if not lambd_value:
#             u, sig, v = csvd(h_mtx)
#             lambd_value = l_cuve(u, sig, pm, plotit=False)
#             # ## Choosing the method to find the P(k)
#             # # print('reg par: {}'.format(lambd_value))
#             if method == 'scipy':
#                 from scipy.sparse.linalg import lsqr, lsmr
#                 x = lsqr(h_mtx, self.pres_s[:,jf], damp=lambd_value)[0]
#                 # self.pk[:,jf] = x[0][0:self.n_waves]
#                 # self.pk_ev.append(x[0][self.n_waves:])
#             elif method == 'direct':
#                 Hm = np.matrix(h_mtx)
#                 x = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
#                 # print(x.shape)
#                 # self.pk[:,jf] = x[0:self.n_waves]
#                 # self.pk_ev.append(x[self.n_waves:])
#             elif method == 'Ridge':
#                 # Form a real H2 matrix and p2 measurement
#                 H2 = np.vstack((np.hstack((h_mtx.real, -h_mtx.imag)),
#                     np.hstack((h_mtx.imag, h_mtx.real))))
#                 p2 = np.vstack((pm.real,pm.imag)).flatten()
#                 # u, sig, v = csvd(H2)
#                 # lambd_value = l_cuve(u, sig, p2, plotit=False)
#                 # form Ridge regressor using the regularization from L-curve
#                 regressor = Ridge(alpha=lambd_value, fit_intercept = False, solver = 'svd')
#                 x2 = regressor.fit(H2, p2).coef_
#                 x = x2[:h_mtx.shape[1]]+1j*x2[h_mtx.shape[1]:]
#                 # print(x.shape)
#                 # separate propagating from evanescent
#                 # self.pk[:,jf] = x[0:self.n_waves]
#                 # self.pk_ev.append(x[self.n_waves:])
#             elif method == 'tikhonov':
#                 u, sig, v = csvd(h_mtx)
#                 x = tikhonov(u, sig, v, pm, lambd_value)
#                 # self.pk[:,jf] = x[0:self.n_waves]
#                 # self.pk_ev.append(x[self.n_waves:])
#                 # print(x.shape)
#             # #### Performing the Tikhonov inversion with cvxpy #########################
#             else:
#                 H = h_mtx.astype(complex)
#                 x_cvx = cp.Variable(h_mtx.shape[1], complex = True)
#                 lambd = cp.Parameter(nonneg=True)
#                 lambd.value = lambd_value[0]
#                 # Create the problem and solve
#                 problem = cp.Problem(cp.Minimize(objective_fn(H, pm, x_cvx, lambd)))
#                 # problem.solve()
#                 problem.solve(solver=cp.SCS, verbose=False) # Fast but gives some warnings
#                 # problem.solve(solver=cp.ECOS, abstol=1e-3) # slow
#                 # problem.solve(solver=cp.ECOS_BB) # slow
#                 # problem.solve(solver=cp.NAG) # not installed
#                 # problem.solve(solver=cp.CPLEX) # not installed
#                 # problem.solve(solver=cp.CBC)  # not installed
#                 # problem.solve(solver=cp.CVXOPT) # not installed
#                 # problem.solve(solver=cp.MOSEK) # not installed
#                 # problem.solve(solver=cp.OSQP) # did not work
#                 # self.pk[:,jf] = x.value[0:self.n_waves]
#                 # self.pk_ev.append(x.value[self.n_waves:])
#                 x = x_cvx.value
#             self.pk[:,jf] = x[0:self.n_waves]
#             if include_evan:
#                 self.pk_ev.append(x[self.n_waves:])
#             # bar.next()
#             bar.update(1)
#         # bar.finish()
#         bar.close()
#         # sys.stdout.write("]\n")
#         # return self.pk

#     def pk_constrained(self, snr=30, headroom = 0, include_evan = False):
#         '''
#         Method to estimate wave number spectrum based on constrained optimization matrix inversion technique.
#         Inputs:
#             epsilon - upper bound of noise floor vector
#         '''
#         # loop over frequencies
#         if include_evan:
#             bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Constrained Optim. (with evanescent waves)...')
#         else:
#             bar = tqdm(total = len(self.controls.k0), desc = 'Calculating Constrained Optim. (without evanescent waves)...')
#         self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=np.csingle)
#         # print(self.pk.shape)
#         self.kx_ef = [] # Filtered version
#         self.ky_ef = [] # Filtered version
#         self.pk_ev = []
#         for jf, k0 in enumerate(self.controls.k0):
#             k_vec = k0 * self.dir
#             h_p = np.exp(-1j*self.receivers.coord @ k_vec.T)
#             if include_evan:
#                 # Then, we have to form the remaining evanescent wave-numbers and evanescent sensing matrix
#                 kx_e, ky_e, n_e = filter_evan(k0, self.kx_e, self.ky_e, plot=False)
#                 # print('Number of evanescent is {}'.format(self.n_evan))
#                 kz_e = np.sqrt(k0**2 - kx_e**2 - ky_e**2+0j)
#                 k_ev = np.array([kx_e, ky_e, kz_e]).T
#                 # Fkz_ev = np.sqrt(k0/np.abs(k_ev[:,2]))
#                 # h_ev = Fkz_ev * np.exp(1j*self.receivers.coord @ k_ev.T)
#                 h_ev = np.exp(1j*self.receivers.coord @ k_ev.T)
#                 self.kx_ef.append(kx_e)
#                 self.ky_ef.append(ky_e)
#                 # Form H matrix
#                 h_mtx = np.hstack((h_p, h_ev))
#             else:
#                 h_mtx = h_p
#             # Form H matrix
#             # h_mtx = np.exp(1j*self.receivers.coord @ k_vec.T)
#             H = h_mtx.astype(complex) # cvxpy does not accept floats, apparently
#             # measured data
#             pm = self.pres_s[:,jf].astype(complex)
#             #### Performing the Tikhonov inversion with cvxpy #########################
#             x_cvx = cp.Variable(h_mtx.shape[1], complex = True) # create x variable
#             # Create the problem
#             epsilon = 10**(-(snr-headroom)/10)
#             problem = cp.Problem(cp.Minimize(cp.norm2(x_cvx)**2),
#                 [cp.pnorm(pm - cp.matmul(H, x_cvx), p=2) <= epsilon])
#             problem.solve(solver=cp.SCS, verbose=False)
#             # problem.solve(verbose=False)
#             x = x_cvx.value
#             self.pk[:,jf] = x[0:self.n_waves]
#             if include_evan:
#                 self.pk_ev.append(x[self.n_waves:])
#             # bar.next()
#             bar.update(1)
#         # bar.finish()
#         bar.close()
#         # return self.pk

#     def pk_cs(self, lambd_value = [], method = 'scipy'):
#         '''
#         Method to estimate wave number spectrum based on the l1 inversion technique.
#         This is supposed to give us a sparse solution for the sound field decomposition.
#         Inputs:
#             method: string defining the method to be used on finding the correct P(k).
#             It can be:
#                 (1) - 'scipy': using scipy.linalg.lsqr
#                 (2) - 'direct': via x= (Hm^H) * ((Hm * Hm^H + lambd_value * I)^-1) * pm
#                 (3) - else: via cvxpy
#         '''
#         # loop over frequencies
#         bar = ChargingBar('Calculating CS inversion...', max=len(self.controls.k0), suffix='%(percent)d%%')
#         self.pk = np.zeros((self.n_waves, len(self.controls.k0)), dtype=np.csingle)
#         # print(self.pk.shape)
#         for jf, k0 in enumerate(self.controls.k0):
#             # wave numbers
#             k_vec = k0 * self.dir
#             # Form H matrix
#             h_mtx = np.exp(-1j*self.receivers.coord @ k_vec.T)
#             # measured data
#             pm = self.pres_s[:,jf].astype(complex)
#             ## Choosing the method to find the P(k)
#             if method == 'scipy':
#                 # from scipy.sparse.linalg import lsqr, lsmr
#                 # x = lsqr(h_mtx, self.pres_s[:,jf], damp=np.sqrt(lambd_value))
#                 # self.pk[:,jf] = x[0]
#                 pass
#             elif method == 'direct':
#                 # Hm = np.matrix(h_mtx)
#                 # self.pk[:,jf] = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + lambd_value*np.identity(len(pm))) @ pm
#                 pass
#             # print('x values: {}'.format(x[0]))
#             #### Performing the Tikhonov inversion with cvxpy #########################
#             else:
#                 H = h_mtx.astype(complex)
#                 x = cp.Variable(h_mtx.shape[1], complex = True)
#                 objective = cp.Minimize(cp.pnorm(x, p=1))
#                 constraints = [H*x == pm]
#                 # Create the problem and solve
#                 problem = cp.Problem(objective, constraints)
#                 # problem.solve()
#                 # problem.solve(verbose=False) # Fast but gives some warnings
#                 problem.solve(solver=cp.SCS, verbose=True) # Fast but gives some warnings
#                 # problem.solve(solver=cp.ECOS, abstol=1e-3) # slow
#                 # problem.solve(solver=cp.ECOS_BB) # slow
#                 # problem.solve(solver=cp.NAG) # not installed
#                 # problem.solve(solver=cp.CPLEX) # not installed
#                 # problem.solve(solver=cp.CBC)  # not installed
#                 # problem.solve(solver=cp.CVXOPT) # not installed
#                 # problem.solve(solver=cp.MOSEK) # not installed
#                 # problem.solve(solver=cp.OSQP) # did not work
#                 self.pk[:,jf] = x.value
#             bar.next()
#         bar.finish()
#         return self.pk

#     def pk_oct_interpolate(self, nband = 3):
#         '''
#         method to interpolate pk over an octave or 1/3 ocatave band
#         '''
#         # Set flag to true
#         self.flag_oct_interp = True
#         self.freq_oct, flower, fupper = octave_freq(self.controls.freq, nband = nband)
#         self.pk_oct = np.zeros((self.n_waves, len(self.freq_oct)), dtype=complex)
#         # octave avg each direction
#         for jdir in np.arange(0, self.n_waves):
#             self.pk_oct[jdir,:] = octave_avg(self.controls.freq, self.pk[jdir, :], self.freq_oct, flower, fupper)

#     def tukey_win(self, start_cut = 1, end_cut = 2):
#         '''
#         A method to apply a Tukey window to the evanescent part of the
#         wavenumber spectrum
#         '''
#         kx = np.linspace(start = -np.pi/self.receivers.ax,
#             stop = np.pi/self.receivers.ax, num = self.n_evan)
#         ky = np.linspace(start = -np.pi/self.receivers.ay,
#             stop = np.pi/self.receivers.ay, num = self.n_evan)
#         bar = tqdm(total = len(self.controls.k0), desc = 'Applying window to evanescent waves')
#         for jf, k0 in enumerate(self.controls.k0):
#             # print('maximum kx is {0:.2f} bigger than k0 = {1:.1f}'.format(end_cut*k0, k0))
#             # Create 1D zeros at the end of transition band
#             nzeros_kx = len(np.where(np.abs(kx) > end_cut*k0)[0])
#             nzeros_ky = len(np.where(np.abs(ky) > end_cut*k0)[0])
#             # Create 1D Tukey windows
#             nwin_kx = len(kx)-nzeros_kx
#             n1s_kx = len(kx[np.abs(kx)<= start_cut*k0])
#             tukey_kx = signal.tukey(nwin_kx, alpha = (nwin_kx-n1s_kx)/nwin_kx)
#                 # alpha=-len(kx[kx<= start_cut*k0]))#len(kx[kx<= start_cut*k0])/(end_cut * len(kx)))
#             nwin_ky = len(ky)-nzeros_ky
#             n1s_ky = len(ky[np.abs(ky)<= start_cut*k0])
#             tukey_ky = signal.tukey(nwin_ky, alpha = (nwin_ky-n1s_ky)/nwin_ky)
#             tukey_kx = np.concatenate((np.zeros(int(np.floor(nzeros_kx/2))),
#                 tukey_kx, np.zeros(int(np.ceil(nzeros_kx/2)))))
#             tukey_ky = np.concatenate((np.zeros(int(np.floor(nzeros_ky/2))),
#                 tukey_ky, np.zeros(int(np.ceil(nzeros_ky/2)))))
#             # create 2D window
#             tukey_kx2, tukey_ky2 = np.meshgrid(tukey_kx,tukey_ky)
#             tukey_kxy = tukey_kx2*tukey_ky2 #np.sqrt(tukey_kx2**2 + tukey_ky2**2)
#             tukey_kxy_f = tukey_kxy.flatten()
#             # Exclude the radiation circle
#             ke_norm = (self.kx_e**2 + self.ky_e**2)**0.5
#             tukey_2dwin = tukey_kxy_f[ke_norm > k0]
#             # Apply the window
#             self.pk_ev[jf] = tukey_2dwin * self.pk_ev[jf]
#             ### debug plot
#             # color_par = np.abs(self.pk_ev[jf])
#             # import matplotlib.tri as tri
#             # x = self.kx_ef[jf]
#             # y = self.ky_ef[jf]
#             # triang = tri.Triangulation(x, y)
#             # triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
#             # y[triang.triangles].mean(axis=1)) < k0)
#             # fig = plt.figure()
#             # fig.canvas.set_window_title('Tukey window')
#             # p = plt.tricontourf(triang, color_par, levels = 40)
#             # fig.colorbar(p)
#             # # plt.plot(kx, tukey_kx, '--b', label = 'win on kx', linewidth = 3)
#             # # plt.plot(ky, tukey_ky, '-r', label = 'win on ky')
#             # # plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
#             # #     k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
#             # plt.xlabel('kx')
#             # plt.ylabel('win')
#             # # plt.legend()
#             # plt.show()
#             bar.update(1)
#         bar.close()

#     def tukey_win2(self, percentage = 50, start_cut = 1.0):
#         '''
#         A method to apply a Tukey window to the evanescent part of the
#         wavenumber spectrum
#         '''
#         kx = np.linspace(start = -np.pi/self.receivers.ax,
#             stop = np.pi/self.receivers.ax, num = self.n_evan)
#         ky = np.linspace(start = -np.pi/self.receivers.ay,
#             stop = np.pi/self.receivers.ay, num = self.n_evan)
#         bar = tqdm(total = len(self.controls.k0), desc = 'Applying window to evanescent waves')
#         for jf, k0 in enumerate(self.controls.k0):
#             # print('maximum kx is {0:.2f} bigger than k0 = {1:.1f}'.format(end_cut*k0, k0))
#             # delta_kxk0 = np.amax(kx) - k0
#             # # Number of points out of the radiation circle
#             # nout_kx = len(np.where(np.abs(kx) > k0)[0])
#             # nout_ky = len(np.where(np.abs(ky) > k0)[0])
#             # Number of zeros at the end of transition band
#             nzeros_kx = int((percentage/100)*len(kx))#int((percentage/100)*nout_kx)
#             nzeros_ky = int((percentage/100)*len(ky))#int((percentage/100)*nout_ky)
#             # Create 1D Tukey windows
#             nwin_kx = len(kx)-nzeros_kx
#             n1s_kx = len(kx[np.abs(kx)<= start_cut*k0])
#             if n1s_kx >= nwin_kx:
#                 n1s_kx = int(0.9*nwin_kx)
#             tukey_kx = signal.tukey(nwin_kx, alpha = (nwin_kx-n1s_kx)/nwin_kx)
#                 # alpha=-len(kx[kx<= start_cut*k0]))#len(kx[kx<= start_cut*k0])/(end_cut * len(kx)))
#             nwin_ky = len(ky)-nzeros_ky
#             n1s_ky = len(ky[np.abs(ky)<= start_cut*k0])
#             if n1s_ky >= nwin_ky:
#                 n1s_ky = int(0.9*nwin_ky)
#             tukey_ky = signal.tukey(nwin_ky, alpha = (nwin_ky-n1s_ky)/nwin_ky)
#             tukey_kx = np.concatenate((np.zeros(int(np.floor(nzeros_kx/2))),
#                 tukey_kx, np.zeros(int(np.ceil(nzeros_kx/2)))))
#             tukey_ky = np.concatenate((np.zeros(int(np.floor(nzeros_ky/2))),
#                 tukey_ky, np.zeros(int(np.ceil(nzeros_ky/2)))))
#             # create 2D window
#             tukey_kx2, tukey_ky2 = np.meshgrid(tukey_kx,tukey_ky)
#             tukey_kxy = tukey_kx2*tukey_ky2 #np.sqrt(tukey_kx2**2 + tukey_ky2**2)
#             tukey_kxy_f = tukey_kxy.flatten()
#             # Exclude the radiation circle
#             ke_norm = (self.kx_e**2 + self.ky_e**2)**0.5
#             tukey_2dwin = tukey_kxy_f[ke_norm > k0]
#             # Apply the window
#             self.pk_ev[jf] = tukey_2dwin * self.pk_ev[jf]
#             ### debug plot
#             color_par = np.abs(self.pk_ev[jf])
#             import matplotlib.tri as tri
#             x = self.kx_ef[jf]
#             y = self.ky_ef[jf]
#             triang = tri.Triangulation(x, y)
#             triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
#             y[triang.triangles].mean(axis=1)) < k0)
#             fig = plt.figure()
#             fig.canvas.set_window_title('Tukey window')
#             p = plt.tricontourf(triang, color_par, levels = 40)
#             fig.colorbar(p)
#             # plt.plot(kx, tukey_kx, '--b', label = 'win on kx', linewidth = 3)
#             # plt.plot(ky, tukey_ky, '-r', label = 'win on ky')
#             # plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
#             #     k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
#             plt.xlabel('kx')
#             plt.ylabel('win')
#             plt.title('start at {0:.1f}, k0 = {1:.1f}'.format(start_cut*k0, k0))
#             plt.legend()
#             plt.show()
#             bar.update(1)
#         bar.close()

#     def reconstruct_pu(self, receivers):
#         '''
#         reconstruct sound pressure and particle velocity at a receiver object
#         '''
#         self.p_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
#         self.uz_recon = np.zeros((receivers.coord.shape[0], len(self.controls.k0)), dtype=complex)
#         bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing sound field...')
#         for jf, k0 in enumerate(self.controls.k0):
#             # First, we form the sensing matrix
#             k_p = k0 * self.dir
#             # h_p = np.exp(-1j*receivers.coord @ k_p.T)
#             try:
#                 kz_e = np.sqrt(k0**2 - self.kx_ef[jf]**2 - self.ky_ef[jf]**2+0j)
#                 k_ev = np.array([self.kx_ef[jf], self.ky_ef[jf], kz_e]).T
#                 k_vec = np.vstack((k_p, k_ev))
#                 # h_ev = np.exp(1j*receivers.coord @ k_ev.T)
#                 # h_mtx = np.hstack((h_p, h_ev))
#                 h_mtx = np.exp(-1j*receivers.coord @ k_vec.T)
#                 # pressure and particle velocity at surface
#                 self.p_recon[:,jf] = h_mtx @ np.concatenate((self.pk[:,jf],self.pk_ev[jf]))
#                 self.uz_recon[:,jf] = -((np.divide(k_vec[:,2], k0)) * h_mtx) @ np.concatenate((self.pk[:,jf], self.pk_ev[jf]))
#             except:
#                 h_mtx = np.exp(-1j*receivers.coord @ k_p.T)
#                 self.p_recon[:,jf] = h_mtx @ self.pk[:,jf]
#                 self.uz_recon[:,jf] = -((np.divide(k_p[:,2], k0)) * h_mtx) @ self.pk[:,jf]
#             # self.p_s[:,jf] =  p_surf_mtx
#             #  =  uz_surf_mtx
#             bar.update(1)
#         bar.close()

#     def plot_condnum(self, save = False, path = '', fname = ''):
#         '''
#         Method to plot the condition number
#         '''
#         fig = plt.figure()
#         fig.canvas.set_window_title('Condition number')
#         plt.title('Condition number - {}'.format(self.decomp_type))
#         plt.loglog(self.controls.freq, self.cond_num, color = 'black', label = self.decomp_type, linewidth = 2)
#         plt.grid(linestyle = '--', which='both')
#         plt.legend(loc = 'best')
#         plt.xticks([50, 100, 500, 1000, 2000, 4000, 8000, 10000],
#             ['50', '100', '500', '1k', '2k', '4k', '8k', '10k'])
#         plt.xlabel('Frequency [Hz]')
#         plt.ylabel(r'$\kappa$ [-]')
#         plt.ylim((0.0001, 2*np.amax(self.cond_num)))
#         plt.xlim((0.8*self.controls.freq[0], 1.2*self.controls.freq[-1]))
#         if save:
#             filename = path + fname
#             plt.savefig(fname = filename, format='pdf')

#     def pk_interpolate(self, npts=100):
#         '''
#         Method to interpolate the wave number spectrum.
#         '''
#         # Recover the actual measured points
#         r, theta, phi = cart2sph(self.dir[:,0], self.dir[:,1], self.dir[:,2])
#         # r, theta, phi = cart2sph(self.dir[:,2], self.dir[:,1], self.dir[:,0])
#         thetaphi_pts = np.transpose(np.array([phi, theta]))
#         # Create a grid to interpolate on
#         nphi = int(2*(npts+1))
#         ntheta = int(npts+1)
#         # sorted_phi = np.sort(phi)
#         # new_phi = np.linspace(sorted_phi[0], sorted_phi[-1], nphi)
#         # sorted_theta = np.sort(theta)
#         # new_theta = np.linspace(sorted_theta[0], sorted_theta[-1], ntheta)#(0, np.pi, nn)
#         new_phi = np.linspace(-np.pi, np.pi, nphi)
#         new_theta = np.linspace(-np.pi/2, np.pi/2, ntheta)#(0, np.pi, nn)
#         self.grid_phi, self.grid_theta = np.meshgrid(new_phi, new_theta)
#         # interpolate
#         from scipy.interpolate import griddata
#         self.grid_pk = []
#         # bar = ChargingBar('Interpolating the grid for P(k)',\
#         #     max=len(self.controls.k0), suffix='%(percent)d%%')
#         bar = tqdm(total = len(self.controls.k0), desc = 'Interpolating the grid for P(k)')

#         if self.flag_oct_interp:
#             for jf, f_oct in enumerate(self.freq_oct):
#                 # update_progress(jf/len(self.freq_oct))
#                 ###### Cubic with griddata #################################
#                 self.grid_pk.append(griddata(thetaphi_pts, self.pk_oct[:,jf],
#                     (self.grid_phi, self.grid_theta), method='cubic', fill_value=np.finfo(float).eps, rescale=False))
#         else:
#             for jf, k0 in enumerate(self.controls.k0):
#                 # update_progress(jf/len(self.controls.k0))
#                 ###### Cubic with griddata #################################
#                 self.grid_pk.append(griddata(thetaphi_pts, np.abs(self.pk[:,jf]),
#                     (self.grid_phi, self.grid_theta), method='cubic', fill_value=np.finfo(float).eps, rescale=False))
#                 bar.update(1)
#             #     bar.next()
#             # bar.finish()
#         bar.close()

#     def plot_pk_sphere(self, freq = 1000, db = False, dinrange = 40, save = False, name=''):
#         '''
#         Method to plot the magnitude of the spatial fourier transform on the surface of a sphere.
#         It is a normalized version of the magnitude, either between 0 and 1 or between -dinrange and 0.
#         inputs:
#             freq - Which frequency you want to see. If the calculated spectrum does not contain it
#                 we plot the closest frequency before the asked one.
#             dB (bool) - Whether to plot in linear scale (default) or decibel scale.
#             dinrange - You can specify a dinamic range for the decibel scale. It will not affect the
#             linear scale.
#             save (bool) - Whether to save or not the figure. PDF file with simple standard name
#         '''
#         id_f = np.where(self.controls.freq <= freq)
#         id_f = id_f[0][-1]
#         fig = plt.figure()
#         fig.canvas.set_window_title('Scatter plot of |P(k)| for freq {} Hz'.format(self.controls.freq[id_f]))
#         ax = fig.gca(projection='3d')
#         if db:
#             color_par = 20*np.log10(np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f])))
#             # color_par = 20*np.log10(np.abs(self.pk_oct[:,id_f])/np.amax(np.abs(self.pk_oct[:,id_f])))
#             id_outofrange = np.where(color_par < -dinrange)
#             color_par[id_outofrange] = -dinrange
#         else:
#             color_par = np.abs(self.pk[:,id_f])/np.amax(np.abs(self.pk[:,id_f]))
#         p=ax.scatter(self.dir[:,0], self.dir[:,1], self.dir[:,2],
#             c = color_par)
#         fig.colorbar(p)
#         ax.set_xlabel('X axis')
#         ax.set_ylabel('Y axis')
#         ax.set_zlabel('Z axis')
#         plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - ' + name)
#         plt.tight_layout()
#         if save:
#             filename = 'data/colormaps/cmat_' + str(int(freq)) + 'Hz_' + name
#             plt.savefig(fname = filename, format='pdf')

#     def plot_pk_map(self, freq = 1000, db = False, dinrange = 40, phase = False,
#         save = False, name='', path = '', fname='', color_code = 'viridis'):
#         '''
#         Method to plot the magnitude of the spatial fourier transform on a map of interpolated theta and phi.
#         It is a normalized version of the magnitude, either between 0 and 1 or between -dinrange and 0.
#         inputs:
#             freq - Which frequency you want to see. If the calculated spectrum does not contain it
#                 we plot the closest frequency before the asked one.
#             dB (bool) - Whether to plot in linear scale (default) or decibel scale.
#             dinrange - You can specify a dinamic range for the decibel scale. It will not affect the
#             linear scale.
#             save (bool) - Whether to save or not the figure. PDF file with simple standard name
#         '''
#         if self.flag_oct_interp:
#             id_f = np.where(self.freq_oct <= freq)
#         else:
#             id_f = np.where(self.controls.freq <= freq)
#         id_f = id_f[0][-1]
#         fig = plt.figure()
#         fig.canvas.set_window_title('Interpolated map of |P(k)| for freq {} Hz'.format(self.controls.freq[id_f]))
#         if db:
#             color_par = 20*np.log10(np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f])))
#             color_range = np.linspace(-dinrange, 0, dinrange+1)
#         else:
#             color_par = np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f]))
#             color_range = np.linspace(0, 1, 21)
        
#         # if db:
#         #     color_par = 20*np.log10(np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f])))
#         #     id_outofrange = np.where(color_par < -dinrange)
#         #     color_par[id_outofrange] = -dinrange
#         # else:
#         #     if phase:
#         #         color_par = np.rad2deg(np.angle(self.grid_pk[id_f]))
#         #     else:
#         #         # color_par = np.abs(self.grid_pk[id_f])/np.amax(np.abs(self.grid_pk[id_f]))
#         #         color_par = np.abs(self.grid_pk[id_f])
#         p=plt.contourf(np.rad2deg(self.grid_phi), 90-np.rad2deg(self.grid_theta), color_par,
#             color_range, extend='both', cmap = color_code)
#         fig.colorbar(p)
#         plt.xlabel(r'$\phi$ (azimuth) [deg]')
#         plt.ylabel(r'$\theta$ (elevation) [deg]')
#         if self.flag_oct_interp:
#             plt.title('|P(k)| at ' + str(self.freq_oct[id_f]) + 'Hz - '+ name)
#         else:
#             plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - P decomp. '+ name)
#         plt.tight_layout()
#         if save:
#             filename = path + fname + '_' + str(int(freq)) + 'Hz'
#             plt.savefig(fname = filename, format='png')

#     def plot_pk_evmap(self, freq = 1000, db = False, dinrange = 12, save = False, name='', path = '', fname='', contourplot = True, plot_kxky = False):
#         '''
#         Method to plot the magnitude of the spatial fourier transform of the evanescent components
#         The map of interpolated to a kx and ky wave numbers.
#         It is a normalized version of the magnitude, either between 0 and 1 or between -dinrange and 0.
#         inputs:
#             freq (float) - Which frequency you want to see. If the calculated spectrum does not contain it
#                 we plot the closest frequency before the asked one.
#             dB (bool) - Whether to plot in linear scale (default) or decibel scale.
#             dinrange (float) - You can specify a dinamic range for the decibel scale. It will not affect the
#             linear scale.
#             save (bool) - Whether to save or not the figure (png file)
#             path (str) - path to save fig
#             fname (str) - name file of the figure
#             plot_kxky (bool) - whether to plot or not the kx and ky points that are part of the evanescent map.
#         '''
#         import matplotlib.tri as tri
#         id_f = np.where(self.controls.freq <= freq)
#         id_f = id_f[0][-1]
#         k0 = self.controls.k0[id_f]
#         if db:
#             color_par = 20*np.log10(np.abs(self.pk_ev[id_f])/np.amax(np.abs(self.pk_ev[id_f])))
#             id_outofrange = np.where(color_par < -dinrange)
#             color_par[id_outofrange] = -dinrange
#         else:
#             color_par = np.abs(self.pk_ev[id_f])#/np.amax(np.abs(pk_ev_grid))
#         ############### Countourf ##########################
#         # Create the Triangulation; no triangles so Delaunay triangulation created.
#         x = self.kx_ef[id_f]
#         y = self.ky_ef[id_f]
#         triang = tri.Triangulation(x, y)
#         # Mask off unwanted triangles.
#         triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
#             y[triang.triangles].mean(axis=1)) < k0)
#         fig = plt.figure()
#         fig.canvas.set_window_title('Filtered evanescent waves')
#         plt.plot(k0*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
#             k0*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
#         if contourplot:
#             p = plt.tricontourf(triang, color_par,
#                 levels=dinrange)
#         else:
#             p=plt.scatter(self.kx_ef[id_f], self.ky_ef[id_f], c = color_par)
#         fig.colorbar(p)
#         if plot_kxky:
#             plt.scatter(self.kx_ef[id_f], self.ky_ef[id_f], c = 'grey', alpha = 0.4)
#         plt.xlabel(r'$k_x$ rad/m')
#         plt.ylabel(r'$k_y$ rad/m')
#         plt.title("|P(k)| (evanescent) at {0:.1f} Hz (k = {1:.2f} rad/m) {2}".format(self.controls.freq[id_f],k0, name))
#         plt.tight_layout()
#         if save:
#             filename = path + fname + '_' + str(int(freq)) + 'Hz'
#             plt.savefig(fname = filename, format='png')

#     def save(self, filename = 'array_zest', path = '/home/eric/dev/insitu/data/zs_recovery/'):
#         '''
#         This method is used to save the simulation object
#         '''
#         filename = filename# + '_Lx_' + str(self.Lx) + 'm_Ly_' + str(self.Ly) + 'm'
#         self.path_filename = path + filename + '.pkl'
#         f = open(self.path_filename, 'wb')
#         pickle.dump(self.__dict__, f, 2)
#         f.close()

#     def load(self, filename = 'array_zest', path = '/home/eric/dev/insitu/data/zs_recovery/'):
#         '''
#         This method is used to load a simulation object. You build a empty object
#         of the class and load a saved one. It will overwrite the empty one.
#         '''
#         lpath_filename = path + filename + '.pkl'
#         f = open(lpath_filename, 'rb')
#         tmp_dict = pickle.load(f)
#         f.close()
#         self.__dict__.update(tmp_dict)