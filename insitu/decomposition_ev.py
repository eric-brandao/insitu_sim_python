import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
# from matplotlib import cm
# from insitu.controlsair import load_cfg
# import scipy.integrate as integrate
# import scipy as spy
from scipy.interpolate import griddata
from sklearn.linear_model import Ridge
import time
from tqdm import tqdm
import sys
# from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
#from tqdm._tqdm_notebook import tqdm
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
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

class DecompositionEv(object):
    """ Decomposition of the sound field using propagating and evanescent waves.

    The class has several methods to perform sound field decomposition into a set of
    incident and reflected plane waves. These sets of plane waves are composed of
    propagating and evanescent waves. We create a regular grid on the kx and ky plane,
    wich will contain the evanescent and propagating waves. In the end, we combine
    the grid of evanescent and propagating waves onto two grids - one for the incident
    and another for the reflected sound field.

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
    kx : numpy 1darray
        contains the kx waves.
    ky : numpy 1darray
        contains the ky waves.
    decomp_type : str
        Decomposition description
    f_ref : float
        The amplitude of the reflected waves (or radiating waves)
    f_inc : float
        The amplitude of the incident waves.
    zp : float
        Virtual source plane location (reflected waves).
    zm : float
        Virtual source plane location (incident waves).
    kx : numpy 1darray
        kx wave numbers (for later plots)
    ky : numpy 1darray
        ky wave numbers (for later plots)
    delta_kx : float
        spacing in kx
    delta_ky : float
        spacing in ky
    kx_grid : numpy ndarray
        Uniform grid on kx
    ky_grid : numpy ndarray
        Uniform grid on ky
    kx_f : numpy 1darray
        Flattened version of kx_grid
    ky_f : numpy 1darray
        Flattened version of ky_grid
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
    uz_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed particle vel (z)
        at all the field points

    Methods
    ----------
    create_kx_ky(n_kx = 20, n_ky = 20, plot=False, freq = 1000)
        Create a regular grid of kx and ky.

    pk_tikhonov_ev(method = 'direct', f_ref = 1.0, f_inc = 1.0, factor = 1, z0 = 1.5, plot_l = False)
        Wave number spectrum estimation using Tikhonov inversion

    reconstruct_pu(receivers)
        Reconstruct the sound pressure and particle velocity at a receiver object

    plot_pk_scatter(freq = 1000, db = False, dinrange = 40, save = False, name='name', travel=True)
        plot the magnitude of P(k) as a scatter plot of evanescent and propagating waves

    plot_colormap(self, freq = 1000, total_pres = True)
        Plots a color map of the pressure field.

    plot_pkmap(freq = 1000, db = False, dinrange = 20, save = False, name='name')
        Plot wave number spectrum as a 2D maps (vs. kx and ky) - uses tricontourf

    plot_pkmap_v2(freq = 1000, db = False, dinrange = 20,
        save = False, name='name', color_code = 'viridis')
        Plot wave number spectrum as a 2D maps (vs. kx and ky) - uses contourf

    plot_pkmap_prop(freq = 1000, db = False, dinrange = 20,
        save = False, name='name', color_code = 'viridis'):
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
        # self.pres_s = sim_field.pres_s[source_num] #FixMe
        # self.air = sim_field.air
        self.controls = controls
        self.material = material
        # self.sources = sim_field.sources
        self.receivers = receivers
        self.pres_s = p_mtx
        self.flag_oct_interp = False

    def create_kx_ky(self, n_kx = 20, n_ky = 20, plot=False, freq = 1000):
        """ Create a regular grid of kx and ky.

        This will be used to create all propagating and evanescent waves at the decompostition time.

        Parameters
        ----------
        n_kx : int
            number of elementary waves in kx direction
        n_ky : int
            number of elementary waves in ky direction
        plot : bool
            whether you plot or not the directions in space for k = 2 pi freq/c0
        freq : float
            frequency for which the L waves are ploted (just for visualization)
        """
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

    def pk_tikhonov_ev(self, method = 'direct', f_ref = 1.0, f_inc = 1.0, factor = 1, z0 = 1.5, plot_l = False):
        """ Wave number spectrum estimation using Tikhonov inversion

        Estimate the wave number spectrum using regularized Tikhonov inversion.
        The choice of the regularization parameter is baded on the L-curve criterion.
        This sound field is modelled by a set of propagating and evanescent waves. We
        use a grid for the incident and another for the reflected sound field. This
        method is very similar of SONAH, implemented in:
            Hald, J. Basic theory and properties of statistically optimized near-field acoustical
            holography, J Acoust Soc Am. 2009 Apr;125(4):2105-20. doi: 10.1121/1.3079773

        The inversion steps are: (i) - We will use regular grid (evanescent and propagating);
        (ii) - create the correct kz for propagating and evanescent waves (reflected/radiating);
        (iii) - use Eqs. (16 - 19) to form reflected matrix using Eq. (2)
        (iv) - use Eqs. (16 - 19) to form incident matrix using Eq. (2) - if requested
        (v) - form the sensing matrix; (vi) - compute SVD of the sensing matix;
        (vi) - compute the regularization parameter (L-curve); (vii) - matrix inversion.

        Parameters
        ----------
        method : string
            Mathod to compute wave number spectrum. Default is 'direct'. You can also try
            'scipy' (from scipy sparse - not that precise), 'Ridge' from sklearn and
            'cvx' from cvxpy (the three are possibly slower than direct and not more precise)
        f_ref : float
            The amplitude of the reflected waves (or radiating waves). Default value is 1.0
        f_inc : float
            The amplitude of the incident waves . Default value is 1.0.
            If you only have a radiatig structure you can set this to 0.0.
            The algorithm will use the appropiate basis for the decomposition.
        factor : float
            This parameter controls the position of your source-plane. It is combined
            with the microphone spacing on the array. This is an hyper-parameter important
            for regularization of the problem. Typically, the location of the source plane
            will be a multiple of the array spacing - this is what factor represents.
            Default value is 2.5 - found optimal on a large set of simulations.
        z0 : float
            The location of the source plane for the incident sound field.
            It should be a vaule somewhat close to the array. The recomendations is to
            use some sensible value that promotes some simetry relative to the array
            distance from the sample and the array thickness.
        plot_l : bool
            Whether to plot the L-curve or not. Default is false.
        """
        self.decomp_type = 'Tikhonov (transparent array) w/ evanescent waves'
        # Incident and reflected amplitudes
        self.f_ref = f_ref
        self.f_inc = f_inc
        # reflected and incident virtual source plane distances
        self.zp = -factor * np.amax([self.receivers.ax, self.receivers.ay])
        self.zm = z0 + factor * np.amax([self.receivers.ax, self.receivers.ay]) # Try
        # Initialize variables
        self.cond_num = np.zeros(len(self.controls.k0))
        if f_inc != 0:
            self.pk = np.zeros((2*len(self.kx_f), len(self.controls.k0)), dtype=complex)
        else:
            self.pk = np.zeros((len(self.kx_f), len(self.controls.k0)), dtype=complex)
        self.kx_ef = [] # Filtered version
        self.ky_ef = [] # Filtered version
        self.pk_ev = []
        # Initializa bar
        bar = tqdm(total = len(self.controls.k0),
            desc = 'Calculating Tikhonov inversion (with evanescent waves)...')
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # For smooth transition from continous to discrete k domain
            kappa = np.sqrt(self.delta_kx*self.delta_ky/(2*np.pi*k0**2))
            # compute kz
            kz_f = form_kz(k0, self.kx_f, self.ky_f, plot=False)
            k_vec_ref = np.array([self.kx_f, self.ky_f, kz_f])
            # Reflected or radiating part
            fz_ref = f_ref * np.sqrt(k0/np.abs(kz_f)) # compensate for higher density near the poles (propagating)
            recs = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
                self.receivers.coord[:,2]-self.zp]).T
            psi_ref = fz_ref * kappa * np.exp(-1j * recs @ k_vec_ref)
            # Incident part
            if f_inc != 0:
                k_vec_inc = np.array([self.kx_f, self.ky_f, -kz_f])
                fz_inc = f_inc * np.sqrt(k0/np.abs(kz_f))
                recs = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
                    self.receivers.coord[:,2]-self.zm]).T
                psi_inc = fz_inc * kappa * np.exp(-1j * recs @ k_vec_inc)
            # Forming the sensing matrix
            if f_inc == 0:
                h_mtx = psi_ref
            else:
                h_mtx = np.hstack((psi_inc, psi_ref))
            self.cond_num[jf] = np.linalg.cond(h_mtx)
            # Measured data
            pm = self.pres_s[:,jf].astype(complex)
            # Compute SVD
            u, sig, v = csvd(h_mtx)
            # Find the optimal regularization parameter.
            lambd_value = l_cuve(u, sig, pm, plotit=plot_l)
            # Choosing the method to find the P(k)
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
            # Performing the Tikhonov inversion with cvxpy #########################
            else:
                H = h_mtx.astype(complex)
                x_cvx = cp.Variable(h_mtx.shape[1], complex = True)
                lambd = cp.Parameter(nonneg=True)
                lambd.value = lambd_value[0]
                # Create the problem and solve
                problem = cp.Problem(cp.Minimize(objective_fn(H, pm, x_cvx, lambd)))
                problem.solve(solver=cp.SCS, verbose=False) # Fast but gives some warnings
                x = x_cvx.value
            # Get correct complex wave number spk
            self.pk[:,jf] = x
            bar.update(1)
        bar.close()

    def reconstruct_pu(self, receivers):
        """ Reconstruct the sound pressure and particle velocity at a receiver object

        Reconstruct the pressure and particle velocity at a set of desired field points.
        This can be used on impedance estimation or to plot spatial maps of pressure,
        velocity, intensity.

        The steps are: (i) - We will use regular grid (evanescent and propagating);
        (ii) - create the correct kz for propagating and evanescent waves (reflected/radiating);
        (iii) - use Eqs. (16 - 19) to form reflected matrix using Eq. (2)
        (iv) - use Eqs. (16 - 19) to form incident matrix using Eq. (2) - if requested
        (v) - form the new sensing matrix; (vi) - compute p and u.
        """
        self.fpts = receivers
        # Initialize variables
        self.p_recon = np.zeros((self.fpts.coord.shape[0], len(self.controls.k0)), dtype=complex)
        self.uz_recon = np.zeros((self.fpts.coord.shape[0], len(self.controls.k0)), dtype=complex)
        # Initialize bar
        bar = tqdm(total = len(self.controls.k0), desc = 'Reconstructing sound field...')
        for jf, k0 in enumerate(self.controls.k0):
            # For smooth transition from continous to discrete k domain
            kappa = np.sqrt(self.delta_kx*self.delta_ky/(2*np.pi*k0**2))
            # compute kz
            kz_f = form_kz(k0, self.kx_f, self.ky_f)
            k_vec_ref = np.array([self.kx_f, self.ky_f, kz_f])
            # Reflected or radiating part
            fz_ref = self.f_ref * np.sqrt(k0/np.abs(kz_f))
            recs = np.array([self.fpts.coord[:,0], self.fpts.coord[:,1],
                self.fpts.coord[:,2]-self.zp]).T
            psi_ref = fz_ref * kappa * np.exp(-1j * recs @ k_vec_ref)
            # Incident part
            if self.f_inc != 0:
                k_vec_inc = np.array([self.kx_f, self.ky_f, -kz_f])
                fz_inc = self.f_inc * np.sqrt(k0/np.abs(kz_f))
                recs = np.array([self.fpts.coord[:,0], self.fpts.coord[:,1],
                    self.fpts.coord[:,2]-self.zm]).T
                psi_inc = fz_inc * kappa * np.exp(-1j * recs @ k_vec_inc)
            # Forming the sensing matrix
            if self.f_inc == 0:
                h_mtx = psi_ref
            else:
                h_mtx = np.hstack((psi_inc, psi_ref))
            # Compute p and uz
            self.p_recon[:,jf] = h_mtx @ self.pk[:,jf]
            if self.f_inc == 0:
                self.uz_recon[:,jf] = -((np.divide(kz_f, k0)) * h_mtx) @ self.pk[:,jf]
            else:
                self.uz_recon[:,jf] = -((np.divide(np.concatenate((-kz_f, kz_f)), k0)) * h_mtx) @ self.pk[:,jf]
            bar.update(1)
        bar.close()

    def plot_pk_scatter(self, freq = 1000, db = False, dinrange = 40, save = False, name='name', travel=True):
        """ plot the magnitude of P(k) as a scatter plot of evanescent and propagating waves

        Plot the magnitude of the wave number spectrum as a scatter plot of
        propagating  and evanescent waves.
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
            travel : bool
                Whether to plot travel direction or arrival direction. Default is True
        """
        id_f = np.where(self.controls.freq <= freq)
        # id_f = np.where(self.freq_oct <= freq)
        id_f = id_f[0][-1]
        pk = self.pk[:,id_f]
        pk_i = self.pk[:len(self.kx_f),id_f]
        pk_r = self.pk[len(self.kx_f):,id_f]
        kz_f = form_kz(self.controls.k0[id_f], self.kx_f, self.ky_f, plot=False)
        # Calculate colors
        if db:
            color_par_i = 20*np.log10(np.abs(pk_i)/np.amax(np.abs(pk)))
            color_par_r = 20*np.log10(np.abs(pk_r)/np.amax(np.abs(pk)))
        else:
            color_par_i = np.abs(pk_i)
            color_par_r = np.abs(pk_r)
        kx = self.kx_f
        ky = self.ky_f
        kz = np.real(kz_f)
        # Figure
        fig = plt.figure()
        fig.canvas.set_window_title('Scatter plot of wavenumber spectrum')
        ax = fig.gca(projection='3d')
        if travel:
            p=ax.scatter(kx, ky, -kz-self.controls.k0[id_f]/8, c = color_par_i,
                vmin=-dinrange, vmax=0, s=int(dinrange))
            p=ax.scatter(kx, ky, kz+self.controls.k0[id_f]/8, c = color_par_r,
                vmin=-dinrange, vmax=0, s=int(dinrange))
        else: #arival
            p=ax.scatter(kx, ky, kz+self.controls.k0[id_f]/8, c = color_par_i,
                vmin=-dinrange, vmax=0, s=int(dinrange))
            p=ax.scatter(kx, ky, -kz-self.controls.k0[id_f]/8, c = color_par_r,
                vmin=-dinrange, vmax=0, s=int(dinrange))
            fig.colorbar(p)
        ax.set_xlabel(r'$k_x$ [rad/m]')
        ax.set_ylabel(r'$k_y$ [rad/m]')
        ax.set_zlabel(r'$k_z$ [rad/m]')
        ax.view_init(elev=10, azim=0)
        plt.title('|P(k)| at ' + str(self.controls.freq[id_f]) + 'Hz - ' + name)
        if save:
            filename = 'data/colormaps/cmat_' + str(int(freq)) + 'Hz_' + name
            plt.savefig(fname = filename, format='pdf')

    def plot_pkmap(self, freq = 1000, db = False, dinrange = 20, save = False, name='name'):
        """ Plot wave number spectrum as a 2D maps (vs. kx and ky) - uses tricontourf

        Plot the magnitude of the wave number spectrum as two 2D maps of
        evanescent and propagating waves. It is a normalized version of the
        magnitude, either between 0 and 1 or between -dinrange and 0.
        The maps are ploted as color as function of kx and ky.
        The radiation circle is also ploted.

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
        """
        id_f = np.where(self.controls.freq <= freq)
        # id_f = np.where(self.freq_oct <= freq)
        id_f = id_f[0][-1]
        k0 = self.controls.k0[id_f]
        kappa = np.sqrt(self.delta_kx*self.delta_ky/(2*np.pi*k0**2))
        pk = self.pk[:,id_f]
        pk_i = self.pk[:len(self.kx_f),id_f] # incident
        pk_r = self.pk[len(self.kx_f):,id_f] # reflected
        # Calculate colors
        if db:
            color_par_i = 20*np.log10(np.abs(pk_i)/np.amax(np.abs(pk)))
            color_par_r = 20*np.log10(np.abs(pk_r)/np.amax(np.abs(pk)))
            color_range = np.linspace(-dinrange, 0, dinrange+1)
        else:
            color_par_i = np.abs(pk_i)/np.amax(np.abs(pk))
            color_par_r = np.abs(pk_r)/np.amax(np.abs(pk))
            color_range = np.linspace(0, 1, 21)
        # Create triangulazition
        triang = tri.Triangulation(self.kx_f, self.ky_f)
        # Figure
        fig = plt.figure(figsize=(8, 8))
        # fig = plt.figure()
        fig.canvas.set_window_title('2D plot of wavenumber spectrum')
        # Incident
        plt.subplot(2, 1, 1)
        plt.title('Incident field')
        plt.plot(self.controls.k0[id_f]*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
            self.controls.k0[id_f]*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
        p = plt.tricontourf(triang, color_par_i, color_range, extend='both')
        fig.colorbar(p)
        plt.ylabel(r'$k_y$ [rad/m]')
        # Reflected
        plt.subplot(2, 1, 2)
        plt.title('Reflected field')
        plt.plot(self.controls.k0[id_f]*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
            self.controls.k0[id_f]*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
        p = plt.tricontourf(triang, color_par_r, color_range, extend='both')
        fig.colorbar(p)
        plt.xlabel(r'$k_x$ [rad/m]')
        plt.ylabel(r'$k_y$ [rad/m]')
        plt.tight_layout()
        if save:
            filename = 'data/colormaps/cmat_' + str(int(freq)) + 'Hz_' + name
            plt.savefig(fname = filename, format='pdf')

    def plot_pkmap_v2(self, freq = 1000, db = False, dinrange = 20,
        save = False, name='name', color_code = 'viridis'):
        """ Plot wave number spectrum as a 2D maps (vs. kx and ky) - uses contourf

        Plot the magnitude of the wave number spectrum as two 2D maps of
        evanescent and propagating waves. The map is first interpolated into
        a regular grid. It is a normalized version of the magnitude, either between
        0 and 1 or between -dinrange and 0. The maps are ploted as color as function
        of kx and ky. The radiation circle is also ploted. 

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
            color_code : str
                Can be anything that matplotlib supports. Some recomendations given below:
                'viridis' (default) - Perceptually Uniform Sequential
                'Greys' - White (cold) to black (hot)
                'seismic' - Blue (cold) to red (hot) with a white in the middle
        """
        id_f = np.where(self.controls.freq <= freq)
        # id_f = np.where(self.freq_oct <= freq)
        id_f = id_f[0][-1]
        # Quatinties from simulation
        k0 = self.controls.k0[id_f]
        pk = self.pk[:,id_f]
        pk_i = self.pk[:len(self.kx_f),id_f] # incident
        pk_r = self.pk[len(self.kx_f):,id_f] # reflected
        # kz_f = form_kz(self.controls.k0[id_f], self.kx_f, self.ky_f, plot=False)
        # Interpolate
        kxy = np.transpose(np.array([self.kx_f, self.ky_f]))
        pk_i_grid = griddata(kxy, np.abs(pk_i), (self.kx_grid, self.ky_grid),
            method='cubic', fill_value=np.finfo(float).eps, rescale=False)
        pk_r_grid = griddata(kxy, np.abs(pk_r), (self.kx_grid, self.ky_grid),
            method='cubic', fill_value=np.finfo(float).eps, rescale=False)
        # Calculate colors
        if db:
            color_par_i = 20*np.log10(np.abs(pk_i_grid)/np.amax(np.abs(pk)))
            color_par_r = 20*np.log10(np.abs(pk_r_grid)/np.amax(np.abs(pk)))
            color_range = np.linspace(-dinrange, 0, dinrange+1)
        else:
            color_par_i = np.abs(pk_i_grid)/np.amax(np.abs(pk))
            color_par_r = np.abs(pk_r_grid)/np.amax(np.abs(pk))
            color_range = np.linspace(0, 1, 21)
        # Figure
        fig = plt.figure(figsize=(8, 8))
        # fig = plt.figure()
        fig.canvas.set_window_title('2D plot of wavenumber spectrum - PE')
        # Incident
        plt.subplot(2, 1, 1)
        plt.title('Incident field')
        plt.plot(self.controls.k0[id_f]*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
            self.controls.k0[id_f]*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
        p = plt.contourf(self.kx_grid, self.ky_grid, color_par_i,
            color_range, extend='both', cmap = color_code)
        fig.colorbar(p)
        plt.ylabel(r'$k_y$ [rad/m]')
        # Reflected
        plt.subplot(2, 1, 2)
        plt.title('Reflected field')
        plt.plot(self.controls.k0[id_f]*np.cos(np.arange(0, 2*np.pi+0.01, 0.01)),
                self.controls.k0[id_f]*np.sin(np.arange(0, 2*np.pi+0.01, 0.01)), 'r')
        p = plt.contourf(self.kx_grid, self.ky_grid, color_par_r,
            color_range, extend='both', cmap = color_code)
        fig.colorbar(p)
        plt.xlabel(r'$k_x$ [rad/m]')
        plt.ylabel(r'$k_y$ [rad/m]')
        plt.tight_layout()
        if save:
            filename = 'data/colormaps/cmat_' + str(int(freq)) + 'Hz_' + name
            plt.savefig(fname = filename, format='pdf')

    def plot_pkmap_prop(self, freq = 1000, db = False, dinrange = 20,
        save = False, name='name', color_code = 'viridis'):
        """ Plot wave number spectrum  - propagating only (vs. phi and theta)

        Plot the magnitude of the wave number spectrum as a map of
        propagating waves. The map is first interpolated into
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
            color_code : str
                Can be anything that matplotlib supports. Some recomendations given below:
                'viridis' (default) - Perceptually Uniform Sequential
                'Greys' - White (cold) to black (hot)
                'seismic' - Blue (cold) to red (hot) with a white in the middle
        """
        id_f = np.where(self.controls.freq <= freq)
        # id_f = np.where(self.freq_oct <= freq)
        id_f = id_f[0][-1]
        # Quatinties from simulation
        k0 = self.controls.k0[id_f]
        # Incident and reflected with evanescent
        pk_i = self.pk[:len(self.kx_f),id_f] # incident
        pk_r = self.pk[len(self.kx_f):,id_f] # reflected
        # We must select only the propagating wave components
        ke_norm = (self.kx_f**2 + self.ky_f**2)**0.5
        kx_p = self.kx_f[ke_norm <= k0]
        ky_p = self.ky_f[ke_norm <= k0]
        kz_p = np.sqrt(k0**2 - (kx_p**2+ky_p**2))
        # The directions
        directions = np.vstack((np.array([-kx_p, -ky_p, -kz_p]).T,
            np.array([-kx_p, -ky_p, kz_p]).T)) / k0
        # Pk - propgating
        pk_ip = pk_i[ke_norm <= k0]
        pk_rp = pk_r[ke_norm <= k0]
        pk_p = np.hstack((pk_ip, pk_rp))
        # Transform uninterpolated data to spherical coords
        r, theta, phi = cart2sph(directions[:,0], directions[:,1], directions[:,2])
        thetaphi = np.transpose(np.array([phi, theta]))
        # Create the new grid to iterpolate
        new_phi = np.linspace(-np.pi, np.pi, len(self.kx))
        new_theta = np.linspace(-np.pi/2, np.pi/2,  len(self.kx))#(0, np.pi, nn)
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


    def plot_pk_evmap(self, freq = 1000, db = False, dinrange = 12, save = False, name='',
        path = '', fname='', contourplot = True):
        """ Plot wave number spectrum as a 2D maps (vs. kx and ky) without radiating circle

        Plot the magnitude of the wave number spectrum as two 2D maps of
        evanescent waves (excludes propagating). 
        It is a normalized version of the magnitude, either between
        0 and 1 or between -dinrange and 0. The maps are ploted as color as function
        of kx and ky. The radiation circle is also ploted. 

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
            countourplot : bool
                Whether to plot a contourplot (using tricontourf) or a scatter plot.
                Default is True
        """
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
        # Create the Triangulation; no triangles so Delaunay triangulation created.
        x = self.kx_f
        y = self.ky_f
        triang = tri.Triangulation(x, y)
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
        plt.xlabel(r'$k_x$ rad/m')
        plt.ylabel(r'$k_y$ rad/m')
        plt.title("|P(k)| (evanescent) at {0:.1f} Hz (k = {1:.2f} rad/m) {2}".format(self.controls.freq[id_f],k0, name))
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

def form_kz(k0, kx_f, ky_f, plot=False):
    """ Form kz according to kx and ky

    Parameters
    ----------
    kx_f : numpy 1darray
        contains all the kx on the search grid
    ky_f : numpy 1darray
        contains all the ky on the search grid
    plot : bool
        whether to plot the grid or not (for checking)
    """
    ke_norm = (kx_f**2 + ky_f**2)**0.5
    kz_f = np.zeros(len(kx_f), dtype = complex)
    # propagating part
    idp = np.where(ke_norm <= k0)[0]
    kz_f[idp] = np.sqrt(k0**2 - (kx_f[idp]**2+ky_f[idp]**2))
    # evanescent part
    ide = np.where(ke_norm > k0)[0]
    kz_f[ide] = -1j*np.sqrt(kx_f[ide]**2+ky_f[ide]**2-k0**2)
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
        plt.show()
    return kz_f

def loss_fn(H, pm, x):
    """ Loss function for cvx (residual)
    """
    return cp.pnorm(cp.matmul(H, x) - pm, p=2)**2

def regularizer(x):
    """ l2 norm of the result (for cvx)
    """
    return cp.pnorm(x, p=2)**2

def objective_fn(H, pm, x, lambd):
    """ Objective function for cvx
    """
    return loss_fn(H, pm, x) + lambd * regularizer(x)


###############################################################################################
###############################################################################################
###############################################################################################

# class DecompositionEv2(DecompositionEv):
#     '''
#     This class inherits the DecompositionEv class and adds the methods for decomposition based on
#     a separate grid approach for propagating and evanescent waves.
#     '''
#     def __init__(self, p_mtx = None, controls = None, material = None, receivers = None):
#         '''
#         Init - we first retrive general data, then we process some receiver data
#         '''
#         DecompositionEv.__init__(self, p_mtx, controls, material, receivers)
#         super().__init__(p_mtx, controls, material, receivers)

#     def prop_dir(self, n_waves = 642, plot = False):
#         '''
#         This method is used to create wave number directions uniformily distributed over the surface of
#         a hemisphere. The directions of propagation that later will become propagating wave-number vectors.
#         The directions of propagation are calculated with the triangulation of an icosahedron used previously
#         in the generation of omnidirectional rays (originally implemented in a ray tracing algorithm).
#         Inputs:
#             n_waves - The number of directions (wave-directions) to generate (integer)
#             plot - whether you plot or not the wave points in space (bool)
#         '''
#         directions = RayInitialDirections()
#         directions, n_sph = directions.isotropic_rays(Nrays = int(n_waves))
#         id_dir = np.where(directions[:,2]>=0)
#         self.pdir = directions[id_dir[0],:]
#         self.n_prop = len(self.pdir[:,0])
#         if plot:
#             fig = plt.figure()
#             fig.canvas.set_window_title('Dir test')
#             ax = fig.gca(projection='3d')
#             p=ax.scatter(self.pdir[:,0], self.pdir[:,1], self.pdir[:,2])

#     def pk_tikhonov_ev_ig(self, f_ref = 1.0, f_inc = 1.0, factor = 1, z0 = 1.5, plot_l = False):
#         '''
#         Method to estimate wave number spectrum based on the Tikhonov matrix inversion technique.
#         This version includes the evanescent waves
#         Inputs:
#         '''
#         self.decomp_type = 'Tikhonov (transparent array) w/ evanescent waves - uses irregular grid'
#         self.f_ref = f_ref
#         self.f_inc = f_inc
#         # self.factor = factor
#         # self.z0 = z0
#         self.zp = -factor * np.amax([self.receivers.ax, self.receivers.ay])
#         print('freq: {}'.format(self.controls.freq))
#         print('zp: {}, rel to lam {}'.format(self.zp, self.zp/(2*np.pi/self.controls.k0)))
#         self.zm = factor * (z0 + np.amax([self.receivers.ax, self.receivers.ay]))
#         # self.zm = z0 + factor * np.amax([self.receivers.ax, self.receivers.ay]) # Try
#         # Generate kx and ky for evanescent wave grid
#         kx = np.linspace(start = -np.pi/self.receivers.ax,
#             stop = np.pi/self.receivers.ax, num = int(self.n_prop**0.5))
#         ky = np.linspace(start = -np.pi/self.receivers.ay,
#             stop = np.pi/self.receivers.ay, num = int(self.n_prop**0.5))
#         kx_grid, ky_grid = np.meshgrid(kx,ky)
#         kx_e = kx_grid.flatten()
#         ky_e = ky_grid.flatten()
#         # Initialize variables
#         self.cond_num = np.zeros(len(self.controls.k0))
#         self.pk = []
#         # Initializa bar
#         bar = tqdm(total = len(self.controls.k0),
#             desc = 'Calculating Tikhonov inversion (with evanescent waves and irregular grid)...')
#         # Freq loop
#         for jf, k0 in enumerate(self.controls.k0):
#             # print(self.controls.freq[jf])
#             # 1 - for smooth transition from continous to discrete k domain
#             # kappa = np.sqrt(self.delta_kx*self.delta_ky/(2*np.pi*k0**2))
#             # 2 - form kz
#             kx_eig, ky_eig, n_e = filter_evan(k0, kx_e, ky_e, plot=False)
#             kz_eig = -1j*np.sqrt(kx_eig**2 + ky_eig**2 - k0**2)
#             # 3 Wavenumbers
#             k_vec_inc = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], -self.pdir[:,2]]).T,
#                 np.array([kx_eig, ky_eig, -kz_eig]).T))
#             k_vec_ref = np.vstack((k0 * np.array([self.pdir[:,0], self.pdir[:,1], self.pdir[:,2]]).T,
#                 np.array([kx_eig, ky_eig, kz_eig]).T))
#             # k_vec_inc = np.vstack((k0 * self.pdir, np.array([kx_eig, ky_eig, -kz_eig]).T))
#             # k_vec_ref = np.vstack((k0 * self.pdir, np.array([kx_eig, ky_eig, kz_eig]).T))
#             # fz_ref = f_ref * np.sqrt(k0/np.abs(kz_f))
#             # 4 - receivers 
#             recs_inc = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
#                 self.receivers.coord[:,2]-self.zm]).T
#             recs_ref = np.array([self.receivers.coord[:,0], self.receivers.coord[:,1],
#                 self.receivers.coord[:,2]-self.zp]).T
#             # 5 - psi and sensing matrix
#             psi_inc = np.exp(-1j * recs_inc @ k_vec_inc.T)
#             psi_ref = np.exp(-1j * recs_ref @ k_vec_ref.T)
#             h_mtx = np.hstack((psi_inc, psi_ref))
#             self.cond_num[jf] = np.linalg.cond(h_mtx)
#             # measured data
#             pm = self.pres_s[:,jf].astype(complex)
#             # 6 - finding the optimal regularization parameter.
#             u, sig, v = csvd(h_mtx)
#             lambd_value = l_cuve(u, sig, pm, plotit=plot_l)
#             Hm = np.matrix(h_mtx)
#             x = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
#             self.pk.append(x)#[0:self.n_waves]
#             bar.update(1)
#         bar.close()

