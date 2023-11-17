import numpy as np
from tqdm import tqdm
from receivers import Receiver
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# from lcurve_functions import csvd, l_curve, tikhonov
import lcurve_functions as lc

SMALL_SIZE = 11
BIGGER_SIZE = 18
plt.rcParams.update({'font.family': 'sans-serif'})
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('figure', titlesize=BIGGER_SIZE)   # font size of the figure title
plt.rc('legend', fontsize=BIGGER_SIZE)    # font size of the figure subtitle
plt.rc('axes', titlesize=BIGGER_SIZE)     # font size of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # font size of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # font size of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # font size of the tick labels


class Decomposition_2M(object):
    """ Decomposition of the sound field using only propagating waves.

    The class has methods to perform sound field decomposition into a set of
    incident and reflected spherical waves.

    Attributes
    ----------
    controls : object (AlgControls)
        Controls of the decomposition (frequency spam)
    material : object (PorousAbsorber)
        Contains the material properties (surface impedance). This can be used as reference
        when simulations is what you want to do.
    receivers : object (Receiver)
        The receivers in the field - this contains the information of the coordinates of
        the microphones in your array
    pk : list
        Estimated amplitudes of all plane waves.
        Each element in the list is relative to a frequency of the measurement spectrum.
    p_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed sound pressure
        at all the field points
    uz_recon : (N_rec x N_freq) numpy array
        A matrix containing the complex amplitudes of the reconstructed particle vel (z)
        at all the field points

    Methods
    ----------
    pk_tikhonov(self, plot_l = False, method = 'direct')
        Wave number spectrum estimation using Tikhonov inversion

    zs(self, Lx=0.1, n_x=21, Ly=0.1, n_y=21, theta=None, avgZs=True)
        Surface impedance reconstruction and absorption coefficient estimation

    reconstruct_pu(receivers)
        Reconstruction of the sound pressure and particle velocity at a receiver object
    """

    def __init__(self, p_mtx=None, controls=None, material=None, receivers=None, source_coord=None):
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
        source_coord : tuple
            The source coordinates in the field

        The objects are stored as attributes in the class (easier to retrieve).
        """

        self.pres_s = p_mtx
        self.controls = controls
        self.material = material
        self.receivers = receivers
        self.source_coord = source_coord
        self.image_source = np.array([self.source_coord[0], self.source_coord[1], -self.source_coord[2]])
        self.pk = []

        #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    def pk_tikhonov(self, plot_l=False, method='direct'):
        """ Wave number spectrum estimation using Tikhonov inversion.

        Estimate the wave number spectrum using regularized Tikhonov inversion.
        The choice of the regularization parameter is based on the L-curve criterion.
        This sound field is modelled by spherical waves. We use a monopole for the incident
        sound field and another for the reflected sound field. This method is an adaptation of
        one of the methods to perform sound field decomposition implemented by Eric Brandão.
        GitHub repository link: https://github.com/eric-brandao/insitu_sim_python

        The inversion steps are:
        (i) - Get the distance from sources in relation to each receiver;
        (ii) - form the sensing matrix;
        (iii) - compute SVD of the sensing matrix;
        (iv) - compute the regularization parameter (L-curve);
        (vii) - matrix inversion.

        Parameters
        ----------
        method : str
            Determines which method to use to compute the pseudo-inverse.
                - 'direct' (default) - analytical solution - fastest, but maybe
                inaccurate on noiseless situations. The following uses optimization
                algorithms
                -'Ridge' - uses sklearn Ridge regression - slower, but accurate.
                -'Tikhonov' - uses cvxpy - slower, but accurate
        plot_l : bool
            Whether to plot the L-curve or not. Default is false.
        """

        # Initialize variables
        r_norm = np.zeros((self.receivers.coord.shape[0], 2))  # Source-receivers distances array

        # Initialize bar
        bar = tqdm(total=len(self.controls.k0), desc='Calculating Tikhonov inversion (with 2 monopoles)...')

        # loop over sources
        for js, s_coord in enumerate([self.source_coord, self.image_source]):  # 1st real source, then the image source
            # loop over receivers
            for jrec, r_coord in enumerate(self.receivers.coord):
                r_norm[jrec, js] = np.linalg.norm(s_coord - r_coord)  # |r_source - r_receiver|

        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):

            # Forming the sensing matrix (M x 2)
            h_mtx = (np.exp(-1j * k0 * r_norm)) / r_norm

            # Measured data
            pm = self.pres_s[:, jf].astype(complex)

            # Compute SVD
            u, sig, v = lc.csvd(h_mtx)

            # Find the optimal regularization parameter.
            lambd_value = lc.l_curve(u, sig, pm, plotit=plot_l)

            # Matrix inversion
            if method == 'Ridge':
                # Form a real H2 matrix and p2 measurement
                H2 = np.vstack((np.hstack((h_mtx.real, -h_mtx.imag)),
                                np.hstack((h_mtx.imag, h_mtx.real))))
                p2 = np.vstack((pm.real, pm.imag)).flatten()
                regressor = Ridge(alpha=lambd_value, fit_intercept=False, solver='svd')
                x2 = regressor.fit(H2, p2).coef_
                x = x2[:h_mtx.shape[1]]+1j*x2[h_mtx.shape[1]:]
            elif method == 'Tikhonov':
                # phi_factors = (sig**2)/(sig**2+lambd_value**2)
                # # because csvd takes the hermitian of h_mtx and only the first m collumns of v
                # x = (v @ np.diag(phi_factors/sig) @ np.conjugate(u)) @ pm
                x = lc.tikhonov(u, sig, v, pm, lambd_value)
            else:
                Hm = np.matrix(h_mtx)
                x = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value**2)*np.identity(len(pm))) @ pm
            self.pk.append(x)
            bar.update(1)
        bar.close()

    def zs(self, Lx=0.1, n_x=21, Ly=0.1, n_y=21, theta=None, avgZs=True):
        """ Reconstruct the surface impedance and estimate the absorption.

        Reconstruct pressure and particle velocity at a grid of points on the
        absorber's surface (z = 0.0). The absorption coefficient is also calculated.

        Parameters
        ----------
        Lx : float
            The length of calculation aperture
        Ly : float
            The width of calculation aperture
        n_x : int
            The number of calculation points in x
        n_y : int
            The number of calculation points in y
        theta : list
            Target angles to calculate the absorption from reconstructed impedance
        avgZs : bool
            Whether to average over <Zs> (default - True) or over <p>/<uz> (if False)

        Returns
        -------
        alpha : (N_theta x N_freq) numpy ndarray
            The absorption coefficients for each target incident angle.
        """

        if theta is None:
            theta = [0]

        # Set the grid used to reconstruct the surface impedance
        grid = Receiver()
        grid.planar_array(x_len=Lx, n_x=n_x, y_len=Ly, n_y=n_y, zr=0.0)

        # Allocate some memory prior to loop
        self.Zs = np.zeros(len(self.controls.k0), dtype=complex)
        self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        self.p_s = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.uz_s = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)

        # Initialize variables
        r_norm_grid = np.zeros((grid.coord.shape[0], 2))
        hs = [self.source_coord[2] - grid.coord[0, 2], -(self.source_coord[2] - grid.coord[0, 2])]  # [(hs-z), -(hs+z)]

        # Initialize bar
        bar = tqdm(total=len(self.controls.k0), desc='Calculating the surface impedance (Zs)...')

        # loop over sources
        for js, s_coord in enumerate([self.source_coord, self.image_source]):  # 1st real source, then the image source
            # loop over receivers
            for jrec, g_coord in enumerate(grid.coord):
                r_norm_grid[jrec, js] = np.linalg.norm(s_coord - g_coord)  # |r_source - r_grid|

        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):

            # Forming the sensing matrix (M x 2)
            h_mtx = (np.exp(-1j * k0 * r_norm_grid)) / r_norm_grid  # e^(-jk0|r1 - r2|) / |r1 - r2|

            # complex amplitudes of all waves
            x = self.pk[jf]

            # reconstruct pressure and particle velocity at surface
            p_surf_mtx = h_mtx @ x
            uz_surf_mtx = (h_mtx * ((1 / (1j * k0 * r_norm_grid)) + 1) * (hs / r_norm_grid)) @ x
            self.p_s[:, jf] = p_surf_mtx
            self.uz_s[:, jf] = uz_surf_mtx

            # Average impedance at grid
            if avgZs:
                Zs_pt = np.divide(p_surf_mtx, uz_surf_mtx)
                self.Zs[jf] = np.mean(Zs_pt)
            else:
                self.Zs[jf] = np.mean(p_surf_mtx) / (np.mean(uz_surf_mtx))
            bar.update(1)
            # bar.next()
        # bar.finish()

        # Calculate the sound absorption coefficient for targeted angles
        for jtheta, dtheta in enumerate(theta):
            self.alpha[jtheta, :] = 1 - (np.abs(np.divide((self.Zs * np.cos(dtheta) - 1),
                                                          (self.Zs * np.cos(dtheta) + 1)))) ** 2
        return self.alpha

    def reconstruct_pu(self, hz=0.01, n_pts=1):
        """ Reconstruct pressure and particle velocity.

        The reconstruction is done at a grid of points.

        Parameters
        ----------
        hz : float
            The height of the point
        n_pts : int
            Number of points

        Returns
        -------
        p_recon : (N_theta x N_freq) numpy ndarray
            The reconstructed pressure.
        uz_recon : (N_theta x N_freq) numpy ndarray
            The reconstructed particle velocity.
        """

        grid = Receiver()
        grid.line_array(startat=hz, n_rec=n_pts, direction='z')
        r_norm_grid = np.zeros((grid.coord.shape[0], 2))

        # Allocate some memory prior to loop
        self.Zs = np.zeros(len(self.controls.k0), dtype=complex)
        self.p_recon = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.uz_recon = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.uz_recon_inc = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.uz_recon_ref = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)

        # Initialize variables
        hs = [self.source_coord[2] - grid.coord[0, 2], self.image_source[2] - grid.coord[0, 2]]  # sources heights

        # Initialize bar
        bar = tqdm(total=len(self.controls.k0), desc='Calculating the reconstructed pressure and particle velocity...')

        # loop over sources
        for js, s_coord in enumerate([self.source_coord, self.image_source]):  # 1st real source, then the image source
            # loop over receivers
            for jrec, g_coord in enumerate(grid.coord):
                r_norm_grid[jrec, js] = np.linalg.norm(s_coord - g_coord)  # |r_source - r_grid|

        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):

            # Forming the sensing matrix (M x 2)
            h_mtx = (np.exp(-1j * k0 * r_norm_grid)) / r_norm_grid  # e^(-jk0|r1 - r2|) / |r1 - r2|

            # complex amplitudes of all waves
            x = self.pk[jf]

            # reconstruct pressure and particle velocity at surface
            p_surf_mtx = h_mtx @ x
            uz_surf_mtx = (h_mtx * ((1 / (1j * k0 * r_norm_grid)) + 1) * (hs / r_norm_grid)) @ x
            uz_inc = (h_mtx[0, 0] * ((1 / (1j * k0 * r_norm_grid[0, 0])) + 1) *
                      (hs[0] / r_norm_grid[0, 0])) * x[0]
            uz_ref = (h_mtx[0, 1] * ((1 / (1j * k0 * r_norm_grid[0, 1])) + 1) *
                      (hs[1] / r_norm_grid[0, 1])) * x[1]

            self.p_recon[0, jf] = p_surf_mtx
            self.uz_recon[0, jf] = uz_surf_mtx
            self.uz_recon_inc[0, jf] = uz_inc
            self.uz_recon_ref[0, jf] = uz_ref

            bar.update(1)
        bar.close()
        return self.p_recon, self.uz_recon, self.uz_recon_inc, self.uz_recon_ref
