import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.tri as tri
from receivers import Receiver
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.special import roots_legendre
# from lcurve_functions_EU import csvd, l_curve, tikhonov
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


def kernel_p(k0, rq):  # pressure kernel function
    iq = (np.exp(-1j * k0 * rq)) / rq
    return iq


def kernel_uz(k0, rq, hz_q):  # particle velocity kernel function
    iq = ((np.exp(-1j * k0 * rq)) / rq) * (hz_q / rq) * (1 + (1 / (1j * k0 * rq)))
    return iq


class Decomposition_QDT(object):
    """ Decomposition of the sound field using only propagating waves.

    The class has methods to perform sound field decomposition into a set of incident
    spherical waves and by representing the plane-wave reflection coefficient as the
    Laplace transform of an image source distribution, a well-behaved image integral,
    proposed by Di and Gilbert (1993). doi: https://doi.org/10.1121/1.405435.
    The image integral is approximated by the Gauss-Legendre quadrature.


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

    reconstruct_p(receivers)
        Reconstruction of the sound pressure at a receiver object

    reconstruct_uz(receivers)
        Reconstruction of the particle velocity at a receiver object

    zs(self, Lx=0.1, n_x=21, Ly=0.1, n_y=21, theta=None, avgZs=True)
        Surface impedance reconstruction and absorption coefficient estimation

    plot_colormap(freq = 1000):
        Plots a color map of the pressure field.

    plot_colormap2():
        Plots a color map of the pressure field for all frequencies at once.
    """

    def __init__(self, p_mtx=None, controls=None, material=None, receivers=None, source_coord=None, quad_order=None):
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
        quad_order : int
            The quadrature order

        The objects are stored as attributes in the class (easier to retrieve).
        """

        self.pres_s = p_mtx
        self.controls = controls
        self.material = material
        self.receivers = receivers
        self.source_coord = source_coord
        self.quad_order = quad_order
        self.pk = []

        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    def pk_tikhonov(self, plot_l=False, method='direct', a=-1, b=1, retraction=0):
        """ Wave number spectrum estimation using Tikhonov inversion.

        Estimate the wave number spectrum using regularized Tikhonov inversion.
        The choice of the regularization parameter is based on the L-curve criterion.
        This sound field is modelled by spherical waves. We use a monopole for the incident
        sound field and image source distribution approximated by the Gauss-Legendre quadrature
        for the reflected sound field. This method is an adaptation of one of the methods to perform
        sound field decomposition implemented by Eric Brandão. GitHub repository link:
        https://github.com/eric-brandao/insitu_sim_python

        The inversion steps are:
        (i) - Get the distance from sources in relation to each receiver;
        (ii) - form the sensing matrix;
        (iii) - compute SVD of the sensing matrix;
        (iv) - compute the regularization parameter (L-curve);
        (vii) - matrix inversion.

        Parameters
        ----------
        a : int
            lower bond of the integral
        b : float
            upper bound of the integral
        method : str
            Determines which method to use to compute the pseudo-inverse.
                'direct' (default) - analytical solution - fastest, but maybe
                inaccurate on noiseless situations. The following uses optimization
                algorithms
                'Ridge' - uses sklearn Ridge regression - slower, but accurate.
                'Tikhonov' - uses cvxpy - slower, but accurate
        plot_l : bool
            Whether to plot the L-curve or not. Default is false.
        retraction : float
            Retraction value of the source height. Default is 1 cm.
        """

        # Initialize variables
        roots, weights = roots_legendre(self.quad_order)  # roots and weights of G-L polynomials
        q = (((b - a) / 2) * roots + ((a + b) / 2))  # variable interval change
        h_mtx = np.zeros((self.receivers.coord.shape[0], (1 + self.quad_order)), dtype=np.clongdouble)  # sensing matrix
        hs = self.source_coord[2] + retraction  # source height with retraction

        # Initialize bar
        bar = tqdm(total=len(self.controls.k0),
                   desc='Calculating Tikhonov inversion (for the Quadrature Method)...', ascii=False)

        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):

            # loop over receivers
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((self.source_coord[0] - r_coord[0]) ** 2 + (
                        self.source_coord[1] - r_coord[1]) ** 2) ** 0.5  # Horizontal distance (S-R)
                zr = r_coord[2]  # Receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5  # Euclidean dist. related to the real source
                rq = (r ** 2 + (hs + zr - 1j * q) ** 2) ** 0.5  # Euclidean dist. related to the image sources
                Iq = (b - a) / 2 * weights * kernel_p(k0, rq)  # Integral related to the quadrature
                p_inc = (np.exp(-1j * k0 * r1)) / r1  # Incident pressure

                # Forming the sensing matrix (M x (1 + ng))
                h_mtx[jrec] = np.concatenate((p_inc, Iq), axis=None)

            # Measured data
            pm = self.pres_s[:, jf].astype(complex)

            # Compute SVD
            u, sig, v = lc.csvd(h_mtx.astype(complex))

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
                x = x2[:h_mtx.shape[1]] + 1j * x2[h_mtx.shape[1]:]
            elif method == 'Tikhonov':
                # phi_factors = (sig**2)/(sig**2+lambd_value**2)
                # # because csvd takes the hermitian of h_mtx and only the first m collumns of v
                # x = (v @ np.diag(phi_factors/sig) @ np.conjugate(u)) @ pm
                x = lc.tikhonov(u, sig, v, pm, lambd_value)
            else:
                Hm = np.matrix(h_mtx)
                x = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() + (lambd_value ** 2) * np.identity(len(pm))) @ pm
            self.pk.append(x)
            bar.update(1)
        bar.close()

    def reconstruct_p(self, a=-1, b=1, hz=0.01, n_pts=1, retraction=0, colormap=False):
        """ Reconstruct pressure.

        The reconstruction is done at a grid of points.

        Parameters
        ----------
        a : int
            lower bond of the integral
        b : float
            upper bound of the integral
        hz : float
            The height of the point
        n_pts : int
            Number of points
        retraction : float
            Retraction value of the source height. Default is 1 cm
        colormap: boolean
            Determines whether the receiver grid is for a colormap or not. Default is False

        Returns
        -------
        p_recon : (N_theta x N_freq) numpy ndarray
            The reconstructed pressure.
        p_recon_inc: (N_theta x N_freq) numpy ndarray
            The incident reconstructed pressure.
        p_recon_ref: (N_theta x N_freq) numpy ndarray
            The reflected reconstructed pressure.
        """

        # Set the grid used to reconstruct the surface impedance
        grid = Receiver()

        if colormap:
            grid.planar_xz(x_len=0.10, n_x=21, z0=0, z_len=0.10, n_z=21, yr=0.0)
        else:
            grid.line_array(startat=hz, n_rec=n_pts, direction='z')

        # Initialize variables
        roots, weights = roots_legendre(self.quad_order)  # roots and weights of G-L polynomials
        q = (((b - a) / 2) * roots + ((a + b) / 2))  # variable interval change
        h_mtx = np.zeros((grid.coord.shape[0], (1 + self.quad_order)), dtype=np.clongdouble)
        hs = self.source_coord[2] + retraction  # source height with retraction

        # Allocate some memory prior to loop
        self.p_recon = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.p_recon_inc = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.p_recon_ref = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)

        # Initialize bar
        bar = tqdm(total=len(self.controls.k0), desc='Calculating the reconstructed pressure (p_recon)...')

        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):

            # loop over receivers
            for jrec, g_coord in enumerate(grid.coord):
                r = ((self.source_coord[0] - g_coord[0]) ** 2 + (
                        self.source_coord[1] - g_coord[1]) ** 2) ** 0.5  # horizontal distance (S-R)
                zr = g_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5  # Euclidean dist. related to the real source
                rq = (r ** 2 + (hs + zr - 1j * q) ** 2) ** 0.5  # Euclidean dist. related to the image sources
                Iq = (b - a) / 2 * weights * kernel_p(k0, rq)  # Integral related to the quadrature
                p_inc = (np.exp(-1j * k0 * r1)) / r1  # Incident pressure

                # Forming the sensing matrix (M x (1 + ng))
                h_mtx[jrec] = np.concatenate((p_inc, Iq), axis=None)

            # complex amplitudes of all waves
            x = self.pk[jf]

            # pressure reconstruction
            self.p_recon[:, jf] = h_mtx @ x  # total pressure
            self.p_recon_inc[:, jf] = h_mtx[:, 0] * x[0]  # incident pressure
            self.p_recon_ref[:, jf] = h_mtx[:, 1:] @ x[1:]  # reflected pressure
            bar.update(1)
        bar.close()
        return self.p_recon, self.p_recon_inc, self.p_recon_ref

    def reconstruct_uz(self, a=-1, b=1, hz=0.01, n_pts=1, retraction=0):
        """ Reconstruct particle velocity.

        The reconstruction is done at a grid of points.

        Parameters
        ----------
        a : int
            lower bond of the integral
        b : float
            upper bound of the integral
        hz : float
            The height of the point
        n_pts : int
            Number of points
        retraction : float
            Retraction value of the source height. Default is 1 cm.

        Returns
        -------
        uz_recon : (N_theta x N_freq) numpy ndarray
            The reconstructed particle velocity.
        uz_recon_inc: (N_theta x N_freq) numpy ndarray
            The incident reconstructed particle velocity.
        uz_recon_ref: (N_theta x N_freq) numpy ndarray
            The reflected reconstructed particle velocity.
        """

        # Set the grid used to reconstruct the surface impedance
        grid = Receiver()
        grid.line_array(startat=hz, n_rec=n_pts, direction='z')

        # Initialize variables
        roots, weights = roots_legendre(self.quad_order)  # roots and weights of G-L polynomials
        q = (((b - a) / 2) * roots + ((a + b) / 2))  # variable interval change
        h_mtx = np.zeros((grid.coord.shape[0], (1 + self.quad_order)), dtype=np.clongdouble)
        hs = self.source_coord[2] + retraction  # source height with retraction

        # Allocate some memory prior to loop
        self.uz_recon = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.uz_recon_inc = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.uz_recon_ref = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)

        # Initialize bar
        bar = tqdm(total=len(self.controls.k0), desc='Calculating the reconstructed particle velocity (uz_recon)...')

        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):

            # loop over receivers
            for jrec, g_coord in enumerate(grid.coord):
                r = ((self.source_coord[0] - g_coord[0]) ** 2 + (
                        self.source_coord[1] - g_coord[1]) ** 2) ** 0.5  # horizontal distance (S-R)
                zr = g_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5  # Euclidean dist. related to the real source
                hz_q = hs + zr - 1j * q  # image source height
                rq = ((r ** 2) + (hz_q ** 2)) ** 0.5  # Euclidean dist. related to the image sources
                Iq = (b - a) / 2 * weights * kernel_uz(k0, rq, hz_q)  # Integral related to the quadrature
                uz_inc = ((np.exp(-1j * k0 * r1)) / r1) * ((hs - zr) / r1) * (1 + (1 / (1j * k0 * r1)))  # Incident uz

                # Forming the sensing matrix (M x (1 + ng))
                h_mtx[jrec] = np.concatenate((uz_inc, -Iq), axis=None)

            # complex amplitudes of all waves
            x = self.pk[jf]

            # particle velocity reconstruction
            self.uz_recon[0, jf] = h_mtx @ x  # total particle velocity
            self.uz_recon_inc[0, jf] = h_mtx[:, 0] * x[0]  # incident particle velocity
            self.uz_recon_ref[0, jf] = h_mtx[:, 1:] @ x[1:]  # reflected particle velocity
            bar.update(1)
        bar.close()
        return self.uz_recon, self.uz_recon_inc, self.uz_recon_ref

    def zs(self, a=-1, b=1, Lx=0.1, n_x=21, Ly=0.1, n_y=21, theta=None, avgZs=True, retraction=0):
        """ Reconstruct the surface impedance and estimate the absorption

        Reconstruct pressure and particle velocity at a grid of points on the
        surface of the absorber (z = 0.0). The absorption coefficient is also calculated.

        Parameters
        ----------
        a : int
            lower bond of the integral
        b : float
            upper bound of the integral
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
            Whether to average over <Zs> (default - True) or over <p>/<uz> (if False).
        retraction : float
            Retraction value of the source height. Default is 1 cm.

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

        # Initialize variables
        roots, weights = roots_legendre(self.quad_order)  # roots and weights of G-L polynomials
        q = (((b - a) / 2) * roots + ((a + b) / 2))  # variable interval change
        h_mtx = np.zeros((grid.coord.shape[0], (1 + self.quad_order)), dtype=np.clongdouble)
        h_mtx_uz = np.zeros((grid.coord.shape[0], (1 + self.quad_order)), dtype=np.clongdouble)
        hs = self.source_coord[2] + retraction  # source height with retraction

        # Allocate some memory prior to loop
        self.Zs = np.zeros(len(self.controls.k0), dtype=complex)
        self.alpha = np.zeros((len(theta), len(self.controls.k0)))
        self.p_s = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.uz_s = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)

        # Initialize bar
        bar = tqdm(total=len(self.controls.k0), desc='Calculating the surface impedance (Zs)...', ascii=False)

        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # loop over receivers
            for jrec, g_coord in enumerate(grid.coord):
                r = ((self.source_coord[0] - g_coord[0]) ** 2 + (
                        self.source_coord[1] - g_coord[1]) ** 2) ** 0.5  # horizontal distance (S-R)
                zr = g_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5  # Euclidean dist. related to the real source
                hz_q = hs + zr - 1j * q  # image source height
                rq = ((r ** 2) + (hz_q ** 2)) ** 0.5  # Euclidean dist. related to the image sources

                Iq_p = (b - a) / 2 * weights * kernel_p(k0, rq)  # Integral related to the quadrature - Pressure
                Iq_uz = (b - a) / 2 * weights * kernel_uz(k0, rq, hz_q)  # Integral related to the quadrature - Uz
                print(hs)
                p_inc = (np.exp(-1j * k0 * r1)) / r1  # Incident pressure
                uz_inc = ((np.exp(-1j * k0 * r1)) / r1) * ((hs - zr) / r1) * (1 + (1 / (1j * k0 * r1)))  # Incident uz

                # Forming the sensing matrix (M x (1 + ng))
                h_mtx[jrec] = np.concatenate((p_inc, Iq_p), axis=None)  # pressure sensing matrix
                h_mtx_uz[jrec] = np.concatenate((uz_inc, -Iq_uz), axis=None)  # uz sensing matrix

            # complex amplitudes of all waves
            x = self.pk[jf]

            # reconstruct pressure and particle velocity at surface
            p_surf_mtx = h_mtx @ x
            uz_surf_mtx = h_mtx_uz @ x

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

    def plot_colormap(self, press=None, freq=None, name='', dinrange=20):
        """Plots a color map of the pressure field.

        Parameters
        ----------
        press : list
            spectrum of the sound pressure for all receivers.
        freq : float
            desired frequency of the color map. If the frequency does not exist
            on the simulation, then it will choose the frequency just before the target.
        name : str
            pressure characteristic name. Measured or reconstructed
        dinrange : float
            Dynamic range of the color map

        Returns
        ---------
        plt : Figure object
        """

        # Set the grid plane used to the plot
        receivers = Receiver()
        receivers.planar_xz(x_len=0.1, n_x=21, z0=0, z_len=0.1, n_z=21, yr=0.0)

        # Initialize variables
        self.fpts = receivers
        id_f = np.where(self.controls.freq <= freq)

        id_f = id_f[0][-1]
        color_par = 20 * np.log10(np.abs(press[:, id_f]) / np.amax(np.abs(press[:, id_f])))

        # Create triangulation
        triang = tri.Triangulation(self.fpts.coord[:, 0], self.fpts.coord[:, 2])

        # Figure
        fig = plt.figure()
        plt.title(str(self.controls.freq[id_f]) + ' [Hz] - |P(f)| ' + name)
        p = plt.tricontourf(triang, color_par, np.linspace(-dinrange, 0, int(dinrange)), cmap=plt.get_cmap('jet'))
        fig.colorbar(p)
        plt.xlabel(r'$x$ [m]')
        plt.ylabel(r'$z$ [m]')

        return plt

    def plot_colormap2(self, press=None, name='', dinrange=20):
        """Plots a color map of the pressure field for all frequencies at once.

        Parameters
        ----------
        press : list
            spectrum of the sound pressure for all receivers.
        name : str
            pressure characteristic name. Measured or reconstructed
        dinrange : float
            Dynamic range of the color map

        Returns
        ---------
        plt : Figure object
        """

        # Set the grid plane used to the plot
        receivers = Receiver()
        receivers.planar_xz(x_len=0.1, n_x=21, z0=0, z_len=0.1, n_z=21, yr=0.0)

        self.fpts = receivers

        for n in range(len(self.controls.freq)):

            id_f = n
            color_par = 20*np.log10(np.abs(press[:, id_f])/np.amax(np.abs(press[:, id_f])))

            # Create triangulation
            triang = tri.Triangulation(self.fpts.coord[:, 0], self.fpts.coord[:, 2])

            # Figure
            fig = plt.figure()
            plt.title(str(self.controls.freq[n]) + ' [Hz] - |P(f)| ' + name)
            p = plt.tricontourf(triang, color_par, np.linspace(-dinrange, 0, int(dinrange)), cmap=plt.get_cmap('jet'))
            fig.colorbar(p)
            plt.xlabel(r'$x$ [m]')
            plt.ylabel(r'$z$ [m]')
        return plt

    def gradient(self, pres_s=None, r_coord=None):
        """ Reconstruct pressure.

                The reconstruction is done at a grid of points.

                Parameters
                ----------
                pres_s : list
                    spectrum of the sound pressure for all receivers
                r_coord : ndarray
                    the coordinates of the microphones to calculate the pressure gradient
                """

        # Initialize variables
        r_norm = np.linalg.norm(r_coord[1] - r_coord[0])

        # Allocate some memory prior to loop
        self.uz_grad = np.zeros((1, len(self.controls.k0)), dtype=complex)

        # Initialize bar
        bar = tqdm(total=len(self.controls.k0), desc='Calculating the gradient...')

        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            # particle velocity - gradient
            self.uz_grad[0, jf] = (1 / 1j*k0*r_norm)*(pres_s[1][jf] - pres_s[0][jf])
            bar.update(1)
        bar.close()

    def zs_wd(self, a=-1, b=1, Lx=0.1, n_x=21, Ly=0.1, n_y=21, theta=None, avgZs=True, retraction=0,
              direction=None, overlap=0.5, n_moves=1):
        """ Reconstruct the surface impedance and estimate the absorption

        Reconstruct pressure and particle velocity at a grid of points on the
        surface of the absorber (z = 0.0). The absorption coefficient is also calculated.

        Parameters
        ----------
        a : int
            lower bond of the integral
        b : float
            upper bound of the integral
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
            Whether to average over <Zs> (default - True) or over <p>/<uz> (if False).
        retraction : float
            Retraction value of the source height. Default is 1 cm.

        Returns
        -------
        alpha : (N_theta x N_freq) numpy ndarray
            The absorption coefficients for each target incident angle.
        """

        if theta is None:
            theta = [0]

        # Set the grid used to reconstruct the surface impedance
        grid = Receiver()
        grid.moving_planar_array(x_len=Lx, n_x=n_x, y_len=Ly, n_y=n_y, zr=0.0,
                                 direction=direction, overlap=overlap, n_moves=n_moves)

        # Initialize variables
        roots, weights = roots_legendre(self.quad_order)  # roots and weights of G-L polynomials
        q = (((b - a) / 2) * roots + ((a + b) / 2))  # variable interval change
        h_mtx = np.zeros((grid.coord.shape[0], (1 + self.quad_order)), dtype=np.clongdouble)
        h_mtx_uz = np.zeros((grid.coord.shape[0], (1 + self.quad_order)), dtype=np.clongdouble)
        hs = self.source_coord[2] + retraction  # source height with retraction

        # Allocate some memory prior to loop
        self.Zs = np.zeros(len(self.controls.k0), dtype=complex)
        alpha = np.zeros((len(theta), len(self.controls.k0)))
        self.p_s = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.uz_s = np.zeros((len(grid.coord), len(self.controls.k0)), dtype=complex)
        self.alphas_wd = []

        # Initialize bar
        bar = tqdm(total=len(self.controls.k0), desc='Calculating the surface impedance (Zs)...', ascii=False)

        # Initialize bar
        for n in (range(0, len(grid.wd_array))):
            # Freq loop
            for jf, k0 in enumerate(self.controls.k0):
                # loop over receivers
                for jrec, g_coord in enumerate(grid.wd_array[n]):
                    r = ((self.source_coord[0] - g_coord[0]) ** 2 + (
                            self.source_coord[1] - g_coord[1]) ** 2) ** 0.5  # horizontal distance (S-R)
                    zr = g_coord[2]  # receiver height
                    r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5  # Euclidean dist. related to the real source
                    hz_q = hs + zr - 1j * q  # image source height
                    rq = ((r ** 2) + (hz_q ** 2)) ** 0.5  # Euclidean dist. related to the image sources

                    Iq_p = (b - a) / 2 * weights * kernel_p(k0, rq)  # Integral related to the quadrature - Pressure
                    Iq_uz = (b - a) / 2 * weights * kernel_uz(k0, rq, hz_q)  # Integral related to the quadrature - Uz

                    p_inc = (np.exp(-1j * k0 * r1)) / r1  # Incident pressure
                    uz_inc = ((np.exp(-1j * k0 * r1)) / r1) * ((hs - zr) / r1) * (1 + (1 / (1j * k0 * r1)))  # Incident uz

                    # Forming the sensing matrix (M x (1 + ng))
                    h_mtx[jrec] = np.concatenate((p_inc, Iq_p), axis=None)  # pressure sensing matrix
                    h_mtx_uz[jrec] = np.concatenate((uz_inc, -Iq_uz), axis=None)  # uz sensing matrix

                # complex amplitudes of all waves
                x = self.pk[jf]

                # reconstruct pressure and particle velocity at surface
                p_surf_mtx = h_mtx @ x
                uz_surf_mtx = h_mtx_uz @ x

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
                alpha[jtheta, :] = 1 - (np.abs(np.divide((self.Zs * np.cos(dtheta) - 1),
                                                              (self.Zs * np.cos(dtheta) + 1)))) ** 2
            self.alphas_wd.append(alpha)
            alpha = np.zeros((len(theta), len(self.controls.k0)))
        return self.alphas_wd
    # def save(self, filename = 'my_quad_tes', path = '/home/eric/dev/insitu/data/'):
    #     """
    #     This method is used to save the simulation object
    #     """
    #     self.path_filename = path + filename + '.pkl'
    #     f = open(self.path_filename, 'wb')
    #     pickle.dump(self.__dict__, f, 2)
    #     f.close()
    #
    # def load(self, filename = 'my_quad_test', path = '/home/eric/dev/insitu/data/'):
    #     """
    #     This method is used to load a simulation object. You build a empty object
    #     of the class and load a saved one. It will overwrite the empty one.
    #     """
    #     lpath_filename = path + filename + '.pkl'
    #     f = open(lpath_filename, 'rb')
    #     tmp_dict = pickle.load(f)
    #     f.close()
    #     self.__dict__.update(tmp_dict)

    # def plot_tests(self, path='', ng_array=None, b_max=None, type_test = 0):
    #     """Plots a color map of the pressure field.
    #
    #        Parameters
    #        ----------
    #         path : str
    #             the path
    #         ng_array : ndarray
    #             array of quadrature orders
    #         b_max : ndarray
    #             array of upper limits
    #         type_test : int
    #             type of the test. 1: quadrature order, 2: integral limit
    #
    #         Returns
    #         ---------
    #             plt : Figure object
    #         """
    #     if type_test == 1:
    #         for i in ng_array:
    #             name = f'ng{int(i)}_a0_b{int(b_max)}_d5cm_el0d_az0d_r30cm_resist50k'
    #             self.load(path=path, filename=name)
    #             zss = self.
    #
    #         plt.figure(figsize=(16, 9))
    #         plt.title('Reconstructed surface impedance')
    #         for i in np.linspace(0, p_hz, 5):
    #             plt.semilogx(controls.freq, np.real(p_recon_ref[aux][0, :]), '-', linewidth=2,
    #                          markersize=5, alpha=0.8, label=r'$P_\mathrm{Real}$ - p = %.3f [m]' % i)
    #             plt.semilogx(controls.freq, np.imag(p_recon_ref[aux][0, :]), '--', linewidth=2,
    #                          markersize=5, alpha=0.8, label=r'$P_\mathrm{Imag}$ - p = %.3f [m]' % i)
    #             aux = aux+1
    #         plt.xlabel('Frequency [Hz]')
    #         plt.ylabel('Amplitude [Pa]')
    #         plt.xticks([50, 100, 500, 1000, 2000, 3000], ['50', '100', '500', '1k', '2k', '3k'])
    #         plt.grid(linestyle='--', which='both')
    #         plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #         plt.xlim((80, 3000))
    #         plt.tight_layout()
    #         plt.show()
