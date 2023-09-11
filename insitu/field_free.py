import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from insitu.controlsair import load_cfg
# import scipy.integrate as integrate
import scipy as spy
import time
import sys
# from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
import pickle
import time
from controlsair import sph2cart, cart2sph
from controlsair import plot_spk
from sources import Source

class FreeField(object):
    """ Calculates the sound field in free-field conditions.

    It is used to calculate the sound pressure using
    the free-field formulations of plane and spherical waves

    Attributes
    ----------
    pres_s - list of receiver pressure spectrums for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.
        Each line of the matrix is a spectrum of a sound pressure for a receiver.
        Each column is a set of sound pressure at all receivers for a frequency.

    Methods
    ----------
    planewave_ff(theta = 0, phi = 0, Ap = 1, calc_ev = False, kx_f = 2, ky_f = 2, Ae = 1)
        Calculates the sound field due to a plane wave in free field

    monopole_ff(sources)
        Calculates the sound field due to a monopole wave in free field

    mirrorsource(sources)
        Calculates the sound field due to a monopole and its image

    add_noise(snr = 30, uncorr = False)
        Add gaussian noise to the simulated data.

    plot_scene(vsam_size = 2, mesh = True)
        Plot of the scene using matplotlib - not redered

    plot_pres()
        Plot the spectrum of the sound pressure for all receivers
    """
    def __init__(self, air, controls, receivers):
        """

        Parameters
        ----------
        air : object (AirProperties)
            The relevant properties of the air: c0 (sound speed) and rho0 (air density)
        controls : object (AlgControls)
            Controls of the simulation (frequency spam)
        receivers : object (Receiver)
            The receivers in the field

        The objects are stored as attributes in the class (easier to retrieve).
        """
        self.air = air
        self.controls = controls
        self.material = []
        self.receivers = receivers
        self.pres_s = []
        # self.uz_s = []

    def planewave_ff(self, theta = 0, phi = 0, Ap = 1, calc_ev = False, kx_f = 2, ky_f = 2, Ae = 1):
        """ Calculates the sound field due to a plane wave in free field

        Parameters
        ----------
        theta : float
            Elevation angle of the propagating plane wave in [rad]
        phi : float
            Azimuth angle of the propagating plane wave in [rad]
        Ap : float
            Amplitude of the propagating plane wave
        calc_ev : bool
            Whether to include an evanescent wave or not in the calculations.
            Default is False - just a propagating wave
        kx_f : float
            Evanescent wave kx factor: kx = kx_f * k0
        ky_f : float
            Evanescent wave ky factor: ky = ky_f * k0
        Ap : float
            Amplitude of the evanescent plane wave
        """
        self.sources = Source([np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta), np.cos(theta)])
        self.pres_s = []
        # for js, s_coord in enumerate(self.sources.coord):
            # hs = s_coord[2] # source height
        # Initialize
        pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
        # Loop over receivers
        for jrec, r_coord in enumerate(self.receivers.coord):
            for jf, k0 in enumerate(self.controls.k0):
                # Propagating wave number
                kx = k0 * np.cos(phi) * np.sin(theta)
                ky = k0 * np.sin(phi) * np.sin(theta)
                kz = k0 * np.cos(theta)
                k_vec = np.array([kx, ky, kz])
                if calc_ev:
                    # Evanescent wave number
                    kxe = k0 * kx_f
                    kye = k0 * ky_f
                    kze = (kxe**2 + kye**2 - k0**2)**0.5
                    # Evanescent pressure
                    p_ev = (np.exp(-kze * r_coord[2])) * (np.exp(-1j * (kxe * r_coord[0] + kye * r_coord[1])))
                    pres_rec[jrec, jf] = Ap * np.exp(-1j * np.dot(k_vec, r_coord)) + Ae * p_ev
                else:
                    pres_rec[jrec, jf] = Ap * np.exp(-1j * np.dot(k_vec, r_coord))
        self.pres_s.append(pres_rec)

    def monopole_ff(self, sources):
        """ Calculates the sound field due to a monopole wave in free field

        Parameters
        ----------
        sources : object (Source)
            The sound sources in the field
        """
        self.sources = sources
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            # hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = np.linalg.norm(r_coord - s_coord) # distance source-receiver
                pres_rec[jrec, :] = (np.exp(-1j * self.controls.k0 * r)) / r
            self.pres_s.append(pres_rec)

    def mirrorsource(self, sources):
        """ Calculates the sound field due to a monopole and its image

        Calculate the half-space response due to a monopole with a given source coordinate.
        The original monopole is mirroed against the plane z=0 and its strength is the same 
        as the original monopole

        Parameters
        ----------
        sources : object (Source)
            The sound sources in the field
        """

        self.sources = sources
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            # s_coord_is = s_coord
            # # s_coord_is[2] = -s_coord[2]
            # print(s_coord)
            # print(s_coord_is)
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r1 = np.linalg.norm(r_coord - s_coord) # distance source-receiver
                r2 = np.linalg.norm(r_coord - np.array([s_coord[0], s_coord[1], -s_coord[2]])) # distance source-receiver
                pres_rec[jrec, :] = (np.exp(-1j * self.controls.k0 * r1)) / r1 +\
                    (np.exp(-1j * self.controls.k0 * r2)) / r2
            self.pres_s.append(pres_rec)

    def add_noise(self, snr = 30, uncorr = False):
        """ Add gaussian noise to the simulated data.

        The function is used to add noise to the pressure data.
        it reads the clean signal and estimate its power. Then, it estimates the power
        of the noise that would lead to the target SNR. Then it draws random numbers
        from a Normal distribution with standard deviation =  noise power

        Parameters
        ----------
        snr : float
            The signal to noise ratio you want to emulate
        uncorr : bool
            If added noise to each receiver is uncorrelated or not.
            If uncorr is True the the noise power is different for each receiver
            and frequency. If uncorr is False the noise power is calculated from
            the average signal magnitude of all receivers (for each frequency).
            The default value is False
        """
        signal = self.pres_s[0]
        if uncorr:
            signalPower_lin = (np.abs(signal)/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
        else:
            # signalPower_lin = (np.abs(np.mean(signal, axis=0))/np.sqrt(2))**2
            signalPower_lin = ((np.mean(np.abs(signal), axis=0))/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
        np.random.seed(0)
        noise = np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape)
        self.pres_s[0] = signal + noise

    def plot_pres(self):
        """ Plot the spectrum of the sound pressure for all receivers
        """
        plot_spk(self.controls.freq, self.pres_s, ref = 20e-6)

    def plot_scene(self, vsam_size = 2):
        """ Plot of the scene using matplotlib - not redered

        Parameters
        ----------
        vsam_size : float
            Scene size. Just to make the plot look nicer. You can choose any value.
            An advice is to choose a value bigger than the sample's largest dimension.
        """
        fig = plt.figure()
        # fig.canvas.set_window_title("Measurement scene")
        ax = plt.axes(projection ="3d")
        vertices = np.array([[-vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, vsam_size/2, 0.0],
            [-vsam_size/2, vsam_size/2, 0.0]])
        verts = [list(zip(vertices[:,0],
                vertices[:,1], vertices[:,2]))]
        # patch plot
        collection = Poly3DCollection(verts,
            linewidths=2, alpha=0.3, edgecolor = 'black', zorder=2)
        collection.set_facecolor('silver')
        ax.add_collection3d(collection)
        # plot source
        for s_coord in self.sources.coord:
            ax.scatter(s_coord[0], s_coord[1], s_coord[2],
                color='red',  marker = "o", s=20)
        # plot receiver
        for r_coord in self.receivers.coord:
            ax.scatter(r_coord[0], r_coord[1], r_coord[2],
                color='blue',  marker = "o")
        ax.set_xlabel('X axis')
        # plt.xticks([], [])
        ax.set_ylabel('Y axis')
        # plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='both')
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.view_init(elev=30, azim=-50)


    # def plot_cmap(self, freq = 1000, name = '', save = False, path='', fname=''):
    #     '''
    #     plot color map from plane_xz
    #     '''
    #     # id_f = np.where(self.controls.freq <= freq)
    #     # id_f = id_f[0][-1]
    #     color_par = np.real((self.p_grid))
    #     fig = plt.figure()
    #     p=plt.contourf(self.receivers.x_grid, self.receivers.z_grid,
    #         color_par)
    #     fig.colorbar(p)
    #     plt.xlabel('x [m]')
    #     plt.ylabel('z [m]')
    #     plt.title('|p(f)| at ' + str(freq) + 'Hz - '+ name)
    #     if save:
    #         filename = path + fname
    #         plt.savefig(fname = filename, format='png')

    
    # def planewave_diffuse(self, randomize = True, seed = 0):
    #     '''
    #     Method used to calculate the free field response due to multiple plane wave incidence.
    #     The directions are supposed to come from a list of unity vectors contained in sources.coord
    #     Inputs:
    #         theta - the elevation angle
    #         phi - the azimuth angle
    #     '''
    #     # r, theta, phi = sph2cart(self.sources.coord[:,0], self.sources.coord[:,1], self.sources.coord[:,2])
    #     pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
    #     bar = ChargingBar('Calculating sound pressure at each receiver', max=len(self.receivers.coord), suffix='%(percent)d%%')
    #     for jrec, r_coord in enumerate(self.receivers.coord):
    #         # r = np.linalg.norm(r_coord) # distance source-receiver
    #         for jf, k0 in enumerate(self.controls.k0):
    #             k_vec = k0 * self.sources.coord
    #             if randomize:
    #                 np.random.seed(seed)
    #                 q = np.random.randn(1) + 1j*np.random.randn(1)
    #             else:
    #                 q = 1
    #             pres_rec[jrec, jf] = np.sum(q * np.exp(1j * np.dot(k_vec, r_coord)))
    #         bar.next()
    #     bar.finish()
    #     self.pres_s = [pres_rec]

    # def pw_ev_grid_ff(self, theta_p = 0, phi_p = 0, Ap=1, kx_f = 2, ky_f = 2, Ae = 1, freq = 1000):
    #     '''
    #     Method used to calculate the free field response due to a propagating plane wave 
    #     and an evanescent wave (at a grid)
    #     Inputs:
    #         theta_p - the elevation angle of the propagating plane wave
    #         phi_p - the azimuth angle of the propagating plane wave
    #     '''
    #     id_f = np.where(self.controls.freq <= freq)
    #     id_f = id_f[0][-1]
    #     # K vector of propagating wave
    #     k0 = self.controls.k0[id_f]
    #     kx = k0 * np.cos(phi_p) * np.sin(theta_p)
    #     ky = k0 * np.sin(phi_p) * np.sin(theta_p)
    #     kz = k0 * np.cos(theta_p)
    #     k_vec = np.array([kx, ky, kz])
    #     # K vector of evanescent wave
    #     kxe = k0 * kx_f
    #     kye = k0 * ky_f
    #     # ky = k0 * np.sin(phi_e) * np.sin(theta_e)
    #     kze = (kxe**2 + kye**2 - k0**2)**0.5
    #     # k_ev = np.array([kxe, 0, kze])
    #     # Loop the receivers
    #     self.p_grid = np.zeros(self.receivers.x_grid.shape, dtype = complex)
    #     for jx in np.arange(self.receivers.x_grid.shape[0]):
    #         for jy in np.arange(self.receivers.x_grid.shape[1]):
    #             r_coord = np.array([self.receivers.x_grid[jx, jy],
    #                 0.0,
    #                 self.receivers.z_grid[jx, jy]])
    #             p_prop = Ap * np.exp(1j * np.dot(k_vec, r_coord))
    #             p_ev = Ae * (np.exp(-kze * r_coord[2])) *\
    #                 (np.exp(1j * (kxe * r_coord[0] + kye * r_coord[1])))
    #             self.p_grid[jx, jy] = p_prop + p_ev




    # def add_noise(self, snr = np.Inf):
    #     '''
    #     Method used to artificially add gaussian noise to the measued data
    #     '''
    #     for js, s_coord in enumerate(self.sources.coord):
    #         # hs = s_coord[2] # source height
    #         pres_rec = self.pres_s[js]
    #         # N = (pres_rec).shape
    #         # print(N)
    #         for jrec, r_coord in enumerate(self.receivers.coord):
    #             for jf, k0 in enumerate(self.controls.k0):
    #                 # Here, the signal power is 2 (1 for real and imaginary component)
    #                 signal = pres_rec[jrec, jf]
    #                 signalPower_lin = np.abs(signal)**2
    #                 signalPower_dB = 10 * np.log10(signalPower_lin)
    #                 noisePower_dB = signalPower_dB - snr
    #                 noisePower_lin = 10 ** (noisePower_dB/10)
    #                 noise = np.sqrt(noisePower_lin/2)*(np.random.randn(1) + 1j*np.random.randn(1))
    #                 self.pres_s[js][jrec][jf] = signal + noise
