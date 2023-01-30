import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
#from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
import pickle
from controlsair import plot_spk

class LocallyReactiveInfSph(object):
    """ Calculates the sound field using the q-term formulation.

    Calculate the sound pressure and particle velocity using the q-term
    formulation (exact for spherical waves above locally reactive and
    infinite samples)

    Attributes
    ----------
    pres_s - list of receiver pressure spectrums for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.
        Each line of the matrix is a spectrum of a sound pressure for a receiver.
        Each column is a set of sound pressure at all receivers for a frequency.
    uz_s - list of receiver velocity spectrums (z-dir) for each source.
        Each element of the list has a (N_rec x N_freq) matrix for a given source.

    Methods
    ----------
    planewave_ff(theta = 0, phi = 0, Ap = 1, calc_ev = False, kx_f = 2, ky_f = 2, Ae = 1)
        Calculates the sound field due to a plane wave in free field

    monopole_ff(sources)
        Calculates the sound field due to a monopole wave in free field

    mirrorsource(sources)
        Calculates the sound field due to a monopole and its image

    Methods
        ---------
    p_loc(upper_int_limit = 20)
        Calculates the sound pressure spectrum for all sources and receivers

    uz_loc(upper_int_limit = 20)
        Calculates the z-dir particle velocity spectrum for all sources and receivers

    p_mult(upper_int_limit = 10, randomize = False, amp_min = 0.0002, amp_max = 20)
        Calculates the sound pressure spectrum under diffuse field for receivers

    add_noise(snr = 30, uncorr = False)
        Add gaussian noise to the simulated data.

    plot_scene(vsam_size = 2, mesh = True)
        Plot of the scene using matplotlib - not redered

    plot_pres()
        Plot the spectrum of the sound pressure for all receivers

    plot_uz()
        Plot the spectrum of the particle velocity in zdir for all receivers

    save(filename = 'my_bemflush', path = '/home/eric/dev/insitu/data/bem_simulations/')
        To save the simulation object as pickle

    load(filename = 'my_qterm', path = '/home/eric/dev/insitu/data/bem_simulations/')
        Load a simulation object.
    """

    def __init__(self, air, controls, material, sources, receivers,
                 bar_mode = 'terminal'):
        """

        Parameters
        ----------
        air : object (AirProperties)
            The relevant properties of the air: c0 (sound speed) and rho0 (air density)
        controls : object (AlgControls)
            Controls of the simulation (frequency spam)
        material : object (PorousAbsorber)
            Contains the material properties (surface impedance)
        sources : object (Source)
            The sound sources in the field
        receivers : object (Receiver)
            The receivers in the field

        The objects are stored as attributes in the class (easier to retrieve).
        """

        self.air = air
        self.controls = controls
        self.material = material
        self.sources = sources
        self.receivers = receivers
        self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance
        self.pres_s = []
        self.uz_s = []
        if bar_mode == 'notebook':
            from tqdm.notebook import trange, tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm

    def p_loc(self, upper_int_limit = 20):
        """ Calculates the sound pressure spectrum for all sources and receivers

        Parameters
        ----------
        upper_int_limit : float
            upper bound of the integral
        """
        # self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1])**2)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                # setup progressbar
                print('Calculate sound pressure for source {} and receiver {}'.format(js+1, jrec+1))
                #bar = ChargingBar('Processing sound pressure (q-term)', max=len(self.controls.k0), suffix='%(percent)d%%')
                bar = self.tqdm(total = len(self.controls.k0),
                           desc = 'Processing sound pressure (q-term)')
                # pres = []
                for jf, k0 in enumerate(self.controls.k0):
                    f_qr = lambda q: np.real((np.exp(-q * k0 * self.beta[jf])) *
                        ((np.exp(-1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5)) /
                        ((r**2 + (hs + zr - 1j*q)**2) ** 0.5)))
                    f_qi = lambda q: np.imag((np.exp(-q * k0 * self.beta[jf])) *
                        ((np.exp(-1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5)) /
                        ((r**2 + (hs + zr - 1j*q)**2) ** 0.5)))
                    Iq_real = integrate.quad(f_qr, 0.0, upper_int_limit)
                    Iq_imag = integrate.quad(f_qi, 0.0, upper_int_limit)
                    # Iq_real = integrate.quadrature(f_qr, 0.0, upper_int_limit, maxiter = 500)
                    # Iq_imag = integrate.quadrature(f_qi, 0.0, upper_int_limit, maxiter = 500)
                    Iq = Iq_real[0] + 1j * Iq_imag[0]
                    pres_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1) + (np.exp(-1j * k0 * r2) / r2) - 2 * k0 * self.beta[jf] * Iq
                    # pres.append((np.exp(-1j * k0 * r1) / r1) +
                    #     (np.exp(-1j * k0 * r2) / r2) - 2 * k0 * self.beta[jf] * Iq)
                    # Progress bar stuff
                    bar.update(1)
                bar.close()
            self.pres_s.append(pres_rec)

    def uz_loc(self, upper_int_limit = 20):
        """ Calculates the z-dir particle velocity spectrum for all sources and receivers

        Parameters
        ----------
        upper_int_limit : float
            upper bound of the integral
        """
        # setup progressbar
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            uz_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((s_coord[0] - r_coord[0])**2.0 + (s_coord[1] - r_coord[1])**2.0)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                print('Calculate particle vel. (z-dir) for source {} and receiver {}'.format(js+1, jrec+1))
                # bar = ChargingBar('Processing particle velocity z-dir (q-term)',
                #     max=len(self.controls.k0), suffix='%(percent)d%%')
                bar = self.tqdm(total = len(self.controls.k0),
                           desc = 'Processing particle velocity z-dir (q-term)')
                for jf, k0 in enumerate(self.controls.k0):
                    # f_qr = lambda q: np.real((np.exp(-q * k0 * self.beta[jf])) *
                    #     ((np.exp(-1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5)) /
                    #     ((r**2 + (hs + zr - 1j*q)**2) ** 0.5)))
                    f_qr = lambda q: np.real(((np.exp(-q * k0 * self.beta[jf])) *
                        ((np.exp(-1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5)) /
                        ((r**2 + (hs + zr - 1j*q)**2) ** 0.5))) *
                        ((hs + zr - 1j*q) / (r**2 + (hs + zr - 1j*q)**2)**0.5) *
                        (1 + (1 / (1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5))))

                    f_qi = lambda q: np.imag(((np.exp(-q * k0 * self.beta[jf])) *
                        ((np.exp(-1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5)) /
                        ((r**2 + (hs + zr - 1j*q)**2) ** 0.5))) *
                        ((hs + zr - 1j*q) / (r**2 + (hs + zr - 1j*q)**2)**0.5) *
                        (1 + (1 / (1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5))))
                    Iq_real = integrate.quad(f_qr, 0.0, upper_int_limit)
                    Iq_imag = integrate.quad(f_qi, 0.0, upper_int_limit)
                    # Iq_real = integrate.quadrature(f_qr, 0.0, upper_int_limit, maxiter = 500)
                    # Iq_imag = integrate.quadrature(f_qi, 0.0, upper_int_limit, maxiter = 500)
                    Iq = Iq_real[0] + 1j * Iq_imag[0]
                    uz_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1)*\
                        (1 + (1 / (1j * k0 * r1)))* ((hs - zr)/r1)-\
                        (np.exp(-1j * k0 * r2) / r2) *\
                        (1 + (1 / (1j * k0 * r2))) * ((hs + zr)/r2)+\
                        2 * k0 * self.beta[jf] * Iq
                    # pres.append((np.exp(-1j * k0 * r1) / r1) +
                    #     (np.exp(-1j * k0 * r2) / r2) - 2 * k0 * self.beta[jf] * Iq)
                    # Progress bar stuff
                    bar.update(1)
                bar.close()
            self.uz_s.append(uz_rec)

    def p_mult(self, upper_int_limit = 10, randomize = False, amp_min = 0.0002, amp_max = 20):
        """ Calculates the sound pressure spectrum under diffuse field for receivers

        This method calculates the sound pressure spectrum for a distribution of sources at all receivers.
        It considers that the integration can be done once by considering a sumation of the contributions of
        all sound sources in the integrand.

        Parameters
        ----------
        upper_int_limit : float
            upper bound of the integral
        randomize : bool
            wether each sound source has a random amplitude (default is False)
        amp_min : float
            minimum amplitude of sound sources (default 0.0002 Pa or 20 dB)
        amp_max : float
            maximum amplitude of sound sources (default 20 Pa or 120 dB)
        """
        # get array of source heights
        hs = self.sources.coord[:,2]
        # initialize
        pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
        # loop over receivers
        for jrec, r_coord in enumerate(self.receivers.coord):
            # get arrays of distances
            r = ((self.sources.coord[:,0] - r_coord[0])**2 + (self.sources.coord[:,1] - r_coord[1])**2)**0.5 # horizontal distance source-receiver
            zr = r_coord[2]  # receiver height
            r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
            r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
            # setup progressbar
            print('Calculate sound pressure for receiver {}'.format(jrec+1))
            # bar = ChargingBar('Processing sound pressure (NLR)', max=len(self.controls.k0), suffix='%(percent)d%%')
            bar = self.tqdm(total = len(self.controls.k0),
                           desc = 'Processing diffuse sound pressure')
            # seed randomizition
            np.random.seed(0)
            # self.q = np.zeros((len(self.controls.freq), len(self.sources.coord)), dtype = complex)
            for jf, k0 in enumerate(self.controls.k0):
                # randomization of pressure
                if randomize:
                    amp = np.random.uniform(low = amp_min, high = amp_max, size = len(self.sources.coord))
                    phase = np.random.uniform(low = 0, high = 2*np.pi, size = len(self.sources.coord))
                    qs = amp * np.exp(1j*phase)
                    # self.q[jf,:] = q
                else:
                    qs = np.ones(len(self.sources.coord))
                # integrand
                fq = lambda q: (np.exp(-q * k0 * self.beta[jf])) *\
                    np.sum(qs * ((np.exp(-1j * k0 * (r**2 + (hs + zr - 1j*q)**2)**0.5)) /\
                    ((r**2 + (hs + zr - 1j*q)**2) ** 0.5)))
                fs_r = lambda q: np.real(fq(q))
                fs_i = lambda q: np.imag(fq(q))
                # Integrate
                Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit)
                Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit)
                Iq = Iq_real[0] + 1j * Iq_imag[0]
                # Pressure
                pres_rec[jrec, jf] = (np.sum(qs * (np.exp(-1j * k0 * r1) / r1 + np.exp(-1j * k0 * r2) / r2)) -\
                    2 * k0 * self.beta[jf] * Iq)
                bar.update(1)
            bar.close()
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

    def plot_uz(self):
        """ Plot the spectrum of the particle velocity in zdir for all receivers
        """
        plot_spk(self.controls.freq, self.uz_s, ref = 5e-8)

    def plot_scene(self, vsam_size = 50):
        """ Plot of the scene using matplotlib - not redered

        Parameters
        ----------
        vsam_size : float
            Scene size. Just to make the plot look nicer. You can choose any value.
            An advice is to choose a value bigger than the sample's largest dimension.
        """
        fig = plt.figure()
        fig.canvas.set_window_title("Measurement scene")
        ax = fig.gca(projection='3d')
        # vertexes plot
        vertices = np.array([[-vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, -vsam_size/2, 0.0],
            [vsam_size/2, vsam_size/2, 0.0],
            [-vsam_size/2, vsam_size/2, 0.0]])
        # ax.scatter(vertices[:,0], vertices[:,1],
        #     vertices[:,2], color='blue')
        verts = [list(zip(vertices[:,0],
                vertices[:,1], vertices[:,2]))]
        # patch plot
        collection = Poly3DCollection(verts,
            linewidths=1, alpha=0.9, edgecolor = 'gray')
        collection.set_facecolor('silver')
        ax.add_collection3d(collection)
        for s_coord in self.sources.coord:
            ax.scatter(s_coord[0], s_coord[1], s_coord[2],
                color='black',  marker = "*", s=500)
        for r_coord in self.receivers.coord:
            ax.scatter(r_coord[0], r_coord[1], r_coord[2],
                color='blue',  marker = "o")
        ax.set_xlabel('X axis')
        plt.xticks([], [])
        ax.set_ylabel('Y axis')
        plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='none')
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.view_init(elev=5, azim=-55)
        # ax.invert_zaxis()
        plt.show() # show plot

    def save(self, filename = 'my_qterm', path = '/home/eric/dev/insitu/data/'):
        """ To save the simulation object as pickle

        Parameters
        ----------
        filename : str
            name of the file
        pathname : str
            path of folder to save the file
        """
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'my_qterm', path = '/home/eric/dev/insitu/data/'):
        """ Load a simulation object.

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

def load_simu(filename = 'my_qterm', path = '/home/eric/dev/insitu/data/'):
    '''
    This function is used to load a simulation object. You build a empty object
    of the class and load a saved one. It will overwrite the empty one.
    '''
    # with open(path + filename+'.pkl', 'rb') as input:
    #     simu_data = pickle.load(input)
    # return simu_data
    lpath_filename = path + filename + '.pkl'
    f = open(lpath_filename, 'rb')
    tmp_dict = pickle.load(f)
    f.close()
    return tmp_dict.update(tmp_dict)


