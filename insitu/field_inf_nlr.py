import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy.special as sp
from scipy import optimize
import scipy as spy
import time
import sys
from tqdm import tqdm
import pickle
from controlsair import plot_spk
import utils_insitu as ut_is
# import quadpy.line_segment._gauss_lobatto as qp

class NLRInfSph(object):
    '''
    A class to calculate the sound pressure and particle velocity
    using the non-locally reactive formulation (exact for spherical waves on 
    non-locally reactive and infinite samples)
    The inputs are the objects: air, controls, material, sources, receivers
    '''
    def __init__(self, air = [], controls = [], material = [], sources = [], receivers = [], sing_step = 0.01):
        self.limits = np.arange(start = 0, stop = 1 + sing_step, step = sing_step)
        ## air
        self.air = air
        self.controls = controls
        self.material = material
        ## material
        try:
            self.Zp = material.Zp
            self.kp = material.kp
            # self.thick = material.thickness
            self.cp = np.divide(2*np.pi*self.controls.freq, self.kp)
            self.rhop = np.divide(self.Zp * self.kp, 2*np.pi*self.controls.freq)
            self.m = np.divide(self.air.rho0, self.rhop)
            self.n = np.divide(self.kp, self.controls.k0)
            self.thickness = material.thickness
            self.beta = (self.air.rho0 * self.air.c0) / material.Zs  # normalized surface admitance
        except:
            self.Zp = []
            self.kp = []
            # self.thick = material.thickness
            self.cp = []
            self.rhop = []
            self.m = []
            self.n = []
            self.thickness = []
            self.beta = []  # normalized surface admitance
        ## sources / receivers
        self.sources = sources
        self.receivers = receivers
        ## quantities
        self.pres_s = []
        self.uz_s = []
        self.singularities_r = []
        self.singularities_i = []

    def determine_singularities(self, ):
        '''
        A method to find the singulatities in the integrands. They are independent of source-receiver configuration,
        and the same for pressure and particle velocity
        '''
        bar = tqdm(total = len(self.controls.k0), desc = 'Finding singularities')
        for jf, k0 in enumerate(self.controls.k0):
            # the denominator
            fs_den = lambda s: (np.sqrt(s**2-1+0j))+self.m[jf]*(np.sqrt(s**2-self.n[jf]**2+0j))*\
                np.tanh(k0*self.thickness*(np.sqrt(s**2-self.n[jf]**2+0j)))
            fs_den_r = lambda s: np.real(fs_den(s))
            fs_den_i = lambda s: np.imag(fs_den(s))
            sol_r = optimize.root_scalar(fs_den_r,x0 = 1, x1 = 1-self.limits[1], method='secant')
            sol_i = optimize.root_scalar(fs_den_i,x0 = 1, x1 = 1-self.limits[1], method='secant')
            self.singularities_r.append(sol_r.root)
            self.singularities_i.append(sol_i.root)
            bar.update(1)
        bar.close()

    def p_nlr(self, upper_int_limit = 10):
        '''
        This method calculates the sound pressure spectrum for all sources and receivers
        Inputs:
            upper_int_limit (default 20) - upper integral limit for truncation
        Outputs:
            pres_s - this is an array of objects. Inside each object there is a
            (N_rec x N_freq) matrix. Each line of the matrix is a spectrum of a sound
            pressure for a receiver. Each column is a set of sound pressures measured
            by the receivers for a given frequency
        '''
        # self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
            bar = tqdm(total = len(self.controls.k0)*self.receivers.coord.shape[0], desc = 'Processing sound pressure (NLR)')
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1])**2)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                # setup progressbar
                #print('Calculate sound pressure for source {} and receiver {}'.format(js+1, jrec+1))
                
                # pres = []
                for jf, k0 in enumerate(self.controls.k0):
                    # integrand
                    fs = lambda s: ((2*np.exp(-k0*(np.sqrt(s**2-1+0j))*(hs+zr)))*\
                        k0*s*sp.jv(0,k0*s*r))/\
                        ((np.sqrt(s**2-1+0j))+self.m[jf]*(np.sqrt(s**2-self.n[jf]**2+0j))*\
                        np.tanh(k0*self.thickness*(np.sqrt(s**2-self.n[jf]**2+0j))))
                    fs_r = lambda s: np.real(fs(s))
                    fs_i = lambda s: np.imag(fs(s))
                    # Integrate
                    if not self.singularities_r: # in case singularities were not computed
                        Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit)
                        Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit)
                    else:
                        Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit,
                            limit = int(len(self.limits)+1), points = self.singularities_r[jf])
                        Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit,
                            limit = int(len(self.limits)+1), points = self.singularities_i[jf])
                    # Iq_real = qp(fs_r, [0.0, upper_int_limit])
                    # Iq_imag = qp(fs_i, [0.0, upper_int_limit])
                    I_nlr = Iq_real[0] + 1j * Iq_imag[0]
                    pres_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1) - (np.exp(-1j * k0 * r2) / r2) + I_nlr
                    bar.update(1)
            bar.close()
            self.pres_s.append(pres_rec)

    def uz_nlr(self, upper_int_limit = 10):
        '''
        This method calculates the sound particle velocity spectrum (z-dir) for all sources and receivers
        Inputs:
            upper_int_limit (default 10) - upper integral limit for truncation
        Outputs:
            uz_s - this is an array of objects. Inside each object there is a
            (N_rec x N_freq) matrix. Each line of the matrix is a spectrum of a particle
            velocity (z-dir) for a receiver. Each column is a set of particle velocity (z-dir)
            measured by the receivers for a given frequency
        '''
        # setup progressbar
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            uz_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = complex)
            bar = tqdm(total = len(self.controls.k0)*self.receivers.coord.shape[0], 
                       desc = 'Processing particle velocity z-dir (NLR)')
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((s_coord[0] - r_coord[0])**2.0 + (s_coord[1] - r_coord[1])**2.0)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                #print('Calculate particle vel. (z-dir) for source {} and receiver {}'.format(js+1, jrec+1))
                for jf, k0 in enumerate(self.controls.k0):
                    # fs = lambda s: ((2*((s**2-1)**0.5)*np.exp(-k0*((s**2-1)**0.5)*(hs+zr)))*k0*s*sp.jv(0,k0*s*r))/\
                    #     (((s**2-1)**0.5)+self.m[jf]*((s**2-self.n[jf]**2)**0.5)*\
                    #     np.tanh(k0*self.material.thickness*((s**2-self.n[jf]**2)**0.5)))

                    fs = lambda s: ((2*(np.sqrt(s**2-1+0j))*np.exp(-k0*(np.sqrt(s**2-1+0j))*(hs+zr)))*k0*s*sp.jv(0,k0*s*r))/\
                        ((np.sqrt(s**2-1+0j))+self.m[jf]*(np.sqrt(s**2-self.n[jf]**2+0j))*\
                        np.tanh(k0*self.thickness*(np.sqrt(s**2-self.n[jf]**2+0j))))
                    fs_r = lambda s: np.real(fs(s))
                    fs_i = lambda s: np.imag(fs(s))
                    # Integrate
                    if not self.singularities_r: # in case singularities were not computed
                        Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit)
                        Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit)
                    else:
                        Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit,
                            limit = int(len(self.limits)+1), points = self.singularities_r[jf])
                        Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit,
                            limit = int(len(self.limits)+1), points = self.singularities_i[jf])
                    Iq = Iq_real[0] + 1j * Iq_imag[0]
                    # particle velocity
                    uz_rec[jrec, jf] = ((np.exp(-1j * k0 * r1) / r1)*\
                        (1 + (1 / (1j * k0 * r1)))* ((hs - zr)/r1)+\
                        (np.exp(-1j * k0 * r2) / r2) *\
                        (1 + (1 / (1j * k0 * r2))) * ((hs + zr)/r2) -(1/1j)*Iq)

                    bar.update(1)
            bar.close()
            self.uz_s.append(uz_rec)

    def p_mult(self, upper_int_limit = 10, randomize = False, amp_min = 0.0002, amp_max = 20):
        '''
        This method calculates the sound pressure spectrum for a distribution of sources at all receivers.
        It considers that the integration can be done once by considering a sumation of the contributions of
        all sound sources in the integrand.
        Inputs:
            upper_int_limit (default 20) - upper integral limit for truncation
            randomize (default False) - boolean - wether each sound source has a random amplitude
        Outputs:
            pres_s - this is an array of objects. Inside each object there is a
            (N_rec x N_freq) matrix. Each line of the matrix is a spectrum of a sound
            pressure for a receiver. Each column is a set of sound pressures measured
            by the receivers for a given frequency
        '''
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
            bar = tqdm(total = len(self.controls.k0), desc = 'Processing sound pressure (NLR)')
            # seed randomizition
            np.random.seed(0)
            self.q = np.zeros((len(self.controls.freq), len(self.sources.coord)), dtype = complex)
            for jf, k0 in enumerate(self.controls.k0):
                # randomization of pressure
                if randomize:
                    amp = np.random.uniform(low = amp_min, high = amp_max, size = len(self.sources.coord))
                    phase = np.random.uniform(low = 0, high = 2*np.pi, size = len(self.sources.coord))
                    q = amp * np.exp(1j*phase)
                    self.q[jf,:] = q
                else:
                    q = np.ones(len(self.sources.coord))
                # integrand
                fs = lambda s: np.sum(q*(2*np.exp(-k0*(np.sqrt(s**2-1+0j))*(hs+zr)))*\
                    k0*s*sp.jv(0,k0*s*r))/\
                    ((np.sqrt(s**2-1+0j))+self.m[jf]*(np.sqrt(s**2-self.n[jf]**2+0j))*\
                    np.tanh(k0*self.thickness*(np.sqrt(s**2-self.n[jf]**2+0j))))
                fs_r = lambda s: np.real(fs(s))
                fs_i = lambda s: np.imag(fs(s))
                # Integrate
                if not self.singularities_r: # in case singularities were not computed
                    Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit)
                    Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit)
                else:
                    Iq_real = integrate.quad(fs_r, 0.0, upper_int_limit,
                        limit = int(len(self.limits)+1), points = self.singularities_r[jf])
                    Iq_imag = integrate.quad(fs_i, 0.0, upper_int_limit,
                        limit = int(len(self.limits)+1), points = self.singularities_i[jf])
                # Iq_real = qp(fs_r, [0.0, upper_int_limit])
                # Iq_imag = qp(fs_i, [0.0, upper_int_limit])
                I_nlr = Iq_real[0] + 1j * Iq_imag[0]
                pres_rec[jrec, jf] = (np.sum(q * (np.exp(-1j * k0 * r1) / r1 - np.exp(-1j * k0 * r2) / r2)) + I_nlr)
                bar.update(1)
            bar.close()
        self.pres_s.append(pres_rec)

    def plot_scene(self, vsam_size = 3):
        '''
        a simple plot of the scene using matplotlib - not redered
        '''
        fig = plt.figure()
        ax = plt.axes(projection ="3d")
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
            linewidths=1, alpha=0.5, edgecolor = 'gray')
        collection.set_facecolor('silver')
        ax.add_collection3d(collection)

        ax.scatter(self.sources.coord[:,0], self.sources.coord[:,1],
                   self.sources.coord[:,2], marker = "o", s=100, 
                   color='red', label = "Source")
        ax.scatter(self.receivers.coord[:,0], self.receivers.coord[:,1],
                   self.receivers.coord[:,2], color='blue',  marker = "o", label = "Receivers")
        ax.legend()
        ax.set_xlabel('X axis')
        plt.xticks([], [])
        ax.set_ylabel('Y axis')
        plt.yticks([], [])
        ax.set_zlabel('Z axis')
        # ax.grid(linestyle = ' ', which='none')
        ax.set_xlim((-vsam_size/2, vsam_size/2))
        ax.set_ylim((-vsam_size/2, vsam_size/2))
        # ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        ax.set_zlim((0, 1.2*np.amax(np.linalg.norm(self.sources.coord[0][2]))))
        ax.set_zticks((0, 1.2*np.amax(np.linalg.norm(self.sources.coord))))
        #ax.view_init(elev=5, azim=-55)
        # ax.invert_zaxis()
        plt.show() # show plot

    def save(self, filename = 'qdt', path = ''):
        """ To save the decomposition object as pickle
        """
        ut_is.save(self, filename = filename, path = path)

    def load(self, filename = 'qdt', path = ''):
        """ To load the decomposition object as pickle

        You can instantiate an empty object of the class and load a saved one.
        It will overwrite the empty object.
        """
        ut_is.load(self, filename = filename, path = path)
        
    def add_noise(self, snr = 30, uncorr = False):
        """ Add gaussian noise to the simulated data.

        The function is used to add noise to the pressure and particle velocity data.
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
        try:
            signal_u = self.uz_s[0]
        except:
            signal_u = np.zeros(1)
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
            if signal_u.any() != 0:
                signalPower_lin_u = (np.abs(np.mean(signal_u, axis=0))/np.sqrt(2))**2
                signalPower_dB_u = 10 * np.log10(signalPower_lin_u)
                noisePower_dB_u = signalPower_dB_u - snr
                noisePower_lin_u = 10 ** (noisePower_dB_u/10)
        np.random.seed(0)
        noise = np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape)
        # noise = 2*np.sqrt(noisePower_lin)*\
        #     (np.random.randn(signal.shape[0], signal.shape[1]) + 1j*np.random.randn(signal.shape[0], signal.shape[1]))
        self.pres_s[0] = signal + noise
        if signal_u.any() != 0:
            # print('Adding noise to particle velocity')
            noise_u = np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape)
            self.uz_s[0] = signal_u + noise_u