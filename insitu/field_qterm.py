import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar
import pickle
from controlsair import plot_spk

class LocallyReactiveInfSph(object):
    '''
    A class to calculate the sound pressure and particle velocity
    using the q-term formulation (exact for spherical waves on locally reactive and
    infinite samples)
    The inputs are the objects: air, controls, material, sources, receivers
    '''
    def __init__(self, air, controls, material, sources, receivers):
        self.air = air
        self.controls = controls
        self.material = material
        self.sources = sources
        self.receivers = receivers
        self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance
        self.pres_s = []
        self.uz_s = []
        # print(self.beta)
        # self.beta = (self.air.rho0 * self.air.c0) / self.material.Zs  # normalized surface admitance

        # self.c0 = c0
        # self.rho0 = rho0
        # self.freq = freq
        # self.w = 2 * np.pi * freq
        # self.k0 = self.w / c0
        # self.Zs = Zs / (rho0 * c0)
        # self.beta = 1 / self.Zs  # normalized surface admitance
        # self.h_s = h_s      # source height
        # self.r_r = r_r      # horizontal distance source-receiver
        # self.z_r = z_r      # receiver height
        # self.r_1 = (r_r ** 2 + (h_s - z_r) ** 2) ** 0.5
        # self.r_2 = (r_r ** 2 + (h_s + z_r) ** 2) ** 0.5

    def p_loc(self, upper_int_limit = 20):
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
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = ((s_coord[0] - r_coord[0])**2 + (s_coord[1] - r_coord[1])**2)**0.5 # horizontal distance source-receiver
                zr = r_coord[2]  # receiver height
                r1 = (r ** 2 + (hs - zr) ** 2) ** 0.5
                r2 = (r ** 2 + (hs + zr) ** 2) ** 0.5
                # setup progressbar
                print('Calculate sound pressure for source {} and receiver {}'.format(js+1, jrec+1))
                bar = ChargingBar('Processing sound pressure (q-term)', max=len(self.controls.k0), suffix='%(percent)d%%')
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
                    bar.next()
                bar.finish()
            self.pres_s.append(pres_rec)

    def uz_loc(self, upper_int_limit = 20):
        '''
        This method calculates the sound particle velocity spectrum (z-dir) for all sources and receivers
        Inputs:
            upper_int_limit (default 20) - upper integral limit for truncation
        Outputs:
            uz_s - this is an array of objects. Inside each object there is a
            (N_rec x N_freq) matrix. Each line of the matrix is a spectrum of a particle
            velocity (z-dir) for a receiver. Each column is a set of particle velocity (z-dir)
            measured by the receivers for a given frequency
        '''
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
                bar = ChargingBar('Processing particle velocity z-dir (q-term)',
                    max=len(self.controls.k0), suffix='%(percent)d%%')
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
                    bar.next()
                bar.finish()
            self.uz_s.append(uz_rec)

        # uz = []
        # for jf, k0 in enumerate(self.k0):
        #     f_qr = lambda q: np.real(((np.exp(-q * k0 * self.beta[jf])) *
        #         ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
        #         ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
        #         ((self.h_s + self.z_r - 1j*q) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
        #         (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
        #     f_qi = lambda q: np.imag(((np.exp(-q * k0 * self.beta[jf])) *
        #         ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
        #         ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
        #         ((self.h_s + self.z_r - 1j*q) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
        #         (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
        #     Iq_real = integrate.quad(f_qr, 0.0, 20.0)
        #     Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
        #     Iq = Iq_real[0] + 1j * Iq_imag[0]
        #     uz.append((np.exp(-1j * k0 * self.r_1) / self.r_1) *
        #         (1 + (1 / (1j * k0 * self.r_1))) * ((self.h_s - self.r_r)/self.r_1) -
        #         (np.exp(-1j * k0 * self.r_2) / self.r_2) *
        #         (1 + (1 / (1j * k0 * self.r_2))) * ((self.h_s + self.r_r)/self.r_2) +
        #         2 * k0 * self.beta[jf] * Iq)
        #     # Progress bar stuff
        #     bar.next()
        # bar.finish()
        # self.uz = np.array(uz, dtype = np.csingle)
        # return self.uz

    def p_mult(self, upper_int_limit = 10, randomize = False, amp_min = 0.0002, amp_max = 20):
        '''
        This method calculates the sound pressure spectrum for a distribution of sources at all receivers.
        It considers that the integration can be done once by considering a sumation of the contributions of
        all sound sources in the integrand.
        Inputs:
            upper_int_limit (default 20) - upper integral limit for truncation
            randomize (default False) - boolean - wether each sound source has a random amplitude
            amp_min (default 0.0002 Pa or 20 dB) - minimum amplitude of sound sources
            amp_max (default 20 Pa or 120 dB) - maximum amplitude of sound sources
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
            bar = ChargingBar('Processing sound pressure (NLR)', max=len(self.controls.k0), suffix='%(percent)d%%')
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
                bar.next()
            bar.finish()
        self.pres_s.append(pres_rec)

    # def ur_loc(self):
    #     # setup progressbar
    #     bar = ChargingBar('Processing particle velocity r-dir (q-term)', max=len(self.k0), suffix='%(percent)d%%')
    #     ur = []
    #     for jf, k0 in enumerate(self.k0):
    #         f_qr = lambda q: np.real(((np.exp(-q * k0 * self.beta[jf])) *
    #             ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
    #             ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
    #             ((self.r_r) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
    #             (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
    #         f_qi = lambda q: np.imag(((np.exp(-q * k0 * self.beta[jf])) *
    #             ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
    #             ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
    #             ((self.r_r) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
    #             (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
    #         Iq_real = integrate.quad(f_qr, 0.0, 20.0)
    #         Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
    #         Iq = Iq_real[0] + 1j * Iq_imag[0]
    #         ur.append((np.exp(-1j * k0 * self.r_1) / self.r_1) *
    #             (1 + (1 / (1j * k0 * self.r_1))) * ((- self.r_r)/self.r_1) +
    #             (np.exp(-1j * k0 * self.r_2) / self.r_2) *
    #             (1 + (1 / (1j * k0 * self.r_2))) * ((-self.r_r)/self.r_2) +
    #             2 * k0 * self.beta[jf] * Iq)
    #         # Progress bar stuff
    #         bar.next()
    #     bar.finish()
    #     self.ur = np.array(ur, dtype = np.csingle)
    #     return self.ur

    def plot_pres(self):
        '''
        Method to plot the spectrum of the sound pressure
        '''
        plot_spk(self.controls.freq, self.pres_s, ref = 20e-6)

    def plot_uz(self):
        '''
        Method to plot the spectrum of the particle velocity in zdir
        '''
        plot_spk(self.controls.freq, self.uz_s, ref = 5e-8)


    # def plot_ur(self):
    #     # plt.figure(2)
    #     figur, axs = plt.subplots(2,1)
    #     axs[0].semilogx(self.freq, 20 * np.log10(np.abs(self.ur) / 50e-9), 'k-', label='ur q-term')
    #     axs[0].grid(linestyle = '--', which='both')
    #     axs[0].legend(loc = 'upper right')
    #     # axs[0].set(xlabel = 'Frequency [Hz]')
    #     axs[0].set(ylabel = '|u_r(f)| [dB]')
    #     axs[1].semilogx(self.freq, np.angle(self.pres), 'k-', label='pres q-term')
    #     axs[1].grid(linestyle = '--', which='both')
    #     axs[1].set(xlabel = 'Frequency [Hz]')
    #     axs[1].set(ylabel = 'phase [-]')
    #     plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
    #     xticklabels=['50', '100', '500', '1000', '5000', '10000'])
    #     plt.setp(axs, xlim=(0.8 * self.freq[0], 1.2*self.freq[-1]))
    #     # plt.show()

    def plot_scene(self, vsam_size = 50):
        '''
        a simple plot of the scene using matplotlib - not redered
        '''
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
        '''
        This method is used to save the simulation object
        '''
        self.path_filename = path + filename + '.pkl'
        f = open(self.path_filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename = 'my_qterm', path = '/home/eric/dev/insitu/data/'):
        '''
        This method is used to load a simulation object. You build a empty object
        of the class and load a saved one. It will overwrite the empty one.
        '''
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

# class LocallyReactiveInfSph(object):
#     '''
#     A class to calculate the sound pressure and particle velocity
#     using the q-term formulation (exact for spherical waves on locally reactive and
#     infinite samples)
#     '''
#     def __init__(self, freq, Zs, h_s, r_r, z_r, c0, rho0):
#         self.c0 = c0
#         self.rho0 = rho0
#         self.freq = freq
#         self.w = 2 * np.pi * freq
#         self.k0 = self.w / c0
#         self.Zs = Zs / (rho0 * c0)
#         self.beta = 1 / self.Zs  # normalized surface admitance
#         self.h_s = h_s      # source height
#         self.r_r = r_r      # horizontal distance source-receiver
#         self.z_r = z_r      # receiver height
#         self.r_1 = (r_r ** 2 + (h_s - z_r) ** 2) ** 0.5
#         self.r_2 = (r_r ** 2 + (h_s + z_r) ** 2) ** 0.5

#     def p_loc(self):
#         # setup progressbar
#         bar = ChargingBar('Processing sound pressure (q-term)', max=len(self.k0), suffix='%(percent)d%%')
#         pres = []
#         for jf, k0 in enumerate(self.k0):
#             f_qr = lambda q: np.real((np.exp(-q * k0 * self.beta[jf])) *
#                 ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
#                 ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5)))
#             f_qi = lambda q: np.imag((np.exp(-q * k0 * self.beta[jf])) *
#                 ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
#                 ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5)))
#             Iq_real = integrate.quad(f_qr, 0.0, 20.0)
#             Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
#             Iq = Iq_real[0] + 1j * Iq_imag[0]
            # pres.append((np.exp(-1j * k0 * self.r_1) / self.r_1) +
            #     (np.exp(-1j * k0 * self.r_2) / self.r_2) - 2 * k0 * self.beta[jf] * Iq)
#             # Progress bar stuff
#             bar.next()
#         bar.finish()
#         self.pres = np.array(pres, dtype = np.csingle)
#         return self.pres

#     def uz_loc(self):
#         # setup progressbar
#         bar = ChargingBar('Processing particle velocity z-dir (q-term)', max=len(self.k0), suffix='%(percent)d%%')
#         uz = []
#         for jf, k0 in enumerate(self.k0):
#             f_qr = lambda q: np.real(((np.exp(-q * k0 * self.beta[jf])) *
#                 ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
#                 ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
#                 ((self.h_s + self.z_r - 1j*q) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
#                 (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
#             f_qi = lambda q: np.imag(((np.exp(-q * k0 * self.beta[jf])) *
#                 ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
#                 ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
#                 ((self.h_s + self.z_r - 1j*q) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
#                 (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
#             Iq_real = integrate.quad(f_qr, 0.0, 20.0)
#             Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
#             Iq = Iq_real[0] + 1j * Iq_imag[0]
#             uz.append((np.exp(-1j * k0 * self.r_1) / self.r_1) *
#                 (1 + (1 / (1j * k0 * self.r_1))) * ((self.h_s - self.r_r)/self.r_1) -
#                 (np.exp(-1j * k0 * self.r_2) / self.r_2) *
#                 (1 + (1 / (1j * k0 * self.r_2))) * ((self.h_s + self.r_r)/self.r_2) +
#                 2 * k0 * self.beta[jf] * Iq)
#             # Progress bar stuff
#             bar.next()
#         bar.finish()
#         self.uz = np.array(uz, dtype = np.csingle)
#         return self.uz

#     def ur_loc(self):
#         # setup progressbar
#         bar = ChargingBar('Processing particle velocity r-dir (q-term)', max=len(self.k0), suffix='%(percent)d%%')
#         ur = []
#         for jf, k0 in enumerate(self.k0):
#             f_qr = lambda q: np.real(((np.exp(-q * k0 * self.beta[jf])) *
#                 ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
#                 ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
#                 ((self.r_r) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
#                 (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
#             f_qi = lambda q: np.imag(((np.exp(-q * k0 * self.beta[jf])) *
#                 ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
#                 ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
#                 ((self.r_r) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
#                 (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
#             Iq_real = integrate.quad(f_qr, 0.0, 20.0)
#             Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
#             Iq = Iq_real[0] + 1j * Iq_imag[0]
#             ur.append((np.exp(-1j * k0 * self.r_1) / self.r_1) *
#                 (1 + (1 / (1j * k0 * self.r_1))) * ((- self.r_r)/self.r_1) +
#                 (np.exp(-1j * k0 * self.r_2) / self.r_2) *
#                 (1 + (1 / (1j * k0 * self.r_2))) * ((-self.r_r)/self.r_2) +
#                 2 * k0 * self.beta[jf] * Iq)
#             # Progress bar stuff
#             bar.next()
#         bar.finish()
#         self.ur = np.array(ur, dtype = np.csingle)
#         return self.ur

#     def plot_pres(self):
#         # plt.figure(1)
#         figp, axs = plt.subplots(2,1)
#         axs[0].semilogx(self.freq, 20 * np.log10(np.abs(self.pres) / 20e-6), 'k-', label='pres q-term')
#         axs[0].grid(linestyle = '--', which='both')
#         axs[0].legend(loc = 'upper right')
#         # axs[0].set(xlabel = 'Frequency [Hz]')
#         axs[0].set(ylabel = '|p(f)| [dB]')
#         axs[1].semilogx(self.freq, np.angle(self.pres), 'k-', label='pres q-term')
#         axs[1].grid(linestyle = '--', which='both')
#         axs[1].set(xlabel = 'Frequency [Hz]')
#         axs[1].set(ylabel = 'phase [-]')
#         plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
#         xticklabels=['50', '100', '500', '1000', '5000', '10000'])
#         plt.setp(axs, xlim=(0.8 * self.freq[0], 1.2*self.freq[-1]))
#         # f.show()

#     def plot_uz(self):
#         # plt.figure(2)
#         figuz, axs = plt.subplots(2,1)
#         axs[0].semilogx(self.freq, 20 * np.log10(np.abs(self.uz) / 50e-9), 'k-', label='uz q-term')
#         axs[0].grid(linestyle = '--', which='both')
#         axs[0].legend(loc = 'upper right')
#         # axs[0].set(xlabel = 'Frequency [Hz]')
#         axs[0].set(ylabel = '|u_z(f)| [dB]')
#         axs[1].semilogx(self.freq, np.angle(self.pres), 'k-', label='pres q-term')
#         axs[1].grid(linestyle = '--', which='both')
#         axs[1].set(xlabel = 'Frequency [Hz]')
#         axs[1].set(ylabel = 'phase [-]')
#         plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
#         xticklabels=['50', '100', '500', '1000', '5000', '10000'])
#         plt.setp(axs, xlim=(0.8 * self.freq[0], 1.2*self.freq[-1]))
#         # plt.show()

#     def plot_ur(self):
#         # plt.figure(2)
#         figur, axs = plt.subplots(2,1)
#         axs[0].semilogx(self.freq, 20 * np.log10(np.abs(self.ur) / 50e-9), 'k-', label='ur q-term')
#         axs[0].grid(linestyle = '--', which='both')
#         axs[0].legend(loc = 'upper right')
#         # axs[0].set(xlabel = 'Frequency [Hz]')
#         axs[0].set(ylabel = '|u_r(f)| [dB]')
#         axs[1].semilogx(self.freq, np.angle(self.pres), 'k-', label='pres q-term')
#         axs[1].grid(linestyle = '--', which='both')
#         axs[1].set(xlabel = 'Frequency [Hz]')
#         axs[1].set(ylabel = 'phase [-]')
#         plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
#         xticklabels=['50', '100', '500', '1000', '5000', '10000'])
#         plt.setp(axs, xlim=(0.8 * self.freq[0], 1.2*self.freq[-1]))
#         # plt.show()
