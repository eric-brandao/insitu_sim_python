import numpy as np
import matplotlib.pyplot as plt
# from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar

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
        self.pres_s = []
        for js, s_coord in enumerate(self.sources.coord):
            hs = s_coord[2] # source height
            pres_rec = np.zeros((self.receivers.coord.shape[0], len(self.controls.freq)), dtype = np.csingle)
            for jrec, r_coord in enumerate(self.receivers.coord):
                r = (r_coord[0]**2 + r_coord[1]**2)**0.5 # horizontal distance source-receiver
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
                    Iq = Iq_real[0] + 1j * Iq_imag[0]
                    pres_rec[jrec, jf] = (np.exp(-1j * k0 * r1) / r1) + (np.exp(-1j * k0 * r2) / r2) - 2 * k0 * self.beta[jf] * Iq
                    # pres.append((np.exp(-1j * k0 * r1) / r1) +
                    #     (np.exp(-1j * k0 * r2) / r2) - 2 * k0 * self.beta[jf] * Iq)
                    # Progress bar stuff
                    bar.next()
                bar.finish()
            self.pres_s.append(pres_rec)

    # def uz_loc(self):
    #     # setup progressbar
    #     bar = ChargingBar('Processing particle velocity z-dir (q-term)', max=len(self.k0), suffix='%(percent)d%%')
    #     uz = []
    #     for jf, k0 in enumerate(self.k0):
    #         f_qr = lambda q: np.real(((np.exp(-q * k0 * self.beta[jf])) *
    #             ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
    #             ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
    #             ((self.h_s + self.z_r - 1j*q) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
    #             (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
    #         f_qi = lambda q: np.imag(((np.exp(-q * k0 * self.beta[jf])) *
    #             ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
    #             ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
    #             ((self.h_s + self.z_r - 1j*q) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
    #             (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
    #         Iq_real = integrate.quad(f_qr, 0.0, 20.0)
    #         Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
    #         Iq = Iq_real[0] + 1j * Iq_imag[0]
    #         uz.append((np.exp(-1j * k0 * self.r_1) / self.r_1) *
    #             (1 + (1 / (1j * k0 * self.r_1))) * ((self.h_s - self.r_r)/self.r_1) -
    #             (np.exp(-1j * k0 * self.r_2) / self.r_2) *
    #             (1 + (1 / (1j * k0 * self.r_2))) * ((self.h_s + self.r_r)/self.r_2) +
    #             2 * k0 * self.beta[jf] * Iq)
    #         # Progress bar stuff
    #         bar.next()
    #     bar.finish()
    #     self.uz = np.array(uz, dtype = np.csingle)
    #     return self.uz

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
        # plt.figure(1)
        figp, axs = plt.subplots(2,1)
        for js, p_s_mtx in enumerate(self.pres_s):
            for jrec, p_spk in enumerate(p_s_mtx):
                leg = 'source ' + str(js+1) + ' receiver ' + str(jrec+1)
                axs[0].semilogx(self.controls.freq, 20 * np.log10(np.abs(p_spk) / 20e-6), label = leg)
                # axs[0].semilogx(self.controls.freq, np.abs(p_spk), label = leg)
        axs[0].grid(linestyle = '--', which='both')
        axs[0].legend(loc = 'best')
        # axs[0].set(xlabel = 'Frequency [Hz]')
        axs[0].set(ylabel = '|p(f)| [dB]')
        for p_s_mtx in self.pres_s:
            for p_ph in p_s_mtx:
                axs[1].semilogx(self.controls.freq, np.angle(p_ph), label=leg)
        axs[1].grid(linestyle = '--', which='both')
        axs[1].set(xlabel = 'Frequency [Hz]')
        axs[1].set(ylabel = 'phase [-]')
        plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
        xticklabels=['50', '100', '500', '1000', '5000', '10000'])
        plt.setp(axs, xlim=(0.8 * self.controls.freq[0], 1.2*self.controls.freq[-1]))
        plt.show()

    # def plot_uz(self):
    #     # plt.figure(2)
    #     figuz, axs = plt.subplots(2,1)
    #     axs[0].semilogx(self.freq, 20 * np.log10(np.abs(self.uz) / 50e-9), 'k-', label='uz q-term')
    #     axs[0].grid(linestyle = '--', which='both')
    #     axs[0].legend(loc = 'upper right')
    #     # axs[0].set(xlabel = 'Frequency [Hz]')
    #     axs[0].set(ylabel = '|u_z(f)| [dB]')
    #     axs[1].semilogx(self.freq, np.angle(self.pres), 'k-', label='pres q-term')
    #     axs[1].grid(linestyle = '--', which='both')
    #     axs[1].set(xlabel = 'Frequency [Hz]')
    #     axs[1].set(ylabel = 'phase [-]')
    #     plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
    #     xticklabels=['50', '100', '500', '1000', '5000', '10000'])
    #     plt.setp(axs, xlim=(0.8 * self.freq[0], 1.2*self.freq[-1]))
    #     # plt.show()

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
