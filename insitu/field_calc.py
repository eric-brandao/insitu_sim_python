import numpy as np
import matplotlib.pyplot as plt
import toml
from insitu.controlsair import load_cfg
import scipy.integrate as integrate
import scipy as spy
import time
import sys
from progress.bar import Bar, IncrementalBar, FillingCirclesBar, ChargingBar

class LocallyReactive():
    '''
    A class to calculate the sound pressure and particle velocity
    using the q-term formulation (exact for locaaly reactive and
    infinite samples)
    '''
    def __init__(self, freq, Zs, h_s, r_r, z_r, c0, rho0):
        self.c0 = c0
        self.rho0 = rho0
        self.freq = freq
        self.w = 2 * np.pi * freq
        self.k0 = self.w / c0
        self.Zs = Zs / (rho0 * c0)
        self.beta = 1 / self.Zs  # normalized surface admitance
        self.h_s = h_s      # source height
        self.r_r = r_r      # horizontal distance source-receiver
        self.z_r = z_r      # receiver height
        self.r_1 = (r_r ** 2 + (h_s - z_r) ** 2) ** 0.5
        self.r_2 = (r_r ** 2 + (h_s + z_r) ** 2) ** 0.5

    def p_loc(self):
        # setup progressbar
        bar = ChargingBar('Processing sound pressure (q-term)', max=len(self.k0), suffix='%(percent)d%%')
        pres = []
        for jf, k0 in enumerate(self.k0):
            f_qr = lambda q: np.real((np.exp(-q * k0 * self.beta[jf])) *
                ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
                ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5)))
            f_qi = lambda q: np.imag((np.exp(-q * k0 * self.beta[jf])) *
                ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
                ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5)))
            Iq_real = integrate.quad(f_qr, 0.0, 20.0)
            Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
            Iq = Iq_real[0] + 1j * Iq_imag[0]
            pres.append((np.exp(-1j * k0 * self.r_1) / self.r_1) +
                (np.exp(-1j * k0 * self.r_2) / self.r_2) - 2 * k0 * self.beta[jf] * Iq)
            # Progress bar stuff
            bar.next()
        bar.finish()
        self.pres = np.array(pres, dtype = np.csingle)
        return self.pres

    def uz_loc(self):
        # setup progressbar
        bar = ChargingBar('Processing particle velocity z-dir (q-term)', max=len(self.k0), suffix='%(percent)d%%')
        uz = []
        for jf, k0 in enumerate(self.k0):
            f_qr = lambda q: np.real(((np.exp(-q * k0 * self.beta[jf])) *
                ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
                ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
                ((self.h_s + self.z_r - 1j*q) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
                (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
            f_qi = lambda q: np.imag(((np.exp(-q * k0 * self.beta[jf])) *
                ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
                ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
                ((self.h_s + self.z_r - 1j*q) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
                (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
            Iq_real = integrate.quad(f_qr, 0.0, 20.0)
            Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
            Iq = Iq_real[0] + 1j * Iq_imag[0]
            uz.append((np.exp(-1j * k0 * self.r_1) / self.r_1) *
                (1 + (1 / (1j * k0 * self.r_1))) * ((self.h_s - self.r_r)/self.r_1) -
                (np.exp(-1j * k0 * self.r_2) / self.r_2) *
                (1 + (1 / (1j * k0 * self.r_2))) * ((self.h_s + self.r_r)/self.r_2) +
                2 * k0 * self.beta[jf] * Iq)
            # Progress bar stuff
            bar.next()
        bar.finish()
        self.uz = np.array(uz, dtype = np.csingle)
        return self.uz

    def ur_loc(self):
        # setup progressbar
        bar = ChargingBar('Processing particle velocity r-dir (q-term)', max=len(self.k0), suffix='%(percent)d%%')
        ur = []
        for jf, k0 in enumerate(self.k0):
            f_qr = lambda q: np.real(((np.exp(-q * k0 * self.beta[jf])) *
                ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
                ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
                ((self.r_r) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
                (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
            f_qi = lambda q: np.imag(((np.exp(-q * k0 * self.beta[jf])) *
                ((np.exp(-1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5)) /
                ((self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2) ** 0.5))) *
                ((self.r_r) / (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5) *
                (1 + (1 / (1j * k0 * (self.r_r**2 + (self.h_s + self.z_r - 1j*q)**2)**0.5))))
            Iq_real = integrate.quad(f_qr, 0.0, 20.0)
            Iq_imag = integrate.quad(f_qi, 0.0, 20.0)
            Iq = Iq_real[0] + 1j * Iq_imag[0]
            ur.append((np.exp(-1j * k0 * self.r_1) / self.r_1) *
                (1 + (1 / (1j * k0 * self.r_1))) * ((- self.r_r)/self.r_1) +
                (np.exp(-1j * k0 * self.r_2) / self.r_2) *
                (1 + (1 / (1j * k0 * self.r_2))) * ((-self.r_r)/self.r_2) +
                2 * k0 * self.beta[jf] * Iq)
            # Progress bar stuff
            bar.next()
        bar.finish()
        self.ur = np.array(ur, dtype = np.csingle)
        return self.ur

    def plot_pres(self):
        # plt.figure(1)
        figp, axs = plt.subplots(2,1)
        axs[0].semilogx(self.freq, 20 * np.log10(np.abs(self.pres) / 20e-6), 'k-', label='pres q-term')
        axs[0].grid(linestyle = '--', which='both')
        axs[0].legend(loc = 'upper right')
        # axs[0].set(xlabel = 'Frequency [Hz]')
        axs[0].set(ylabel = '|p(f)| [dB]')
        axs[1].semilogx(self.freq, np.angle(self.pres), 'k-', label='pres q-term')
        axs[1].grid(linestyle = '--', which='both')
        axs[1].set(xlabel = 'Frequency [Hz]')
        axs[1].set(ylabel = 'phase [-]')
        plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
        xticklabels=['50', '100', '500', '1000', '5000', '10000'])
        plt.setp(axs, xlim=(0.8 * self.freq[0], 1.2*self.freq[-1]))
        # f.show()

    def plot_uz(self):
        # plt.figure(2)
        figuz, axs = plt.subplots(2,1)
        axs[0].semilogx(self.freq, 20 * np.log10(np.abs(self.uz) / 50e-9), 'k-', label='uz q-term')
        axs[0].grid(linestyle = '--', which='both')
        axs[0].legend(loc = 'upper right')
        # axs[0].set(xlabel = 'Frequency [Hz]')
        axs[0].set(ylabel = '|u_z(f)| [dB]')
        axs[1].semilogx(self.freq, np.angle(self.pres), 'k-', label='pres q-term')
        axs[1].grid(linestyle = '--', which='both')
        axs[1].set(xlabel = 'Frequency [Hz]')
        axs[1].set(ylabel = 'phase [-]')
        plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
        xticklabels=['50', '100', '500', '1000', '5000', '10000'])
        plt.setp(axs, xlim=(0.8 * self.freq[0], 1.2*self.freq[-1]))
        # plt.show()

    def plot_ur(self):
        # plt.figure(2)
        figur, axs = plt.subplots(2,1)
        axs[0].semilogx(self.freq, 20 * np.log10(np.abs(self.ur) / 50e-9), 'k-', label='ur q-term')
        axs[0].grid(linestyle = '--', which='both')
        axs[0].legend(loc = 'upper right')
        # axs[0].set(xlabel = 'Frequency [Hz]')
        axs[0].set(ylabel = '|u_r(f)| [dB]')
        axs[1].semilogx(self.freq, np.angle(self.pres), 'k-', label='pres q-term')
        axs[1].grid(linestyle = '--', which='both')
        axs[1].set(xlabel = 'Frequency [Hz]')
        axs[1].set(ylabel = 'phase [-]')
        plt.setp(axs, xticks=[50, 100, 500, 1000, 5000, 10000],
        xticklabels=['50', '100', '500', '1000', '5000', '10000'])
        plt.setp(axs, xlim=(0.8 * self.freq[0], 1.2*self.freq[-1]))
        # plt.show()
